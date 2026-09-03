"""Ввод-вывод, конфиги и учёт прогонов.

Правило 5 CLAUDE.md: никаких молчаливых дефолтов — конфиг читается как есть,
отсутствующий ключ это ошибка, а не None.
Правило 6: имя весов, sha256, разрешение входа, device, precision, версия трекера
пишутся в run_manifest.json на каждом прогоне.
Правило 8: всё падает громко.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from looq import SCHEMA_VERSION

#: Значение git_sha, когда git недоступен или репозиторий пуст. Не падаем (решение
#: владельца проекта от 2026-09-03), но и не притворяемся, что версия известна.
NO_GIT = "no-git"

MANIFEST_PATH = Path("run_manifest.json")


# --------------------------------------------------------------------------- #
# Хеши и версии
# --------------------------------------------------------------------------- #

def sha256_file(path: str | os.PathLike) -> str | None:
    """sha256 файла или None, если файла нет.

    None здесь — честное "не измерено" (правило 7), а не ноль.
    """
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_sha() -> str:
    """Хеш текущего коммита или NO_GIT."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return NO_GIT
    if out.returncode != 0:
        return NO_GIT
    sha = out.stdout.strip()
    return sha if sha else NO_GIT


def git_dirty() -> bool | None:
    """True, если в рабочем дереве есть незакоммиченные изменения. None — git недоступен."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def library_versions() -> dict[str, str | None]:
    """Версии всего, что может повлиять на числа. Отсутствующее -> None (правило 7)."""
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("numpy", "cv2", "pandas", "pyarrow", "yaml", "torch",
                "ultralytics", "shapely", "scipy"):
        try:
            m = __import__(mod)
            versions[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            versions[mod] = None
    return versions


def gpu_info() -> dict[str, Any]:
    """Что реально исполняет модель. 8 ГБ VRAM — жёсткое ограничение проекта."""
    info: dict[str, Any] = {"cuda_available": False, "device_name": None,
                            "vram_total_mb": None}
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["vram_total_mb"] = int(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            )
    except Exception:
        pass
    return info


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# --------------------------------------------------------------------------- #
# Конфиги
# --------------------------------------------------------------------------- #

class ConfigError(RuntimeError):
    """Конфиг отсутствует, не парсится или в нём нет обязательного ключа."""


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Читает configs/*.yaml. Отсутствие файла — громкая ошибка, не дефолт."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(
            f"конфиг не найден: {p}. Правило 5: молчаливых дефолтов нет, "
            f"каждый порог живёт в configs/*.yaml"
        )
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(
            f"конфиг {p} должен быть YAML-словарём, получено {type(cfg).__name__}"
        )
    cfg["_config_path"] = str(p)
    cfg["_config_sha256"] = sha256_file(p)
    return cfg


def require(cfg: Mapping[str, Any], *keys: str) -> Any:
    """Достаёт вложенный ключ с понятной ошибкой вместо KeyError или None."""
    node: Any = cfg
    trail: list[str] = []
    for k in keys:
        trail.append(k)
        if not isinstance(node, Mapping) or k not in node:
            raise ConfigError(
                f"в конфиге {cfg.get('_config_path', '?')} нет обязательного ключа "
                f"{'.'.join(trail)}"
            )
        node = node[k]
    return node


# --------------------------------------------------------------------------- #
# Атомарная запись
# --------------------------------------------------------------------------- #

def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> Path:
    """Записать целиком или не записать вовсе. Полуфайлов на диске не остаётся."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp_", suffix=p.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def atomic_write_text(path: str | os.PathLike, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: str | os.PathLike, obj: Any) -> Path:
    return atomic_write_text(
        path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    )


def read_json(path: str | os.PathLike) -> Any:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"артефакт не найден: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# run_manifest.json (правило 6)
# --------------------------------------------------------------------------- #

class RunManifest:
    """Провенанс прогона одного этапа.

    Пишется в run_manifest.json в корне проекта: один файл, секция на этап,
    последний прогон этапа перетирает предыдущий. Ответ "используется YOLO"
    недопустим — в манифесте лежит имя файла весов и его sha256.
    """

    def __init__(self, stage: str, config: Mapping[str, Any],
                 path: str | os.PathLike = MANIFEST_PATH) -> None:
        if not stage:
            raise ValueError("RunManifest требует непустое имя этапа")
        self.stage = stage
        self.config = dict(config)
        self.path = Path(path)
        self._entry: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        # Убитый прогон оставляет в манифесте status="running" навсегда, и файл
        # начинает утверждать, что этап якобы выполняется. Это хуже отсутствия
        # манифеста: провенанс, который врёт о состоянии, нельзя предъявлять.
        # Помечаем такую запись как оборванную ПЕРЕД тем, как затереть её новой.
        self._mark_stale_running()
        model = self.config.get("model") or {}
        weights = model.get("weights")
        self._entry = {
            "stage": self.stage,
            "schema_version": SCHEMA_VERSION,
            "started_at": utc_now_iso(),
            "finished_at": None,
            "status": "running",
            "config_path": self.config.get("_config_path"),
            "config_sha256": self.config.get("_config_sha256"),
            "model": {
                # Правило 6. None означает "на этом этапе модели нет либо веса
                # ещё не подключены" — и это видно, а не замаскировано.
                "weights": weights,
                "weights_sha256": sha256_file(weights) if weights else None,
                "weights_present": bool(weights) and Path(str(weights)).is_file(),
                "imgsz": model.get("imgsz"),
                "device": model.get("device"),
                "precision": model.get("precision"),
                "tracker": model.get("tracker"),
                "tracker_version": model.get("tracker_version"),
            },
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "libraries": library_versions(),
            "gpu": gpu_info(),
            "notes": {},
        }
        self._flush()
        return self._entry

    def _mark_stale_running(self) -> None:
        """Прежняя запись этапа осталась в состоянии running — значит прогон убит."""
        if not self.path.is_file():
            return
        try:
            doc = read_json(self.path)
        except (json.JSONDecodeError, OSError):
            return
        prev = (doc.get("stages") or {}).get(self.stage)
        if not isinstance(prev, dict) or prev.get("status") != "running":
            return
        prev["status"] = "aborted"
        prev["finished_at"] = utc_now_iso()
        prev["error"] = ("прогон не завершился: запись осталась в состоянии running "
                         "и была помечена оборванной при следующем запуске этапа")
        history = doc.setdefault("aborted_runs", [])
        history.append({"stage": self.stage, "started_at": prev.get("started_at"),
                        "detected_at": prev["finished_at"]})
        write_json(self.path, doc)

    def note(self, key: str, value: Any) -> None:
        """Числа, которые этап предъявляет: throughput, доли, счётчики пруфов."""
        if not self._entry:
            raise RuntimeError("RunManifest.note() вызван до start()")
        self._entry["notes"][key] = value

    def finish(self, status: str, error: str | None = None) -> dict[str, Any]:
        if not self._entry:
            raise RuntimeError("RunManifest.finish() вызван до start()")
        self._entry["finished_at"] = utc_now_iso()
        self._entry["status"] = status
        if error is not None:
            self._entry["error"] = error
        self._flush()
        return self._entry

    def _flush(self) -> None:
        doc: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "stages": {}}
        if self.path.is_file():
            try:
                existing = read_json(self.path)
                if isinstance(existing, dict) and isinstance(existing.get("stages"), dict):
                    doc = existing
                    doc["schema_version"] = SCHEMA_VERSION
            except (json.JSONDecodeError, OSError):
                # Битый манифест не должен ронять прогон, но и молча теряться не должен.
                doc["previous_manifest_unreadable"] = True
        doc["stages"][self.stage] = self._entry
        write_json(self.path, doc)
