# -*- coding: utf-8 -*-
"""Обезличивание — путь, а не шаг.

Почему этот файл существует
---------------------------
Правило 9 держалось на дисциплине: обезличивание было отдельным вызовом,
который вызывающая сторона могла сделать, а могла забыть. Забыла. Замер
2026-09-06 по опубликованному на GitHub Pages ``out/dashboard.html``: 14
изображений из 51 несли резкие лица, в одном 27 лицевых кейпоинтов из 31.
Причина — ``b64_img(..., anonymise: bool = False)``: один вызов флаг
передавал, второй нет.

Тесты здесь проверяют не поведение функции, а НЕВОЗМОЖНОСТЬ обхода:

  * ``test_no_raw_image_write_outside_the_module`` — разбором синтаксиса, а
    не поиском подстроки: подстрока не видит ``from cv2 import imwrite``;
  * ``test_anonymise_has_no_off_switch`` — у публичного входа нет аргумента,
    которым его можно выключить;
  * ``test_no_sharp_faces_anywhere`` — замер на всех изображениях
    репозитория и на всех, вшитых в публикуемые страницы.
"""

from __future__ import annotations

import ast
import base64
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Единственный файл, которому разрешено кодировать и записывать изображение.
GATEKEEPER = "looq/anonymise.py"

#: Где ищем нарушителей. tests/ не входит намеренно: тесты пишут синтетику
#: во временные каталоги, и это не публикация.
SCANNED = ("looq", "scripts", "verify")

#: Вызовы, которые превращают массив пикселей в файл или в байты.
RAW_WRITERS = {
    ("cv2", "imwrite"), ("cv2", "imencode"),
    ("plt", "savefig"), ("plt", "imsave"), ("pyplot", "savefig"),
    ("matplotlib", "imsave"),
}
#: Методы, которые пишут независимо от того, на чём вызваны: fig.savefig(),
#: Image.save(). Имя объекта тут не помогает — ловим по имени метода.
RAW_METHODS = {"savefig", "imsave"}


def _py_files() -> list[Path]:
    out: list[Path] = []
    for d in SCANNED:
        out += sorted((ROOT / d).rglob("*.py"))
    return out


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


# --------------------------------------------------------------------------- #
# 1. Обход невозможен: другой точки записи в коде нет
# --------------------------------------------------------------------------- #

def test_no_raw_image_write_outside_the_module():
    """Ни один файл, кроме looq/anonymise.py, не кодирует изображение сам.

    Разбор синтаксиса, а не grep. grep по ``cv2.imwrite`` не увидел бы
    ``from cv2 import imwrite as w`` и не отличил бы вызов от упоминания в
    докстроке — а докстрока в looq/evidence.py как раз описывает историю
    этого запрета и содержит нужную подстроку.
    """
    offenders: list[str] = []
    for f in _py_files():
        if _rel(f) == GATEKEEPER:
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:                       # правило 8: не молчим
            pytest.fail(f"{_rel(f)} не разбирается: {e}")

        # from cv2 import imwrite — переименованный вызов не поймать по имени
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in ("cv2",):
                for a in node.names:
                    if a.name in ("imwrite", "imencode"):
                        offenders.append(
                            f"{_rel(f)}:{node.lineno} from cv2 import {a.name}")

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute):
                base = fn.value.id if isinstance(fn.value, ast.Name) else None
                if base is not None and (base, fn.attr) in RAW_WRITERS:
                    offenders.append(f"{_rel(f)}:{node.lineno} {base}.{fn.attr}")
                elif base is None and fn.attr in RAW_METHODS:
                    offenders.append(f"{_rel(f)}:{node.lineno} <...>.{fn.attr}")
                elif fn.attr == "save" and isinstance(fn.value, ast.Name) \
                        and fn.value.id in ("im", "img", "image", "Image"):
                    offenders.append(f"{_rel(f)}:{node.lineno} PIL {fn.value.id}.save")

    assert not offenders, (
        "изображение кодируется в обход looq.anonymise — это возвращает "
        "обезличивание из пути в шаг, который можно забыть:\n  "
        + "\n  ".join(offenders))


def test_anonymise_has_no_off_switch():
    """У публичного входа нет параметра, которым его можно выключить.

    Прежняя подпись ``b64_img(..., anonymise: bool = False)`` была честным
    флагом с честным дефолтом — и именно она пропустила лица на Pages.
    """
    from looq import anonymise as A
    import inspect

    for name in ("anonymise", "encode_image", "save_image", "data_uri",
                 "save_figure"):
        sig = inspect.signature(getattr(A, name))
        banned = [p for p in sig.parameters
                  if p in ("anonymise", "blur", "skip_anonymise", "raw",
                           "no_blur", "anonymize")]
        assert not banned, f"{name}{sig}: параметр-выключатель {banned}"

    # Шов для тестов не должен звучать из рабочего кода: подменить модель на
    # ту, что ничего не находит, — это и есть обход, только длиннее.
    # _anonymise_with разрешён render_overlay: он передаёт НАСТОЯЩИЕ модели,
    # уже загруженные им для рендера, и второй раз грузить их незачем.
    ALLOWED = {"_anonymise_with": {"scripts/render_overlay.py"}}
    leaks: list[str] = []
    for f in _py_files():
        if _rel(f) == GATEKEEPER:
            continue
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            nm = (node.func.id if isinstance(node.func, ast.Name)
                  else node.func.attr if isinstance(node.func, ast.Attribute)
                  else None)
            if nm in ("_install_models_for_tests", "_anonymise_with") \
                    and _rel(f) not in ALLOWED.get(nm, set()):
                leaks.append(f"{_rel(f)}:{node.lineno} {nm}")
    assert not leaks, f"внутренняя дверь обезличивателя вызвана из кода: {leaks}"

    src = (ROOT / GATEKEEPER).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    # encode_image обязан звать anonymise, а не «может звать»
    body = ast.dump(fns["encode_image"])
    assert "'anonymise'" in body or "anonymise" in body, (
        "encode_image перестал вызывать обезличивание")
    # save_image обязан идти через encode_image, а не кодировать сам
    assert "encode_image" in ast.dump(fns["save_image"]), (
        "save_image кодирует в обход encode_image")


# --------------------------------------------------------------------------- #
# 2. Замер: ни одного резкого лица ни на одном изображении
# --------------------------------------------------------------------------- #

#: Метрика — та же, что в test_overlay_anonymised.py, буква в букву.
FACE_KP = (0, 1, 2, 3, 4)
KP_CONF = 0.30
WIN = 15
#: Пятый процентиль резкости вокруг лицевых кейпоинтов на НЕОБЕЗЛИЧЕННЫХ
#: кадрах: даже самое размытое настоящее лицо резче этого.
SHARP_THR = 13.4

_POSE = ROOT / "models" / "yolo11m-pose.pt"
#: Страницы, которые реально уходят на GitHub Pages. report.html и replay
#: сюда не входят: они не публикуются (docs/DECISIONS.md, раздел 12).
PUBLISHED = ("out/dashboard.html", "out/benchmark.html")


def _sharp_at(gray, x, y, win: int = WIN) -> float:
    import cv2
    h, w = gray.shape
    x0, x1 = max(0, int(x) - win), min(w, int(x) + win)
    y0, y1 = max(0, int(y) - win), min(h, int(y) + win)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    return float(np.abs(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F)).mean())


def _exposed(pose, img) -> list[float]:
    """Резкости лицевых кейпоинтов, перешагнувших порог."""
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = pose.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]
    if r.keypoints is None or r.keypoints.conf is None:
        return []
    xy = np.asarray(r.keypoints.xy.cpu())
    cf = np.asarray(r.keypoints.conf.cpu())
    out = []
    for pi in range(cf.shape[0]):
        for j in FACE_KP:
            if cf[pi, j] >= KP_CONF:
                s = _sharp_at(gray, xy[pi, j, 0], xy[pi, j, 1])
                if s >= SHARP_THR:
                    out.append(round(s, 1))
    return out


def _tracked_images() -> list[Path]:
    """Изображения, лежащие в git. Именно они уезжают вместе с репозиторием."""
    try:
        r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if r.returncode != 0:
        return []
    ex = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return [ROOT / ln for ln in r.stdout.splitlines()
            if Path(ln).suffix.lower() in ex and (ROOT / ln).is_file()]


def _embedded_images(page: Path):
    """(метка, кадр) для каждого data:image, вшитого в страницу."""
    import cv2
    t = page.read_text(encoding="utf-8", errors="replace")
    out = []
    for i, b in enumerate(re.findall(r"data:image/\w+;base64,([A-Za-z0-9+/=]+)", t)):
        try:
            arr = np.frombuffer(base64.b64decode(b), dtype=np.uint8)
        except (ValueError, TypeError):
            continue
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is not None:
            out.append((f"{page.name}#img{i}", im))
    return out


def _targets():
    """Всё, что тест обязан проверить: git плюс публикуемые страницы."""
    import cv2
    items = []
    for p in _tracked_images():
        im = cv2.imread(str(p))
        if im is not None:
            items.append((p.relative_to(ROOT).as_posix(), im))
    for rel in PUBLISHED:
        page = ROOT / rel
        if page.is_file():
            items += _embedded_images(page)
    return items


@pytest.mark.skipif(not _POSE.is_file(),
                    reason="нужны веса модели позы для замера")
def test_no_sharp_faces_anywhere():
    """Ни одно изображение репозитория и публикуемых страниц не несёт лица.

    Резкость, а не число кейпоинтов: модель позы находит ГОЛОВУ по плечам и
    корпусу и уверенно ставит «нос» на пикселизованное пятно. Опасно не то,
    что голову видно, а то, что видно лицо.

    На чистом клоне без весов тест пропускается, а не падает: гейт, который
    невозможно выполнить, перестают запускать, и он перестаёт защищать.
    """
    from ultralytics import YOLO

    items = _targets()
    if not items:
        pytest.skip("нет ни одного изображения для проверки")
    pose = YOLO(str(_POSE))

    bad = {}
    for name, img in items:
        vals = _exposed(pose, img)
        if vals:
            bad[name] = vals

    assert not bad, (
        f"резкие лицевые кейпоинты на {len(bad)} изображениях из {len(items)} "
        f"(порог резкости {SHARP_THR}):\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in sorted(
            bad.items(), key=lambda kv: -len(kv[1]))[:20]))
