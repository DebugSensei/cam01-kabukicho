"""Общий каркас этапа.

Каждый этап проходит одну и ту же последовательность, и она задана здесь, а не
переписывается в десяти модулях:

    1. load_config(--config)        конфига нет -> громкая ошибка, не дефолт (правило 5)
    2. validate_inputs()            артефакт предыдущего этапа есть и совпадает по колонкам
    3. RunManifest.start()          провенанс прогона (правило 6)
    4. build_evidence()             ОБЯЗАТЕЛЬНЫЙ шаг: этап объявляет claim-ы из конфига
    5. write_empty_artifact()       файл со всеми колонками, 0 строк, status="skeleton"
    6. RunManifest.finish()
    7. not_implemented()            -> exit 1

Пункт 5 нужен, чтобы валидация входа следующего этапа реально проверялась уже сейчас.
Пункт 7 гарантирует, что пустой артефакт нигде не выдаётся за успех: код возврата
всегда 1, а сам файл помечен status="skeleton", и любой verify обязан его отвергнуть
(правило 8).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from looq import SCHEMA_VERSION, STATUS_SKELETON
from looq.evidence import EvidenceSampler, EvidenceWriter
from looq.io import ConfigError, RunManifest, load_config, write_json

STATUS_KEY = "status"


def _force_utf8_console() -> None:
    """Консоль Windows по умолчанию cp866: русские сообщения об ошибках нечитаемы.

    Вызывается при импорте, потому что этот модуль — общая точка входа и этапов,
    и гейтов. Никакого другого побочного эффекта у импорта нет.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_force_utf8_console()


class StageError(RuntimeError):
    """Этап не может отработать корректно."""


class StageNotImplemented(StageError):
    """Логика этапа ещё не написана."""


# --------------------------------------------------------------------------- #
# Описание колонки: имя, тип, единица И СИСТЕМА КООРДИНАТ
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Col:
    """Одна колонка артефакта.

    frame обязателен и для непространственных колонок ("-"): требование заполнить
    поле заставляет автора схемы задуматься, в чём измеряется значение.
    """
    name: str
    dtype: str          # int8|int16|int32|int64|float32|float64|string|bool
    nullable: bool
    unit: str           # "кадр", "с", "px", "м", "м/с", "градус", "[0,1]", "-"
    frame: str          # "frame_px" | "plane_m" | "-"
    note: str = ""


# dtype контракта -> имя фабрики в pyarrow. Отображение явное, потому что имена
# расходятся: булев тип у pyarrow называется bool_, а не bool.
_ARROW = {
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "float32": "float32", "float64": "float64", "string": "string", "bool": "bool_",
}


def _arrow_schema(cols: Sequence[Col]):
    import pyarrow as pa
    fields = []
    for c in cols:
        if c.dtype not in _ARROW:
            raise StageError(f"неизвестный dtype {c.dtype!r} в колонке {c.name}")
        fields.append(pa.field(c.name, getattr(pa, _ARROW[c.dtype])(), nullable=c.nullable))
    return pa.schema(fields)


# --------------------------------------------------------------------------- #
# Артефакты
# --------------------------------------------------------------------------- #

def write_empty_parquet(path: str | Path, cols: Sequence[Col], stage: str,
                        status: str = STATUS_SKELETON) -> Path:
    """Parquet со всей схемой и нулём строк. status уходит в метаданные файла."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _arrow_schema(cols)
    meta = {
        b"schema_version": SCHEMA_VERSION.encode(),
        b"stage": stage.encode(),
        STATUS_KEY.encode(): status.encode(),
        b"coord_frames": str({c.name: c.frame for c in cols}).encode(),
    }
    schema = schema.with_metadata(meta)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([], schema=schema), p)
    return p


def write_parquet(path: str | Path, cols: Sequence[Col], rows: Sequence[dict],
                  stage: str, status: str) -> Path:
    """Parquet по схеме контракта. Порядок и типы колонок задаёт cols, не данные.

    Если строка содержит ключ не из схемы или в схеме есть ключ, которого нет
    в строке, — падаем. Молча дописать колонку значило бы разойтись с контрактом
    (правило 2), а молча подставить null — выдать пропуск за измерение (правило 7).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    names = [c.name for c in cols]
    nullable = {c.name: c.nullable for c in cols}
    columns: dict[str, list] = {n: [] for n in names}
    for i, row in enumerate(rows):
        extra = set(row) - set(names)
        if extra:
            raise StageError(f"строка {i}: ключи вне схемы {sorted(extra)}")
        for n in names:
            if n not in row:
                raise StageError(f"строка {i}: нет обязательной колонки {n!r}")
            v = row[n]
            if v is None and not nullable[n]:
                raise StageError(f"строка {i}: колонка {n!r} не nullable, а получена None")
            columns[n].append(v)

    schema = _arrow_schema(cols).with_metadata({
        b"schema_version": SCHEMA_VERSION.encode(),
        b"stage": stage.encode(),
        STATUS_KEY.encode(): status.encode(),
        b"coord_frames": str({c.name: c.frame for c in cols}).encode(),
    })
    table = pa.Table.from_pydict(columns, schema=schema)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(p)
    return p


def write_empty_json(path: str | Path, stage: str, status: str = STATUS_SKELETON) -> Path:
    return write_json(path, {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        STATUS_KEY: status,
        "note_ru": "Каркас. Логика этапа не реализована, содержимого нет.",
    })


def write_empty_geojson(path: str | Path, stage: str, status: str = STATUS_SKELETON) -> Path:
    return write_json(path, {
        "type": "FeatureCollection",
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        STATUS_KEY: status,
        # ВНИМАНИЕ: координаты здесь — МЕТРЫ ПЛАНА ЗЕМЛИ, не широта/долгота.
        "coordinate_frame": "plane_m",
        "coordinate_order": "[x_m, y_m]",
        "is_geographic": False,
        "warning_ru": "Координаты — метры плана земли (plane_m), НЕ широта/долгота.",
        "features": [],
    })


def write_empty_html(path: str | Path, stage: str, status: str = STATUS_SKELETON) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"<!doctype html><meta charset=utf-8><title>{stage}</title>"
        f"<p data-status={status}>Каркас. Отчёт не построен.</p>\n",
        encoding="utf-8",
    )
    return p


def read_artifact_status(path: str | Path) -> str | None:
    """status артефакта или None, если файла нет либо статус не записан."""
    p = Path(path)
    if not p.is_file():
        return None
    if p.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            meta = pq.read_schema(p).metadata or {}
            v = meta.get(STATUS_KEY.encode())
            return v.decode() if v else None
        except Exception:
            return None
    if p.suffix in (".json", ".geojson"):
        try:
            import json
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh).get(STATUS_KEY)
        except Exception:
            return None
    return None


def parquet_columns(path: str | Path) -> list[str]:
    import pyarrow.parquet as pq
    return list(pq.read_schema(Path(path)).names)


def validate_inputs(inputs: Sequence[str], stage: str) -> list[str]:
    """Артефакты предыдущих этапов существуют. Возвращает список предупреждений."""
    warnings: list[str] = []
    for path in inputs:
        p = Path(path)
        if not p.is_file():
            raise StageError(
                f"[{stage}] нет входного артефакта {p}. Этапы общаются только через файлы "
                f"(правило 2): сначала прогоните предыдущий этап"
            )
        st = read_artifact_status(p)
        if st == STATUS_SKELETON:
            warnings.append(f"вход {p} помечен status=skeleton: в нём нет данных")
    return warnings


# --------------------------------------------------------------------------- #
# Пруфы — обязательный шаг этапа
# --------------------------------------------------------------------------- #

def build_evidence(cfg: dict[str, Any], stage: str,
                   manifest: RunManifest) -> EvidenceSampler:
    """Создаёт сборщик пруфов и объявляет claim-ы, которые этап ОБЯЗАН набрать.

    Не опция. Этап без объявленных claim-ов не запускается: число, для которого
    нельзя показать кадры, в отчёт не идёт (правило 1).
    """
    claims = cfg.get("evidence_claims")
    if not isinstance(claims, list) or not claims:
        raise ConfigError(
            f"[{stage}] в конфиге {cfg.get('_config_path')} нет непустого списка "
            f"evidence_claims. Каждый этап обязан объявить, какие утверждения он "
            f"подкрепляет кадрами (правило 1)"
        )
    ev = cfg.get("evidence") or {}
    priv = cfg.get("privacy") or {}
    sampling = cfg.get("sampling") or {}
    model = cfg.get("model") or {}

    writer = EvidenceWriter(
        root=ev.get("root", "evidence"),
        stage=stage,
        model_name=str(model.get("weights") or "none"),
        face_blur_top_frac=float(priv.get("face_blur_top_frac", 0.22)),
        blur_kernel_frac=float(priv.get("blur_kernel_frac", 0.35)),
        blur_sigma_frac=float(priv.get("blur_sigma_frac", 1 / 3)),
        pixelate_factor=int(priv.get("pixelate_factor", 16)),
        jpeg_quality=int(ev.get("jpeg_quality", 90)),
    )
    sampler = EvidenceSampler(
        writer,
        n_per_claim=int(sampling.get("n_per_claim", 12)),
        n_strata=int(sampling.get("n_strata", 3)),
        seed=int(sampling.get("seed", 20260903)),
        max_candidates_per_claim=int(sampling.get("max_candidates_per_claim", 2000)),
    )
    sampler.declare(claims)
    manifest.note("evidence_claims_declared", sampler.declared_claims)
    return sampler


def finalize_evidence(sampler: EvidenceSampler, manifest: RunManifest) -> Path:
    """Вызывается реальным этапом в конце. Падает, если пруфы не набраны."""
    path = sampler.finalize()
    manifest.note("evidence_index", str(path))
    manifest.note("evidence_sampling_stats", sampler.stats())
    return path


# --------------------------------------------------------------------------- #
# Точка входа этапа
# --------------------------------------------------------------------------- #

def not_implemented(stage: str) -> None:
    raise StageNotImplemented(
        f"этап {stage} не реализован: каркас отработал, CV-логики нет. "
        f"Артефакт записан пустым с status={STATUS_SKELETON} и в качестве результата "
        f"не годится"
    )


def _write_empty(path: str, kind: str, cols: Sequence[Col], stage: str) -> Path:
    if kind == "parquet":
        return write_empty_parquet(path, cols, stage)
    if kind == "json":
        return write_empty_json(path, stage)
    if kind == "geojson":
        return write_empty_geojson(path, stage)
    if kind == "html":
        return write_empty_html(path, stage)
    raise StageError(f"неизвестный тип артефакта {kind!r}")


def stage_main(
    stage: str,
    inputs: Sequence[str],
    output: str,
    output_kind: str,
    output_cols: Sequence[Col] = (),
    argv: Sequence[str] | None = None,
    extra_outputs: Sequence[tuple[str, str, Sequence[Col]]] = (),
) -> int:
    """Каркас прогона этапа. Всегда возвращает 1, пока этап не реализован."""
    parser = argparse.ArgumentParser(prog=f"python -m looq.stages.{stage}")
    parser.add_argument("--config", required=True, help=f"configs/{stage}.yaml")
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest: RunManifest | None = None
    try:
        cfg = load_config(args.config)
        if cfg.get("stage") != stage:
            raise ConfigError(
                f"конфиг {args.config} объявляет stage={cfg.get('stage')!r}, "
                f"а запущен {stage!r}"
            )

        warnings = validate_inputs(inputs, stage)
        for w in warnings:
            print(f"[{stage}] ВНИМАНИЕ: {w}", file=sys.stderr)

        manifest = RunManifest(stage, cfg)
        manifest.start()
        for w in warnings:
            manifest.note("input_warning", w)

        build_evidence(cfg, stage, manifest)

        written_all = [_write_empty(output, output_kind, output_cols, stage)]
        for extra_path, extra_kind, extra_cols in extra_outputs:
            written_all.append(_write_empty(extra_path, extra_kind, extra_cols, stage))

        manifest.note("output_artifacts", [str(w) for w in written_all])
        manifest.note("output_rows", 0)
        manifest.note("output_status", STATUS_SKELETON)
        for w in written_all:
            print(f"[{stage}] записан пустой артефакт {w} "
                  f"(status={STATUS_SKELETON}, строк 0)")

        manifest.finish("not_implemented")
        not_implemented(stage)
        return 0  # недостижимо

    except StageNotImplemented as exc:
        print(f"[{stage}] ОШИБКА: {exc}", file=sys.stderr)
        return 1
    except (StageError, ConfigError, OSError) as exc:
        if manifest is not None:
            manifest.finish("failed", error=str(exc))
        print(f"[{stage}] ОШИБКА: {exc}", file=sys.stderr)
        return 1
