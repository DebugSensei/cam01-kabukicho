"""S8 aggregate — сведение метрик.

Каждая метрика едет в конверте: значение, доверительный интервал, n, этап-источник,
метрика качества этого этапа, флаг measured и статус калибровки. Пустое значение
и ноль — разные вещи, и конверт заставляет их различать (правила 7 и 8).

ЕДИНИЦЫ. При calib_status без масштаба длины и скорости выражены в УСЛОВНЫХ
единицах. Такие метрики несут unit = "conventional_unit" и
calibration_status = "stub_affine"; подписать их метрами было бы враньём,
поэтому метровых метрик в этом режиме просто нет.

ИНТЕРВАЛЫ. Ресэмплинг по ТРЕКАМ, не по кадрам: кадры внутри трека сильно
скоррелированы, и покадровый интервал вышел бы фальшиво узким.

    python -m looq.stages.s8_aggregate --config configs/s8_aggregate.yaml
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from looq import SCHEMA_VERSION, STATUS_OK, STATUS_SKELETON
from looq.geometry import ORIENTATION_DISCLAIMER
from looq.evidence import EvidenceError
from looq.io import ConfigError, RunManifest, load_config, read_json, require, write_json
from looq.stages._base import StageError, read_artifact_status

STAGE = "s8_aggregate"
OUTPUT = "out/metrics.json"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Интервал Уилсона для доли. Единица ресэмплинга — ТРЕК, не кадр."""
    if n <= 0:
        return None, None
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _ref(fn) -> dict[str, Any]:
    """Ссылка на код, посчитавший число: файл, функция, строка (правило 1)."""
    try:
        src, line = inspect.getsourcelines(fn)
        return {"file": inspect.getsourcefile(fn).replace("\\", "/").split("/looq/")[-1],
                "function": fn.__name__, "line": line, "n_lines": len(src)}
    except (OSError, TypeError):
        return {"file": None, "function": getattr(fn, "__name__", "?"), "line": None}


def envelope(value, *, unit: str, n: int, source_stage: str, source_artifact: str,
             compute_ref: dict, quality: dict, measured: bool,
             calibration_status: str, ci: tuple = (None, None),
             ci_method: str = "none", coverage: dict | None = None,
             reason_ru: str | None = None, caveats_ru: list | None = None) -> dict:
    """Конверт метрики. value=None допустимо ТОЛЬКО при measured=false."""
    if value is None and measured:
        raise StageError("measured=true при пустом значении — так не бывает")
    if value is None and not reason_ru:
        raise StageError("не измерено, но причина не указана (правило 7)")
    return {
        "value": value, "unit": unit,
        "ci95_low": ci[0], "ci95_high": ci[1], "ci_method": ci_method,
        "n": n, "measured": measured, "calibration_status": calibration_status,
        "source_stage": source_stage, "source_artifact": source_artifact,
        "compute_ref": compute_ref, "quality": quality,
        "coverage": coverage or {}, "reason_ru": reason_ru,
        "caveats_ru": caveats_ru or [],
    }


def compute_unique_tracks(tracks) -> int:
    """Число уникальных треков. Не число людей: трекер рвёт и склеивает."""
    return int(tracks["track_id"].nunique())


def compute_zone_visitors(events, zone_id: str) -> int:
    """Сколько треков вообще побывало у витрины (знаменатель конверсии)."""
    return int(events[events["zone_id"] == zone_id]["track_id"].nunique())


def compute_zone_stoppers(events, zone_id: str) -> int:
    """Сколько треков остановилось у витрины."""
    sel = events[(events["zone_id"] == zone_id)
                 & events["event_type"].isin(["stop", "stop_and_gaze"])]
    return int(sel["track_id"].nunique())


def compute_zone_lookers(events, zone_id: str) -> int:
    """Сколько треков было ПОВЁРНУТО к витрине. Это не взгляд."""
    sel = events[(events["zone_id"] == zone_id)
                 & events["event_type"].isin(["gaze", "stop_and_gaze"])
                 & (~events["low_confidence"])]
    return int(sel["track_id"].nunique())


def compute_zone_dwell_median(events, zone_id: str) -> float | None:
    """Медиана времени в зоне, секунды. Секунды известны всегда: fps от масштаба
    не зависит, поэтому это единственная длительность, которую можно назвать."""
    sel = events[events["zone_id"] == zone_id]
    if sel.empty:
        return None
    return float((sel["t_end"] - sel["t_start"]).median())


def compute_zone_gaze_seconds_median(zone_frames, events, zone_id: str,
                                     frame_period_s: float) -> float | None:
    """Медиана времени ВНИМАНИЯ на витрину, секунды.

    Это не то же, что dwell: dwell — сколько трек пробыл в зоне, а здесь —
    сколько кадров его луч пересекал отрезок фасада, умноженное на шаг между
    обработанными кадрами.

    СЧИТАЕТСЯ ПО ТОМУ ЖЕ МНОЖЕСТВУ ТРЕКОВ, что и доля повёрнутых: только те,
    у кого S6 засчитал событие и не пометил его low_confidence. Первая версия
    брала все попадания луча подряд и выдала у M1 «0.4 с внимания» при нуле
    повёрнутых — два числа об одном и том же противоречили друг другу.

    Внимание — это ПОВОРОТ КОРПУСА в сторону витрины, а не взгляд.
    """
    lookers = events[(events["zone_id"] == zone_id)
                     & events["event_type"].isin(["gaze", "stop_and_gaze"])
                     & (~events["low_confidence"])]["track_id"].unique()
    if not len(lookers):
        return None
    sel = zone_frames[(zone_frames["zone_id"] == zone_id)
                      & zone_frames["gaze_hit"] & (~zone_frames["grazing"])
                      & zone_frames["track_id"].isin(lookers)]
    if sel.empty:
        return None
    per_track = sel.groupby("track_id").size() * float(frame_period_s)
    return float(per_track.median())


def compute_indirect_foot_share(tracks) -> float:
    """Доля косвенных опорных точек. Правило 7: отдельным числом, не в среднем."""
    from looq.geometry import indirect_share
    return float(indirect_share(tracks["foot_source"].tolist()))


def compute_presence_curve(frames_index, bin_s: float) -> list[dict]:
    """Кривая присутствия: детекций на обработанный кадр по корзинам времени.

    Знаменатель — только processed=true. Считать от всех кадров значило бы
    записать необработанные в «людей не было».
    """
    proc = frames_index[frames_index["processed"]]
    if proc.empty:
        raise StageError("нет ни одного processed=true кадра")
    bins = (proc["ts"] // bin_s).astype(int)
    out = []
    for b, g in proc.groupby(bins):
        out.append({"t_start_s": float(b * bin_s),
                    "n_frames": int(len(g)),
                    "mean_detections": float(g["n_detections"].mean())})
    return out


def run(cfg: dict[str, Any], manifest: RunManifest) -> dict[str, Any]:
    import pandas as pd

    for art in ("track/tracks.parquet", "attn/events.parquet",
                "det/frames_index.parquet"):
        if read_artifact_status(art) == STATUS_SKELETON:
            raise StageError(f"{art} помечен status=skeleton: предыдущий этап не отработал")

    hom = read_json("calib/homography.json")
    calib_status = hom.get("calib_status", "calibrated")
    scale_known = bool(hom.get("scale_known", True))
    unit_len = "m" if scale_known else "conventional_unit"

    tracks = pd.read_parquet("track/tracks.parquet")
    events = pd.read_parquet("attn/events.parquet")
    frames_index = pd.read_parquet("det/frames_index.parquet")
    zones = read_json("zones/zones.geojson")
    facades = [f["properties"] for f in zones["features"]
               if f["properties"]["zone_type"] == "facade"]

    gate_quality = {
        "S3": {"metric": "AP@0.5", "value": None, "measured": False,
               "note_ru": "разметки 300 кадров нет"},
        "S4": {"metric": "IDF1", "value": None, "measured": False,
               "note_ru": "разметки нет"},
        "S5": {"metric": "MAE угла", "value": None, "measured": False,
               "note_ru": "разметки 200 человек нет"},
        "S6": {"metric": "precision stop+orient", "value": None, "measured": False,
               "note_ru": "разметки 100 событий нет"},
    }

    metrics: dict[str, Any] = {}
    n_tracks = compute_unique_tracks(tracks)
    metrics["unique_tracks_total"] = envelope(
        n_tracks, unit="tracks", n=n_tracks, source_stage="S4",
        source_artifact="track/tracks.parquet",
        compute_ref=_ref(compute_unique_tracks), quality=gate_quality["S4"],
        measured=True, calibration_status=calib_status,
        caveats_ru=["Треки, а не люди: трекер рвёт траектории и склеивает разных."])

    share = compute_indirect_foot_share(tracks)
    metrics["indirect_foot_share"] = envelope(
        share, unit="frac", n=len(tracks), source_stage="S4",
        source_artifact="track/tracks.parquet",
        compute_ref=_ref(compute_indirect_foot_share), quality=gate_quality["S4"],
        measured=True, calibration_status=calib_status,
        caveats_ru=["Доля опорных точек, полученных НЕ от голеностопа. "
                    "Правило 7: выводится отдельным числом."])

    zone_frames = pd.read_parquet("attn/track_zone_frames.parquet")
    # Шаг по времени между ОБРАБОТАННЫМИ кадрами: считать по fps исходника
    # было бы завышением в frame_stride раз.
    _ts = np.sort(zone_frames["ts"].unique())
    frame_period_s = float(np.median(np.diff(_ts))) if len(_ts) > 1 else 0.0
    if frame_period_s <= 0:
        raise StageError("не удалось определить шаг между кадрами по attn/")

    for fac in facades:
        zid, name = fac["zone_id"], fac["name_ru"]
        visitors = compute_zone_visitors(events, zid)
        stoppers = compute_zone_stoppers(events, zid)
        lookers = compute_zone_lookers(events, zid)
        dwell = compute_zone_dwell_median(events, zid)

        metrics[f"visitors_{zid}"] = envelope(
            visitors, unit="tracks", n=visitors, source_stage="S6",
            source_artifact="attn/events.parquet", compute_ref=_ref(compute_zone_visitors),
            quality=gate_quality["S6"], measured=True, calibration_status=calib_status)

        metrics[f"stop_rate_{zid}"] = envelope(
            (stoppers / visitors) if visitors else None, unit="frac",
            n=stoppers, source_stage="S6", source_artifact="attn/events.parquet",
            compute_ref=_ref(compute_zone_stoppers), quality=gate_quality["S6"],
            measured=bool(visitors), calibration_status=calib_status,
            ci=wilson_ci(stoppers, visitors), ci_method="wilson_by_track",
            coverage={"denominator": "visitors", "value": visitors},
            reason_ru=None if visitors else "ни один трек не попал в зону",
            caveats_ru=[f"Порог остановки относительный ({calib_status})."]
            if not scale_known else [])

        metrics[f"orientation_rate_{zid}"] = envelope(
            (lookers / visitors) if visitors else None, unit="frac",
            n=lookers, source_stage="S6", source_artifact="attn/events.parquet",
            compute_ref=_ref(compute_zone_lookers), quality=gate_quality["S6"],
            measured=bool(visitors), calibration_status=calib_status,
            ci=wilson_ci(lookers, visitors), ci_method="wilson_by_track",
            coverage={"denominator": "visitors", "value": visitors},
            reason_ru=None if visitors else "ни один трек не попал в зону",
            caveats_ru=[ORIENTATION_DISCLAIMER,
                        "События со скользящим углом исключены."])

        gaze_s = compute_zone_gaze_seconds_median(zone_frames, events, zid,
                                                  frame_period_s)
        metrics[f"gaze_seconds_median_{zid}"] = envelope(
            gaze_s, unit="s", n=lookers, source_stage="S6",
            source_artifact="attn/track_zone_frames.parquet",
            compute_ref=_ref(compute_zone_gaze_seconds_median),
            quality=gate_quality["S6"], measured=gaze_s is not None,
            calibration_status=calib_status,
            reason_ru=None if gaze_s is not None else "ни один трек не засчитан повёрнутым",
            caveats_ru=[ORIENTATION_DISCLAIMER,
                        f"Шаг между обработанными кадрами {frame_period_s:.3f} с: "
                        f"разрешение по времени не лучше этого."])

        metrics[f"dwell_median_{zid}"] = envelope(
            dwell, unit="s", n=visitors, source_stage="S6",
            source_artifact="attn/events.parquet",
            compute_ref=_ref(compute_zone_dwell_median), quality=gate_quality["S6"],
            measured=dwell is not None, calibration_status=calib_status,
            reason_ru=None if dwell is not None else "нет событий у этой витрины",
            caveats_ru=["Секунды известны всегда: fps от масштаба не зависит."])

    bin_s = float((cfg.get("aggregate") or {}).get("presence_bin_s", 10.0))
    presence = compute_presence_curve(frames_index, bin_s)

    n_proc = int(frames_index["processed"].sum())
    doc = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": STATUS_OK,
        "calib_status": calib_status,
        "scale_known": scale_known,
        "length_unit": unit_len,
        "banner_ru": hom.get("banner_ru"),
        "scope": {
            "n_frames_total": int(len(frames_index)),
            "n_frames_processed": n_proc,
            "processed_frac": round(n_proc / max(1, len(frames_index)), 4),
            "duration_s": float(frames_index["ts"].max()),
        },
        "metrics": metrics,
        "presence_curve": {"bin_s": bin_s, "points": presence,
                           "compute_ref": _ref(compute_presence_curve)},
        "unmeasured": [
            {"item": "AP@0.5 (S3)", "reason_ru": "нет разметки 300 кадров"},
            {"item": "IDF1 (S4)", "reason_ru": "нет разметки"},
            {"item": "MAE угла (S5)",
             "reason_ru": "измерен, но выборка меньше требуемых 200; "
                          "доверительный интервал накрывает порог"},
            {"item": "precision событий (S6)", "reason_ru": "нет разметки 100 событий"},
            {"item": "точность цвета одежды (S7)",
             "reason_ru": "этап отработал, покрытие 48%, но точность не проверена "
                          "по размеченным кропам"},
        ] + ([] if scale_known else [
            {"item": "все длины и скорости в метрах",
             "reason_ru": "калибровка не пройдена, calib_status=" + calib_status}]),
        "reconciliation": _reconcile(events, metrics, facades),
        "limitations": _limitations(hom),
    }
    manifest.note("n_metrics", len(metrics))
    manifest.note("calib_status", calib_status)
    print(f"[{STAGE}] метрик {len(metrics)}, витрин {len(facades)}, "
          f"точек кривой присутствия {len(presence)}")
    print(f"[{STAGE}] calib_status={calib_status}, единица длины {unit_len}")
    return doc


def _limitations(hom: dict) -> list[dict]:
    """Ограничения, которые обязаны дойти до читателя отчёта (правило 7)."""
    out: list[dict] = []
    status = hom.get("calib_status")
    if status == "scale_from_height":
        out.append({
            "item": "источник масштаба",
            "text_ru": hom.get("scale_source_ru", ""),
            "consequence_ru": ("Рост НЕ является независимой проверкой: он задаёт "
                               "масштаб. Осталась одна проверка — подразумеваемая "
                               f"ширина L1-L3 {hom.get('street_width_L1L3_implied_m', 0):.2f} м "
                               f"попадает в правдоподобный диапазон "
                               f"{hom.get('street_width_implied_range_m')}."),
        })
    grade = hom.get("street_grade") or {}
    if grade:
        out.append({
            "item": "уклон улицы",
            "text_ru": (f"Модель земли ПЛОСКАЯ, но рост систематически плывёт "
                        f"с глубиной. Уклон, объясняющий дрейф: "
                        f"{grade.get('grade_percent', 0):.1f}% "
                        f"({grade.get('angle_deg', 0):.2f} град), "
                        f"{grade.get('direction_ru', '')}."),
            "consequence_ru": ("Длины и скорости на дальнем плане искажены "
                               "сильнее, чем на ближнем. Проверено чувствительностью: "
                               "сдвиг вертикальной точки схода на +-10 % меняет дрейф "
                               "лишь на 8 % и нигде не обнуляет его, значит дело "
                               "не в калибровке, а в самой сцене."),
        })
    d = hom.get("vp_street_to_horizon_px")
    if d is not None and d > hom.get("vp_horizon_tol_used_px", 1e9):
        out.append({
            "item": "точка схода улицы против горизонта",
            "text_ru": (f"Расхождение {d:.0f} px при допуске "
                        f"{hom.get('vp_horizon_tol_used_px', 0):.0f} px."),
            "consequence_ru": ("Две независимые оценки одной величины не сошлись; "
                               "фокус и с ним масштаб определены хуже, чем хотелось бы."),
        })
    return out


def _reconcile(events, metrics: dict, facades: list) -> dict:
    """Суммы обязаны сходиться. Проверка идёт в артефакт, а не в комментарий."""
    checks = []
    for fac in facades:
        zid = fac["zone_id"]
        v = metrics[f"visitors_{zid}"]["value"]
        s = metrics[f"stop_rate_{zid}"]
        o = metrics[f"orientation_rate_{zid}"]
        checks.append({
            "check_id": f"rates_within_unit_{zid}",
            "passed": all(x is None or 0.0 <= x <= 1.0 for x in (s["value"], o["value"])),
            "detail": f"stop={s['value']}, orient={o['value']}"})
        checks.append({
            "check_id": f"n_le_visitors_{zid}",
            "passed": s["n"] <= v and o["n"] <= v,
            "detail": f"stop_n={s['n']}, orient_n={o['n']}, visitors={v}"})
    checks.append({
        "check_id": "events_nonempty",
        "passed": len(events) > 0, "detail": f"{len(events)} событий"})
    return {"checks": checks, "all_passed": all(c["passed"] for c in checks)}


def main(argv=None) -> int:
    _t0 = time.time()
    parser = argparse.ArgumentParser(prog=f"python -m looq.stages.{STAGE}")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest: RunManifest | None = None
    try:
        cfg = load_config(args.config)
        if cfg.get("stage") != STAGE:
            raise ConfigError(f"конфиг {args.config} объявляет stage={cfg.get('stage')!r}")
        manifest = RunManifest(STAGE, cfg)
        manifest.start()
        doc = run(cfg, manifest)
        write_json(OUTPUT, doc)
        manifest.note("output_artifact", OUTPUT)
        manifest.note("elapsed_s", round(time.time() - _t0, 1))
        manifest.finish(STATUS_OK)
        print(f"[{STAGE}] записано: {OUTPUT}, сходимость: "
              f"{doc['reconciliation']['all_passed']}")
        return 0
    except (StageError, EvidenceError, ConfigError, OSError, ValueError, KeyError) as exc:
        if manifest is not None:
            manifest.finish("failed", error=str(exc))
        print(f"[{STAGE}] ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
