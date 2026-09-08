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


def compute_stop_dwell_grid(tracks, speed_thr_mps: float, bin_m: float,
                            max_gap_s: float = 1.0, roi_m=None):
    """Человеко-секунды стояния по клеткам плана. Ответ на «где останавливаются».

    ПОЧЕМУ ВРЕМЯ, А НЕ ЧИСЛО ОСТАНОВОК. Счётчик остановок требует решить, что
    считать одной остановкой, а значит — допуск на разрыв. Замер 2026-09-08
    показал, что от этого допуска результат зависит сильно и немонотонно:
    0.11 с -> 851 остановка у 385 треков, 1.0 с -> 934 у 502, 3.0 с -> 821 у
    678. Один трек давал до 25 «остановок» в одном квадрате — дребезг оценки
    скорости вокруг порога. Сумма времени такого параметра не имеет: каждый
    кадр учитывается ровно один раз.

    ``max_gap_s`` ограничивает вклад одного кадра: если трек прервался на
    минуту, эта минута не превращается в минуту стояния.

    ``roi_m`` ОБЯЗАТЕЛЕН по смыслу: за границей ROI recall детектора неизвестен
    (граница задана порогом высоты рамки 70 px, а не измеренной кривой), и
    стояние там означает «столько мы увидели», а не «столько было». Весь
    остальной конвейер ROI уважает, и карта остановок обязана тоже — иначе
    дальний конец улицы попадёт в отчёт наравне с ближним.

    Возвращает (клетки, всего_стояния_с, всего_наблюдения_с).
    """
    import numpy as np
    from collections import defaultdict

    from looq.geometry import point_in_polygon_m

    sec: dict[tuple[int, int], float] = defaultdict(float)
    who: dict[tuple[int, int], set] = defaultdict(set)
    total_stop = total_seen = 0.0
    for tid, g in tracks.sort_values(["track_id", "frame_idx"]).groupby(
            "track_id", sort=False):
        v = g["speed_mps"].to_numpy(dtype=np.float64)
        t = g["ts"].to_numpy(dtype=np.float64)
        x = g["foot_x_m"].to_numpy(dtype=np.float64)
        y = g["foot_y_m"].to_numpy(dtype=np.float64)
        ok = np.isfinite(v) & np.isfinite(x) & np.isfinite(y)
        if roi_m is not None:
            inside = np.array([point_in_polygon_m((float(a_), float(b_)), roi_m)
                               if np.isfinite(a_) and np.isfinite(b_) else False
                               for a_, b_ in zip(x, y)])
            ok &= inside
        if ok.sum() < 2:
            continue
        dt = np.clip(np.diff(t, prepend=t[0]), 0.0, float(max_gap_s))
        total_seen += float(dt[ok].sum())
        for i in np.flatnonzero(ok & (v < speed_thr_mps)):
            c = (int(x[i] // bin_m), int(y[i] // bin_m))
            sec[c] += float(dt[i])
            who[c].add(int(tid))
            total_stop += float(dt[i])
    cells = [{"x_m": ix * bin_m, "y_m": iy * bin_m,
              "stop_seconds": round(s, 1), "n_tracks": len(who[(ix, iy)])}
             for (ix, iy), s in sec.items()]
    cells.sort(key=lambda d: -d["stop_seconds"])
    return cells, total_stop, total_seen


def compute_stoppers(tracks, speed_thr_mps: float, min_duration_s: float,
                     roi_m=None) -> int:
    """Сколько РАЗНЫХ треков хоть раз простояли не меньше min_duration_s.

    Уникальные треки, а не число остановок: число остановок зависит от допуска
    на разрыв, а факт «этот трек стоял» — нет.

    ROI тот же, что у карты остановок. Два числа про остановки с разными
    знаменателями в одном отчёте — это тот самый дефект, из-за которого реплей
    печатал 279 попаданий против 3 на дашборде.
    """
    import numpy as np

    from looq.geometry import point_in_polygon_m

    n = 0
    for _, g in tracks.sort_values(["track_id", "frame_idx"]).groupby(
            "track_id", sort=False):
        v = g["speed_mps"].to_numpy(dtype=np.float64)
        t = g["ts"].to_numpy(dtype=np.float64)
        slow = np.isfinite(v) & (v < speed_thr_mps)
        if roi_m is not None:
            x = g["foot_x_m"].to_numpy(dtype=np.float64)
            y = g["foot_y_m"].to_numpy(dtype=np.float64)
            slow &= np.array([point_in_polygon_m((float(a_), float(b_)), roi_m)
                              if np.isfinite(a_) and np.isfinite(b_) else False
                              for a_, b_ in zip(x, y)])
        idx = np.flatnonzero(slow)
        if idx.size == 0:
            continue
        brk = np.flatnonzero(np.diff(t[idx]) > 1.0) + 1
        if any(t[g_[-1]] - t[g_[0]] >= min_duration_s for g_ in np.split(idx, brk)):
            n += 1
    return n


def compute_track_duration_hist(tracks, edges_s):
    """Распределение длительности треков. Для витрины, где нужен профиль, а не медиана.

    Треки, а не визиты: трекер рвёт траектории, и «визит 47 минут» с одной
    камеры без ReID получить нечем. Медиана длительности трека 13 с — это
    длительность НАБЛЮДЕНИЯ, и подпись обязана это говорить.
    """
    import numpy as np

    d = tracks.groupby("track_id")["ts"].agg(lambda s: float(s.max() - s.min()))
    v = d.to_numpy(dtype=np.float64)
    out = []
    for lo, hi in zip(edges_s[:-1], edges_s[1:]):
        n = int(((v >= lo) & (v < hi)).sum())
        out.append({"from_s": float(lo),
                    "to_s": None if hi == float("inf") else float(hi),
                    "n": n, "share": round(n / max(1, v.size), 4)})
    return {"bins": out, "median_s": round(float(np.median(v)), 1),
            "p90_s": round(float(np.percentile(v, 90)), 1),
            "max_s": round(float(v.max()), 1), "n_tracks": int(v.size)}


def compute_approached_tracks_total(events) -> int:
    """РАЗНЫХ треков, подошедших хоть к одной витрине ближе gaze.max_dist_m.

    Ни сумма по витринам (10 043, каждый трек посчитан у каждой витрины), ни
    максимум по витринам (2 706, нижняя оценка) — а именно разные треки: 2 979.
    Три способа сложить одно и то же дают три числа, и только одно из них
    отвечает на вопрос «сколько человек подошло».
    """
    return int(events["track_id"].nunique())


def compute_turned_tracks_total(events) -> int:
    """РАЗНЫХ треков, засчитанных повёрнутыми хоть к одной витрине.

    Не сумма по витринам. Один трек может быть засчитан к двум витринам, и
    сумма n по зонам даёт 683 там, где разных треков 514. Складывать доли по
    зонам — это ровно тот дефект, из-за которого страница реплея печатала 279
    попаданий против 3 на дашборде.
    """
    good = events[(~events["low_confidence"])
                  & events["event_type"].isin(["gaze", "stop_and_gaze"])]
    return int(good["track_id"].nunique())


def compute_colour_spatial_bias(attrs, tracks):
    """Диагностика: не привязан ли класс цвета к МЕСТУ, а не к одежде.

    ЗАЧЕМ. Владелец заметил глазом: «оранжевый не всегда оранжевая одежда,
    иногда там оранжевое освещение». Проверка подтвердила и дала число. Медиана
    позиции по классам: синий x 13.9, серый 12.2, чёрный 11.2, красный 12.2,
    белый 14.1 — все кучно в середине улицы. ОРАНЖЕВЫЙ: x 2.7, y 7.2, другой
    угол улицы. В квадрате 6x6 м вокруг этой точки оранжевых 35 % треков, на
    остальной улице 1.4 % — разница в 25 раз, критерий Манна-Уитни p = 5e-18
    по x и 3e-25 по y.

    Одежда по улице не распределена пятнами. Освещение — распределено. Значит
    в этом месте классификатор меряет подсветку, а не ткань.

    КАК СЧИТАЕТСЯ. Для каждого класса берётся медианная позиция его треков и
    доля класса в квадрате ``patch_m`` вокруг неё против доли на остальной
    улице. Отношение этих долей — ``concentration``. Класс, размазанный по
    улице как все, даёт отношение около 1. Класс, привязанный к месту, даёт
    десятки.

    Порог здесь НЕ ставится и никто не выбрасывается: это диагностика, которая
    печатается в отчёт (правило 7). Выбрасывать класс по этому числу без
    размеченных кропов значило бы чинить одну неизмеренную величину другой.
    """
    import numpy as np

    ok = attrs[attrs["top_color_status"] == "ok"]
    if ok.empty:
        return []
    pos = tracks.groupby("track_id")[["foot_x_m", "foot_y_m"]].median()
    m = ok.set_index("track_id").join(pos).dropna(subset=["foot_x_m", "foot_y_m"])
    if m.empty:
        return []
    patch = 3.0                      # полуразмер квадрата, метры плана
    out = []
    for name, g in m.groupby("top_color_name"):
        if len(g) < 3:
            continue
        cx, cy = float(g["foot_x_m"].median()), float(g["foot_y_m"].median())
        near = ((m["foot_x_m"] - cx).abs() < patch) & ((m["foot_y_m"] - cy).abs() < patch)
        n_near = int(near.sum())
        k_near = int((near & (m["top_color_name"] == name)).sum())
        k_far = int((~near & (m["top_color_name"] == name)).sum())
        n_far = int((~near).sum())
        share_near = k_near / max(1, n_near)
        share_far = k_far / max(1, n_far)
        out.append({
            "colour": str(name), "n_tracks": int(len(g)),
            "centre_x_m": round(cx, 1), "centre_y_m": round(cy, 1),
            "share_in_patch": round(share_near, 4),
            "share_elsewhere": round(share_far, 4),
            "concentration": round(share_near / share_far, 1) if share_far > 0 else None,
        })
    out.sort(key=lambda d: -(d["concentration"] or 0))
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
    # S7 опционален: без него цветовой диагностики просто не будет.
    _attr_path = Path("attr/tracks_attr.parquet")
    attrs = pd.read_parquet(_attr_path) if _attr_path.is_file() else None
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

    # Остановки по всей улице. Порог скорости берётся ТОТ ЖЕ, что у S6, из
    # configs/s6_attn.yaml: два разных порога «стоит» в одном отчёте — это
    # ровно тот дефект, из-за которого страница реплея печатала 279, а
    # дашборд 3.
    s6cfg = load_config("configs/s6_attn.yaml")["stop"]
    med_speed = float(tracks["speed_mps"].median())
    stop_thr = (med_speed * float(s6cfg["speed_thr_frac_of_median"])
                if s6cfg.get("mode") == "relative"
                else float(s6cfg["speed_thr_mps"]))
    bin_m = float(cfg.get("stops", {}).get("bin_m", 2.0))
    n_appr = compute_approached_tracks_total(events)
    metrics["approached_tracks_total"] = envelope(
        n_appr, unit="tracks", n=n_tracks, source_stage="S6",
        source_artifact="attn/events.parquet",
        compute_ref=_ref(compute_approached_tracks_total), quality=gate_quality["S6"],
        measured=True, calibration_status=calib_status,
        caveats_ru=["РАЗНЫЕ треки. Сумма по витринам даёт 10 043, максимум по "
                    "витринам 2 706 — оба отвечают не на этот вопрос.",
                    "«Подошёл», а не «вошёл»: метрики входа в проекте нет, и с "
                    "этого ракурса исчезновение трека у двери неотличимо от "
                    "перекрытия прохожим."])

    n_turned = compute_turned_tracks_total(events)
    metrics["turned_tracks_total"] = envelope(
        n_turned, unit="tracks", n=n_tracks, source_stage="S6",
        source_artifact="attn/events.parquet",
        compute_ref=_ref(compute_turned_tracks_total), quality=gate_quality["S6"],
        measured=True, calibration_status=calib_status,
        caveats_ru=["РАЗНЫЕ треки, а не сумма по витринам: один трек может быть "
                    "засчитан к двум витринам, и сумма даёт 683 там, где треков 514.",
                    "Поворот корпуса, а не взгляд. Считается по покрытой "
                    "подвыборке (ориентация есть у 45.6 % строк), значит доля "
                    "смещена в сторону крупных неперекрытых людей."])

    roi_m = None
    for f in zones["features"]:
        if (f.get("properties") or {}).get("zone_type") == "roi":
            import numpy as _np
            roi_m = _np.asarray(f["geometry"]["coordinates"][0], dtype=float)
    cells, stop_s, seen_s = compute_stop_dwell_grid(tracks, stop_thr, bin_m,
                                                   roi_m=roi_m)
    n_stoppers = compute_stoppers(tracks, stop_thr, float(s6cfg["min_duration_s"]),
                                  roi_m=roi_m)

    metrics["stop_seconds_total"] = envelope(
        round(stop_s, 1), unit="person-seconds", n=len(tracks), source_stage="S4",
        source_artifact="track/tracks.parquet",
        compute_ref=_ref(compute_stop_dwell_grid), quality=gate_quality["S4"],
        measured=True, calibration_status=calib_status,
        caveats_ru=[f"Суммарное время, проведённое треками медленнее "
                    f"{stop_thr:.3f} м/с. Порог НЕ ОТКАЛИБРОВАН: он равен "
                    f"{s6cfg['speed_thr_frac_of_median']} медианной скорости "
                    f"потока и не проверен на размеченных остановках.",
                    f"Знаменатель — {seen_s:.0f} чел-с наблюдения, то есть "
                    f"{stop_s / max(1.0, seen_s) * 100:.1f} % времени.",
                    "Время, а не число остановок: счётчик остановок зависит от "
                    "допуска на разрыв, и замер показал разброс 851-934 при "
                    "допуске от 0.11 до 3 с. У суммы времени такого параметра нет."])
    metrics["stoppers_total"] = envelope(
        n_stoppers, unit="tracks", n=n_stoppers, source_stage="S4",
        source_artifact="track/tracks.parquet",
        compute_ref=_ref(compute_stoppers), quality=gate_quality["S4"],
        measured=True, calibration_status=calib_status,
        caveats_ru=[f"Треки, простоявшие подряд не менее "
                    f"{s6cfg['min_duration_s']} с. Треки, а не люди.",
                    "Считается ГДЕ УГОДНО на улице. stop_rate_* считает только "
                    "внутри фартука витрины, и числа расходятся на два порядка "
                    "— это разные вопросы, а не расхождение."])

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
        "track_duration": {
            **compute_track_duration_hist(
                tracks, [0.0, 5.0, 10.0, 20.0, 40.0, float("inf")]),
            "compute_ref": _ref(compute_track_duration_hist),
        },
        "colour_spatial_bias": {
            "patch_half_m": 3.0,
            "note_ru": "Класс цвета, привязанный к МЕСТУ, а не к одежде: "
                       "concentration — во сколько раз доля класса выше в его "
                       "собственном пятне, чем на остальной улице. Одежда пятнами "
                       "не распределена, освещение распределено. Порог не ставится "
                       "и классы не выбрасываются: это диагностика (правило 7).",
            "classes": compute_colour_spatial_bias(attrs, tracks)
                       if attrs is not None else [],
            "compute_ref": _ref(compute_colour_spatial_bias),
        },
        "stop_hotspots": {
            "bin_m": bin_m,
            "speed_thr_mps": round(stop_thr, 4),
            "min_duration_s": float(s6cfg["min_duration_s"]),
            "stop_seconds_total": round(stop_s, 1),
            "observed_seconds_total": round(seen_s, 1),
            "stop_share": round(stop_s / max(1.0, seen_s), 4),
            "n_stopper_tracks": n_stoppers,
            "cells": cells,
            "compute_ref": _ref(compute_stop_dwell_grid),
        },
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
    top = cells[0] if cells else None
    print(f"[{STAGE}] стояние: {stop_s:.0f} чел-с из {seen_s:.0f} наблюдаемых "
          f"({stop_s / max(1.0, seen_s) * 100:.1f} %), треков со стоянкой "
          f">= {s6cfg['min_duration_s']} с: {n_stoppers}; ячеек {len(cells)}")
    if top:
        print(f"[{STAGE}] главная точка: x {top['x_m']:.0f}..{top['x_m'] + bin_m:.0f}, "
              f"y {top['y_m']:.0f}..{top['y_m'] + bin_m:.0f} — "
              f"{top['stop_seconds']:.0f} чел-с ({top['stop_seconds'] / max(1.0, stop_s) * 100:.1f} % "
              f"всего стояния) от {top['n_tracks']} разных треков")
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
