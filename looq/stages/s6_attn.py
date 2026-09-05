"""S6 attention — остановки и ориентация на витрину.

Формулы из CLAUDE.md, менять только с записью в docs/DECISIONS.md:

    stop_score : доля времени в apron-полигоне со скоростью ниже порога,
                 длительностью > min_duration_s
    gaze_score : доля времени, когда СЕКТОР ОРИЕНТАЦИИ пересекает facade-отрезок
                 (луч длиной <= max_dist, полуширина = неопределённость yaw)

ПОРОГ ОСТАНОВКИ БЕЗ МЕТРОВ. При calib_status без масштаба порог в м/с
бессмыслен: 0.3 «м/с» в условных единицах — выдуманное число. В режиме relative
порог берётся как доля МЕДИАННОЙ СКОРОСТИ ПОТОКА этого же прогона. Отношение
скоростей от масштаба не зависит, поэтому порог защитим и вдобавок
самонастраивается под сцену.

СКОЛЬЗЯЩИЙ УГОЛ. Если луч падает на фасад под углом больше grazing_max_deg
от нормали, фасад виден с ребра и пересечение ненадёжно. Событие помечается
low_confidence: в основной агрегат не идёт, но СОХРАНЯЕТСЯ в данных (правило 7).

    python -m looq.stages.s6_attn --config configs/s6_attn.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np

from looq import STATUS_OK, STATUS_SKELETON
from looq.geometry import grazing_angle_deg, point_in_polygon_m, sector_hits_segment_m
from looq.evidence import EvidenceError
from looq.io import ConfigError, RunManifest, load_config, read_json, require
from looq.stages._base import (commit_parquet,
                               
    Col,
    StageError,
    build_evidence,
    finalize_evidence,
    read_artifact_status,
    write_parquet,
)

STAGE = "s6_attn"
INPUTS = ["zones/zones.geojson", "calib/homography.json",
          "track/tracks.parquet", "pose/orient.parquet"]
OUTPUT = "attn/events.parquet"

# Покадровый след: без него отчёт и оверлей вынуждены ПЕРЕСЧИТЫВАТЬ попадания
# луча, то есть заводить вторую реализацию формулы. Правило 1 требует обратного:
# всё, что показано, посчитано этапом и лежит на диске.
FRAMES_OUTPUT = "attn/track_zone_frames.parquet"
FRAMES_COLS: list[Col] = [
    Col("frame_idx",  "int64",   False, "кадр",  "-",       "номер кадра"),
    Col("track_id",   "int32",   False, "-",     "-",       "идентификатор трека"),
    Col("zone_id",    "string",  False, "-",     "-",       "зона из zones.geojson"),
    Col("ts",         "float64", False, "с",     "-",       "секунды от начала записи"),
    Col("x_m",        "float32", True,  "м",     "plane_m", "положение на плане, x"),
    Col("y_m",        "float32", True,  "м",     "plane_m", "положение на плане, y"),
    Col("in_apron",   "bool",    False, "-",     "-",       "точка внутри прифасадной полосы"),
    Col("is_slow",    "bool",    False, "-",     "-",       "скорость ниже порога остановки"),
    Col("gaze_hit",   "bool",    False, "-",     "-",       "сектор ОРИЕНТАЦИИ пересёк фасад"),
    Col("gaze_dist_m", "float32", True, "м",     "plane_m", "дистанция до точки пересечения"),
    Col("grazing",    "bool",    False, "-",     "-",       "скользящий угол: пересечение ненадёжно"),
    Col("yaw_deg",    "float32", True,  "градус", "plane_m", "использованный угол ОРИЕНТАЦИИ"),
]

OUTPUT_COLS: list[Col] = [
    Col("event_id",   "string",  False, "-",     "-", "детерминированный: track:zone:type:frame_start"),
    Col("track_id",   "int32",   False, "-",     "-", "идентификатор трека"),
    Col("zone_id",    "string",  False, "-",     "-", "зона из zones/zones.geojson"),
    Col("t_start",    "float64", False, "с",     "-", "начало эпизода, секунды от начала записи"),
    Col("t_end",      "float64", False, "с",     "-", "конец эпизода, включительно"),
    Col("stop_score", "float32", True,  "[0,1]", "-", "доля времени в apron ниже порога скорости"),
    Col("gaze_score", "float32", True,  "[0,1]", "-", "доля времени, когда сектор ОРИЕНТАЦИИ пересекает фасад"),
    Col("event_type", "string",  False, "-",     "-", "stop | gaze | stop_and_gaze | pass_by"),
    # Расширение схемы: скользящий угол обязан быть виден в данных (правило 7).
    Col("low_confidence", "bool", False, "-",    "-", "пересечение под скользящим углом либо ориентация не измерена"),
]


def _load_zones(path: str, homography_path: str) -> tuple[dict, dict, dict, bool]:
    doc = read_json(path)
    if doc.get("status") == STATUS_SKELETON:
        raise StageError(f"{path} помечен status=skeleton: S2 не отработал")

    # Зоны в метрах имеют смысл только вместе с той гомографией, которой их
    # спроецировали. Молчаливое использование устаревшего geojson с новой
    # калибровкой один раз уже произошло: фасады вышли по 0.5 м вместо 2.3 м,
    # и заметно это стало только на картинке.
    from looq.io import sha256_file
    want = sha256_file(homography_path)
    got = doc.get("homography_sha256")
    if got is None:
        raise StageError(
            f"{path} записан без homography_sha256 — нечем убедиться, что зоны "
            f"спроецированы текущей калибровкой. Перегоните S2")
    if got != want:
        raise StageError(
            f"{path} спроецирован ДРУГОЙ гомографией (sha {got[:12]} против "
            f"{want[:12]}). Калибровка менялась после S2 — перегоните S2: "
            f"python -m looq.stages.s2_zones --config configs/s2_zones.yaml")
    facades, aprons = {}, {}
    roi = None
    for f in doc["features"]:
        pr = f["properties"]
        if pr["zone_type"] == "facade":
            facades[pr["storefront_id"]] = {
                "zone_id": pr["zone_id"], "name": pr["name_ru"],
                "seg": np.asarray(f["geometry"]["coordinates"], dtype=np.float64)}
        elif pr["zone_type"] == "apron":
            aprons[pr["storefront_id"]] = {
                "zone_id": pr["zone_id"],
                "poly": np.asarray(f["geometry"]["coordinates"][0][:-1], dtype=np.float64)}
        elif pr["zone_type"] == "roi":
            roi = np.asarray(f["geometry"]["coordinates"][0][:-1], dtype=np.float64)
    if not facades:
        raise StageError(f"{path}: нет ни одного facade — S6 нечего считать")
    return facades, aprons, roi, bool(doc.get("scale_known", True))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Непрерывные отрезки True. Возвращает пары индексов [начало, конец]."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def run(cfg: dict[str, Any], manifest: RunManifest, sampler) -> dict[str, Any]:
    import pandas as pd

    for art in ("track/tracks.parquet", "pose/orient.parquet"):
        if read_artifact_status(art) == STATUS_SKELETON:
            raise StageError(f"{art} помечен status=skeleton: предыдущий этап не отработал")

    facades, aprons, roi, scale_known = _load_zones(
        require(cfg, "input", "zones"), require(cfg, "input", "homography"))
    stop_cfg = require(cfg, "stop")
    gaze_cfg = require(cfg, "gaze")

    tracks = pd.read_parquet("track/tracks.parquet")
    orient = pd.read_parquet("pose/orient.parquet")
    df = tracks.merge(orient, on=["track_id", "frame_idx"], how="left",
                      suffixes=("", "_pose"))

    # Уточнённая опорная точка из S5, если есть; иначе базовая из S4.
    if "foot_x_m_refined" in df.columns:
        use_refined = df["foot_x_m_refined"].notna()
        df["x"] = np.where(use_refined, df["foot_x_m_refined"], df["foot_x_m"])
        df["y"] = np.where(use_refined, df["foot_y_m_refined"], df["foot_y_m"])
        refined_share = float(use_refined.mean())
    else:
        df["x"], df["y"] = df["foot_x_m"], df["foot_y_m"]
        refined_share = 0.0
    df = df[df["x"].notna() & df["y"].notna()]
    if df.empty:
        raise StageError("после отбрасывания точек без координат плана не осталось строк")

    # Единицы зон и треков обязаны совпадать. Совпадения sha гомографии для
    # этого НЕ достаточно: один и тот же файл содержит две матрицы в разных
    # единицах, и S2 однажды взял не ту. Сверяем порядок величин напрямую.
    zone_pts = np.vstack([f["seg"] for f in facades.values()])
    zone_span = float(np.ptp(zone_pts[:, 0]) + np.ptp(zone_pts[:, 1]))
    track_span = float(np.ptp(df["x"]) + np.ptp(df["y"]))
    if zone_span > 0 and not (0.05 <= zone_span / track_span <= 20.0):
        raise StageError(
            f"зоны и треки в разных единицах: разброс зон {zone_span:.2f} против "
            f"{track_span:.2f} у треков, отношение {zone_span / track_span:.3f}. "
            f"Скорее всего S2 спроецировал не той матрицей — в артефакте S1 их две, "
            f"H (метры) и H_px_to_unit (высоты камеры)")

    # Порог остановки.
    mode = str(stop_cfg.get("mode", "relative"))
    speeds = df["speed_mps"].dropna()
    median_speed = float(speeds.median()) if len(speeds) else None
    if mode == "absolute":
        if not scale_known:
            raise StageError(
                "stop.mode=absolute, но масштаб не определён: порог в м/с был бы "
                "выдуманным числом. Поставьте mode: relative")
        stop_thr = float(stop_cfg["speed_thr_mps"])
    else:
        if median_speed is None or not np.isfinite(median_speed) or median_speed <= 0:
            raise StageError("медианная скорость потока не определена — "
                             "относительный порог не от чего считать")
        stop_thr = float(stop_cfg["speed_thr_frac_of_median"]) * median_speed
    min_dur = float(stop_cfg["min_duration_s"])

    max_dist = (float(gaze_cfg["max_dist_m"]) if scale_known
                else float(gaze_cfg.get("max_dist_units_when_unscaled", 0.5)))
    half_width = float(gaze_cfg["yaw_uncertainty_deg"])
    min_gaze_score = float(gaze_cfg.get("min_score_for_event", 0.0))
    grazing_max = float(gaze_cfg["grazing_max_deg"])

    print(f"[{STAGE}] строк {len(df)}, треков {df['track_id'].nunique()}, "
          f"уточнённая опорная точка у {refined_share:.1%}")
    print(f"[{STAGE}] порог остановки: режим {mode}, {stop_thr:.4f} "
          f"({'м/с' if scale_known else 'усл.ед/с'}), медиана потока "
          f"{median_speed:.4f}")
    print(f"[{STAGE}] сектор ориентации: длина {max_dist}, полуширина +-{half_width} град, "
          f"порог по доле времени {min_gaze_score:.0%}, скользящий угол > {grazing_max} град")

    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    stats = {"stop": 0, "gaze": 0, "stop_and_gaze": 0, "pass_by": 0,
             "low_conf": 0, "thin_window": 0}

    for track_id, g in df.groupby("track_id"):
        g = g.sort_values("frame_idx")
        xs = g["x"].to_numpy(dtype=np.float64)
        ys = g["y"].to_numpy(dtype=np.float64)
        ts = g["ts"].to_numpy(dtype=np.float64)
        sp = g["speed_mps"].to_numpy(dtype=np.float64)
        yaw = (g["body_yaw_deg"].to_numpy(dtype=np.float64)
               if "body_yaw_deg" in g.columns else np.full(len(g), np.nan))
        head = (g["head_yaw_deg"].to_numpy(dtype=np.float64)
                if "head_yaw_deg" in g.columns else np.full(len(g), np.nan))
        # Голова информативнее корпуса, если измерена; иначе корпус.
        yaw_used = np.where(np.isfinite(head), head, yaw)
        frame0 = int(g["frame_idx"].iloc[0])

        for sid, fac in facades.items():
            apron = aprons.get(sid)
            seg_a, seg_b = fac["seg"][0], fac["seg"][1]

            in_apron = np.array([
                point_in_polygon_m((x, y), apron["poly"]) if apron else False
                for x, y in zip(xs, ys)])
            slow = np.isfinite(sp) & (sp < stop_thr)
            stop_mask = in_apron & slow

            gaze_mask = np.zeros(len(xs), dtype=bool)
            grazing = np.zeros(len(xs), dtype=bool)
            gaze_dist = np.full(len(xs), np.nan)
            for i in range(len(xs)):
                if not np.isfinite(yaw_used[i]):
                    continue
                dist, pt = sector_hits_segment_m((xs[i], ys[i]), yaw_used[i],
                                                 half_width, seg_a, seg_b, max_dist)
                if dist is None:
                    continue
                gaze_mask[i] = True
                gaze_dist[i] = dist
                if grazing_angle_deg(pt, (xs[i], ys[i]), seg_a, seg_b) > grazing_max:
                    grazing[i] = True

            # Знаменатель «проходил мимо витрины»: трек приближался к фасаду
            # на дистанцию сектора, независимо от того, куда был повёрнут.
            # Считать знаменателем только apron нельзя: полоса узкая, и почти
            # все прохожие выпадали бы из знаменателя, завышая любые доли.
            # Дистанция до ОТРЕЗКА, а не до его середины. Середина давала
            # круг вокруг центра фасада: 4807 из 35114 засчитанных кадров
            # лежали дальше 8 м от неё, то есть числитель не был подмножеством
            # знаменателя и доля могла превысить единицу.
            seg_v = seg_b - seg_a
            seg_len2 = float(seg_v @ seg_v) or 1e-12
            tt = np.clip(((xs - seg_a[0]) * seg_v[0]
                          + (ys - seg_a[1]) * seg_v[1]) / seg_len2, 0.0, 1.0)
            near = np.hypot(xs - (seg_a[0] + tt * seg_v[0]),
                            ys - (seg_a[1] + tt * seg_v[1])) <= max_dist
            frames_arr = g["frame_idx"].to_numpy(dtype=np.int64)
            for i in range(len(xs)):
                if not (in_apron[i] or gaze_mask[i] or near[i]):
                    continue          # строка ни о чём — не пишем
                frame_rows.append({
                    "frame_idx": int(frames_arr[i]), "track_id": int(track_id),
                    "zone_id": fac["zone_id"], "ts": float(ts[i]),
                    "x_m": float(xs[i]), "y_m": float(ys[i]),
                    "in_apron": bool(in_apron[i]),
                    "is_slow": bool(slow[i]),
                    "gaze_hit": bool(gaze_mask[i]),
                    "gaze_dist_m": None if not np.isfinite(gaze_dist[i]) else float(gaze_dist[i]),
                    "grazing": bool(grazing[i]),
                    "yaw_deg": None if not np.isfinite(yaw_used[i]) else float(yaw_used[i]),
                })

            n = len(xs)
            in_zone_n = int(in_apron.sum())
            # Знаменатель — кадры трека В ОКНЕ с измеренным углом, ровно как
            # в CLAUDE.md и docs/CONTRACTS.md 8.1: «доля времени трека В ОКНЕ».
            # Раньше делилось на ВСЕ кадры трека с измеренным углом, включая те,
            # где человек был за пределами восьми метров и попасть в фасад не
            # мог физически. Числитель и знаменатель жили на разных множествах,
            # и доля размывалась длиной трека, а не вниманием: медиана
            # знаменателя 64 кадра против 31 в окне.
            orient_n = int((near & np.isfinite(yaw_used)).sum())
            stop_runs = [(i, j) for i, j in _runs(stop_mask)
                         if ts[j] - ts[i] >= min_dur]
            stop_time = sum(ts[j] - ts[i] for i, j in stop_runs)
            stop_score = (stop_time / (ts[-1] - ts[0])
                          if in_zone_n and ts[-1] > ts[0] else None)
            gaze_score = (float(gaze_mask.sum()) / orient_n) if orient_n else None

            has_stop = bool(stop_runs)
            # Порог по ДОЛЕ ВРЕМЕНИ, а не по факту одного кадра: мгновенный
            # поворот головы у проходящего мимо — не внимание к витрине.
            has_gaze = gaze_score is not None and gaze_score >= min_gaze_score
            if has_stop and has_gaze:
                etype = "stop_and_gaze"
            elif has_stop:
                etype = "stop"
            elif has_gaze:
                etype = "gaze"
            elif in_zone_n or near.any():
                etype = "pass_by"      # знаменатель: прошёл мимо, не остановился
            else:
                continue

            # Порог по ДОЛЕ теряет смысл, когда знаменатель меньше 1/порог:
            # там уже ОДИН засчитанный кадр перешагивает порог, и правило
            # вырождается в «хотя бы раз посмотрел» — ровно то, против чего
            # порог и введён (см. комментарий к min_score_for_event).
            # Граница выведена из самого порога, а не назначена: при
            # min_score_for_event = 0.20 это 5 кадров, полсекунды при 10 fps.
            # Замер до введения границы: 39% засчитанных событий имели
            # знаменатель меньше 10 кадров, 27% — меньше 5, у витрины M4
            # медиана знаменателя была 3 кадра.
            min_obs = int(np.ceil(1.0 / min_gaze_score)) if min_gaze_score > 0 else 1
            thin = has_gaze and orient_n < min_obs
            low_conf = bool(grazing.any()) or orient_n == 0 or thin
            stats[etype] += 1
            if low_conf:
                stats["low_conf"] += 1
            if thin:
                stats["thin_window"] += 1
            rows.append({
                "event_id": f"{int(track_id)}:{fac['zone_id']}:{etype}:{frame0}",
                "track_id": int(track_id), "zone_id": fac["zone_id"],
                "t_start": float(ts[0]), "t_end": float(ts[-1]),
                "stop_score": None if stop_score is None else float(min(1.0, stop_score)),
                "gaze_score": None if gaze_score is None else float(min(1.0, gaze_score)),
                "event_type": etype, "low_confidence": low_conf,
            })

    if not rows:
        raise StageError("ни одного события: ни один трек не попал ни в одну зону")

    manifest.note("events_by_type", stats)
    manifest.note("median_speed", median_speed)
    manifest.note("stop_threshold", stop_thr)
    manifest.note("stop_mode", mode)
    manifest.note("refined_foot_share", round(refined_share, 4))
    manifest.note("scale_known", scale_known)
    print(f"[{STAGE}] событий {len(rows)}: {stats}")
    print(f"[{STAGE}] из них low_confidence {stats['low_conf']} "
          f"(скользящий угол или ориентация не измерена — в основной агрегат не идут)")
    print(f"[{STAGE}] покадровый след: {len(frame_rows)} строк -> {FRAMES_OUTPUT}")
    return {"rows": rows, "frame_rows": frame_rows, "stats": stats,
            "median_speed": median_speed, "stop_threshold": stop_thr,
            "scale_known": scale_known}


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
        sampler = build_evidence(cfg, STAGE, manifest)
        res = run(cfg, manifest, sampler)
        # Оба артефакта читаются S8 совместно, поэтому подменяются разом.
        tmps = [
            write_parquet(OUTPUT, OUTPUT_COLS, res["rows"], STAGE, STATUS_OK,
                          inputs=INPUTS, commit=False),
            write_parquet(FRAMES_OUTPUT, FRAMES_COLS, res["frame_rows"], STAGE,
                          STATUS_OK, inputs=INPUTS, commit=False),
        ]
        commit_parquet(tmps)
        manifest.note("output_artifacts", [OUTPUT, FRAMES_OUTPUT])
        manifest.note("elapsed_s", round(time.time() - _t0, 1))
        manifest.finish(STATUS_OK)
        print(f"[{STAGE}] записано: {OUTPUT} ({len(res['rows'])} строк)")
        print(f"[{STAGE}] пруфы не собраны: S6 работает по артефактам, кадров не читает")
        return 0
    except (StageError, EvidenceError, ConfigError, OSError, ValueError, KeyError) as exc:
        if manifest is not None:
            manifest.finish("failed", error=str(exc))
        print(f"[{STAGE}] ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
