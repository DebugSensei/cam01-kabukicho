"""S5 pose+orient — ориентация корпуса и головы.

Это ПОВОРОТ КОРПУСА И ГОЛОВЫ, а не направление взгляда. Формулировка идёт
в UI и в отчёт дословно (looq.geometry.ORIENTATION_DISCLAIMER).

Как считается угол
------------------
Отрезок плеч в мире горизонтален, значит его направление на плане полностью
определяется ПРЯМОЙ, на которой он лежит в кадре. Гомография переводит эту
прямую в прямую на плане, и её направление и есть направление плеч. Высота
плеч при этом не нужна вовсе, как и матрицы K и R.

Это важно практически: запасной путь калибровки (manual_ground_plane) даёт
только плоскость земли, без камеры. Метод через план работает с обоими путями,
а прежний, через обратную проекцию на высоту плеч, работал бы только с одним.
Заодно исчезли два неоткалиброванных параметра — доли роста для плеч и ушей.

Неоднозначность 180 градусов разрешается сама: модель размечает плечи
анатомически, левое и правое. Человек, стоящий лицом в направлении f, держит
левое плечо слева от f, поэтому поворот вектора «левое -> правое» на +90
против часовой стрелки и есть направление корпуса. Гадать по движению нельзя:
тогда ориентация перестала бы быть независимой от траектории, а весь смысл
S6 в том, чтобы сравнивать их между собой.

Почему S5 снова читает видео. В track/tracks.parquet по контракту нет рамок —
только опорная точка. Кропа взять неоткуда, поэтому поза считается по кадрам,
а к трекам детекции привязываются по ближайшей опорной точке.

    python -m looq.stages.s5_orient --config configs/s5_orient.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from looq import STATUS_OK, STATUS_SKELETON
from looq.geometry import smooth_yaw_series_deg
from looq.calib import CalibError, apply_h, yaw_from_pair_via_horizon
from looq.evidence import EvidenceError
from looq.io import ConfigError, RunManifest, load_config, read_json, require
from looq.pilot import PilotError, infer_params, iter_frames
from looq.stages._base import (
    Col,
    StageError,
    build_evidence,
    finalize_evidence,
    read_artifact_status,
    validate_inputs,
    write_parquet,
)

STAGE = "s5_orient"

INPUTS = ["track/tracks.parquet", "calib/homography.json"]
OUTPUT = "pose/orient.parquet"

OUTPUT_COLS: list[Col] = [
    Col("track_id",     "int32",   False, "-",      "-",       "идентификатор трека"),
    Col("frame_idx",    "int64",   False, "кадр",   "-",       "номер кадра"),
    Col("ts",           "float64", False, "с",      "-",       "секунды от начала записи"),
    Col("body_yaw_deg", "float32", True,  "градус", "plane_m", "ПОВОРОТ КОРПУСА, не взгляд. 0 = +x плана, против часовой, [0,360). null = не измерено"),
    Col("head_yaw_deg", "float32", True,  "градус", "plane_m", "ПОВОРОТ ГОЛОВЫ, не взгляд. Та же конвенция. null = не измерено"),
    Col("yaw_conf",     "float32", True,  "[0,1]",  "-",       "уверенность угла"),
    Col("n_kpts_valid", "int8",    False, "шт",     "-",       "видимых keypoints, 0..17; 0 — валидное значение"),
    # Уточнённая опорная точка. S4 пишет bbox_bottom (косвенное), но поза здесь
    # уже посчитана, и лодыжки достаются бесплатно — без второго прохода модели.
    # S6 берёт refined, если он есть, иначе базовую точку из S4.
    Col("foot_x_m_refined", "float32", True, "м", "plane_m", "опорная точка по лодыжкам, x; null если лодыжек не видно"),
    Col("foot_y_m_refined", "float32", True, "м", "plane_m", "опорная точка по лодыжкам, y; null если лодыжек не видно"),
    Col("foot_refined_source", "string", True, "-", "-",      "ankle | null. ПРАВИЛО 7: ankle — ПРЯМОЕ измерение, в отличие от bbox_bottom в S4"),
]

# COCO-17: индексы, которые нам нужны.
KP_LEFT_EAR, KP_RIGHT_EAR = 3, 4
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_ANKLE, KP_RIGHT_ANKLE = 15, 16


def _ground_homography(path: str) -> tuple[np.ndarray, bool]:
    """Гомография кадр -> план и признак, известен ли масштаб.

    Работает с обоими путями калибровки: auto_vp_height даёт метры,
    manual_ground_plane — условные единицы. Угол от масштаба не зависит,
    поэтому ориентация считается одинаково; а вот подписывать длины метрами
    во втором случае нельзя, и scale_known это фиксирует.
    """
    doc = read_json(path)
    if doc.get("status") == STATUS_SKELETON:
        raise StageError(f"{path} помечен status=skeleton: S1 не отработал")
    h = doc.get("H_px_to_unit") or doc.get("H")
    if h is None:
        raise StageError(f"{path}: нет ни H_px_to_unit, ни H")
    hm = np.asarray(h, dtype=np.float64)
    if hm.shape != (3, 3):
        raise StageError(f"{path}: гомография должна быть 3x3, получено {hm.shape}")
    return hm, bool(doc.get("scale_known", True))


def _pair_yaw(h_px_to_unit: np.ndarray, kps: np.ndarray, confs: np.ndarray,
              i_left: int, i_right: int,
              kp_conf_thr: float) -> tuple[float | None, float | None]:
    """Угол по анатомической паре точек. None, если пара не видна уверенно."""
    if confs[i_left] < kp_conf_thr or confs[i_right] < kp_conf_thr:
        return None, None
    yaw = yaw_from_pair_via_horizon(h_px_to_unit, kps[i_left], kps[i_right])
    if not np.isfinite(yaw):
        return None, None
    # Уверенность угла — по ХУДШЕЙ из двух точек: угол определяется парой
    # целиком, и одна плохая точка портит его так же, как две.
    return float(yaw), float(min(confs[i_left], confs[i_right]))


def run(cfg: dict[str, Any], manifest: RunManifest, sampler) -> dict[str, Any]:
    import pandas as pd
    from ultralytics import YOLO

    warnings = validate_inputs(INPUTS, STAGE)
    for w in warnings:
        print(f"[{STAGE}] ВНИМАНИЕ: {w}", file=sys.stderr)
    if read_artifact_status("track/tracks.parquet") == STATUS_SKELETON:
        raise StageError(
            "track/tracks.parquet помечен status=skeleton: S4 не отработал. "
            "Поза без треков дала бы углы, которые не к чему привязать (правило 8)")

    h_px_to_unit, scale_known = _ground_homography(require(cfg, "input", "homography"))
    if not scale_known:
        print(f"[{STAGE}] масштаб не определён: длины в условных единицах. "
              f"На УГОЛ это не влияет — он от масштаба не зависит")
    clip = Path(require(cfg, "input", "clip"))
    if not clip.is_file():
        raise StageError(f"нет видео {clip}")

    orient_cfg = require(cfg, "orient")
    kp_conf_thr = float(orient_cfg["kp_conf_thr"])
    min_kpts = int(orient_cfg["min_kpts_valid"])
    min_bbox_h = float(orient_cfg["min_bbox_h_px"])
    match_max_dist = float(orient_cfg["match_max_dist_px"])

    params = infer_params(cfg)
    weights = params["weights"]
    if not Path(str(weights)).is_file():
        raise StageError(
            f"нет весов позы {weights}. Скачайте их сами: автоматическая загрузка "
            f"весов в этом проекте запрещена")
    model = YOLO(weights)

    tracks = pd.read_parquet("track/tracks.parquet")
    if tracks.empty:
        raise StageError("track/tracks.parquet пуст")
    frames = sorted(tracks["frame_idx"].astype(int).unique())
    by_frame = {int(f): g for f, g in tracks.groupby("frame_idx")}
    print(f"[{STAGE}] треков {tracks['track_id'].nunique()}, кадров {len(frames)}")

    rows: list[dict[str, Any]] = []
    stats = {"pose_persons": 0, "matched": 0, "unmatched": 0, "body_ok": 0,
             "head_ok": 0, "ankle_ok": 0, "too_small": 0, "few_kpts": 0}
    t0 = time.time()

    for frame_idx, frame in iter_frames(clip, np.asarray(frames, dtype=np.int64)):
        grp = by_frame.get(int(frame_idx))
        if grp is None:
            continue
        res = model.predict(frame, imgsz=params["imgsz"], conf=params["conf"],
                            device=params["device"], half=params["half"], verbose=False)
        kp = res[0].keypoints
        boxes = res[0].boxes
        if kp is None or boxes is None or len(boxes) == 0:
            continue
        xy = kp.xy.cpu().numpy()                       # (N, 17, 2)
        kconf = kp.conf.cpu().numpy() if kp.conf is not None else np.zeros(xy.shape[:2])
        xyxy = boxes.xyxy.cpu().numpy()
        stats["pose_persons"] += len(xyxy)

        # Опорная точка каждой позы — низ рамки по центру, как в S4.
        pose_foot = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]], axis=1)
        track_foot = grp[["foot_x_px", "foot_y_px"]].to_numpy(dtype=np.float64)
        track_ids = grp["track_id"].to_numpy(dtype=np.int64)
        ts_by_track = dict(zip(track_ids.tolist(), grp["ts"].astype(float).tolist()))

        used: set[int] = set()
        for t_i, (tx, ty) in enumerate(track_foot):
            # Ближайшая свободная поза, но не дальше match_max_dist_px.
            # Одна поза — одному треку: иначе двое рядом получили бы один угол.
            d = np.hypot(pose_foot[:, 0] - tx, pose_foot[:, 1] - ty)
            p_i = None
            for cand in np.argsort(d):
                cand = int(cand)
                if cand in used:
                    continue
                if d[cand] <= match_max_dist:
                    p_i = cand
                break
            track_id = int(track_ids[t_i])
            ts = float(ts_by_track[track_id])

            if p_i is None:
                stats["unmatched"] += 1
                # Строка всё равно пишется: «позу пытались посчитать и не смогли»
                # и «попытки не было» — разные вещи, и доля непокрытия считается
                # только если обе видны (правило 7).
                rows.append({"track_id": track_id, "frame_idx": int(frame_idx), "ts": ts,
                             "body_yaw_deg": None, "head_yaw_deg": None,
                             "yaw_conf": None, "n_kpts_valid": 0,
                             "foot_x_m_refined": None, "foot_y_m_refined": None,
                             "foot_refined_source": None})
                continue
            used.add(p_i)
            stats["matched"] += 1

            confs = kconf[p_i]
            kps = xy[p_i]
            n_valid = int((confs >= kp_conf_thr).sum())
            x1, y1, x2, y2 = xyxy[p_i]
            bbox_h = float(y2 - y1)

            body = head = None
            body_conf = head_conf = None
            if bbox_h <= min_bbox_h:
                stats["too_small"] += 1
            elif n_valid < min_kpts:
                stats["few_kpts"] += 1
            else:
                body, body_conf = _pair_yaw(h_px_to_unit, kps, confs,
                                            KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
                                            kp_conf_thr)
                head, head_conf = _pair_yaw(h_px_to_unit, kps, confs,
                                            KP_LEFT_EAR, KP_RIGHT_EAR, kp_conf_thr)
            if body is not None:
                stats["body_ok"] += 1
            if head is not None:
                stats["head_ok"] += 1

            # Уточнённая опорная точка: середина между лодыжками. Это ПРЯМОЕ
            # измерение, в отличие от низа рамки, который в толпе лежит на
            # чужой спине. Поза уже посчитана, второй проход модели не нужен.
            fx = fy = None
            refined_source = None
            if (confs[KP_LEFT_ANKLE] >= kp_conf_thr
                    and confs[KP_RIGHT_ANKLE] >= kp_conf_thr):
                mid = (kps[KP_LEFT_ANKLE] + kps[KP_RIGHT_ANKLE]) / 2.0
                try:
                    m = apply_h(h_px_to_unit, mid[None, :])[0]
                    if np.all(np.isfinite(m)):
                        fx, fy = float(m[0]), float(m[1])
                        refined_source = "ankle"
                        stats["ankle_ok"] += 1
                except CalibError:
                    pass

            confs_used = [c for c in (body_conf, head_conf) if c is not None]
            rows.append({
                "track_id": track_id, "frame_idx": int(frame_idx), "ts": ts,
                "body_yaw_deg": body, "head_yaw_deg": head,
                "yaw_conf": float(max(confs_used)) if confs_used else None,
                "n_kpts_valid": int(min(n_valid, 127)),
                "foot_x_m_refined": fx,
                "foot_y_m_refined": fy,
                "foot_refined_source": refined_source,
            })
            _offer_evidence(sampler, frame, xyxy[p_i], track_id, frame_idx, ts,
                            body, body_conf, head)

    elapsed = time.time() - t0
    if not rows:
        raise StageError("ни одной строки ориентации — поза не нашла людей на кадрах треков")

    n = len(rows)
    body_cov = stats["body_ok"] / n
    head_cov = stats["head_ok"] / n
    manifest.note("elapsed_s", round(elapsed, 1))
    manifest.note("pose_stats", stats)
    ankle_cov = stats["ankle_ok"] / n
    manifest.note("body_yaw_coverage", round(body_cov, 4))
    manifest.note("head_yaw_coverage", round(head_cov, 4))
    manifest.note("foot_refined_coverage", round(ankle_cov, 4))
    manifest.note("scale_known", scale_known)
    print(f"[{STAGE}] строк {n}, поз найдено {stats['pose_persons']}, "
          f"привязано {stats['matched']}, без пары {stats['unmatched']}")
    print(f"[{STAGE}] покрытие: корпус {body_cov:.1%}, голова {head_cov:.1%} "
          f"(правило 7: непокрытие идёт в отчёт отдельным числом, "
          f"а не превращается в «не смотрел»)")
    print(f"[{STAGE}] уточнённая опорная точка по лодыжкам: {ankle_cov:.1%} строк "
          f"(остальные останутся на bbox_bottom из S4 — косвенные, правило 7)")
    print(f"[{STAGE}] отсеяно: мелкие {stats['too_small']}, мало keypoints {stats['few_kpts']}")
    n_smoothed = _smooth_rows(rows, int(orient_cfg.get("smooth_window_frames", 0)))
    if n_smoothed:
        manifest.note("smooth_window_frames", int(orient_cfg["smooth_window_frames"]))
        manifest.note("rows_smoothed", n_smoothed)
        print(f"[{STAGE}] углы сглажены круговой медианой, окно "
              f"{orient_cfg['smooth_window_frames']} кадров: изменено {n_smoothed} строк. "
              f"Причина и цена — в configs/s5_orient.yaml")

    return {"rows": rows, "body_coverage": body_cov, "head_coverage": head_cov,
            "foot_refined_coverage": ankle_cov}


def _smooth_rows(rows: list[dict[str, Any]], window: int) -> int:
    """Сглаживает углы по каждому треку НА МЕСТЕ. Возвращает число изменённых строк.

    Делается ПОСЛЕ сбора всех строк, а не по ходу: медиана окна требует и
    будущих кадров, а поток идёт по времени. Пруфы уже отобраны по сырым
    значениям, и это правильно — на кропе видно то, что выдала модель.
    """
    if window < 3:
        return 0
    by_track: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_track.setdefault(int(r["track_id"]), []).append(i)
    changed = 0
    for idx in by_track.values():
        idx.sort(key=lambda i: rows[i]["frame_idx"])
        for col in ("body_yaw_deg", "head_yaw_deg"):
            raw = np.array([np.nan if rows[i][col] is None else float(rows[i][col])
                            for i in idx], dtype=np.float64)
            sm = smooth_yaw_series_deg(raw, window)
            for k, i in enumerate(idx):
                if np.isfinite(sm[k]) and (not np.isfinite(raw[k])
                                           or abs(sm[k] - raw[k]) > 1e-9):
                    rows[i][col] = float(sm[k])
                    changed += 1
    return changed


def _offer_evidence(sampler, frame, box, track_id, frame_idx, ts,
                    body, body_conf, head) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 4:
        return
    if body is not None:
        sampler.offer("claim.orient.body_yaw", track_id=track_id, frame_idx=int(frame_idx),
                      ts=float(ts), crop_bgr=crop, value=float(body),
                      confidence=float(body_conf or 0.0),
                      extra={"body_yaw_deg": round(float(body), 1),
                             "head_yaw_deg": None if head is None else round(float(head), 1)})
        if (body_conf or 0.0) < 0.5:
            sampler.offer("claim.orient.low_conf", track_id=track_id,
                          frame_idx=int(frame_idx), ts=float(ts), crop_bgr=crop,
                          value=float(body), confidence=float(body_conf or 0.0))
    else:
        # Пруфы непокрытия так же важны, как пруфы измеренного: по ним видно,
        # у кого именно угол не посчитался.
        sampler.offer("claim.orient.missing", track_id=track_id, frame_idx=int(frame_idx),
                      ts=float(ts), crop_bgr=crop, value=0.0, confidence=0.0)


def main(argv=None) -> int:
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
        write_parquet(OUTPUT, OUTPUT_COLS, res["rows"], STAGE, STATUS_OK,
                      inputs=INPUTS)
        index = finalize_evidence(sampler, manifest)

        manifest.note("output_artifact", OUTPUT)
        manifest.finish(STATUS_OK)
        print(f"[{STAGE}] записано: {OUTPUT} ({len(res['rows'])} строк), пруфы {index}")
        return 0

    except (StageError, EvidenceError, CalibError, PilotError, ConfigError, OSError,
            ValueError, KeyError) as exc:
        if manifest is not None:
            manifest.finish("failed", error=str(exc))
        print(f"[{STAGE}] ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
