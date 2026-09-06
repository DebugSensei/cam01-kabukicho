"""S1 calib — автоматическая калибровка камеры по точкам схода и росту людей.

Ручных замеров геометрии нет. Порядок работы — docs/S1_TASK.md:

    клики-затравки -> точки схода (RANSAC) -> плоскость земли с точностью
    до масштаба -> пилотная детекция -> масштаб по медиане роста
    -> независимые проверки

S1 самодостаточен: он НЕ ждёт артефактов S3/S4 и не может их ждать — S3 пишет
foot_x_m, для чего нужна уже готовая гомография. Порядок строго S1 -> S3 -> S4.
Пилотные результаты живут в calib/pilot/ и НИКОГДА не пишутся в det/ и track/.

    python -m looq.stages.s1_calib --config configs/s1_calib.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from looq import SCHEMA_VERSION, STATUS_OK
from looq.calib import (
    CalibError,
    camera_from_focal_and_vps,
    homography_from_ground_rect,
    apply_h,
    camera_from_vps,
    detect_segments,
    fit_line_px,
    height_depth_slope,
    lines_common_vp,
    person_height_units,
    ransac_vp,
    scale_from_heights,
    vp_candidates_on_line,
)
from looq.geometry import ORIENTATION_DISCLAIMER, facade_lines_separation_m, height_spread_stats
from looq.anonymise import save_figure, save_image
from looq.evidence import EvidenceError
from looq.io import (ConfigError, RunManifest, load_config, read_json,
                     require, sha256_file, write_json)
from looq.pilot import (
    PilotError,
    detect_people,
    filter_height_samples,
    infer_params,
    sample_burst_indices,
    track_speeds,
)
from looq.selfcalib import (focal_from_horizon_and_vertical,
                            implied_street_width_m,
                            street_grade_from_height_drift,
                            horizon_from_pedestrian_pairs,
                            point_on_line_distance_px,
                            scale_from_street_width,
                            vertical_vp_from_pedestrians)
from looq.stages._base import StageError, build_evidence, finalize_evidence

STAGE = "s1_calib"


# --------------------------------------------------------------------------- #

def _reference_frame(clip: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise StageError(f"cv2 не открыл {clip}")
    frame = None
    for i in range(frame_idx + 1):
        ok, f = cap.read()
        if not ok:
            break
        if i == frame_idx:
            frame = f
    cap.release()
    if frame is None:
        raise StageError(f"в {clip} нет опорного кадра {frame_idx}")
    return frame


def _clip_fps(clip: Path) -> float:
    cap = cv2.VideoCapture(str(clip))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not (1.0 < fps < 240.0) or n < 2:
        raise StageError(f"неправдоподобные параметры клипа: fps={fps}, кадров={n}")
    return fps, n


def _split_segments(segs: np.ndarray, vp_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Кандидаты для обеих точек схода — ВСЕ отрезки кадра.

    Раньше здесь стояло деление по абсолютному наклону: «горизонтальные» шли
    на уличную VP, «вертикальные» на вертикальную. Это оказалось неверным
    допущением о геометрии сцены. На реальном кадре Kabukicho правая линия
    улицы (L3) имеет наклон 65 градусов и попадала в вертикальную корзину,
    из-за чего уличная VP теряла половину подтверждающих отрезков.

    Отбор делает сам RANSAC: инлаером считается отрезок, чья прямая проходит
    близко от КАНДИДАТА, а кандидат берётся из кликов пользователя. Это
    правильный признак принадлежности семейству, в отличие от наклона в кадре,
    который зависит от того, куда повёрнута камера.
    """
    return segs, segs


# --------------------------------------------------------------------------- #

def _pedestrian_boxes(det_path: str, shape, cfg: dict):
    """Детекции, годные для геометрии. Фильтры те же, что в пилоте S1."""
    import pandas as pd

    h, w = shape
    d = pd.read_parquet(det_path)
    pc = cfg
    bh = d["y2_px"] - d["y1_px"]
    edge = float(pc.get("edge_margin_px", 2))
    keep = ((d["conf"] > float(pc.get("min_conf", 0.5)))
            & (bh > float(pc.get("min_bbox_h_px_for_geometry", 80)))
            & (d["x1_px"] > edge) & (d["y1_px"] > edge)
            & (d["x2_px"] < w - edge) & (d["y2_px"] < h - edge))
    d = d[keep]
    max_iou = float(pc.get("max_neighbour_iou", 0.2))
    idx = []
    for _, g in d.groupby("frame_idx"):
        b = g[["x1_px", "y1_px", "x2_px", "y2_px"]].to_numpy()
        if len(b) == 1:
            idx.extend(g.index.tolist())
            continue
        a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        ix1 = np.maximum(b[:, None, 0], b[None, :, 0])
        iy1 = np.maximum(b[:, None, 1], b[None, :, 1])
        ix2 = np.minimum(b[:, None, 2], b[None, :, 2])
        iy2 = np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / (a[:, None] + a[None, :] - inter)
        np.fill_diagonal(iou, 0.0)
        idx.extend(g.index[iou.max(1) < max_iou].tolist())
    return d.loc[idx]


def _pose_segments(clip: Path, frames, params: dict, cfg: dict):
    """Отрезки голеностоп-макушка ИЗ ПОЗЫ.

    Из рамки такие отрезки взять нельзя: низ и верх берутся по центру рамки,
    то есть с одинаковым x, отрезок вертикален по построению, и вертикальной
    точки схода из него не получить. Проверено на реальных детекциях.
    Настоящие голеностоп и нос смещены по x на 15 px в медиане — это смещение
    и несёт перспективу.
    """
    from ultralytics import YOLO

    from looq.pilot import iter_frames

    KP_NOSE, KP_L_ANKLE, KP_R_ANKLE = 0, 15, 16
    thr = float((cfg.get("orient") or {}).get("kp_conf_thr", 0.3))
    min_h = float(cfg.get("min_bbox_h_px_for_geometry", 80))
    model = YOLO(params["weights_pose"])
    feet, heads = [], []
    for _, frame in iter_frames(clip, np.asarray(frames, dtype=np.int64)):
        r = model.predict(frame, imgsz=params["imgsz"], conf=0.5,
                          device=params["device"], half=params["half"],
                          verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            continue
        xy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy()
        bb = r.boxes.xyxy.cpu().numpy()
        for k in range(len(xy)):
            if (bb[k, 3] - bb[k, 1]) < min_h:
                continue
            if min(kc[k, KP_L_ANKLE], kc[k, KP_R_ANKLE], kc[k, KP_NOSE]) < thr:
                continue
            feet.append((xy[k, KP_L_ANKLE] + xy[k, KP_R_ANKLE]) / 2.0)
            heads.append(xy[k, KP_NOSE])
    return np.asarray(feet), np.asarray(heads)


def _rescale_from_height(res: dict, cfg: dict[str, Any],
                         manifest: RunManifest) -> dict:
    """Пересчёт масштаба из роста вместо ширины улицы."""
    doc = res["doc"]
    sc = cfg["self_calib"]
    target_h = float(sc.get("target_height_m", 1.68))
    factor = target_h / doc["height_median_m"]

    old_scale = doc["scale_m_per_unit"]
    new_scale = old_scale * factor
    doc["scale_m_per_unit"] = new_scale
    doc["camera_height_m"] = new_scale
    doc["H"] = (np.diag([new_scale, new_scale, 1.0])
                @ np.asarray(doc["H_px_to_unit"], dtype=np.float64)).tolist()
    doc["calib_status"] = "scale_from_height"
    doc["scale_known"] = True
    doc["direction"] = "px_to_m"
    doc["scale_source_ru"] = (
        f"медиана роста {target_h} м. Ширина улицы как источник масштаба "
        f"ОТВЕРГНУТА: эталон {doc['street_width_reference_m']} м даёт медиану "
        f"{doc['height_median_m']:.2f} м, что вне 1.55-1.75.")
    doc["scale_rescale_factor"] = factor
    doc["height_median_m"] = doc["height_median_m"] * factor
    doc["height_iqr_m"] = doc["height_iqr_m"] * factor
    doc["height_p10"] = doc["height_p10"] * factor
    doc["height_p90"] = doc["height_p90"] * factor
    # Наклон НЕ пересчитывается: м/м инвариантен к масштабу. Ниже рост и глубина
    # умножаются на один и тот же factor, а dh/dz от этого не меняется. Прежний
    # код домножал наклон ещё раз и занижал его ровно в factor раз: артефакт
    # хранил -0.01634 там, где гейт по тем же массивам получает -0.01876, и
    # уклон 4.9 % согласовывался со вторым числом, а не с первым.
    doc["pilot_heights_m"] = [round(v * factor, 4) for v in doc["pilot_heights_m"]]
    doc["pilot_depths_m"] = [round(v * factor, 4) for v in doc["pilot_depths_m"]]
    doc["independent_checks_ru"] = [
        "ВНИМАНИЕ: рост БОЛЬШЕ НЕ является независимой проверкой — он задаёт масштаб",
        f"осталась одна проверка: подразумеваемая ширина L1-L3 "
        f"{doc['street_width_L1L3_implied_m']:.2f} м попадает в правдоподобный "
        f"диапазон {doc['street_width_implied_range_m']}",
        "точка схода улицы из кликов проверена на горизонте по пешеходам",
    ]
    doc["banner_ru"] = (
        f"Масштаб взят ИЗ РОСТА, а не из ширины улицы. Подразумеваемое "
        f"расстояние L1-L3 = {doc['street_width_L1L3_implied_m']:.2f} м против "
        f"эталона {doc['street_width_reference_m']} м: эталон замерен "
        f"здание-здание, а кликнут бордюр. Рост при таком масштабе "
        f"НЕ является независимой проверкой.")
    res["heights_m"] = np.asarray(res["heights_m"]) * factor
    res["foot_m"] = np.asarray(res["foot_m"]) * factor
    res["scale"] = new_scale
    manifest.note("calib_status", "scale_from_height")
    manifest.note("scale_rescale_factor", factor)
    print(f"[{STAGE}] МАСШТАБ ИЗ РОСТА: множитель {factor:.4f}, "
          f"высота камеры {new_scale:.2f} м")
    print(f"[{STAGE}] рост больше НЕ независимая проверка — он задаёт масштаб")
    return res


def evaluate_self_calib(doc: dict[str, Any], cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Судит геометрию и масштаб ОТДЕЛЬНО. Это разные вещи с разными следствиями.

    Геометрия плоскости определяет УГЛЫ: луч ориентации, принадлежность к зонам,
    ранжирование витрин. Масштаб определяет МЕТРЫ. Кривой масштаб при верной
    геометрии не портит ни один угол — он портит только подписи длин.

    Поэтому и статусов три, а не два: calibrated, angles_ok_scale_unverified,
    stub_affine. Свалить второй в третий значило бы выбросить рабочие углы
    из-за неверной линейки.
    """
    gates = cfg.get("gates") or {}
    sc = cfg.get("self_calib") or {}
    geom, scale = [], []

    lo, hi = gates.get("focal_over_diagonal_range", [0.3, 4.0])
    fod = doc["focal_over_diagonal"]
    if not (lo <= fod <= hi):
        geom.append(f"фокус {fod:.2f} диагонали вне {lo}-{hi}")
    if doc["horizon_inlier_frac"] < float(sc.get("horizon_min_inlier_frac", 0.2)):
        geom.append(f"горизонт подтверждён {doc['horizon_inlier_frac']:.0%} пар")
    if doc["vertical_inlier_frac"] < float(sc.get("vertical_min_inlier_frac", 0.3)):
        geom.append(f"вертикаль подтверждена {doc['vertical_inlier_frac']:.0%} отрезков")
    # Расхождение "VP улицы против горизонта" бьёт по МАСШТАБУ, а не по углам.
    # Ориентацию плоскости задают две точки схода напрямую; горизонт участвует
    # только в величине фокуса, а фокус — это масштаб и зависимость от глубины.
    # Поэтому проверка внизу, в scale, а не здесь.

    tol = doc.get("vp_horizon_tol_used_px", float(sc.get("vp_horizon_tol_px", 40.0)))
    if doc["vp_street_to_horizon_px"] > tol:
        scale.append(f"точка схода улицы в {doc['vp_street_to_horizon_px']:.0f} px "
                     f"от горизонта при допуске {tol:.0f}: две независимые оценки "
                     f"не сошлись, фокус и с ним масштаб под вопросом")
    if not doc.get("height_depth_slope_covers_zero", True):
        scale.append(f"рост плывёт с глубиной: наклон "
                     f"{doc['height_depth_slope']:+.4f} м/м, ноль не накрыт")

    hlo, hhi = gates.get("median_height_range_m", [1.55, 1.75])
    med = doc["height_median_m"]
    if not (hlo <= med <= hhi):
        scale.append(f"медиана роста {med:.2f} м вне {hlo}-{hhi} "
                     f"(масштаб завышен в {med / ((hlo + hhi) / 2):.2f} раза)")
    if doc["height_iqr_m"] > float(gates.get("height_iqr_max_m", 0.18)):
        scale.append(f"IQR роста {doc['height_iqr_m']:.2f} м при пороге "
                     f"{gates.get('height_iqr_max_m')}")
    return {"geometry": geom, "scale": scale}


def run_self_calib_pedestrians(cfg: dict[str, Any], manifest: RunManifest) -> dict[str, Any]:
    """Калибровка по пешеходам. Кликов требует только для направления улицы.

    Порядок: горизонт по парам людей -> вертикальная точка схода по позам ->
    фокус из ортогональности горизонта и вертикали -> план земли ->
    масштаб из ШИРИНЫ УЛИЦЫ. Рост в подгонке не участвует и потому становится
    независимой проверкой.
    """
    clip = Path(require(cfg, "input", "clip"))
    hints = load_config(require(cfg, "input", "hints"))
    sc = require(cfg, "self_calib")
    control = require(cfg, "control")

    frame_idx = int(hints.get("frame_idx", 0))
    ref = _reference_frame(clip, frame_idx)
    shape = (ref.shape[0], ref.shape[1])
    h_img, w_img = shape

    # --- 1. горизонт по парам людей (рамок хватает и их много) --------------- #
    det = _pedestrian_boxes("det/frames.parquet", shape, sc)
    if det.empty:
        raise StageError("после фильтров не осталось детекций для геометрии")
    feet_box = np.stack([(det.x1_px + det.x2_px) / 2.0, det.y2_px], axis=1)
    heads_box = np.stack([(det.x1_px + det.x2_px) / 2.0, det.y1_px], axis=1)
    hor = horizon_from_pedestrian_pairs(
        feet_box, heads_box, det.frame_idx.to_numpy(),
        min_depth_sep_px=float(sc["min_depth_sep_px"]),
        ransac_tol_px=float(sc["horizon_ransac_tol_px"]),
        min_inlier_frac=float(sc["horizon_min_inlier_frac"]))
    print(f"[{STAGE}] горизонт: пар {hor.n_pairs}, инлаеров {hor.n_inliers} "
          f"({hor.inlier_frac:.0%}), невязка {hor.residual_px:.1f} px")

    # --- 2. вертикальная VP по позам ----------------------------------------- #
    all_frames = sorted(det.frame_idx.unique())
    step = max(1, len(all_frames) // int(sc["pose_frames"]))
    pose_frames = all_frames[::step][:int(sc["pose_frames"])]
    params = {**infer_params(load_config(require(cfg, "input", "detect_config"))),
              "weights_pose": sc["pose_weights"]}
    feet_p, heads_p = _pose_segments(clip, pose_frames, params, sc)
    print(f"[{STAGE}] поз с уверенными лодыжками и носом: {len(feet_p)} "
          f"по {len(pose_frames)} кадрам")
    if len(feet_p) < 20:
        raise StageError(f"поз для вертикальной точки схода {len(feet_p)} при минимуме 20")
    vert = vertical_vp_from_pedestrians(
        feet_p, heads_p, inlier_angle_deg=float(sc["vertical_inlier_angle_deg"]),
        min_inlier_frac=float(sc["vertical_min_inlier_frac"]))
    print(f"[{STAGE}] вертикальная VP {vert.vp_px.round(0).tolist()}, инлаеров "
          f"{vert.inlier_frac:.0%}, невязка {vert.residual_px:.2f} град")

    # --- 3. направление улицы из кликов, проверка на горизонте ---------------- #
    ground_pair = list(hints["ground_pair"])
    lines = [fit_line_px(hints["lines"][k]["points_px"])[0] for k in ground_pair]
    vp_street, street_diag = lines_common_vp(lines)
    d_hor = point_on_line_distance_px(hor.line, vp_street)
    # Допуск масштабируется с удалением точки схода от центра кадра: точность
    # горизонта по пешеходам падает линейно с расстоянием (измерено на синтетике,
    # около 0.024 * расстояние). Фиксированный порог для близкой точки схода
    # слишком мягок, а для далёкой запретителен.
    centre = np.array([w_img / 2.0, h_img / 2.0])
    vp_dist = float(np.linalg.norm(vp_street - centre))
    tol = (float(sc["vp_horizon_tol_px"])
           + float(sc.get("vp_horizon_tol_per_px", 0.072)) * vp_dist)
    print(f"[{STAGE}] VP улицы {vp_street.round(0).tolist()}, до горизонта "
          f"{d_hor:.1f} px при допуске {tol:.0f} "
          f"(масштабирован: точка схода в {vp_dist:.0f} px от центра)")
    if d_hor > tol:
        # Не исключение: это ИЗМЕРЕНИЕ расхождения двух независимых оценок,
        # и оно идёт в артефакт. Решение принять или отвергнуть калибровку
        # выносит evaluate_self_calib ниже, вместе с проверкой по росту.
        print(f"[{STAGE}] ВНИМАНИЕ: расхождение {d_hor:.1f} px больше допуска {tol}",
              file=sys.stderr)

    # --- 4. фокус и план ------------------------------------------------------ #
    focal, fdiag = focal_from_horizon_and_vertical(hor.line, vert.vp_px, shape)
    diag_px = float(np.hypot(w_img, h_img))
    print(f"[{STAGE}] фокус {focal:.1f} px = {focal / diag_px:.2f} диагонали кадра")
    cam = camera_from_focal_and_vps(focal, vp_street, vert.vp_px, shape)

    # --- 5. масштаб из ШИРИНЫ УЛИЦЫ, а не из роста ---------------------------- #
    a_px = np.asarray(hints["lines"][ground_pair[0]]["points_px"], dtype=np.float64)
    b_px = np.asarray(hints["lines"][ground_pair[1]]["points_px"], dtype=np.float64)
    width_m = float(control["street_width_m"])
    scale, sdiag = scale_from_street_width(cam.H_px_to_unit, a_px, b_px, width_m)
    print(f"[{STAGE}] масштаб {scale:.4f} м/ед из ширины улицы {width_m} м "
          f"(на плане {sdiag['street_width_units']:.4f} ед)")
    print(f"[{STAGE}] высота камеры {scale:.2f} м")
    h_px_to_m = np.diag([scale, scale, 1.0]) @ cam.H_px_to_unit

    # --- 6. рост как НЕЗАВИСИМАЯ проверка ------------------------------------- #
    units, usable = [], []
    for _, r in det.iterrows():
        foot = ((r.x1_px + r.x2_px) / 2.0, r.y2_px)
        head = ((r.x1_px + r.x2_px) / 2.0, r.y1_px)
        try:
            hu = person_height_units(cam, foot, head)
        except CalibError:
            continue
        if hu > 0:
            units.append(hu)
            usable.append(r)
    if len(units) < int(cfg["gates"].get("min_people", 200)):
        raise StageError(f"людей для проверки роста {len(units)}")
    heights_m = np.asarray(units) * scale
    spread = height_spread_stats(heights_m)
    foot_m = apply_h(h_px_to_m, np.array([[(r.x1_px + r.x2_px) / 2.0, r.y2_px]
                                          for r in usable]))
    slope = height_depth_slope(heights_m, foot_m[:, 0])
    print(f"[{STAGE}] РОСТ (независимая проверка): медиана {spread['median_m']:.3f} м, "
          f"IQR {spread['iqr_m']:.3f}, p90-p10 {spread['p90_p10_m']:.3f}")
    print(f"[{STAGE}] дрейф роста по глубине {slope['slope_m_per_m']:+.5f} м/м, "
          f"ноль {'накрыт' if slope['covers_zero'] else 'НЕ НАКРЫТ'}")

    # --- 7. обратная задача: какая ширина даёт правильный рост ---------------- #
    target_h = float(sc.get("target_height_m", 1.68))
    implied_w = implied_street_width_m(width_m, target_h, spread["median_m"])
    lo_w, hi_w = sc.get("implied_width_plausible_range_m", [4.6, 5.6])
    width_plausible = bool(lo_w <= implied_w <= hi_w)
    print(f"[{STAGE}] подразумеваемая ширина L1-L3 = {implied_w:.3f} м "
          f"(та, при которой медиана роста стала бы {target_h} м)")
    print(f"[{STAGE}]   диапазон правдоподобия {lo_w}-{hi_w} м: "
          f"{'ПОПАДАЕТ' if width_plausible else 'НЕ ПОПАДАЕТ'}; "
          f"отступ от эталона {width_m - implied_w:+.2f} м")

    # --- 8. уклон улицы, объясняющий дрейф ------------------------------------ #
    grade = street_grade_from_height_drift(slope["slope_m_per_m"],
                                           spread["median_m"], scale)
    print(f"[{STAGE}] уклон улицы из дрейфа: {grade['grade_percent']:.1f}% "
          f"({grade['angle_deg']:.2f} град), {grade['direction_ru']}")

    manifest.note("focal_px", focal)
    manifest.note("street_width_L1L3_implied_m", implied_w)
    manifest.note("street_grade_percent", grade["grade_percent"])
    manifest.note("scale_m_per_unit", scale)
    manifest.note("height_median_m", spread["median_m"])

    return {"doc": {
        "schema_version": SCHEMA_VERSION, "stage": STAGE, "status": STATUS_OK,
        "method": "self_calib_pedestrians", "direction": "px_to_m",
        "calib_status": "calibrated", "scale_known": True,
        "clip": str(clip).replace("\\", "/"), "reference_frame_idx": frame_idx,
        # ПО ЧЕМУ посчитано. Без этих sha калибровка невоспроизводима: S3
        # перезаписывает det/frames.parquet при каждом прогоне на новом
        # материале, и артефакт калибровки, снятый по прежним детекциям,
        # внешне ничем не отличается от снятого по нынешним.
        "inputs_sha256": {
            "det/frames.parquet": sha256_file("det/frames.parquet"),
            str(clip).replace("\\", "/"): sha256_file(clip),
        },
        "frame_w_px": w_img, "frame_h_px": h_img,
        "H": h_px_to_m.tolist(), "H_px_to_unit": cam.H_px_to_unit.tolist(),
        "K": cam.K.tolist(), "R": cam.R.tolist(), "focal_px": focal,
        "focal_over_diagonal": focal / diag_px,
        "horizon_line": hor.line.tolist(),
        "horizon_n_pairs": hor.n_pairs, "horizon_inlier_frac": hor.inlier_frac,
        "horizon_residual_px": hor.residual_px,
        "vp_vertical": vert.vp_px.tolist(),
        "vertical_inlier_frac": vert.inlier_frac,
        "vertical_n_segments": vert.n_pairs,
        "vp_horizontal": vp_street.tolist(),
        "vp_street_to_horizon_px": d_hor,
        "vp_horizon_tol_used_px": tol,
        "vp_street_dist_from_centre_px": vp_dist,
        "scale_m_per_unit": scale, "camera_height_m": scale,
        "street_width_reference_m": width_m,
        "street_width_units": sdiag["street_width_units"],
        "street_width_L1L3_implied_m": implied_w,
        "street_width_implied_plausible": width_plausible,
        "street_width_implied_range_m": [lo_w, hi_w],
        "street_grade": grade,
        "n_people_used": len(units),
        "height_median_m": spread["median_m"], "height_iqr_m": spread["iqr_m"],
        "height_p10": spread["p10_m"], "height_p90": spread["p90_m"],
        "height_depth_slope": slope["slope_m_per_m"],
        "height_depth_slope_ci95": [slope["ci95_low"], slope["ci95_high"]],
        "height_depth_slope_covers_zero": slope["covers_zero"],
        "pilot_heights_m": [round(float(v), 4) for v in heights_m],
        "pilot_depths_m": [round(float(v), 4) for v in foot_m[:, 0]],
        "axis_convention": {
            "x": "вдоль улицы, в направлении точки схода улицы",
            "y": "поперёк улицы",
            "angle_zero": "0 градусов = +x плана, против часовой стрелки, [0, 360)",
            "orientation_note_ru": ORIENTATION_DISCLAIMER,
        },
        "assumptions_ru": [
            "главная точка в центре кадра, квадратный пиксель",
            "люди в кадре стоят вертикально",
            "рост людей примерно одинаков (разброс входит как шум)",
            f"ширина улицы между линиями {ground_pair} равна {width_m} м",
        ],
        "independent_checks_ru": [
            "рост НЕ участвует в подгонке масштаба и потому проверяет его",
            "точка схода улицы из кликов проверена на горизонте по пешеходам",
        ],
    }, "ref": ref, "cam": cam, "hints": hints, "usable": usable,
        "heights_m": heights_m, "foot_m": foot_m, "scale": scale,
        "slope": slope, "segs": (np.empty((0, 4)), np.empty((0, 4))),
        "vp": (None, None)}


def evaluate_vp_quality(doc: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Годна ли калибровка по точкам схода. Список причин негодности.

    Считается ЗДЕСЬ, до записи артефакта, чтобы этап мог сам решить, переходить
    ли на заглушку. Гейт потом посчитает то же самое независимо — это не
    дублирование, а разделение ролей: этап выбирает путь, гейт судит результат.
    """
    gates = cfg.get("gates") or {}
    control = cfg.get("control") or {}
    bad: list[str] = []

    lo, hi = gates.get("median_height_range_m", [1.55, 1.75])
    if not (lo <= doc["height_median_m"] <= hi):
        bad.append(f"медиана роста {doc['height_median_m']:.2f} м вне {lo}-{hi}")
    if doc["height_iqr_m"] > float(gates.get("height_iqr_max_m", 0.18)):
        bad.append(f"IQR роста {doc['height_iqr_m']:.2f} м при пороге "
                   f"{gates.get('height_iqr_max_m')}")
    span = float(doc["height_p90"]) - float(doc["height_p10"])
    if span > float(gates.get("height_p90_p10_max_m", 0.35)):
        bad.append(f"p90-p10 роста {span:.2f} м при пороге "
                   f"{gates.get('height_p90_p10_max_m')}")
    if not doc.get("height_depth_slope_covers_zero", False):
        bad.append(f"дрейф роста по глубине {doc['height_depth_slope']:+.4f} м/м, "
                   f"ноль не накрыт")
    slo, shi = gates.get("median_speed_range_mps", [1.0, 1.6])
    sm = doc.get("speed_median_mps")
    if sm is None or not (slo <= sm <= shi):
        bad.append(f"медианная скорость {sm} м/с вне {slo}-{shi}")
    ref = control.get("street_width_m")
    if ref is not None:
        limit = float(control["street_width_tolerance_m"]) * float(
            gates.get("street_width_tolerance_factor", 2.0))
        if abs(doc["street_width_delta_m"]) > limit:
            bad.append(f"ширина улицы разошлась на {doc['street_width_delta_m']:+.2f} м "
                       f"при допуске +-{limit:.2f}")
    return bad


def run_stub_affine(cfg: dict[str, Any], manifest: RunManifest,
                    reasons: list[str]) -> dict[str, Any]:
    """АФФИННАЯ ЗАГЛУШКА. Плоскости земли нет, перспектива не исправляется.

    Что это на самом деле: координаты кадра, поделённые на высоту кадра.
    Никакой геометрии земли здесь не восстановлено. Дальний человек и ближний
    в одинаковых «единицах» имеют РАЗНЫЙ физический размер, поэтому:

      * метры не показываются нигде и никогда;
      * длины и скорости сравнимы только внутри похожей глубины;
      * ранжирование витрин остаётся осмысленным — витрины стоят на схожей
        глубине, и относительный порядок «у какой больше остановок» переживает
        искажение;
      * углы считаются ПРИБЛИЖЁННО: это углы в кадре, а не на земле.

    Заглушка существует ровно для того, чтобы труба работала целиком, пока
    калибровка не доведена. Каждое число, полученное на ней, обязано нести
    пометку stub_affine до самого отчёта.
    """
    clip = Path(require(cfg, "input", "clip"))
    hints = load_config(require(cfg, "input", "hints"))
    frame_idx = int(hints.get("frame_idx", 0))
    ref = _reference_frame(clip, frame_idx)
    h, w = ref.shape[0], ref.shape[1]

    scale = 1.0 / float(h)          # единица длины = высота кадра
    h_px_to_unit = np.array([[scale, 0.0, 0.0],
                             [0.0, scale, 0.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)

    print(f"[{STAGE}] ЗАГЛУШКА stub_affine: калибровка по точкам схода забракована")
    for r in reasons:
        print(f"[{STAGE}]    причина: {r}")
    print(f"[{STAGE}] МЕТРОВ НЕТ И НЕ БУДЕТ на этом пути. Углы приближённые, "
          f"длины в условных единицах, сравнимы только внутри похожей глубины")

    manifest.note("calib_status", "stub_affine")
    manifest.note("stub_reasons", reasons)
    manifest.note("scale_known", False)

    return {"doc": {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": STATUS_OK,
        "method": "stub_affine",
        "direction": "px_to_unit",
        "calib_status": "stub_affine",
        "scale_known": False,
        "scale_m_per_unit": None,
        "stub_reasons_ru": reasons,
        "banner_ru": ("КАЛИБРОВКА НЕ ПРОЙДЕНА. Показаны условные единицы, "
                      "а не метры. Углы приближённые. Числа годятся для "
                      "сравнения витрин между собой и ни для чего больше."),
        "unit_note_ru": ("Единица длины = высота кадра. Это НЕ план земли: "
                         "перспектива не исправлена, дальние и ближние длины "
                         "несопоставимы."),
        "clip": str(clip).replace("\\", "/"),
        "reference_frame_idx": frame_idx,
        "frame_w_px": w, "frame_h_px": h,
        "H": h_px_to_unit.tolist(),
        "H_px_to_unit": h_px_to_unit.tolist(),
        "horizon_line": [0.0, 1.0, 0.0],
        "axis_convention": {
            "x": "вправо по кадру (НЕ вдоль улицы)",
            "y": "вниз по кадру (НЕ поперёк улицы)",
            "angle_zero": "0 градусов = +x кадра, против часовой, [0, 360)",
            "orientation_note_ru": ORIENTATION_DISCLAIMER,
        },
        "assumptions_ru": [
            "ЗАГЛУШКА: плоскость земли не восстановлена",
            "перспектива не исправлена",
            "углы приближённые, метров нет",
        ],
    }, "ref": ref, "hints": hints, "cam": None}


def run_manual_ground_plane(cfg: dict[str, Any], manifest: RunManifest) -> dict[str, Any]:
    """Запасной путь: плоскость земли по прямоугольнику мостовой.

    Даёт углы и принадлежность к зонам, НЕ даёт метров. Масштаб остаётся
    неизвестным, длины в условных единицах, calib_status это фиксирует.
    Пилотная детекция здесь не нужна: она служила только подбору масштаба,
    а масштаба тут нет.
    """
    rect_cfg = require(cfg, "ground_rect")
    src = Path(rect_cfg["source"])
    if not src.is_file():
        raise StageError(
            f"нет файла участка {src}. Сначала обведите прямоугольник мостовой:\n"
            f"    python scripts/pick_ground.py --config {cfg['_config_path']}")
    doc = read_json(src)
    aspect = float(rect_cfg["aspect_tl_tr_over_tl_bl"])
    if abs(float(doc.get("aspect_tl_tr_over_tl_bl", aspect)) - aspect) > 1e-9:
        raise StageError(
            f"пропорция в конфиге ({aspect}) не совпадает с той, по которой обводили "
            f"({doc.get('aspect_tl_tr_over_tl_bl')}). Перекликайте или верните пропорцию: "
            f"от неё зависит вся геометрия плоскости")

    rect_px = np.asarray(doc["rect_px"], dtype=np.float64)
    h_px_to_unit, diag = homography_from_ground_rect(rect_px, aspect)
    print(f"[{STAGE}] плоскость земли по участку {src}: невязка углов "
          f"{diag['corner_residual_units']:.2e} усл.ед., план развёрнут "
          f"{diag['plane_y_flipped']}")
    print(f"[{STAGE}] МЕТРОВ НЕТ: масштаб не определён, длины в условных единицах")

    clip = Path(require(cfg, "input", "clip"))
    frame_idx = int(doc.get("frame_idx", 0))
    ref = _reference_frame(clip, frame_idx)
    shape = (ref.shape[0], ref.shape[1])

    manifest.note("calib_status", rect_cfg["calib_status"])
    manifest.note("scale_known", False)

    return {"doc": {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": STATUS_OK,
        "method": "manual_ground_plane",
        "direction": "px_to_unit",
        "calib_status": rect_cfg["calib_status"],
        "scale_known": False,
        "unit_note_ru": ("Длины в УСЛОВНЫХ ЕДИНИЦАХ: сторона TL->BL обведённого "
                         "участка принята за 1.0. Метров нет, и число, названное "
                         "метром, было бы враньём."),
        "clip": str(clip).replace("\\", "/"),
        "reference_frame_idx": frame_idx,
        "frame_w_px": shape[1], "frame_h_px": shape[0],
        "H": h_px_to_unit.tolist(),
        "H_px_to_unit": h_px_to_unit.tolist(),
        "scale_m_per_unit": None,
        "horizon_line": diag["horizon_line"],
        "ground_rect_px": doc["rect_px"],
        "aspect_tl_tr_over_tl_bl": aspect,
        "plane_y_flipped": diag["plane_y_flipped"],
        "corner_residual_units": diag["corner_residual_units"],
        "axis_convention": {
            "x": "вдоль стороны TL->TR обведённого участка",
            "y": "вдоль стороны TL->BL, принятой за 1.0 условной единицы",
            "angle_zero": "0 градусов = +x плана, против часовой стрелки, [0, 360)",
            "orientation_note_ru": ORIENTATION_DISCLAIMER,
        },
        "assumptions_ru": [
            "обведённый участок мостовой — прямоугольник",
            f"его пропорция TL->TR : TL->BL равна {aspect}",
            "участок лежит в плоскости земли",
        ],
        # Полей роста, скорости и ширины улицы здесь НЕТ намеренно: без масштаба
        # они не считаются, и пустые ключи выглядели бы как «посчитали и вышло ноль».
    }, "ref": ref, "hints": None, "cam": None}


def run(cfg: dict[str, Any], manifest: RunManifest) -> dict[str, Any]:
    clip = Path(require(cfg, "input", "clip"))
    hints_path = Path(require(cfg, "input", "hints"))
    if not clip.is_file():
        raise StageError(f"нет клипа {clip}")
    if not hints_path.is_file():
        raise StageError(
            f"нет файла подсказок {hints_path}. Сначала кликните затравки:\n"
            f"    python scripts/pick_hints.py --config {cfg['_config_path']}"
        )
    hints = load_config(hints_path)
    detect_cfg = load_config(require(cfg, "input", "detect_config"))
    params = infer_params(detect_cfg)
    vp_cfg = require(cfg, "vp")
    pilot_cfg = require(cfg, "pilot")
    scale_cfg = require(cfg, "scale")

    fps, n_frames = _clip_fps(clip)
    frame_idx = int(hints.get("frame_idx", 0))
    ref = _reference_frame(clip, frame_idx)
    shape = (ref.shape[0], ref.shape[1])
    print(f"[{STAGE}] клип {clip}: {n_frames} кадров, {fps:.2f} fps, "
          f"{shape[1]}x{shape[0]}; опорный кадр {frame_idx}")

    # --- шаг 2. точки схода --------------------------------------------- #
    if str(hints.get("schema_version")) != "2":
        raise StageError(
            f"{hints_path}: schema_version={hints.get('schema_version')!r}, ожидается '2'. "
            f"Старый формат на 8 кликов больше не поддерживается — перекликайте "
            f"затравки: python scripts/pick_hints.py"
        )
    hint_lines = hints["lines"]
    ground_pair = list(hints["ground_pair"])
    if len(ground_pair) != 2 or any(k not in hint_lines for k in ground_pair):
        raise StageError(f"{hints_path}: ground_pair={ground_pair} не указывает на две линии")

    # Все линии с ролью street/street_ground параллельны улице в мире, значит
    # сходятся в одной точке. МНК по четырём обусловлен лучше, чем пересечение
    # двух: линии на стенах идут высоко и дают широкую базу.
    residuals: dict[str, float] = {}
    street_lines = []
    for line_id, ln in hint_lines.items():
        line, res = fit_line_px(ln["points_px"])
        residuals[line_id] = round(float(res), 3)
        if ln["role"] in ("street", "street_ground"):
            street_lines.append((line_id, line))
    if len(street_lines) < 2:
        raise StageError("нужны минимум две линии с ролью street/street_ground")

    max_res = float(vp_cfg.get("hint_line_max_residual_px", 6.0))
    hint_residual = max(residuals.values())
    for line_id, res in sorted(residuals.items()):
        flag = "  <-- ПОДОЗРИТЕЛЬНО" if res > max_res else ""
        print(f"[{STAGE}] разброс кликов {line_id}: {res:.2f} px{flag}")
    if hint_residual > max_res:
        # Предупреждение, а не ошибка: перекликать или нет — решение человека.
        print(f"[{STAGE}] ВНИМАНИЕ: разброс {hint_residual:.2f} px выше порога "
              f"{max_res} px, точка схода может уехать", file=sys.stderr)

    seed_street, street_diag = lines_common_vp([ln for _, ln in street_lines])
    print(f"[{STAGE}] затравка VP улицы по {len(street_lines)} линиям "
          f"({', '.join(i for i, _ in street_lines)}): "
          f"{seed_street.round(1).tolist()}, разброс попарных пересечений "
          f"{street_diag['pairwise_spread_px']:.1f} px")

    segs = detect_segments(ref, min_len_px=float(vp_cfg["min_segment_len_px"]))
    street_segs, vert_segs = _split_segments(segs, vp_cfg)
    print(f"[{STAGE}] отрезков в кадре: {len(segs)} "
          f"(горизонтальных {len(street_segs)}, вертикальных {len(vert_segs)})")

    # Вертикаль задана ОДНОЙ линией, а одна прямая точку схода не определяет.
    # Зато настоящая вертикальная VP обязана на ней лежать, поэтому кандидаты
    # ищутся только на этой прямой: RANSAC с жёстким ограничением.
    v1_line = fit_line_px(hint_lines["V1"]["points_px"])[0]
    seed_vertical = np.asarray(vp_candidates_on_line(v1_line, vert_segs))

    common = dict(inlier_tol_px=float(vp_cfg["inlier_tol_px"]),
                  min_inliers=int(vp_cfg["min_inliers"]),
                  holdout_frac=float(vp_cfg["holdout_frac"]),
                  seed=int(vp_cfg["seed"]), frame_shape=shape,
                  seed_max_deviation_deg=float(vp_cfg.get("seed_max_deviation_deg", 25.0)))
    vp_h = ransac_vp(street_segs, seed_street, **common)
    vp_v = ransac_vp(vert_segs, seed_vertical, **common)
    print(f"[{STAGE}] VP улицы {vp_h.vp_px.round(1).tolist()}, инлаеров {vp_h.n_inliers}, "
          f"невязка удержанных {vp_h.holdout_residual_px:.2f} px")
    print(f"[{STAGE}] VP вертикали {vp_v.vp_px.round(1).tolist()}, инлаеров {vp_v.n_inliers}, "
          f"невязка удержанных {vp_v.holdout_residual_px:.2f} px")

    # --- шаг 3. плоскость земли с точностью до масштаба ------------------ #
    cam = camera_from_vps(vp_h.vp_px, vp_v.vp_px, shape)
    print(f"[{STAGE}] фокус {cam.focal_px:.1f} px")

    # --- шаг 4. пилотная детекция ---------------------------------------- #
    indices, burst_id = sample_burst_indices(
        n_frames, int(pilot_cfg["n_bursts"]), int(pilot_cfg["burst_frames"]))
    dets, det_shape = detect_people(clip, indices, params)
    kept, filt_stats = filter_height_samples(dets, det_shape, pilot_cfg)
    print(f"[{STAGE}] детекций {filt_stats['total']}, годных для замера роста "
          f"{filt_stats['kept']} (отсеяно: conf {filt_stats['low_conf']}, "
          f"край {filt_stats['touches_edge']}, толпа {filt_stats['crowded']}, "
          f"мелкие {filt_stats['too_small']})")
    if not kept:
        raise StageError("после фильтров не осталось ни одной детекции для замера роста")

    # --- шаг 5. масштаб по росту ------------------------------------------ #
    units, usable = [], []
    for d in kept:
        try:
            h = person_height_units(cam, d.foot_px, d.head_px)
        except CalibError:
            continue
        if h > 0:
            units.append(h)
            usable.append(d)
    if len(units) < 2:
        raise StageError(f"рост удалось померить лишь у {len(units)} человек")

    fit = scale_from_heights(units, float(scale_cfg["target_height_m"]),
                             float(scale_cfg["trim_frac"]))
    scale = fit["scale_m_per_unit"]
    heights_m = np.asarray(units) * scale
    print(f"[{STAGE}] масштаб {scale:.4f} м/единицу (высота камеры {scale:.2f} м), "
          f"людей в выборке {len(units)}")

    h_px_to_m = np.diag([scale, scale, 1.0]) @ cam.H_px_to_unit
    foot_m = apply_h(h_px_to_m, np.array([d.foot_px for d in usable]))

    spread = height_spread_stats(heights_m)
    # Глубина — координата вдоль улицы: именно по ней рост не должен плыть.
    slope = height_depth_slope(heights_m, foot_m[:, 0])
    print(f"[{STAGE}] рост: медиана {spread['median_m']:.3f}, IQR {spread['iqr_m']:.3f}, "
          f"p90-p10 {spread['p90_p10_m']:.3f} м")
    print(f"[{STAGE}] дрейф роста по глубине: наклон {slope['slope_m_per_m']:+.5f} м/м, "
          f"CI95 [{slope['ci95_low']:+.5f}, {slope['ci95_high']:+.5f}], "
          f"ноль {'накрыт' if slope['covers_zero'] else 'НЕ НАКРЫТ'}")

    # --- шаг 6. независимые проверки -------------------------------------- #
    # ТОЛЬКО ground_pair: L2 и L4 идут высоко по стенам, расстояние между ними
    # шириной улицы не является. Спутниковый эталон меряет землю.
    g0, g1 = ground_pair
    left_m = apply_h(h_px_to_m, np.asarray(hint_lines[g0]["points_px"], dtype=np.float64))
    right_m = apply_h(h_px_to_m, np.asarray(hint_lines[g1]["points_px"], dtype=np.float64))
    sep = facade_lines_separation_m(left_m[[0, -1]], right_m[[0, -1]])
    reference_width = float(require(cfg, "control", "street_width_m"))
    print(f"[{STAGE}] ширина улицы {sep['width_mean_m']:.3f} м "
          f"(эталон {reference_width}, расхождение {sep['width_mean_m'] - reference_width:+.3f})")

    speeds = track_speeds(clip, indices, burst_id, params, h_px_to_m, fps,
                          float(pilot_cfg["min_track_s"]),
                          int(pilot_cfg["burst_frames"]))
    speed_median = float(np.median(speeds["speeds_mps"])) if speeds["speeds_mps"] else None
    print(f"[{STAGE}] пачек {int(pilot_cfg['n_bursts'])} по "
          f"{int(pilot_cfg['burst_frames'])} кадров "
          f"({speeds['burst_duration_s']} с каждая); треков {speeds['n_tracks_total']}, "
          f"длиннее {speeds['min_track_s']} с — {speeds['n_tracks_long']}, "
          f"длиннее 2 с — {speeds['n_tracks_over_2s']}; "
          f"медианная скорость {speed_median if speed_median is None else round(speed_median, 3)} м/с")

    manifest.note("pilot_filter_stats", filt_stats)
    manifest.note("n_people_used", len(units))
    manifest.note("scale_m_per_unit", scale)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": STATUS_OK,
        "method": "auto_vp_height",
        "direction": "px_to_m",
        "clip": str(clip).replace("\\", "/"),
        "reference_frame_idx": frame_idx,
        "frame_w_px": shape[1], "frame_h_px": shape[0],

        "H": h_px_to_m.tolist(),
        "H_px_to_unit": cam.H_px_to_unit.tolist(),
        "focal_px": cam.focal_px,
        "K": cam.K.tolist(),
        "R": cam.R.tolist(),
        "vp_horizontal": vp_h.vp_px.tolist(),
        "vp_vertical": vp_v.vp_px.tolist(),
        "horizon_line": cam.horizon_line.tolist(),
        "scale_m_per_unit": scale,
        "camera_height_m": scale,
        "origin_px": [shape[1] / 2.0, float(shape[0])],
        "axis_convention": {
            "x_m": "вдоль улицы, в направлении горизонтальной точки схода (вглубь кадра)",
            "y_m": "поперёк улицы",
            "origin": "проекция центра нижнего края кадра на плоскость земли",
            "angle_zero": "0 градусов = +x плана, против часовой стрелки, [0, 360)",
            "orientation_note_ru": ORIENTATION_DISCLAIMER,
        },
        "assumptions_ru": [
            "главная точка в центре кадра",
            "квадратный пиксель, нулевой перекос",
            "направление улицы и вертикаль ортогональны",
            f"медианный рост выборки равен {scale_cfg['target_height_m']} м",
        ],

        "hint_line_residual_px": {**residuals, "max": hint_residual},
        "hint_line_max_residual_px": max_res,
        "hint_lines_used_for_vp": [i for i, _ in street_lines],
        "ground_pair": ground_pair,
        "vp_seed_pairwise_spread_px": street_diag["pairwise_spread_px"],
        "vp_inliers_used": {"horizontal": vp_h.n_inliers, "vertical": vp_v.n_inliers},
        "vp_holdout_residual_px": {
            "horizontal": vp_h.holdout_residual_px,
            "vertical": vp_v.holdout_residual_px,
            "max": max(vp_h.holdout_residual_px, vp_v.holdout_residual_px),
        },
        "vp_holdout_n": {"horizontal": vp_h.n_holdout, "vertical": vp_v.n_holdout},

        "street_width_L1L3_implied_m": implied_w,
        "street_width_implied_plausible": width_plausible,
        "street_width_implied_range_m": [lo_w, hi_w],
        "street_grade": grade,
        "n_people_used": len(units),
        "pilot_filter_stats": filt_stats,
        "height_median_m": spread["median_m"],
        "height_iqr_m": spread["iqr_m"],
        "height_p10": spread["p10_m"],
        "height_p90": spread["p90_m"],
        "height_depth_slope": slope["slope_m_per_m"],
        "height_depth_slope_ci95": [slope["ci95_low"], slope["ci95_high"]],
        "height_depth_slope_covers_zero": slope["covers_zero"],

        "speed_median_mps": speed_median,
        "speed_n_tracks": speeds["n_tracks_long"],
        "speed_n_tracks_total": speeds["n_tracks_total"],
        # Доля треков длиннее 2 с остаётся по требованию владельца. При пачке
        # короче 2 с она равна нулю ПО ПОСТРОЕНИЮ — рядом лежит burst_duration_s.
        "speed_n_tracks_over_2s": speeds["n_tracks_over_2s"],
        "speed_frac_tracks_over_2s": speeds["frac_tracks_over_2s"],
        "pilot_burst_frames": speeds["burst_frames"],
        "pilot_burst_duration_s": speeds["burst_duration_s"],
        "pilot_n_bursts": int(pilot_cfg["n_bursts"]),

        "street_width_measured_m": sep["width_mean_m"],
        "street_width_reference_m": reference_width,
        "street_width_delta_m": sep["width_mean_m"] - reference_width,
        "street_width_spread_m": sep["width_spread_m"],
        "facade_baselines": [
            {"name": g0, "points_m": left_m[[0, -1]].tolist()},
            {"name": g1, "points_m": right_m[[0, -1]].tolist()},
        ],

        # Пилотные выборки лежат в самом артефакте: гейт считает статистику САМ,
        # а не читает готовые числа, иначе он проверял бы арифметику этапа.
        "pilot_heights_m": [round(float(v), 4) for v in heights_m],
        # Глубины идут рядом с ростами, чтобы гейт ПЕРЕСЧИТАЛ регрессию сам,
        # а не поверил height_depth_slope, записанному этапом.
        "pilot_depths_m": [round(float(v), 4) for v in foot_m[:, 0]],
        "pilot_speeds_mps": [round(float(v), 4) for v in speeds["speeds_mps"]],
    }
    return {"doc": doc, "cam": cam, "ref": ref, "usable": usable,
            "heights_m": heights_m, "foot_m": foot_m, "scale": scale,
            "segs": (street_segs, vert_segs), "vp": (vp_h, vp_v), "hints": hints,
            "slope": slope}


# --------------------------------------------------------------------------- #
# Отладочные картинки: кривая калибровка видна глазом за секунду
# --------------------------------------------------------------------------- #

def write_debug(res: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # 1. вид сверху: улица обязана быть прямой полосой постоянной ширины
    foot = res["foot_m"]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    ax.scatter(foot[:, 0], foot[:, 1], s=6, alpha=0.5)
    for b in res["doc"]["facade_baselines"]:
        pts = np.asarray(b["points_m"])
        ax.plot(pts[:, 0], pts[:, 1], lw=2, label=f"фасад {b['name']}")
    ax.set_aspect("equal", "datalim")
    ax.grid(True, which="both", alpha=0.3, lw=0.5)
    ax.set_xlabel("x, м (вдоль улицы)")
    ax.set_ylabel("y, м (поперёк)")
    ax.set_title("Вид сверху: точки ног. Улица должна быть прямой полосой "
                 "постоянной ширины", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "debug_topdown.png"
    save_figure(fig, p); plt.close(fig); written.append(p)

    # 2. опорный кадр: инлаеры VP, горизонт, клики
    img = res["ref"].copy()
    for segs, color in ((res["segs"][0], (90, 200, 90)), (res["segs"][1], (255, 190, 60))):
        for x1, y1, x2, y2 in segs[:400]:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, cv2.LINE_AA)
    hl = np.asarray(res["doc"]["horizon_line"], dtype=np.float64)
    if abs(hl[1]) > 1e-9:
        w = img.shape[1]
        y0 = int(-(hl[0] * 0 + hl[2]) / hl[1])
        y1 = int(-(hl[0] * w + hl[2]) / hl[1])
        cv2.line(img, (0, y0), (w, y1), (60, 60, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "horizon", (20, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (60, 60, 255), 2, cv2.LINE_AA)
    palette = {"L1": (90, 230, 90), "L2": (60, 190, 255), "L3": (255, 160, 80),
               "L4": (230, 120, 255), "V1": (80, 80, 255)}
    for line_id, ln in res["hints"]["lines"].items():
        color = palette.get(line_id, (255, 255, 255))
        pts = ln["points_px"]
        for x, y in pts:
            cv2.drawMarker(img, (int(x), int(y)), color, cv2.MARKER_CROSS, 24, 2)
        cv2.line(img, tuple(map(int, pts[0])), tuple(map(int, pts[-1])), color, 1, cv2.LINE_AA)
        cv2.putText(img, line_id, (int(pts[0][0]) + 10, int(pts[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    p = out_dir / "debug_vp.png"
    save_image(p, img); written.append(p)

    # 3. рост: гистограмма и дрейф по глубине
    heights, slope = res["heights_m"], res["slope"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    a1.hist(heights, bins=40, color="#1f77b4", alpha=0.8)
    a1.axvline(float(np.median(heights)), color="crimson", lw=1.5,
               label=f"медиана {np.median(heights):.3f} м")
    a1.set_xlabel("est_height_m"); a1.set_ylabel("человек"); a1.legend(fontsize=8)
    a1.set_title("Распределение роста", fontsize=9)

    depth = res["foot_m"][:, 0]
    a2.scatter(depth, heights, s=6, alpha=0.4)
    xs = np.linspace(depth.min(), depth.max(), 50)
    a2.plot(xs, slope["intercept_m"] + slope["slope_m_per_m"] * xs, color="crimson", lw=2,
            label=f"наклон {slope['slope_m_per_m']:+.4f} м/м, "
                  f"ноль {'накрыт' if slope['covers_zero'] else 'НЕ накрыт'}")
    a2.set_xlabel("foot_x_m (глубина вдоль улицы)"); a2.set_ylabel("est_height_m")
    a2.legend(fontsize=8); a2.set_title("Дрейф роста по глубине", fontsize=9)
    fig.tight_layout()
    p = out_dir / "debug_heights.png"
    save_figure(fig, p); plt.close(fig); written.append(p)
    return written


# --------------------------------------------------------------------------- #

def _write_selfcalib_debug(res: dict, out_dir: Path) -> Path:
    """Опорный кадр: горизонт, точки схода, гистограмма роста.

    Проверка глазом: горизонт обязан идти по линии схода улицы, а гистограмма
    роста — стоять около 1.7 м. Смещение гистограммы это смещение масштаба,
    и видно оно за секунду.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = res["doc"]
    img = res["ref"].copy()
    hl = np.asarray(doc["horizon_line"], dtype=np.float64)
    w = img.shape[1]
    if abs(hl[1]) > 1e-9:
        y0 = int(-hl[2] / hl[1])
        y1 = int(-(hl[0] * w + hl[2]) / hl[1])
        cv2.line(img, (0, y0), (w, y1), (60, 60, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "horizon (pedestrians)", (20, max(24, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
    vs = np.asarray(doc["vp_horizontal"], dtype=int)
    if 0 <= vs[0] < img.shape[1] and 0 <= vs[1] < img.shape[0]:
        cv2.drawMarker(img, tuple(vs), (90, 230, 90), cv2.MARKER_TILTED_CROSS, 30, 3)
        cv2.putText(img, "street VP", (int(vs[0]) + 12, int(vs[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 230, 90), 2, cv2.LINE_AA)
    status = doc.get("calib_status", "?")
    if status != "calibrated":
        cv2.rectangle(img, (0, 0), (img.shape[1], 62), (0, 0, 160), -1)
        cv2.putText(img, f"{status.upper()} - metres withheld", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    p_img = out_dir / "debug_selfcalib.png"
    save_image(p_img, img)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    heights = res["heights_m"]
    a1.hist(heights, bins=60, color="#1f77b4", alpha=0.85)
    a1.axvline(float(np.median(heights)), color="crimson", lw=2,
               label=f"медиана {np.median(heights):.3f}")
    a1.axvspan(1.55, 1.75, color="#5ad18e", alpha=0.18, label="ожидаемый диапазон")
    a1.set_xlabel("est_height_m"); a1.legend(fontsize=8)
    a1.set_title("Рост — НЕЗАВИСИМАЯ проверка масштаба", fontsize=9)
    depth = res["foot_m"][:, 0]
    a2.scatter(depth, heights, s=4, alpha=0.3)
    sl = res["slope"]
    xs = np.linspace(float(depth.min()), float(depth.max()), 50)
    a2.plot(xs, sl["intercept_m"] + sl["slope_m_per_m"] * xs, color="crimson", lw=2,
            label=f"наклон {sl['slope_m_per_m']:+.4f}")
    a2.set_xlabel("глубина"); a2.set_ylabel("est_height_m"); a2.legend(fontsize=8)
    a2.set_title("Дрейф роста по глубине", fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_dir / "debug_selfcalib_heights.png")
    plt.close(fig)
    return p_img


def _write_ground_debug(res: dict, out_dir: Path) -> Path:
    """Опорный кадр с обведённым участком и линией горизонта.

    Проверка глазом: горизонт обязан идти примерно по линии схода улицы.
    Если он лёг косо или пересёк мостовую — участок обведён неверно.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img = res["ref"].copy()
    rect_px = res["doc"].get("ground_rect_px")
    if rect_px is not None:
        rect = np.asarray(rect_px, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [rect], (90, 230, 90))
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)
        cv2.polylines(img, [rect], True, (90, 230, 90), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(rect):
            cv2.putText(img, ["TL", "TR", "BR", "BL"][i], (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 230, 90), 2, cv2.LINE_AA)
    banner = res["doc"].get("calib_status")
    if banner == "stub_affine":
        # Картинка обязана кричать о заглушке: иначе её однажды покажут как
        # доказательство работающей калибровки.
        cv2.rectangle(img, (0, 0), (img.shape[1], 70), (0, 0, 160), -1)
        cv2.putText(img, "STUB_AFFINE - NOT CALIBRATED - conventional units",
                    (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        for i, r in enumerate(res["doc"].get("stub_reasons_ru", [])[:5]):
            cv2.putText(img, f"- {r}", (20, 110 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
    hl = np.asarray(res["doc"]["horizon_line"], dtype=np.float64)
    if abs(hl[1]) > 1e-9:
        w = img.shape[1]
        y0 = int(-(hl[2]) / hl[1])
        y1 = int(-(hl[0] * w + hl[2]) / hl[1])
        cv2.line(img, (0, y0), (w, y1), (60, 60, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "horizon", (20, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (60, 60, 255), 2, cv2.LINE_AA)
    p = out_dir / ("debug_stub_affine.png" if banner == "stub_affine"
                   else "debug_ground_rect.png")
    save_image(p, img)
    return p


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

        method = require(cfg, "homography", "method")
        allow_stub = bool(cfg["homography"].get("fallback_to_stub_affine", False))
        if method == "self_calib_pedestrians":
            try:
                res = run_self_calib_pedestrians(cfg, manifest)
                verdict = evaluate_self_calib(res["doc"], cfg)
            except (CalibError, StageError, PilotError) as exc:
                if not allow_stub:
                    raise
                res, verdict = None, {"geometry": [f"самокалибровка упала: {exc}"],
                                      "scale": []}
            if verdict["geometry"]:
                # Геометрия негодна — углы считать нельзя, остаётся заглушка.
                if not allow_stub:
                    raise StageError("геометрия негодна: " + "; ".join(verdict["geometry"]))
                res = run_stub_affine(cfg, manifest,
                                      verdict["geometry"] + verdict["scale"])
                sampler = build_evidence(cfg, STAGE, manifest)
            elif (verdict["scale"] and res is not None
                  and res["doc"].get("street_width_implied_plausible")):
                # Обратная задача дала правдоподобную ширину: версия о том, что
                # эталон 6.06 м замерен здание-здание, а кликнут бордюр,
                # подтверждена. Масштаб берётся из РОСТА.
                #
                # Цена решения названа прямо: рост перестаёт быть независимой
                # проверкой, потому что теперь он задаёт масштаб. Остаётся одна
                # проверка — попадание подразумеваемой ширины в правдоподобный
                # диапазон, и она слабее.
                res = _rescale_from_height(res, cfg, manifest)
            elif verdict["scale"]:
                # Геометрия годна, масштаб нет. Углы остаются рабочими, метры
                # снимаются: подписать условные единицы метрами было бы враньём.
                doc = res["doc"]
                doc["calib_status"] = "angles_ok_scale_unverified"
                doc["scale_known"] = False
                doc["scale_reasons_ru"] = verdict["scale"]
                doc["direction"] = "px_to_unit"
                doc["banner_ru"] = (
                    "ГЕОМЕТРИЯ ПРИНЯТА, МАСШТАБ НЕТ. Углы, зоны и ранжирование "
                    "витрин достоверны. Длины показаны в условных единицах: "
                    + "; ".join(verdict["scale"]))
                doc["stub_reasons_ru"] = verdict["scale"]
                manifest.note("calib_status", "angles_ok_scale_unverified")
                print(f"[{STAGE}] ГЕОМЕТРИЯ ПРИНЯТА, МАСШТАБ ОТВЕРГНУТ:")
                for r in verdict["scale"]:
                    print(f"[{STAGE}]    {r}")
                print(f"[{STAGE}] углы и зоны считаются, метры не показываются")
        elif method == "manual_ground_plane":
            res = run_manual_ground_plane(cfg, manifest)
        elif method in ("vanishing_points", "auto_vp_height"):
            try:
                res = run(cfg, manifest)
                reasons = evaluate_vp_quality(res["doc"], cfg)
            except (CalibError, StageError, PilotError) as exc:
                if not allow_stub:
                    raise
                reasons = [f"путь vanishing_points упал: {exc}"]
                res = None
            if reasons:
                if not allow_stub:
                    raise StageError(
                        "калибровка по точкам схода негодна: " + "; ".join(reasons))
                res = run_stub_affine(cfg, manifest, reasons)
                sampler = build_evidence(cfg, STAGE, manifest)  # прежний уже израсходован
        else:
            raise ConfigError(
                f"homography.method={method!r} не поддерживается; ожидается "
                f"vanishing_points или manual_ground_plane")

        # Пруфы: кропы людей, по которым подобран масштаб. Значение — измеренный
        # рост: по кадрам видно, тем ли людям он приписан.
        # У запасного пути масштаба нет, пилотной выборки тоже — предъявлять
        # нечего, и выдумывать пруфы под claim-ы конфига нельзя.
        # Пруфы по отрезкам точек схода есть только у пути vanishing_points:
        # самокалибровка отрезков кадра не использует, ей предъявлять нечего.
        if res.get("cam") is not None and res.get("vp", (None,))[0] is not None:
            _collect_evidence(sampler, cfg, res)

        out = Path(require(cfg, "output", "homography"))
        write_json(out, res["doc"])
        # write_debug рисует инлаеры точек схода, а у самокалибровки их нет:
        # там отрезки не из кадра, а из людей. Для неё своя картинка.
        has_vp_segments = res.get("vp") is not None and res["vp"][0] is not None
        if res.get("cam") is not None and has_vp_segments:
            debug = write_debug(res, Path(require(cfg, "output", "debug_dir")))
            index = finalize_evidence(sampler, manifest)
        elif res.get("cam") is not None:
            debug = [_write_selfcalib_debug(res, Path(require(cfg, "output", "debug_dir")))]
            index = "не собраны: самокалибровка работает по артефактам S3"
        else:
            # У запасного пути нет ни пилотной выборки, ни ростов: рисовать
            # нечего, и пруфы по claim-ам этого конфига набрать неоткуда.
            debug = [_write_ground_debug(res, Path(require(cfg, "output", "debug_dir")))]
            index = "не собраны: запасной путь калибровки не делает пилотный прогон"

        manifest.note("output_artifact", str(out))
        manifest.note("debug_artifacts", [str(p) for p in debug])
        manifest.finish(STATUS_OK)
        print(f"[{STAGE}] записано: {out}, пруфы {index}, отладка "
              f"{', '.join(p.name for p in debug)}")
        return 0

    except (StageError, EvidenceError, CalibError, PilotError, ConfigError, OSError, ValueError) as exc:
        if manifest is not None:
            manifest.finish("failed", error=str(exc))
        print(f"[{STAGE}] ОШИБКА: {exc}", file=sys.stderr)
        return 1


def _collect_evidence(sampler, cfg: dict, res: dict) -> None:
    """Кропы людей из выборки роста и отрезков-затравок опорного кадра."""
    clip = Path(require(cfg, "input", "clip"))
    from looq.pilot import iter_frames

    by_frame: dict[int, list[tuple]] = {}
    for d, h_m in zip(res["usable"], res["heights_m"]):
        by_frame.setdefault(d.frame_idx, []).append((d, float(h_m)))

    for frame_idx, frame in iter_frames(clip, np.array(sorted(by_frame))):
        for d, h_m in by_frame[frame_idx]:
            x1, y1 = max(0, int(d.x1_px)), max(0, int(d.y1_px))
            x2, y2 = int(d.x2_px), int(d.y2_px)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 4:
                continue
            sampler.offer("claim.calib.scale_check", track_id=0, frame_idx=frame_idx,
                          ts=frame_idx / 30.0, crop_bgr=crop, value=h_m,
                          confidence=d.conf,
                          extra={"est_height_m": round(h_m, 3),
                                 "bbox_h_px": round(d.height_px, 1)})

    # Второй claim: невязка удержанных отрезков — кропы вокруг них на опорном кадре.
    ref = res["ref"]
    vp_h, vp_v = res["vp"]
    for base, (name, segs, resid) in enumerate(
            (("h", res["segs"][0], vp_h.holdout_residual_px),
             ("v", res["segs"][1], vp_v.holdout_residual_px))):
        for k, (x1, y1, x2, y2) in enumerate(segs[:40]):
            i = base * 1000 + k
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            half = 60
            crop = ref[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 4:
                continue
            # track_id = индекс отрезка: ключ индекса пруфов это
            # (claim_id, track_id, frame_idx), а кадр здесь у всех один.
            sampler.offer("claim.calib.holdout_reprojection", track_id=i,
                          frame_idx=int(res["doc"]["reference_frame_idx"]),
                          ts=0.0, crop_bgr=crop, value=float(resid),
                          confidence=float(np.hypot(x2 - x1, y2 - y1)) / 1000.0,
                          extra={"vp": name, "segment_idx": i})


if __name__ == "__main__":
    raise SystemExit(main())
