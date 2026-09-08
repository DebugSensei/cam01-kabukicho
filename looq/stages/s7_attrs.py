"""S7 attrs — цвет верхней одежды по треку.

Одна строка = один трек. Этап опционален и был вырезан из скоупа; возвращён
2026-09-04 по требованию визуала.

ЧТО МЕРЯЕТСЯ. Медианный цвет области ТОРСА, а не всей рамки: голова и ноги
в неё не входят, иначе цвет волос и асфальта попадёт в «верхнюю одежду».
Границы области берутся долями высоты рамки и лежат в конфиге.

ПОЧЕМУ HSV И МЕДИАНА. Яркость на улице меняется от вывесок и теней, а тон
устойчивее. Медиана по пикселям области, а не среднее: среднее уводится
бликами и логотипами.

ЧТО ЗАПРЕЩЕНО. Пол, возраст, этничность — по двум причинам: приватность
(правило 9) и отсутствие ground truth для гейта (правило 7). Это явный отказ,
а не пункт бэклога.

ЧЕГО НЕ ЗНАЕМ. Точность классификации не измерена: разметки нет. Уверенность
в артефакте — это доля пикселей области, согласных с выбранным классом,
а НЕ вероятность правильности. Путать их нельзя.

    python -m looq.stages.s7_attrs --config configs/s7_attrs.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from looq import STATUS_OK, STATUS_SKELETON
from looq.color import (apply_wb, collect_road_pixels, hue_hist, road_mask,
                         sat_median, wb_gains_from_grey)
from looq.evidence import EvidenceError
from looq.io import ConfigError, RunManifest, load_config, read_json, require
from looq.pilot import iter_frames
from looq.stages._base import (
    Col,
    StageError,
    build_evidence,
    finalize_evidence,
    read_artifact_status,
    write_parquet,
)

STAGE = "s7_attrs"
INPUTS = ["track/tracks.parquet", "det/frames.parquet"]
OUTPUT = "attr/tracks_attr.parquet"

OUTPUT_COLS: list[Col] = [
    Col("track_id",        "int32",   False, "-",     "-", "идентификатор трека"),
    Col("top_color_name",  "string",  True,  "-",     "-", "цвет верхней одежды из палитры конфига; null = не определён"),
    Col("top_color_conf",  "float32", True,  "[0,1]", "-", "доля кадров за класс x доля согласных пикселей. НЕ вероятность правильности"),
    Col("top_color_status", "string", False, "-",     "-", "ok | too_small_px | low_agreement | background_match | not_attempted"),
    Col("n_crops_used",    "int16",   False, "шт",    "-", "кропов участвовало в голосовании"),
    Col("n_crops_available", "int16", False, "шт",    "-", "кропов было доступно"),
    Col("hsv_h",           "float32", True,  "0-179", "-", "медианный тон области торса"),
    Col("hsv_s",           "float32", True,  "0-255", "-", "медианная насыщенность"),
    Col("hsv_v",           "float32", True,  "0-255", "-", "медианная яркость"),
    Col("bg_hue_dist_deg", "float32", True,  "градус", "-", "расхождение тона торса и фона по краям кропа. Малое значение при равной насыщенности = цвет от освещения, а не от ткани"),
    Col("bg_sat_ratio",    "float32", True,  "-",     "-", "насыщенность торса / насыщенность фона. Около 1 при малом bg_hue_dist_deg = подсветка"),
]

#: Границы классов в HSV. Комментарии — почему граница там, где она есть.
#: НЕ ОТКАЛИБРОВАНО: подобрано по смыслу, на размеченных кропах не проверялось.
HUE_BANDS = [
    ("red",    [(0, 10), (170, 180)]),   # красный лежит по обе стороны нуля тона
    ("orange", [(10, 22)]),
    ("yellow", [(22, 33)]),
    ("green",  [(33, 85)]),
    ("blue",   [(85, 130)]),
    ("purple", [(130, 160)]),
    ("pink",   [(160, 170)]),
]


def classify_hsv(h: float, s: float, v: float, cfg: dict) -> tuple[str, str]:
    """Цвет по медианному HSV. Ахроматика решается ДО тона.

    Тон у серого и чёрного шумит: при низкой насыщенности он определяется
    случайным перевесом каналов. Поэтому сначала отсекаются чёрный, белый
    и серый, и только потом смотрится тон.
    """
    v_black = float(cfg.get("v_black_max", 55))
    v_white = float(cfg.get("v_white_min", 185))
    s_gray = float(cfg.get("s_achromatic_max", 45))
    if v <= v_black:
        return "black", "achromatic_dark"
    if s <= s_gray:
        return ("white", "achromatic_light") if v >= v_white else ("grey", "achromatic_mid")
    for name, bands in HUE_BANDS:
        for lo, hi in bands:
            if lo <= h < hi:
                return name, "hue"
    return "other", "hue_unmatched"


def _hue_distance(a: float, b: float) -> float:
    """Расстояние между тонами по кругу OpenCV [0, 180). Наивная разность даёт
    179 там, где на самом деле 1."""
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def torso_median_hsv(crop_bgr: np.ndarray, cfg: dict):
    """Медианный HSV торса, согласие пикселей и КОНТРАСТ ТОРСА К ФОНУ.

    Контраст нужен, чтобы отличить цвет одежды от цвета освещения. Подсветка
    красит и человека, и стену за ним одинаково, поэтому у «цвета от лампы»
    торс и фон совпадают по тону и по насыщенности. Замер 2026-09-08 по
    пруф-кропам: у оранжевого расхождение тона 2.1 градуса при равной
    насыщенности (96 против 99), у синего 12.5 градуса при насыщенности 81
    против 38, у чёрного 31.8 при 100 против 42.

    Фон берётся по левой и правой кромкам кропа НА ТОЙ ЖЕ ВЫСОТЕ, что и торс:
    сравнивать торс с небом над головой было бы сравнением разных вещей.
    """
    h, w = crop_bgr.shape[:2]
    top = float(cfg.get("torso_top_frac", 0.22))
    bot = float(cfg.get("torso_bottom_frac", 0.55))
    side = float(cfg.get("torso_side_frac", 0.18))
    y0, y1 = int(h * top), int(h * bot)
    x0, x1 = int(w * side), int(w * (1.0 - side))
    if y1 - y0 < 4 or x1 - x0 < 3:
        return None
    region = crop_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    med = np.median(hsv, axis=0)
    name, _ = classify_hsv(med[0], med[1], med[2], cfg)
    agree = np.mean([classify_hsv(p[0], p[1], p[2], cfg)[0] == name for p in hsv])

    edge = max(1, int(w * float(cfg.get("bg_edge_frac", 0.12))))
    sides = [crop_bgr[y0:y1, :edge], crop_bgr[y0:y1, w - edge:]]
    sides = [s for s in sides if s.size]
    if sides:
        bg = np.median(np.vstack([
            cv2.cvtColor(s, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
            for s in sides]), axis=0)
        d_hue = _hue_distance(med[0], bg[0])
        # Отношение насыщенностей, а не разность: разность в 20 единиц значит
        # разное при насыщенности 30 и при 150.
        r_sat = float(med[1] + 1.0) / float(bg[1] + 1.0)
    else:
        d_hue, r_sat = float("nan"), float("nan")
    return med, float(agree), float(d_hue), float(r_sat)


def run(cfg: dict[str, Any], manifest: RunManifest, sampler) -> dict[str, Any]:
    import pandas as pd

    if read_artifact_status("track/tracks.parquet") == STATUS_SKELETON:
        raise StageError("track/tracks.parquet помечен status=skeleton: S4 не отработал")
    attrs_cfg = require(cfg, "attrs")
    if not bool(attrs_cfg.get("enabled", False)):
        raise StageError(
            "attrs.enabled=false. Этап опционален; чтобы его прогнать, включите "
            "флаг явно — молча считать выключенный этап нельзя")

    video = Path(require(cfg, "input", "video"))
    if not video.is_file():
        raise StageError(f"нет видео {video}")

    tracks = pd.read_parquet("track/tracks.parquet")
    det = pd.read_parquet("det/frames.parquet")
    n_crops = int(attrs_cfg["n_crops_per_track"])
    min_h = float(attrs_cfg["min_crop_h_px"])
    min_agree = float(attrs_cfg.get("min_agreement", 0.35))
    palette = set(attrs_cfg["palette"])

    # Кропы берутся равномерно по жизни трека: подряд с начала дали бы один
    # ракурс и один свет.
    picks: dict[int, list[int]] = {}
    for tid, g in tracks.groupby("track_id"):
        f = g["frame_idx"].to_numpy()
        idx = np.unique(np.linspace(0, len(f) - 1, min(n_crops, len(f)))
                        .round().astype(int))
        picks[int(tid)] = [int(f[i]) for i in idx]
    want: dict[int, list[int]] = {}
    for tid, fl in picks.items():
        for f in fl:
            want.setdefault(f, []).append(tid)

    foot = tracks.set_index(["frame_idx", "track_id"])[["foot_x_px", "foot_y_px"]]
    det_by_frame = {int(f): g for f, g in det.groupby("frame_idx")}
    samples: dict[int, list] = {}
    crops_for_evidence: dict[int, tuple] = {}
    clip_by_track: dict[int, list] = {}

    # ---- баланс белого ---------------------------------------------------- #
    wb_cfg = attrs_cfg.get("white_balance") or {}
    wb_on = bool(wb_cfg.get("enabled", False))
    roi_px = None
    if wb_on:
        traced = read_json("zones/zones.json")
        roi_px = np.asarray(traced["roi_px"], dtype=np.float64)
    order = sorted(want)
    wb_frames = set(order[:int(wb_cfg.get("n_frames", 50))]) if wb_on else set()
    road_px: list[np.ndarray] = []
    per_frame_gains: list[np.ndarray] = []
    excl = {"in_roi_minus_people": 0, "clipped": 0, "too_saturated": 0}
    hist_before = hist_after = None
    sat_before = sat_after = float("nan")
    gains = np.ones(3)
    pending: list[tuple] = []      # кропы, снятые ДО того как коэффициенты готовы

    wb_ready = not wb_on
    for fi, frame in iter_frames(video, np.asarray(order, dtype=np.int64)):
        dets = det_by_frame.get(int(fi))
        if dets is None:
            continue

        if wb_on and int(fi) in wb_frames:
            m = road_mask(frame.shape, roi_px,
                          dets[["x1_px", "y1_px", "x2_px", "y2_px"]].to_numpy(),
                          int(wb_cfg.get("person_dilate_px", 8)))
            px, st = collect_road_pixels(frame, m, wb_cfg)
            for k in excl:
                excl[k] += st[k]
            if len(px):
                road_px.append(px)
                try:                       # покадровые коэффициенты — замер дрейфа
                    per_frame_gains.append(wb_gains_from_grey(
                        px, {**wb_cfg, "min_pixels": 1})[0])
                except ValueError:
                    pass
            if hist_before is None:
                hist_before, sat_before = hue_hist(frame, m), sat_median(frame, m)
            # Коэффициенты готовы, как только пройден последний кадр выборки.
            if int(fi) == max(wb_frames):
                allpx = np.concatenate(road_px) if road_px else np.empty((0, 3))
                gains, wb_meta = wb_gains_from_grey(allpx, wb_cfg)
                corr, _ = apply_wb(frame, gains)
                hist_after, sat_after = hue_hist(corr, m), sat_median(corr, m)
                wb_ready = True
                print(f"[{STAGE}] серая точка BGR {wb_meta['grey_point_bgr']}, "
                      f"коэффициенты {gains.round(3).tolist()}")
        cx = (dets["x1_px"] + dets["x2_px"]) / 2.0
        for tid in want[int(fi)]:
            try:
                fx, fy = foot.loc[(int(fi), int(tid))]
            except KeyError:
                continue
            m = (np.abs(cx - fx) < 1.0) & (np.abs(dets["y2_px"] - fy) < 1.0)
            if not m.any():
                continue
            b = dets[m].iloc[0]
            if (b.y2_px - b.y1_px) < min_h:
                continue
            crop = frame[max(0, int(b.y1_px)):int(b.y2_px),
                         max(0, int(b.x1_px)):int(b.x2_px)]
            if not wb_ready:
                # Коэффициенты ещё не посчитаны. Классифицировать сейчас
                # значило бы смешать в одной колонке скорректированные и
                # нескорректированные кропы.
                pending.append((int(tid), crop, int(fi), float(b.conf)))
                continue
            crop_wb, clip_frac = apply_wb(crop, gains) if wb_on else (crop, 0.0)
            got = torso_median_hsv(crop_wb, attrs_cfg)
            if got is None:
                continue
            samples.setdefault(int(tid), []).append(got)
            clip_by_track.setdefault(int(tid), []).append(clip_frac)
            crops_for_evidence.setdefault(int(tid), (crop_wb, int(fi), float(b.conf)))

    # Кропы, снятые до готовности коэффициентов, обрабатываются теми же
    # коэффициентами: иначе часть треков считалась бы по другому правилу.
    for tid, crop, fi, cf in pending:
        crop_wb, clip_frac = apply_wb(crop, gains) if wb_on else (crop, 0.0)
        got = torso_median_hsv(crop_wb, attrs_cfg)
        if got is None:
            continue
        samples.setdefault(tid, []).append(got)
        clip_by_track.setdefault(tid, []).append(clip_frac)
        crops_for_evidence.setdefault(tid, (crop_wb, fi, cf))
    pending.clear()

    rows: list[dict[str, Any]] = []
    stats = {"ok": 0, "too_small_px": 0, "low_agreement": 0,
             "background_match": 0, "not_attempted": 0}
    # Пороги отделения «цвет лампы» от «цвет ткани». Значения в конфиге.
    bg_hue_min = float(attrs_cfg.get("bg_min_hue_dist_deg", 6.0))
    bg_sat_tol = float(attrs_cfg.get("bg_sat_ratio_tol", 0.25))
    by_color: dict[str, list[int]] = {}
    for tid in sorted(picks):
        avail = len(picks[tid])
        got = samples.get(tid, [])
        if not got:
            stats["too_small_px"] += 1
            rows.append({"track_id": tid, "top_color_name": None, "top_color_conf": None,
                         "top_color_status": "too_small_px", "n_crops_used": 0,
                         "n_crops_available": avail, "hsv_h": None, "hsv_s": None,
                         "hsv_v": None, "bg_hue_dist_deg": None,
                         "bg_sat_ratio": None})
            continue
        # ГОЛОСОВАНИЕ ПО КРОПАМ, а не усреднение HSV между кадрами. Медиана
        # тона по кадрам смешивает разные условия освещения в одно число,
        # которого не было ни на одном кадре: человек под красной вывеской
        # и он же в тени дают "средний" тон, не равный ни одному наблюдению.
        # Голосование выбирает класс, который реально повторился чаще всего.
        votes: dict[str, int] = {}
        for med_i, *_rest in got:
            nm, _ = classify_hsv(med_i[0], med_i[1], med_i[2], attrs_cfg)
            if nm not in palette:
                nm = "other"
            votes[nm] = votes.get(nm, 0) + 1
        name = max(sorted(votes), key=lambda k: votes[k])
        winners = [i for i, (mm, *_r) in enumerate(got)
                   if classify_hsv(mm[0], mm[1], mm[2], attrs_cfg)[0] == name]
        # HSV в артефакте — медиана ТОЛЬКО по кадрам, проголосовавшим за
        # победивший класс: иначе записанный тон противоречил бы записанному
        # классу.
        med = np.median(np.stack([got[i][0] for i in winners]), axis=0)
        # Контраст к фону по тем же кадрам, что дали победивший класс.
        d_hue = float(np.nanmedian([got[i][2] for i in winners]))
        r_sat = float(np.nanmedian([got[i][3] for i in winners]))
        # Две доли перемножаются: сколько кадров сошлись на классе и насколько
        # чисто выглядел торс на этих кадрах. Обе — доли согласия, не
        # вероятности правильности.
        vote_share = votes[name] / len(got)
        pixel_agree = float(np.mean([got[i][1] for i in winners]))
        agree = float(vote_share * pixel_agree)
        status = "ok" if agree >= min_agree else "low_agreement"
        # Торс неотличим от фона И по тону, И по насыщенности — это цвет лампы,
        # а не ткани. Условие «И», а не «ИЛИ»: тёмная одежда на тёмной стене
        # честно совпадает по насыщенности, но расходится по тону.
        if status == "ok" and np.isfinite(d_hue) and np.isfinite(r_sat) \
                and d_hue < bg_hue_min and abs(r_sat - 1.0) < bg_sat_tol:
            status = "background_match"
        stats[status] = stats.get(status, 0) + 1
        if status == "ok":
            by_color.setdefault(name, []).append(tid)
        rows.append({
            "track_id": tid,
            "top_color_name": name if status == "ok" else None,
            "top_color_conf": agree,
            "top_color_status": status,
            "n_crops_used": len(got), "n_crops_available": avail,
            "hsv_h": float(med[0]), "hsv_s": float(med[1]), "hsv_v": float(med[2]),
            "bg_hue_dist_deg": None if not np.isfinite(d_hue) else round(d_hue, 2),
            "bg_sat_ratio": None if not np.isfinite(r_sat) else round(r_sat, 3),
        })

    coverage = stats["ok"] / max(1, len(rows))
    for name, tids in by_color.items():
        for tid in tids:
            crop, fi, conf = crops_for_evidence[tid]
            r = next(x for x in rows if x["track_id"] == tid)
            extra = {"color": name, "hsv": [round(float(r[k]), 1)
                                            for k in ("hsv_h", "hsv_s", "hsv_v")]}
            # Два claim-а на один кроп: обобщённый объявлен в конфиге и держит
            # правило 1 для колонки целиком, поцветной нужен визуалу, чтобы
            # показать пруфы отдельно по каждому цвету.
            for claim in (f"claim.attrs.color.{name}", "claim.attrs.top_color"):
                sampler.offer(claim, track_id=tid, frame_idx=fi, ts=0.0,
                              crop_bgr=crop, value=float(r["hsv_h"]),
                              confidence=float(r["top_color_conf"]), extra=extra)

    manifest.note("color_stats", stats)
    if wb_on:
        gsp = (np.percentile(np.stack(per_frame_gains), [10, 50, 90], axis=0).round(3)
               .tolist() if len(per_frame_gains) > 2 else None)
        manifest.note("white_balance", {
            "applied": True, "reference": "road_surface",
            "reference_is_measured": False,
            "gains_bgr": gains.round(4).tolist(),
            "excluded_by": excl,
            "gain_spread_p10_p50_p90": gsp,
            "sat_road_before": round(sat_before, 1),
            "sat_road_after": round(sat_after, 1),
            "hue_hist_road_before": hist_before,
            "hue_hist_road_after": hist_after,
            "note_ru": ("Падение насыщенности фона к нулю доказывает, что коррекция "
                        "ПРИМЕНИЛАСЬ, а не что она ВЕРНА: коэффициенты считались "
                        "из этого же фона. Единственная некольцевая проверка — "
                        "ручная разметка."),
        })
        print(f"[{STAGE}] насыщенность мостовой: до {sat_before:.1f} -> "
              f"после {sat_after:.1f} (падение к нулю = коррекция применилась, "
              f"НЕ доказательство её верности)")
    manifest.note("coverage", round(coverage, 4))
    manifest.note("by_color", {k: len(v) for k, v in by_color.items()})
    print(f"[{STAGE}] треков {len(rows)}, определён цвет у {stats['ok']} "
          f"({coverage:.1%}), низкое согласие {stats['low_agreement']}, "
          f"мелкие {stats['too_small_px']}, "
          f"неотличимы от фона {stats['background_match']}")
    print(f"[{STAGE}] по цветам: {dict(sorted(((k, len(v)) for k, v in by_color.items()), key=lambda x: -x[1]))}")
    print(f"[{STAGE}] ТОЧНОСТЬ НЕ ИЗМЕРЕНА: разметки нет. top_color_conf — доля "
          f"согласных пикселей, а не вероятность правильности")
    return {"rows": rows, "coverage": coverage, "by_color": by_color, "stats": stats}


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
        write_parquet(OUTPUT, OUTPUT_COLS, res["rows"], STAGE, STATUS_OK,
                      inputs=INPUTS)
        index = finalize_evidence(sampler, manifest)
        manifest.note("output_artifact", OUTPUT)
        manifest.note("elapsed_s", round(time.time() - _t0, 1))
        manifest.finish(STATUS_OK)
        print(f"[{STAGE}] записано: {OUTPUT} ({len(res['rows'])} строк), пруфы {index}")
        return 0
    except (StageError, EvidenceError, ConfigError, OSError, ValueError, KeyError) as exc:
        if manifest is not None:
            manifest.finish("failed", error=str(exc))
        print(f"[{STAGE}] ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
