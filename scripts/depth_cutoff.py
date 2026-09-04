"""Замер: на какой глубине высота рамки падает ниже порога.

ЗАЧЕМ. Дальний конец улицы портит аналитику: за 25+ м детектор на этом
разрешении держит людей ненадёжно, а к витринам они отношения не имеют.
Полноценный замер recall по глубине требует ручной разметки, которой нет.
Прокси без ground truth — ВЫСОТА РАМКИ В ПИКСЕЛЯХ.

ЧТО СЧИТАЕТ. Медианную высоту рамки по бинам глубины плана и глубину, на
которой медиана уходит ниже detect.min_box_height_px. Плюс сколько треков
было и сколько осталось.

ЧЕГО НЕ ЗНАЕМ. Это НЕ recall. Порог зависит от масштаба, а масштаб взят из
медианного роста и независимо не подтверждён — глубина отсечки в метрах
наследует эту неопределённость.

    python scripts/depth_cutoff.py
    make depth-cutoff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.io import load_config, read_json, write_json  # noqa: E402

OUT = Path("out/depth_cutoff.json")
#: Допуск сопоставления трека с детекцией — доля высоты рамки. Замерено:
#: медиана 0.006, p95 0.024, p99 0.044. Тот же порог, что в отрисовке.
TOL_FRAC, TOL_MIN_PX = 0.08, 4.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/s3_detect.yaml")
    ap.add_argument("--bin-m", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    import pandas as pd

    cfg = load_config(args.config)
    thr = float((cfg.get("detect") or {}).get("min_box_height_px", 0.0))
    if thr <= 0:
        raise SystemExit("detect.min_box_height_px не задан в конфиге")
    hom = read_json("calib/homography.json")

    tracks = pd.read_parquet("track/tracks.parquet")
    det = pd.read_parquet("det/frames.parquet")

    # Каждой строке трека — высота его рамки. Сопоставление относительным
    # допуском: трекер сглаживает состояние, точного совпадения не бывает.
    rows = []
    for fi, g in tracks.groupby("frame_idx"):
        dd = det[det["frame_idx"] == fi]
        if dd.empty:
            continue
        cx = ((dd["x1_px"] + dd["x2_px"]) / 2.0).to_numpy()
        y2 = dd["y2_px"].to_numpy()
        bh = (dd["y2_px"] - dd["y1_px"]).to_numpy()
        for r in g.itertuples():
            d = np.hypot(cx - r.foot_x_px, y2 - r.foot_y_px)
            j = int(np.argmin(d))
            if d[j] <= max(TOL_MIN_PX, TOL_FRAC * bh[j]):
                rows.append((float(r.foot_x_m), float(bh[j]), int(r.track_id)))
    if not rows:
        raise SystemExit("ни одна строка трека не сопоставлена с детекцией")
    a = pd.DataFrame(rows, columns=["x_m", "box_h", "track_id"])

    lo = float(np.floor(a["x_m"].min()))
    hi = float(np.ceil(a["x_m"].max()))
    edges = np.arange(lo, hi + args.bin_m, args.bin_m)
    bins, cutoff = [], None
    print(f"порог detect.min_box_height_px = {thr:.0f} px, бин {args.bin_m:.0f} м\n")
    print(" глубина x, м | строк | медиана высоты рамки, px")
    for k in range(len(edges) - 1):
        m = (a["x_m"] >= edges[k]) & (a["x_m"] < edges[k + 1])
        if m.sum() < 10:                    # бин из пяти точек — не медиана
            continue
        med = float(a.loc[m, "box_h"].median())
        bins.append({"x_lo_m": float(edges[k]), "x_hi_m": float(edges[k + 1]),
                     "n_rows": int(m.sum()), "median_box_h_px": round(med, 1)})
        flag = ""
        if med < thr and cutoff is None:
            cutoff = float(edges[k])
            flag = "  <- отсечка"
        print(f"  {edges[k]:>5.0f}-{edges[k+1]:<5.0f} | {int(m.sum()):>5} "
              f"| {med:>6.0f}{flag}")

    n_before = int(a["track_id"].nunique())
    if cutoff is None:
        print(f"\nмедиана НЕ опускается ниже {thr:.0f} px ни в одном бине: "
              f"отсечка не сработает, все {n_before} треков останутся")
        n_after, rows_keep = n_before, 1.0
    else:
        keep = a["x_m"] < cutoff
        n_after = int(a.loc[keep, "track_id"].nunique())
        rows_keep = float(keep.mean())
        print(f"\nотсечка на {cutoff:.0f} м: остаётся {rows_keep:.1%} строк, "
              f"{n_after} треков из {n_before}")

    doc = {
        "schema_version": "1", "stage": "measurement", "status": "ok",
        "source": "scripts/depth_cutoff.py",
        "min_box_height_px": thr,
        "cutoff_depth_m": cutoff,
        "bin_m": args.bin_m,
        "bins": bins,
        "n_tracks_before": n_before, "n_tracks_after": n_after,
        "n_rows_before": int(len(a)), "n_rows_after": int(round(len(a) * rows_keep)),
        "rows_kept_frac": round(rows_keep, 4),
        "calib_status": hom.get("calib_status"),
        "scale_source_note_ru": (
            "Порог задан в пикселях, но его перевод в метры опирается на "
            "масштаб, взятый из медианного роста и независимо не подтверждённый. "
            "Глубина отсечки в метрах наследует эту неопределённость."),
        "not_measured_ru": (
            "Это НЕ recall по глубине. Полноценный замер recall требует ручной "
            "разметки, она не выполнена."),
    }
    write_json(args.out, doc)
    print(f"записано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
