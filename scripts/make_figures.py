"""out/img/*.png — статические картинки для дашборда.

  plan_all.png    вид сверху со ВСЕМИ траекториями за прогон, сетка в метрах
  zones_ref.png   опорный кадр с обведёнными витринами
  gaze_example.png  пример: человек, чей луч попал в фасад

Ничего не пересчитывает: только рисует то, что уже лежит в артефактах.

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_overlay import PlanView, ZONE_COLORS, track_color  # noqa: E402

from looq.geometry import point_in_polygon_m  # noqa: E402
from looq.anonymise import save_image  # noqa: E402
from looq.io import load_config, read_json, require  # noqa: E402

OUT_DIR = Path("out/img")
PLAN_SIZE = (900, 1500)     # ширина, высота


def _facades(zones):
    out, roi = {}, None
    for f in zones["features"]:
        pr, g = f["properties"], f["geometry"]
        if pr["zone_type"] == "roi":
            roi = np.asarray(g["coordinates"][0], dtype=np.float64)
        elif pr["zone_type"] == "facade":
            out[pr["zone_id"]] = {
                "name": pr["name_ru"],
                "color": ZONE_COLORS[len(out) % len(ZONE_COLORS)],
                "poly": np.asarray(pr["polygon_px"], dtype=np.int32),
                "seg_m": np.asarray(g["coordinates"], dtype=np.float64),
            }
    return out, roi


def plan_all(tracks, zones, unit: str, path: Path) -> None:
    """Все траектории за прогон одной картинкой + сетка в метрах.

    Рисуется ТОЛЬКО то, что внутри ROI. Дальний конец улицы за отсечкой в
    аналитику не идёт, и на плане ему делать нечего: клубок из обрывков
    дальних треков читается как хаос и портит проверку калибровки глазом.
    """
    w, h = PLAN_SIZE
    facades, roi = _facades(zones)
    n_all = tracks["track_id"].nunique()
    if roi is not None:
        inside = np.array([point_in_polygon_m((float(x), float(y)), roi)
                           for x, y in tracks[["foot_x_m", "foot_y_m"]].to_numpy()])
        tracks = tracks[inside]
        print(f"  ROI отсекает: строк {inside.mean():.1%}, "
              f"треков {tracks['track_id'].nunique()} из {n_all}")
    pts = [tracks[["foot_x_m", "foot_y_m"]].dropna().to_numpy(np.float64)]
    if roi is not None:
        pts.append(roi)
    pts += [f["seg_m"] for f in facades.values()]
    plan = PlanView(w, h, np.concatenate(pts))
    img = np.full((h, w, 3), 250, np.uint8)

    # Сетка каждый метр: без неё «прямая улица» — впечатление, а не проверка.
    allm = np.concatenate(pts)
    lo, hi = allm.min(axis=0), allm.max(axis=0)
    for xm in range(int(np.floor(lo[0])) - 1, int(np.ceil(hi[0])) + 2):
        a = plan.to_px([xm, lo[1] - 2])[0]
        b = plan.to_px([xm, hi[1] + 2])[0]
        major = xm % 5 == 0
        cv2.line(img, tuple(a), tuple(b), (205, 205, 205) if major else (232, 232, 232),
                 2 if major else 1, cv2.LINE_AA)
        if major:
            cv2.putText(img, f"{xm}", tuple(plan.to_px([xm, lo[1] - 1])[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    for ym in range(int(np.floor(lo[1])) - 1, int(np.ceil(hi[1])) + 2):
        a = plan.to_px([lo[0] - 2, ym])[0]
        b = plan.to_px([hi[0] + 2, ym])[0]
        major = ym % 5 == 0
        cv2.line(img, tuple(a), tuple(b), (205, 205, 205) if major else (232, 232, 232),
                 2 if major else 1, cv2.LINE_AA)

    if roi is not None:
        cv2.polylines(img, [plan.to_px(roi)], True, (150, 150, 150), 2, cv2.LINE_AA)

    n = 0
    for tid, g in tracks.sort_values("frame_idx").groupby("track_id"):
        m = g[["foot_x_m", "foot_y_m"]].dropna().to_numpy(np.float64)
        if len(m) < 3:
            continue
        cv2.polylines(img, [plan.to_px(m)], False, track_color(int(tid)), 1, cv2.LINE_AA)
        n += 1

    for zid, fac in facades.items():
        seg = plan.to_px(fac["seg_m"])
        cv2.line(img, tuple(seg[0]), tuple(seg[1]), fac["color"], 8, cv2.LINE_AA)
        mid = seg.mean(axis=0).astype(int)
        cv2.putText(img, zid.replace("facade_", ""), (mid[0] + 12, mid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)

    cv2.putText(img, f"plan view, grid = 1 {unit} (bold = 5 {unit})", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(img, f"{n} trajectories inside ROI (of {n_all} total)", (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
    save_image(path, img)
    print(f"  {path} ({n} траекторий)")


def zones_ref(video: Path, zones, frame_idx: int, path: Path) -> None:
    facades, _ = _facades(zones)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"не прочитан кадр {frame_idx} из {video}")
    for zid, fac in facades.items():
        ov = frame.copy()
        cv2.fillPoly(ov, [fac["poly"]], fac["color"])
        cv2.addWeighted(ov, 0.22, frame, 0.78, 0, dst=frame)
        cv2.polylines(frame, [fac["poly"]], True, fac["color"], 3, cv2.LINE_AA)
        x0, y0 = int(fac["poly"][:, 0].min()), int(fac["poly"][:, 1].min())
        cv2.putText(frame, zid.replace("facade_", ""), (x0, max(22, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, fac["color"], 3, cv2.LINE_AA)
    save_image(path, frame, quality=90)
    print(f"  {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/s3_detect.yaml")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    import pandas as pd

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    video = Path(require(cfg, "input", "video"))
    hom = read_json("calib/homography.json")
    unit = "m" if hom.get("scale_known") else "unit"
    zones = read_json("zones/zones.geojson")
    tracks = pd.read_parquet("track/tracks.parquet")
    zf = pd.read_parquet("attn/track_zone_frames.parquet")

    print("рисую:")
    plan_all(tracks, zones, unit, args.out_dir / "plan_all.png")

    # Опорный кадр — тот, где в кадре больше всего людей: на пустом кадре
    # зоны не с чем соотнести.
    busiest = int(tracks.groupby("frame_idx").size().idxmax())
    zones_ref(video, zones, busiest, args.out_dir / "zones_ref.jpg")

    hits = zf[zf["gaze_hit"]]
    if len(hits):
        fi = int(hits["frame_idx"].iloc[len(hits) // 2])
        src = Path("out/overlay_frames")
        near = sorted(src.glob("*.jpg"), key=lambda p: abs(int(p.stem[1:]) - fi))
        if near:
            img = cv2.imread(str(near[0]))
            save_image(args.out_dir / "gaze_example.jpg", img, quality=90)
            print(f"  {args.out_dir / 'gaze_example.jpg'} (кадр {near[0].stem})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
