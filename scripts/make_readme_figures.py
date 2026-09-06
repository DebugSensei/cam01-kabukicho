"""docs/img/*.webp — картинки для README.

Всё строится ИЗ АРТЕФАКТОВ, ничего не рисуется от руки: гистограмма роста и
скаттер дрейфа берут pilot_heights_m и pilot_depths_m из calib/homography.json,
горизонт и точка схода — оттуда же.

webp шириной 1200: в README больше не нужно, а jpg того же качества весит
примерно вдвое больше.

    python scripts/make_readme_figures.py
    make readme-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.io import read_json  # noqa: E402
from looq.anonymise import save_image  # noqa: E402

OUT_DIR = Path("docs/img")
WIDTH = 1200
QUALITY = 80

INK = (26, 22, 15)
MUTED = (130, 116, 107)
GRID = (240, 233, 230)
ACCENT = (47, 107, 255)
BG = (252, 250, 248)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def save(out_dir: Path, name: str, img: np.ndarray) -> None:
    if img.shape[1] != WIDTH:
        s = WIDTH / img.shape[1]
        img = cv2.resize(img, (WIDTH, int(round(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    p = out_dir / name
    save_image(p, img, quality=QUALITY)
    print(f"  {p}  {p.stat().st_size / 1024:.0f} KB  {img.shape[1]}x{img.shape[0]}")


def _canvas(w: int, h: int, pad):
    img = np.full((h, w, 3), BG, np.uint8)
    left, right, top, bottom = pad
    cv2.rectangle(img, (left, top), (w - right, h - bottom), GRID, 1)
    return img, (left, w - right, top, h - bottom)


def height_hist(hom, out_dir: Path, name: str) -> None:
    """Гистограмма роста. Медиана задаёт масштаб, зелёная полоса — диапазон
    правдоподобия, по которому спутниковая ширина была отвергнута."""
    hts = np.asarray(hom["pilot_heights_m"], dtype=np.float64)
    hts = hts[(hts > 0.8) & (hts < 2.6)]
    w, h = 1200, 520
    img, (x0, x1, y0, y1) = _canvas(w, h, (70, 30, 46, 62))
    lo, hi = 1.0, 2.4
    counts, edges = np.histogram(hts, bins=56, range=(lo, hi))
    cmax = int(counts.max()) or 1

    def px(v):
        return int(x0 + (x1 - x0) * (v - lo) / (hi - lo))

    cv2.rectangle(img, (px(1.55), y0), (px(1.75), y1), (226, 244, 232), -1)
    for k, c in enumerate(counts):
        a, b = px(edges[k]), px(edges[k + 1])
        top = int(y1 - (y1 - y0) * c / cmax)
        cv2.rectangle(img, (a + 1, top), (b - 1, y1), (196, 176, 166), -1)

    med = float(np.median(hts))
    cv2.line(img, (px(med), y0), (px(med), y1), ACCENT, 2, cv2.LINE_AA)
    cv2.putText(img, f"median {med:.2f} m  (source of scale)",
                (px(med) + 8, y0 + 26), FONT, 0.58, ACCENT, 2, cv2.LINE_AA)
    cv2.line(img, (px(1.93), y0), (px(1.93), y1), (60, 60, 200), 2, cv2.LINE_AA)
    cv2.putText(img, "1.93 m implied by the 6.06 m satellite width",
                (px(1.93) - 470, y1 - 16), FONT, 0.55, (60, 60, 200), 2, cv2.LINE_AA)

    for v in np.arange(1.0, 2.41, 0.2):
        cv2.line(img, (px(v), y1), (px(v), y1 + 6), MUTED, 1)
        cv2.putText(img, f"{v:.1f}", (px(v) - 14, y1 + 26), FONT, 0.5, MUTED,
                    1, cv2.LINE_AA)
    cv2.putText(img, f"pedestrian height, n = {len(hts)}", (x0, y0 - 16),
                FONT, 0.62, INK, 2, cv2.LINE_AA)
    cv2.putText(img, "green band: plausible median 1.55-1.75 m", (x0, y1 + 50),
                FONT, 0.5, MUTED, 1, cv2.LINE_AA)
    save(out_dir, name, img)


def height_drift(hom, out_dir: Path, name: str) -> None:
    """Рост против глубины: тот самый дрейф, который НЕ называется уклоном."""
    d = np.asarray(hom["pilot_depths_m"], dtype=np.float64)
    hts = np.asarray(hom["pilot_heights_m"], dtype=np.float64)
    ok = (hts > 0.8) & (hts < 2.6) & np.isfinite(d) & (d > 0)
    d, hts = d[ok], hts[ok]
    w, h = 1200, 520
    img, (x0, x1, y0, y1) = _canvas(w, h, (70, 30, 46, 62))
    dlo, dhi = float(np.percentile(d, 1)), float(np.percentile(d, 99))
    hlo, hhi = 1.1, 2.3

    def px(v):
        return int(x0 + (x1 - x0) * (v - dlo) / max(dhi - dlo, 1e-6))

    def py(v):
        return int(y1 - (y1 - y0) * (v - hlo) / (hhi - hlo))

    step = max(1, len(d) // 4500)
    for a, b in zip(d[::step], hts[::step]):
        if dlo <= a <= dhi and hlo <= b <= hhi:
            cv2.circle(img, (px(a), py(b)), 1, (206, 190, 182), -1, cv2.LINE_AA)

    # Наклон считается ЗДЕСЬ по тем же массивам, что нарисованы точками, а не
    # берётся из поля height_depth_slope: иначе прямая может не соответствовать
    # облаку. Ровно это и было — этап писал в поле значение, домноженное на
    # scale_rescale_factor.
    slope, icept = np.polyfit(d, hts, 1)
    slope = float(slope)
    med = float(np.median(hts))
    dmid = float(np.median(d))
    cv2.line(img, (px(dlo), py(med + slope * (dlo - dmid))),
             (px(dhi), py(med + slope * (dhi - dmid))), ACCENT, 3, cv2.LINE_AA)
    # CI бутстрапом по тем же точкам, тоже без опоры на поле артефакта.
    rng = np.random.default_rng(20260904)
    boot = [np.polyfit(d[i], hts[i], 1)[0]
            for i in (rng.integers(0, len(d), len(d)) for _ in range(400))]
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    cv2.putText(img, f"slope {slope:.4f} m/m, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]"
                     f" - does not cover zero",
                (x0 + 12, y0 + 28), FONT, 0.58, ACCENT, 2, cv2.LINE_AA)
    cv2.putText(img, "reconstructed height vs depth", (x0, y0 - 16),
                FONT, 0.62, INK, 2, cv2.LINE_AA)
    for v in np.linspace(dlo, dhi, 7):
        cv2.putText(img, f"{v:.0f} m", (px(v) - 18, y1 + 26), FONT, 0.5, MUTED,
                    1, cv2.LINE_AA)
    for v in (1.2, 1.5, 1.8, 2.1):
        cv2.putText(img, f"{v:.1f}", (14, py(v) + 5), FONT, 0.5, MUTED, 1, cv2.LINE_AA)
    save(out_dir, name, img)


def vanishing(hom, video: Path, out_dir: Path, name: str) -> None:
    """Опорный кадр с горизонтом и точкой схода улицы."""
    if not video.is_file():
        print(f"  пропуск {name}: нет {video}")
        return
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"  пропуск {name}: кадр не прочитан")
        return
    h, w = frame.shape[:2]
    ov = frame.copy()

    a, b, c = np.asarray(hom["horizon_line"], dtype=np.float64)
    if abs(b) > 1e-9:
        p0 = (0, int(-c / b))
        p1 = (w, int(-(a * w + c) / b))
        cv2.line(ov, p0, p1, (80, 220, 255), 3, cv2.LINE_AA)
        cv2.putText(ov, "horizon, from pedestrian pairs", (26, max(34, p0[1] - 16)),
                    FONT, 0.85, (80, 220, 255), 3, cv2.LINE_AA)

    vp = np.asarray(hom.get("vp_horizontal") or [np.nan, np.nan], dtype=np.float64)
    if np.all(np.isfinite(vp)) and 0 <= vp[0] < w and 0 <= vp[1] < h:
        p = (int(vp[0]), int(vp[1]))
        cv2.circle(ov, p, 12, (60, 90, 255), -1, cv2.LINE_AA)
        cv2.circle(ov, p, 26, (60, 90, 255), 3, cv2.LINE_AA)
        cv2.putText(ov, "street vanishing point", (p[0] + 36, p[1] + 8),
                    FONT, 0.85, (60, 90, 255), 3, cv2.LINE_AA)

    cv2.addWeighted(ov, 0.9, frame, 0.1, 0, dst=frame)
    txt = (f"f = {hom['focal_px']:.0f} px    camera height "
           f"{hom['camera_height_m']:.2f} m    {hom['n_people_used']} people used")
    cv2.putText(frame, txt, (26, h - 30), FONT, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    save(out_dir, name, frame)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    hom = read_json("calib/homography.json")
    print("рисую:")
    height_hist(hom, args.out, "calib_height_hist.webp")
    height_drift(hom, args.out, "calib_height_drift.webp")
    vanishing(hom, Path(hom["clip"]), args.out, "calib_vanishing.webp")

    for src, dst in (("out/img/plan_all.png", "plan_trajectories.webp"),
                     ("out/img/zones_ref.jpg", "zones_reference.webp")):
        img = cv2.imread(src)
        if img is None:
            print(f"  пропуск {dst}: нет {src}")
            continue
        save(args.out, dst, img)

    frames = sorted(Path("out/overlay_frames").glob("*.jpg"))
    if frames:
        save(args.out, "overlay_frame.webp", cv2.imread(str(frames[len(frames) // 2])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
