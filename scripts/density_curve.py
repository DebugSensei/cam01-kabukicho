"""Кривая присутствия по одному скачанному файлу — обоснование выбора часа.

Число людей в кадре раз в step_sec. Результат идёт в отчёт, поэтому прогон
обязан быть воспроизводимым: время JST считается из аргумента --start-jst,
а НЕ из настенных часов машины. Повторный прогон того же файла с теми же
аргументами обязан дать тот же csv побайтово.

    python scripts/density_curve.py raw/peak_1900-2200JST.ts \\
        --start-jst "2026-09-03 19:00" --step-sec 60

Выход:
    out/density.csv        t_sec, jst_time, n_people
    out/density.png        график с отмеченным лучшим часом
    out/density_meta.json  провенанс: веса, sha256, параметры инференса (правило 6)

Это не этап пайплайна: у него нет гейта и он не пишет артефакт по контракту.
Это инструмент выбора входных данных, и его провенанс всё равно фиксируется —
число из него попадает в отчёт.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.io import library_versions, sha256_file, utc_now_iso, write_json  # noqa: E402

OUT_CSV = Path("out/density.csv")
OUT_PNG = Path("out/density.png")
OUT_META = Path("out/density_meta.json")

# Параметры инференса зафиксированы: они те же, что в configs/s3_detect.yaml,
# иначе кривая присутствия и сам пайплайн считали бы разных людей.
IMGSZ = 1280
CONF = 0.25
CLASSES = [0]
HALF = True
DEVICE = 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="один mp4/ts файл")
    ap.add_argument("--start-jst", required=True, metavar='"YYYY-MM-DD HH:MM"',
                    help="время JST первого кадра файла; из него, а не из "
                         "системных часов, считается колонка jst_time")
    ap.add_argument("--step-sec", type=float, default=60.0,
                    help="шаг выборки кадров, секунды (по умолчанию 60)")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--window-min", type=float, default=60.0,
                    help="длина окна, которое ищем как самое плотное, минуты")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.input.is_file():
        raise SystemExit(f"нет входного файла: {args.input}")
    try:
        start_jst = dt.datetime.strptime(args.start_jst, "%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise SystemExit(f"--start-jst: {exc}") from None
    if args.step_sec <= 0:
        raise SystemExit("--step-sec должен быть больше нуля")

    import torch
    if not torch.cuda.is_available():
        # Правило 8: half=True на CPU молча даёт мусор. Лучше упасть.
        raise SystemExit("CUDA недоступна, а инференс задан с half=True на device=0")

    from ultralytics import YOLO
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"cv2 не открыл {args.input}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not (1.0 < fps < 240.0):
        raise SystemExit(f"неправдоподобный fps={fps} у {args.input}")
    step_frames = max(1, int(round(args.step_sec * fps)))

    # Последовательное чтение, а не перемотка: у .ts из HLS сегментов
    # CAP_PROP_POS_FRAMES врёт на границах сегментов, и выборка поехала бы.
    rows: list[dict] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step_frames == 0:
            res = model.predict(frame, imgsz=IMGSZ, classes=CLASSES, conf=CONF,
                                device=DEVICE, half=HALF, verbose=False)
            n = int(len(res[0].boxes))
            t_sec = frame_idx / fps
            stamp = start_jst + dt.timedelta(seconds=t_sec)
            rows.append({"t_sec": round(t_sec, 3),
                         "jst_time": stamp.strftime("%Y-%m-%d %H:%M:%S"),
                         "n_people": n})
            print(f"  t={t_sec:8.1f}s  JST {stamp:%H:%M:%S}  людей {n}")
        frame_idx += 1
    cap.release()

    if not rows:
        raise SystemExit(f"из {args.input} не декодировано ни одного кадра")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["t_sec", "jst_time", "n_people"])
        w.writeheader()
        w.writerows(rows)

    best = _best_window(rows, args.window_min * 60.0)
    _plot(rows, best, args)

    write_json(OUT_META, {
        "generated_utc": utc_now_iso(),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "start_jst": args.start_jst,
        "step_sec": args.step_sec,
        "fps": fps,
        "n_samples": len(rows),
        "model": {"weights": args.weights, "weights_sha256": sha256_file(args.weights),
                  "imgsz": IMGSZ, "conf": CONF, "classes": CLASSES,
                  "half": HALF, "device": DEVICE},
        "libraries": library_versions(),
        "best_window": best,
    })

    counts = [r["n_people"] for r in rows]
    print()
    print(f"замеров: {len(rows)}, людей в кадре min/медиана/max: "
          f"{min(counts)} / {sorted(counts)[len(counts) // 2]} / {max(counts)}")
    if best:
        print(f"самое плотное окно {args.window_min:.0f} мин: "
              f"{best['start_jst']} — {best['end_jst']}, среднее {best['mean_people']:.1f}")
    print(f"{OUT_CSV}, {OUT_PNG}, {OUT_META}")
    return 0


def _best_window(rows: list[dict], window_sec: float) -> dict | None:
    """Самое плотное непрерывное окно — это и есть обоснование выбора часа."""
    if rows[-1]["t_sec"] - rows[0]["t_sec"] < window_sec:
        return None
    best = None
    for i, start in enumerate(rows):
        # Окно должно целиком помещаться в запись, иначе последние окна
        # усредняются по меньшему числу замеров и выигрывают незаслуженно.
        if start["t_sec"] + window_sec > rows[-1]["t_sec"]:
            break
        inside = [r for r in rows[i:] if r["t_sec"] <= start["t_sec"] + window_sec]
        mean = sum(r["n_people"] for r in inside) / len(inside)
        if best is None or mean > best["mean_people"]:
            best = {"start_jst": inside[0]["jst_time"], "end_jst": inside[-1]["jst_time"],
                    "start_t_sec": inside[0]["t_sec"], "end_t_sec": inside[-1]["t_sec"],
                    "mean_people": round(mean, 2), "n_samples": len(inside)}
    return best


def _plot(rows: list[dict], best: dict | None, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["t_sec"] / 60.0 for r in rows]
    ys = [r["n_people"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=130)
    ax.plot(xs, ys, lw=1.4, color="#1f77b4")
    ax.fill_between(xs, ys, alpha=0.15, color="#1f77b4")

    if best:
        ax.axvspan(best["start_t_sec"] / 60.0, best["end_t_sec"] / 60.0,
                   color="#ff7f0e", alpha=0.18,
                   label=f"выбранный час: {best['start_jst'][11:16]}–"
                         f"{best['end_jst'][11:16]} JST, среднее {best['mean_people']:.1f}")
        ax.legend(loc="upper left", fontsize=8)

    # Подписи по оси X — время JST, а не минуты от начала: читателю отчёта
    # важно время суток, а не смещение в файле.
    step = max(1, len(rows) // 12)
    ax.set_xticks([xs[i] for i in range(0, len(rows), step)])
    ax.set_xticklabels([rows[i]["jst_time"][11:16] for i in range(0, len(rows), step)],
                       fontsize=8)
    ax.set_xlabel("время JST")
    ax.set_ylabel("людей в кадре")
    ax.set_title(f"Присутствие в кадре, замер раз в {args.step_sec:.0f} с "
                 f"(yolo11m, imgsz={IMGSZ}, conf={CONF})", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
