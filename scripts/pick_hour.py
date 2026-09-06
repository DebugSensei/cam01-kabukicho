"""Выбор самого людного часа из записей raw/live_*.ts.

Кликов не требует. По каждому файлу считается присутствие с шагом step_sec,
затем ищется самое людное НЕПРЕРЫВНОЕ окно длиной window_min и файлы,
покрывающие это окно, склеиваются в raw/peak_hour.ts.

График идёт в отчёт как обоснование выбора часа, поэтому прогон обязан быть
воспроизводим: время берётся из ИМЁН ФАЙЛОВ (live_YYYYMMDD_HHMMJST.ts),
а не из системных часов.

Склейка побайтовая. MPEG-TS — потоковый контейнер без глобального заголовка,
поэтому конкатенация файлов даёт корректный поток; перекодировать нечего,
и качество не теряется.

    python scripts/pick_hour.py
    make pick-hour
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.anonymise import save_figure  # noqa: E402
from looq.io import library_versions, load_config, sha256_file, utc_now_iso, write_json  # noqa: E402
from looq.pilot import infer_params  # noqa: E402

NAME_RE = re.compile(r"live_(\d{8})_(\d{4})JST\.ts$")
OUT_TS = Path("raw/peak_hour.ts")
OUT_CSV = Path("out/hour_choice.csv")
OUT_PNG = Path("out/hour_choice.png")
OUT_META = Path("out/hour_choice.json")


def parse_start(path: Path) -> dt.datetime:
    m = NAME_RE.search(path.name)
    if not m:
        raise SystemExit(f"имя {path.name} не разбирается: ждём live_YYYYMMDD_HHMMJST.ts")
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")


def sample_file(path: Path, model, params: dict, step_sec: float) -> list[tuple[float, int]]:
    """(смещение в секундах от начала файла, число людей) с шагом step_sec.

    Перемотка по номеру кадра: последовательное чтение 4.5 часов заняло бы
    часы. У .ts перемотка неточна на границах сегментов, поэтому шаг берётся
    крупный (минуты), и промах в пару кадров ничего не меняет.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cv2 не открыл {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not (1.0 < fps < 240.0) or n < 2:
        cap.release()
        raise SystemExit(f"неправдоподобные параметры {path}: fps={fps}, кадров={n}")
    out = []
    t = 0.0
    while t * fps < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=params["imgsz"], classes=params["classes"],
                          conf=params["conf"], iou=params["iou"],
                          device=params["device"], half=params["half"], verbose=False)
        out.append((t, int(len(r[0].boxes)) if r[0].boxes is not None else 0))
        t += step_sec
    cap.release()
    return out


def best_window(samples: list[dict], window_min: float) -> dict:
    """Самое людное непрерывное окно. Окно обязано целиком помещаться в записи."""
    window_s = window_min * 60.0
    if not samples:
        raise SystemExit("нет ни одного замера")
    span = samples[-1]["t_abs_s"] - samples[0]["t_abs_s"]
    if span < window_s:
        raise SystemExit(f"записи покрывают {span / 60:.0f} мин, окно {window_min:.0f} мин "
                         f"в них не помещается")
    best = None
    for i, s0 in enumerate(samples):
        end = s0["t_abs_s"] + window_s
        if end > samples[-1]["t_abs_s"]:
            break
        inside = [s for s in samples[i:] if s["t_abs_s"] <= end]
        mean = float(np.mean([s["n_people"] for s in inside]))
        if best is None or mean > best["mean_people"]:
            best = {"start_s": s0["t_abs_s"], "end_s": end,
                    "start_jst": s0["jst"], "end_jst": inside[-1]["jst"],
                    "mean_people": round(mean, 2), "n_samples": len(inside)}
    if best is None:
        raise SystemExit("окно не найдено")
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="raw/live_*.ts")
    ap.add_argument("--step-sec", type=float, default=60.0)
    ap.add_argument("--window-min", type=float, default=60.0)
    ap.add_argument("--config", default="configs/s3_detect.yaml")
    ap.add_argument("--no-concat", action="store_true", help="только график, без склейки")
    args = ap.parse_args(argv)

    files = sorted(Path().glob(args.pattern), key=lambda p: parse_start(p))
    if not files:
        raise SystemExit(f"нет файлов по шаблону {args.pattern}")
    params = infer_params(load_config(args.config))
    if not Path(str(params["weights"])).is_file():
        raise SystemExit(f"нет весов {params['weights']}")

    from ultralytics import YOLO
    model = YOLO(params["weights"])

    t0 = parse_start(files[0])
    samples: list[dict] = []
    print(f"файлов {len(files)}, шаг {args.step_sec:.0f} с, начало {t0:%Y-%m-%d %H:%M} JST")
    for f in files:
        start = parse_start(f)
        got = sample_file(f, model, params, args.step_sec)
        for off, n in got:
            when = start + dt.timedelta(seconds=off)
            samples.append({
                "file": f.name,
                "t_abs_s": (when - t0).total_seconds(),
                "jst": when.strftime("%Y-%m-%d %H:%M:%S"),
                "n_people": n,
            })
        counts = [n for _, n in got]
        print(f"  {f.name}  замеров {len(got):3d}  людей медиана "
              f"{int(np.median(counts)) if counts else 0:3d}  max {max(counts) if counts else 0:3d}")

    samples.sort(key=lambda s: s["t_abs_s"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "t_abs_s", "jst", "n_people"])
        w.writeheader()
        w.writerows(samples)

    best = best_window(samples, args.window_min)
    used = sorted({s["file"] for s in samples
                   if best["start_s"] <= s["t_abs_s"] <= best["end_s"]},
                  key=lambda name: parse_start(Path("raw") / name))
    print()
    print(f"самое людное окно {args.window_min:.0f} мин: {best['start_jst']} — "
          f"{best['end_jst']} JST, среднее {best['mean_people']:.1f} человек")
    print("вошли файлы:")
    for name in used:
        print(f"  {name}")

    _plot(samples, best, args)

    if not args.no_concat:
        # MPEG-TS без глобального заголовка: побайтовая склейка даёт корректный
        # поток, перекодировать нечего.
        OUT_TS.parent.mkdir(parents=True, exist_ok=True)
        with OUT_TS.open("wb") as out:
            for name in used:
                with (Path("raw") / name).open("rb") as src:
                    while chunk := src.read(1 << 22):
                        out.write(chunk)
        print(f"склеено: {OUT_TS} ({OUT_TS.stat().st_size / 1e9:.2f} ГБ)")

    write_json(OUT_META, {
        "generated_utc": utc_now_iso(),
        "pattern": args.pattern, "step_sec": args.step_sec,
        "window_min": args.window_min,
        "n_files_scanned": len(files), "n_samples": len(samples),
        "best_window": best, "files_used": used,
        "output_ts": None if args.no_concat else str(OUT_TS),
        "output_sha256": None if args.no_concat else sha256_file(OUT_TS),
        "model": {"weights": params["weights"],
                  "weights_sha256": sha256_file(params["weights"]),
                  "imgsz": params["imgsz"], "conf": params["conf"]},
        "libraries": library_versions(),
    })
    print(f"{OUT_CSV}, {OUT_PNG}, {OUT_META}")
    return 0


def _plot(samples: list[dict], best: dict, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [s["t_abs_s"] / 60.0 for s in samples]
    ys = [s["n_people"] for s in samples]
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=130)
    ax.plot(xs, ys, lw=1.2, color="#1f77b4")
    ax.fill_between(xs, ys, alpha=0.15, color="#1f77b4")
    ax.axvspan(best["start_s"] / 60.0, best["end_s"] / 60.0, color="#ff7f0e", alpha=0.2,
               label=f"выбранный час: {best['start_jst'][11:16]}–{best['end_jst'][11:16]} JST, "
                     f"среднее {best['mean_people']:.1f}")
    step = max(1, len(samples) // 14)
    ax.set_xticks([xs[i] for i in range(0, len(samples), step)])
    ax.set_xticklabels([samples[i]["jst"][11:16] for i in range(0, len(samples), step)],
                       fontsize=8, rotation=45)
    ax.set_xlabel("время JST")
    ax.set_ylabel("людей в кадре")
    ax.set_title(f"Присутствие по записям, замер раз в {args.step_sec:.0f} с", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUT_PNG)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
