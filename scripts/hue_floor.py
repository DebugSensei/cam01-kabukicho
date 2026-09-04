"""Замер: с какой насыщенности тон перестаёт быть тоном сцены.

ЗАЧЕМ. Первый прогон S7 назвал 21 трек из 25 синими, и у всех тон лежал
в 107..124. Медианный тон ВСЕГО кадра — 110. Совпадение означает, что
классифицировался цветовой сдвиг камеры, а не одежда.

ЧТО СЧИТАЕТ. Пиксели нескольких кадров бьются по насыщенности на бины,
в каждом бине берётся медианный тон. Пока медиана держится у тона сцены,
хрома в этом бине — шум. Бин, где медиана отрывается, и задаёт пол
насыщенности s_achromatic_max в configs/s7_attrs.yaml.

Это НЕ подкрутка порога под желаемый результат: значение выводится из записи
и меняется вместе с ней. На другой записи скрипт даст другое число.

    python scripts/hue_floor.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.pilot import iter_frames  # noqa: E402

BINS = [(0, 30), (30, 45), (45, 60), (60, 80), (80, 100),
        (100, 130), (130, 180), (180, 256)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="raw/live_20260904_0911JST.ts")
    ap.add_argument("--frames", type=int, nargs="+",
                    default=[100, 5000, 20000, 40000])
    ap.add_argument("--tol", type=float, default=15.0,
                    help="на сколько градусов тон должен уйти от тона сцены, "
                         "чтобы бин считался несущим реальную хрому")
    args = ap.parse_args(argv)

    acc = []
    for _fi, f in iter_frames(Path(args.video),
                              np.asarray(sorted(args.frames), dtype=np.int64)):
        acc.append(cv2.cvtColor(f, cv2.COLOR_BGR2HSV).reshape(-1, 3))
    if not acc:
        print("ОШИБКА: ни один кадр не прочитан", file=sys.stderr)
        return 1
    hsv = np.concatenate(acc).astype(np.float64)

    scene_hue = float(np.median(hsv[:, 0]))
    print(f"кадров {len(acc)}, пикселей {len(hsv):,}")
    print(f"медианный тон сцены: {scene_hue:.0f}")
    print(f"s-перцентили: {np.percentile(hsv[:, 1], [50, 75, 90, 95, 99]).round(0)}")
    print("\n  s-бин  | доля пикселей | медиана тона | отрыв от сцены")

    floor = None
    for lo, hi in BINS:
        m = (hsv[:, 1] >= lo) & (hsv[:, 1] < hi)
        if m.sum() < 1000:
            continue
        med = float(np.median(hsv[m, 0]))
        # Тон цикличен: 0 и 179 — соседи.
        d = abs(med - scene_hue)
        d = min(d, 180.0 - d)
        broke = d >= args.tol
        if broke and floor is None:
            floor = lo
        print(f"{lo:>4}-{hi:<4} |   {m.mean():9.3f}   |    {med:6.0f}    | "
              f"{d:5.0f}  {'ОТРЫВ' if broke else '-'}")

    if floor is None:
        print(f"\nтон НИ В ОДНОМ бине не отрывается от {scene_hue:.0f} более чем "
              f"на {args.tol:.0f}: хрома в этой записи не измеряется вообще")
        return 1
    print(f"\ns_achromatic_max = {floor}: ниже этой насыщенности тон повторяет "
          f"тон сцены и об одежде ничего не говорит")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
