"""Контактный лист обезличенных кропов — контроль глазами.

Правило 9 проверяется тестами на синтетике, но синтетика не доказательство:
параметры обезличивания подобраны на шуме, а не на реальных лицах в кадре
Kabukicho ночью. Этот скрипт собирает первые N кропов из evidence/ в один jpg,
чтобы владелец посмотрел и подтвердил или потребовал усилить.

Пока подтверждения нет — в docs/DECISIONS.md висит блокер.

    python scripts/check_blur.py                 # первые 20 кропов
    python scripts/check_blur.py --n 40 --claim claim.detect.far_half

Кропы на листе УЖЕ обезличены: скрипт читает то, что лежит на диске, и ничего
не размывает сам. Если на листе видно лицо — значит обезличивание слабое,
и это ровно тот вывод, ради которого лист собирается.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.io import atomic_write_bytes  # noqa: E402

INDEX = Path("evidence/index.parquet")
DEFAULT_OUT = Path("evidence/blur_check.jpg")

#: Размер ячейки на листе. Кропы мельче растягиваются: обезличивание надо
#: смотреть в увеличении, иначе «неразличимо» получается за счёт масштаба,
#: а не за счёт обработки.
CELL_W, CELL_H = 180, 320
COLS = 5
LABEL_H = 22


def _aspect(path: str) -> float:
    """Отношение высота/ширина опубликованного кропа."""
    im = cv2.imread(str(path))
    if im is None or im.shape[1] == 0:
        return float("inf")
    return im.shape[0] / im.shape[1]


def _load_index(claim: str | None, n: int, risky: bool = False):
    if not INDEX.is_file():
        raise SystemExit(
            f"нет {INDEX}. Сначала должен отработать этап, собирающий пруфы "
            f"(S3 и далее). Правило 8: пустой лист не рисуем."
        )
    import pandas as pd

    df = pd.read_parquet(INDEX)
    if claim:
        df = df[df["claim_id"] == claim]
        if df.empty:
            raise SystemExit(f"в индексе нет строк с claim_id={claim!r}")
    if risky:
        # Чем НИЖЕ отношение высота/ширина, тем меньше видно тела и тем большую
        # долю кропа занимает лицо. У ростовой рамки отношение около 2.8 и
        # голова укладывается в верхние ~13%; у обрезанной оно падает к 1.2,
        # и фиксированная доля может лицо не накрыть. Сортируем по возрастанию:
        # первыми идут самые опасные.
        df = df.assign(_ar=df["path"].map(_aspect)).sort_values("_ar").head(n)
        df = df.drop(columns=["_ar"])
    else:
        df = df.sort_values(["claim_id", "frame_idx"]).head(n)
    if df.empty:
        raise SystemExit("индекс пуст — проверять нечего")
    return df


def _cell(row) -> np.ndarray:
    """Одна ячейка листа: кроп в рамке плюс подпись."""
    cell = np.full((CELL_H + LABEL_H, CELL_W, 3), 32, np.uint8)
    img = cv2.imread(str(row["path"]))
    if img is None:
        cv2.putText(cell, "MISSING", (8, CELL_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return cell

    h, w = img.shape[:2]
    scale = min(CELL_W / w, CELL_H / h)
    # INTER_NEAREST: увеличиваем без сглаживания, иначе интерполяция сама
    # «дорисует» плавность и обезличивание покажется лучше, чем оно есть.
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_NEAREST)
    rh, rw = resized.shape[:2]
    y0, x0 = (CELL_H - rh) // 2, (CELL_W - rw) // 2
    cell[y0:y0 + rh, x0:x0 + rw] = resized

    # Граница области, которая должна была быть обезличена.
    try:
        top_frac = json.loads(row["extra_json"]).get("blur_top_frac")
    except (json.JSONDecodeError, TypeError):
        top_frac = None
    if top_frac:
        y_line = y0 + int(round(top_frac * rh))
        cv2.line(cell, (x0, y_line), (x0 + rw, y_line), (0, 200, 255), 1)

    label = f"t{int(row['track_id'])} f{int(row['frame_idx'])} c{row['confidence']:.2f}"
    cv2.putText(cell, label, (4, CELL_H + 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return cell


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="сколько кропов взять")
    ap.add_argument("--claim", default=None, help="ограничить одним claim_id")
    ap.add_argument("--risky", action="store_true",
                    help="самые обрезанные рамки вперёд: там лицо занимает "
                         "бо́льшую долю кропа и полоса скорее его не накроет")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    df = _load_index(args.claim, args.n, risky=args.risky)
    cells = [_cell(r) for _, r in df.iterrows()]

    rows = []
    for i in range(0, len(cells), COLS):
        chunk = cells[i:i + COLS]
        while len(chunk) < COLS:
            chunk.append(np.full_like(cells[0], 32))
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)

    ok, buf = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise SystemExit("не удалось закодировать контактный лист")
    atomic_write_bytes(args.out, buf.tobytes())

    print(f"кропов на листе: {len(df)} из запрошенных {args.n}")
    print(f"claim_id: {sorted(df['claim_id'].unique())}")
    print(f"жёлтая линия — нижняя граница обезличенной области")
    print(f"лист: {args.out}")
    print()
    print("Посмотрите глазами. Если лицо узнаваемо хотя бы на одном кропе —")
    print("поднимайте face_blur_top_frac или pixelate_factor в configs/evidence.yaml")
    print("и снимайте блокер в docs/DECISIONS.md только после повторной проверки.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
