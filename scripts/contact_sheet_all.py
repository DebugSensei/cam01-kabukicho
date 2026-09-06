"""Один лист со ВСЕМИ изображениями проекта — для проверки глазами.

ЗАЧЕМ. Метрики говорят, что уверенных лицевых кейпоинтов не осталось. Но
публикуются картинки, а не метрики, и последнее слово за человеком. Скрипт
собирает всё, что может уехать наружу, в один файл:

  A. фигуры docs/img — то, что лежит в репозитории;
  B. кропы, ВШИТЫЕ в out/*.html — то, что увидит открывший страницу;
  C. кропы на диске в evidence/ — источник, из которого вшивается B.

Группа C важна отдельно: она хранит пороги записи, а B показывает пороги
применённые, и это разные множества.

    python scripts/contact_sheet_all.py
    python scripts/contact_sheet_all.py --out out/check_all.jpg
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from looq.anonymise import save_image  # noqa: E402

#: Размер ячейки и число колонок. Дефолт — крупно и в 12 колонок: лист для
#: проверки глазами, а не для украшения. --compact ужимает его до размера,
#: который проходит по каналу и открывается на телефоне.
CELL_W, CELL_H = 150, 250
COLS = 12
LABEL_H = 20
HEAD_H = 44
BG = 22


def _cell(img: np.ndarray, label: str) -> np.ndarray:
    out = np.full((CELL_H + LABEL_H, CELL_W, 3), BG, np.uint8)
    if img is not None and img.size:
        s = min(CELL_W / img.shape[1], CELL_H / img.shape[0])
        r = cv2.resize(img, (max(1, int(img.shape[1] * s)),
                             max(1, int(img.shape[0] * s))),
                       interpolation=cv2.INTER_CUBIC)
        y0, x0 = (CELL_H - r.shape[0]) // 2, (CELL_W - r.shape[1]) // 2
        out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    if LABEL_H:
        cv2.putText(out, label[:22], (3, CELL_H + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (185, 190, 200), 1, cv2.LINE_AA)
    return out


def _band(title: str, width: int) -> np.ndarray:
    b = np.full((HEAD_H, width, 3), 38, np.uint8)
    cv2.putText(b, title, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (245, 245, 245), 2, cv2.LINE_AA)
    return b


def _grid(cells: list[np.ndarray], width: int) -> np.ndarray:
    if not cells:
        return np.full((1, width, 3), BG, np.uint8)
    rows = []
    for i in range(0, len(cells), COLS):
        chunk = cells[i:i + COLS]
        while len(chunk) < COLS:
            chunk.append(np.full((CELL_H + LABEL_H, CELL_W, 3), BG, np.uint8))
        rows.append(np.hstack(chunk))
    g = np.vstack(rows)
    if g.shape[1] != width:
        pad = np.full((g.shape[0], width - g.shape[1], 3), BG, np.uint8)
        g = np.hstack([g, pad])
    return g


def figures() -> list[np.ndarray]:
    out = []
    for p in sorted(Path("docs/img").glob("*")):
        img = cv2.imread(str(p))
        if img is not None:
            out.append(_cell(img, p.name.replace(".webp", "")))
    return out


def embedded_in_html() -> list[np.ndarray]:
    """Кропы, вшитые в страницы. Это то, что реально увидит читатель."""
    out = []
    for page in sorted(Path("out").glob("*.html")):
        html = page.read_text(encoding="utf-8", errors="replace")
        for k, m in enumerate(re.finditer(r'data:image/(?:jpeg|png);base64,([A-Za-z0-9+/=]+)', html)):
            try:
                buf = np.frombuffer(base64.b64decode(m.group(1)), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                continue
            if img is None or img.shape[0] < 24:
                continue
            # крупные картинки — это фигуры страницы, они уже в группе A
            if img.shape[1] > 500:
                continue
            out.append(_cell(img, f"{page.stem[:9]}#{k}"))
    return out


def on_disk_crops() -> list[np.ndarray]:
    out = []
    for p in sorted(Path("evidence").rglob("*.jpg")):
        img = cv2.imread(str(p))
        if img is not None:
            out.append(_cell(img, p.parent.name[:12]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("out/check_all_images.jpg"))
    ap.add_argument("--compact", action="store_true",
                    help="мельче и в больше колонок: лист влезает в канал")
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args(argv)

    if args.compact:
        global CELL_W, CELL_H, COLS, LABEL_H
        CELL_W, CELL_H, COLS, LABEL_H = 84, 140, 26, 0

    groups = [
        ("A. docs/img — figures committed to the repository", figures()),
        ("B. crops embedded in out/*.html — what a reader of the page sees", embedded_in_html()),
        ("C. crops on disk in evidence/ — the source B is built from", on_disk_crops()),
    ]

    width = COLS * CELL_W
    parts = []
    for title, cells in groups:
        parts.append(_band(f"{title}   [{len(cells)}]", width))
        parts.append(_grid(cells, width))
        print(f"  {title}: {len(cells)}")

    sheet = np.vstack(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(args.out, sheet, quality=args.quality)
    print(f"\nлист: {args.out}  {sheet.shape[1]}x{sheet.shape[0]}, "
          f"{args.out.stat().st_size / 1e6:.1f} МБ")
    print("Смотрите глазами. Если узнаваемо хоть одно лицо — публиковать нельзя.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
