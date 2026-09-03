"""Общие детали интерактивных сборщиков кликов.

Лупа и чтение опорного кадра нужны и обводке зон, и калибровке по прямоугольнику
мостовой. Две копии одного кода разъедутся, и лупа в одном месте начнёт врать,
а в другом нет — поэтому они здесь.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MAG_SIZE = 200          # сторона врезки на экране
MAG_ZOOM = 4            # во сколько раз увеличиваем
MAG_SRC = MAG_SIZE // MAG_ZOOM   # сколько пикселей кадра попадает во врезку


def grab_frame(clip: Path, frame_idx: int) -> np.ndarray:
    """Кадр по номеру. Читаем последовательно: у .ts перемотка врёт на границах
    сегментов, и кадр оказался бы не тем, по которому потом считают."""
    clip = Path(clip)
    if not clip.is_file():
        raise SystemExit(f"нет клипа: {clip}")
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"cv2 не открыл {clip}")
    frame = None
    for i in range(frame_idx + 1):
        ok, f = cap.read()
        if not ok:
            break
        if i == frame_idx:
            frame = f
    cap.release()
    if frame is None:
        raise SystemExit(f"в {clip} нет кадра {frame_idx}")
    return frame


def draw_magnifier(img: np.ndarray, base: np.ndarray, cursor, top_margin: int = 70) -> None:
    """Врезка с увеличением вокруг курсора, в дальнем от курсора углу.

    Увеличение через INTER_NEAREST: сглаживание нарисовало бы промежуточные
    значения, которых в кадре нет, и попадание в пиксель стало бы иллюзией.

    Врезка заодно проверяет координаты: если перекрестие стоит не там, куда
    целится пользователь, значит окно и кадр разъехались по масштабу.
    """
    if cursor is None:
        return
    cx, cy = int(cursor[0]), int(cursor[1])
    h, w = base.shape[:2]
    half = MAG_SRC // 2
    x0 = int(np.clip(cx - half, 0, max(0, w - MAG_SRC)))
    y0 = int(np.clip(cy - half, 0, max(0, h - MAG_SRC)))
    patch = base[y0:y0 + MAG_SRC, x0:x0 + MAG_SRC]
    if patch.shape[0] != MAG_SRC or patch.shape[1] != MAG_SRC:
        return
    mag = cv2.resize(patch, (MAG_SIZE, MAG_SIZE), interpolation=cv2.INTER_NEAREST)

    px = int((cx - x0) * MAG_ZOOM)
    py = int((cy - y0) * MAG_ZOOM)
    cv2.line(mag, (px, 0), (px, MAG_SIZE), (0, 0, 255), 1)
    cv2.line(mag, (0, py), (MAG_SIZE, py), (0, 0, 255), 1)
    cv2.rectangle(mag, (0, 0), (MAG_SIZE - 1, MAG_SIZE - 1), (255, 255, 255), 1)

    margin = 12
    mx = w - MAG_SIZE - margin if cx < w / 2 else margin
    my = h - MAG_SIZE - margin if cy < h / 2 else top_margin
    img[my:my + MAG_SIZE, mx:mx + MAG_SIZE] = mag
    cv2.putText(img, f"x{MAG_ZOOM}  {cx},{cy}", (mx, my - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def header(img: np.ndarray, line1: str, line2: str) -> None:
    """Две строки подсказки поверх кадра. Только ASCII: cv2 кириллицу не рисует."""
    cv2.rectangle(img, (0, 0), (img.shape[1], 64), (20, 20, 20), -1)
    cv2.putText(img, line1, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, line2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)
