"""Баланс белого по опорной поверхности и классификация цвета верха.

ЗАЧЕМ. У камеры глобальный цветовой сдвиг: медианный тон ВСЕГО кадра равен
110, и первый прогон S7 назвал синими 21 трек из 25 — все с тоном 107..124.
Классифицировался сдвиг камеры, а не одежда. Поднимать порог ахроматичности,
чтобы это спрятать, — маскировка. Сдвиг надо КОМПЕНСИРОВАТЬ.

ЧТО ДЕЛАЕМ. Мостовая внутри ROI серая по смыслу. Берём её медианный BGR за
несколько кадров как серую точку и приводим её к нейтральной.

ПОЧЕМУ НОРМИРОВКА НА СРЕДНЕЕ ТРЁХ КАНАЛОВ. Пороги v_black_max и v_white_min
заданы в АБСОЛЮТНЫХ единицах V. Нормировка на один канал (обычная
green-preserving) умножает картинку на m_G/mean и молча смещает смысл этих
порогов. Нормировка на максимум делает все коэффициенты >= 1, всё светлеет и
белых рубашек становится больше из ниоткуда. Только среднее трёх каналов
оставляет яркость опорной поверхности на месте.

ЧЕГО ЭТО НЕ ДОКАЗЫВАЕТ. Падение насыщенности фона к нулю после коррекции —
это проверка, что коррекция ПРИМЕНИЛАСЬ, а не что она ВЕРНА: коэффициенты
из этого же фона и считались. Единственная некольцевая проверка — ручная
разметка.

ЧТО НЕ ИЗМЕРЕНО. Коррекция делается в гамма-кодированном sRGB, а физически
корректный баланс белого — линейная операция. Размер этой погрешности не
измерен. И само допущение «мостовая серая» не проверено: если плитка тёплая,
grey-world по ней перекорректирует в синеву.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

#: Тон в OpenCV лежит в 0..179, а не 0..255.
HUE_MAX = 180


def road_mask(shape, roi_px, boxes_xyxy, dilate_px: int) -> np.ndarray:
    """Маска мостовой: полигон ROI минус люди.

    Полигон заливается через cv2.fillPoly, а не поточечным тестом: точечный
    тест на 2 Мпикс это ~2 млн вызовов питона на кадр.

    Рамки расширяются на dilate_px: они плотно облегают человека, а тень и
    край пальто выходят за них, и без запаса одежда попадёт в «серую» опору.
    """
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.asarray(roi_px, dtype=np.int32).reshape(-1, 1, 2)], 255)
    for x1, y1, x2, y2 in np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4):
        cv2.rectangle(m,
                      (int(x1) - dilate_px, int(y1) - dilate_px),
                      (int(x2) + dilate_px, int(y2) + dilate_px), 0, -1)
    return m.astype(bool)


def collect_road_pixels(frame_bgr, mask, cfg: dict) -> tuple[np.ndarray, dict]:
    """Пиксели мостовой, годные как опора серого, и статистика отсева."""
    px = frame_bgr[mask]
    n0 = len(px)
    if n0 == 0:
        return px, {"in_roi_minus_people": 0, "clipped": 0, "too_saturated": 0}
    lo = int(cfg.get("clip_low", 5))
    hi = int(cfg.get("clip_high", 250))
    # Выбитые в белое пиксели уже потеряли цветность и тянут оценку к
    # нейтрали, то есть ЗАНИЖАЮТ видимый сдвиг. Задавленные в чёрное — шум.
    keep = (px.max(axis=1) < hi) & (px.min(axis=1) > lo)
    n_clip = int((~keep).sum())
    px = px[keep]
    n_sat = 0
    if len(px):
        s = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV
                         ).reshape(-1, 3)[:, 1]
        # Разметка, неон и краска машин — не серое. Отсев ЧАСТИЧНО кольцевой:
        # исключаем пиксели за цветность той самой мерой, которую измеряем.
        # Порог берётся заведомо выше измеренного сдвига, а доля отсева
        # выводится наружу, чтобы кольцевость была видна, а не спрятана.
        ok = s <= float(cfg.get("s_max_for_grey", 160))
        n_sat = int((~ok).sum())
        px = px[ok]
    return px, {"in_roi_minus_people": n0, "clipped": n_clip, "too_saturated": n_sat}


def wb_gains_from_grey(px_bgr: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """Коэффициенты каналов из серой точки. Падает громко, если опора плохая."""
    n_min = int(cfg.get("min_pixels", 20000))
    if len(px_bgr) < n_min:
        raise ValueError(
            f"пикселей мостовой {len(px_bgr)} < {n_min}: серая точка оценивалась "
            f"бы по обрезку, и коэффициенты были бы случайными")
    med = np.median(px_bgr.astype(np.float64), axis=0)
    mean = px_bgr.astype(np.float64).mean(axis=0)
    lvl = float(cfg.get("min_channel_level", 8))
    if float(med.min()) < lvl:
        raise ValueError(f"канал серой точки {med} ниже {lvl}: коэффициент взорвётся")
    gains = med.mean() / med
    gmin, gmax = float(cfg.get("gain_min", 0.5)), float(cfg.get("gain_max", 2.0))
    if gains.min() < gmin or gains.max() > gmax:
        raise ValueError(
            f"коэффициенты {gains.round(3)} вне [{gmin}, {gmax}]: это не мостовая")
    return gains, {
        "grey_point_bgr": [round(float(v), 2) for v in med],
        "grey_point_mean_bgr": [round(float(v), 2) for v in mean],
        # Расхождение медианы и среднего — бесплатный индикатор грязной маски.
        "median_mean_gap": round(float(np.abs(med - mean).max()), 2),
        "grey_point_v": round(float(med.max()), 2),
        "v_after_wb": round(float(med.mean()), 2),
        "norm": "mean_of_three_channel_means",
        "space": "srgb_gamma_encoded",
        "n_pixels": int(len(px_bgr)),
    }


def apply_wb(img_bgr: np.ndarray, gains: np.ndarray) -> tuple[np.ndarray, float]:
    """Умножение на коэффициенты с обрезкой. Возвращает долю упёршихся пикселей.

    Клиппинг важен не яркостью, а ТОНОМ: если упирается один канал, отношение
    B:G:R меняется и тон поворачивается, а тон — весь вход классификатора.
    """
    f = img_bgr.astype(np.float32) * gains.astype(np.float32)[None, None, :]
    clipped = float(np.mean((f > 255.0) | (f < 0.0)))
    return np.clip(f, 0, 255).astype(np.uint8), clipped


def hue_hist(img_bgr: np.ndarray, mask=None, bins: int = 36) -> list[int]:
    """Гистограмма тона. Идёт в отчёт до и после коррекции."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][mask] if mask is not None else hsv[..., 0].reshape(-1)
    return np.histogram(h, bins=bins, range=(0, HUE_MAX))[0].astype(int).tolist()


def sat_median(img_bgr: np.ndarray, mask=None) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1][mask] if mask is not None else hsv[..., 1].reshape(-1)
    return float(np.median(s)) if len(s) else float("nan")
