"""Калибровка по пешеходам из уже посчитанных детекций.

Кликов не требует. Источник геометрии — люди в кадре, а точнее два факта:

  1. Все стопы лежат в плоскости земли, все макушки — в параллельной ей
     плоскости. Значит для пары людей прямая через их стопы и прямая через их
     макушки пересекаются НА ЛИНИИ ГОРИЗОНТА. Точность зависит от того,
     насколько одинаков рост: разброс роста входит как шум, и его гасит
     RANSAC по многим парам.
  2. Отрезок «стопа-макушка» вертикален в мире. Значит отрезки разных людей
     пересекаются в ВЕРТИКАЛЬНОЙ ТОЧКЕ СХОДА.

Почему это лучше прежней схемы. Раньше вертикальная VP бралась из одной
кликнутой линии, а одна прямая точку схода не задаёт: на реальных кликах
V1 оказалась почти идеальной вертикалью в кадре, фокус вышел 541 px, и рост
разлетелся на 21 метр. Пешеходов в кадре тысячи, и каждый даёт свой отрезок.

Масштаб берётся НЕ из роста, а из ширины улицы (замер по спутнику). Тогда
рост становится НЕЗАВИСИМОЙ проверкой: медиана обязана лечь в 1.55-1.75 м,
и проверять её есть чем. В прежней схеме рост задавал масштаб, и проверка
роста была тавтологией.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from looq.calib import CalibError, apply_h, line_through


@dataclass
class HorizonResult:
    line: np.ndarray            # (3,), нормирована a^2 + b^2 = 1
    n_pairs: int
    n_inliers: int
    inlier_frac: float
    residual_px: float          # медианное расстояние инлаеров до прямой


@dataclass
class VerticalVPResult:
    vp_px: np.ndarray
    n_pairs: int
    n_inliers: int
    inlier_frac: float
    residual_px: float          # для вертикали это ГРАДУСЫ, не пиксели


def _fit_line_total_least_squares(pts: np.ndarray) -> np.ndarray:
    """Прямая по облаку точек, нормированная a^2 + b^2 = 1."""
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return np.array([normal[0], normal[1], -float(normal @ centroid)])


def horizon_from_pedestrian_pairs(
    feet_px: np.ndarray,
    heads_px: np.ndarray,
    frame_idx: np.ndarray,
    *,
    min_depth_sep_px: float = 80.0,  # ближе — прямые почти совпадают, точка улетает
    max_pairs: int = 20000,
    ransac_tol_px: float = 25.0,
    ransac_iters: int = 400,
    min_inlier_frac: float = 0.20,
    seed: int = 20260903,
) -> HorizonResult:
    """Линия горизонта по парам людей в ОДНОМ кадре.

    Пары берутся разнесённые по глубине: при близких стопах прямая через них
    почти совпадает с прямой через макушки, и точка пересечения улетает
    в бесконечность от шума. min_depth_sep_px отсекает такие пары.

    Про min_inlier_frac. Доля инлаеров здесь плохой показатель качества: она
    зависит от допуска и от разброса роста, а не от точности результата.
    Замер на синтетике (120 кадров по 6 человек, допуск 25 px):

        разброс роста 0.00 м -> инлаеров 100 %, фокус точный
        разброс роста 0.07 м -> инлаеров  37 %, ошибка фокуса -0.4 %
        разброс роста 0.12 м -> инлаеров  23 %, метод падает

    То есть при реалистичном разбросе (0.07 м) фокус выходит с точностью 0.4 %
    уже на 37 % инлаеров. Настоящие проверки идут ниже по трубе: точка схода
    улицы обязана лечь на найденный горизонт, а медиана роста — в 1.55-1.75 м.
    """
    rng = np.random.default_rng(seed)
    pts: list[np.ndarray] = []

    order = np.argsort(frame_idx, kind="stable")
    feet_px, heads_px, frame_idx = feet_px[order], heads_px[order], frame_idx[order]
    bounds = np.flatnonzero(np.diff(frame_idx)) + 1
    for group in np.split(np.arange(len(frame_idx)), bounds):
        if len(group) < 2:
            continue
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                i, j = group[a], group[b]
                if abs(feet_px[i][1] - feet_px[j][1]) < min_depth_sep_px:
                    continue
                try:
                    foot_line = line_through(feet_px[i], feet_px[j])
                    head_line = line_through(heads_px[i], heads_px[j])
                except CalibError:
                    continue
                cross = np.cross(foot_line, head_line)
                if abs(cross[2]) < 1e-9:
                    continue     # прямые параллельны: пара вырождена
                pts.append(np.array([cross[0] / cross[2], cross[1] / cross[2]]))
                if len(pts) >= max_pairs:
                    break
            if len(pts) >= max_pairs:
                break
        if len(pts) >= max_pairs:
            break

    if len(pts) < 20:
        raise CalibError(
            f"пар людей для горизонта набралось {len(pts)} при минимуме 20: "
            f"мало детекций либо все на одной глубине")
    p = np.asarray(pts)
    # Однородные координаты один раз: расстояние до нормированной прямой это
    # просто |l . p_h|, и считать его надо тысячи раз.
    ph = np.concatenate([p, np.ones((len(p), 1))], axis=1)

    best_line, best_inliers = None, -1
    for _ in range(ransac_iters):
        idx = rng.choice(len(p), size=2, replace=False)
        try:
            cand = line_through(p[idx[0]], p[idx[1]])
        except CalibError:
            continue
        d = np.abs(ph @ cand)
        n = int((d < ransac_tol_px).sum())
        if n > best_inliers:
            best_line, best_inliers = cand, n
    if best_line is None:
        raise CalibError("RANSAC не нашёл ни одной прямой-кандидата для горизонта")

    mask = np.abs(ph @ best_line) < ransac_tol_px
    if int(mask.sum()) >= 2:
        best_line = _fit_line_total_least_squares(p[mask])
        mask = np.abs(ph @ best_line) < ransac_tol_px
    inliers = p[mask]
    frac = len(inliers) / len(p)
    if frac < min_inlier_frac:
        raise CalibError(
            f"горизонт подтверждён лишь {frac:.1%} пар при минимуме "
            f"{min_inlier_frac:.0%}: точки пересечения не ложатся на прямую, "
            f"значит допущение о примерно равном росте не выполняется")
    return HorizonResult(
        line=best_line, n_pairs=len(p), n_inliers=len(inliers), inlier_frac=frac,
        residual_px=float(np.median(np.abs(ph[mask] @ best_line))))


def vertical_vp_from_pedestrians(
    feet_px: np.ndarray,
    heads_px: np.ndarray,
    *,
    max_segments: int = 4000,
    inlier_angle_deg: float = 2.0,
    irls_iters: int = 12,
    tukey_c: float = 4.685,
    min_inlier_frac: float = 0.30,
    min_len_px: float = 60.0,
    seed: int = 20260903,
) -> VerticalVPResult:
    """Вертикальная точка схода по отрезкам «стопа-макушка».

    Клики пользователя здесь не участвуют вообще: одна кликнутая вертикаль
    точку схода не задаёт, а на реальном кадре она оказалась почти идеально
    вертикальной, что и сломало фокус.

    ТОЧКИ ОБЯЗАНЫ БЫТЬ ИЗ ПОЗЫ, А НЕ ИЗ РАМКИ. У bbox низ и верх берутся по
    центру рамки, то есть с ОДИНАКОВЫМ x: отрезок вертикален в кадре по
    построению, все такие отрезки параллельны и не пересекаются нигде.
    Проверено на реальных детекциях — RANSAC не находил ни одного кандидата.
    Настоящие голеностоп и макушка смещены друг относительно друга, и именно
    это смещение несёт перспективу.
    """
    feet_px = np.asarray(feet_px, dtype=np.float64)
    heads_px = np.asarray(heads_px, dtype=np.float64)
    dx = np.abs(heads_px[:, 0] - feet_px[:, 0])
    if len(dx) and float(np.median(dx)) < 1.0:
        raise CalibError(
            "отрезки стопа-макушка вертикальны в кадре (медианное смещение по x "
            f"{float(np.median(dx)):.2f} px): похоже, точки взяты из ЦЕНТРА РАМКИ, "
            "а не из позы. Из рамки вертикальную точку схода получить нельзя")
    rng = np.random.default_rng(seed)
    keep = np.hypot(*(heads_px - feet_px).T) >= min_len_px
    feet_px, heads_px = feet_px[keep], heads_px[keep]
    if len(feet_px) > max_segments:
        sel = rng.choice(len(feet_px), size=max_segments, replace=False)
        feet_px, heads_px = feet_px[sel], heads_px[sel]
    if len(feet_px) < 20:
        raise CalibError(f"отрезков стопа-макушка {len(feet_px)} при минимуме 20")

    lines = np.asarray([line_through(f, h) for f, h in zip(feet_px, heads_px)])

    # РОБАСТНЫЙ МНК, а не RANSAC. Отрезки «стопа-макушка» почти параллельны друг
    # другу (медианный наклон к вертикали 5.4 градуса при разбросе 8.9), и
    # попарные пересечения таких прямых — дикие выбросы. RANSAC на них собирает
    # консенсус на шуме: на реальных данных он давал точку схода ВНУТРИ кадра,
    # что для камеры, наклонённой вниз, геометрически невозможно.
    # МНК по всем прямым сразу с итеративным перевзвешиванием такой ямы не имеет:
    # он дал (900, 5332), то есть далеко под кадром, как и должно быть.
    weights = np.ones(len(lines))
    vp_h = None
    for _ in range(int(irls_iters)):
        wl = lines * weights[:, None]
        _, _, vh = np.linalg.svd(wl.T @ wl)
        v = vh[-1]
        if abs(v[2]) < 1e-12:
            raise CalibError("вертикальная точка схода ушла на бесконечность")
        vp_h = np.array([v[0] / v[2], v[1] / v[2], 1.0])
        resid = np.abs(lines @ vp_h)
        # Тьюки: выбросы получают нулевой вес, а не просто малый.
        c = max(1e-6, 1.4826 * float(np.median(resid)) * float(tukey_c))
        u = np.clip(resid / c, 0.0, 1.0)
        weights = (1.0 - u ** 2) ** 2

    # Инлаер меряется УГЛОМ, а не пикселями. Точка схода стоит в тысячах
    # пикселей от кадра, и там допуск 30 px означает 0.34 градуса — недостижимую
    # точность для отрезка длиной 160 px. Угол между направлением отрезка и
    # направлением от его середины на точку схода от удалённости не зависит.
    mid = (feet_px + heads_px) / 2.0
    seg_dir = heads_px - feet_px
    to_vp = vp_h[:2][None, :] - mid
    cross = np.abs(seg_dir[:, 0] * to_vp[:, 1] - seg_dir[:, 1] * to_vp[:, 0])
    dots = np.abs(seg_dir[:, 0] * to_vp[:, 0] + seg_dir[:, 1] * to_vp[:, 1])
    ang_deg = np.degrees(np.arctan2(cross, dots))
    mask = ang_deg < float(inlier_angle_deg)
    resid = ang_deg
    frac = float(mask.mean())
    if frac < min_inlier_frac:
        raise CalibError(
            f"вертикальная точка схода подтверждена {frac:.1%} отрезков при минимуме "
            f"{min_inlier_frac:.0%}: люди в кадре не выглядят вертикальными")
    return VerticalVPResult(
        vp_px=vp_h[:2], n_pairs=len(lines), n_inliers=int(mask.sum()),
        inlier_frac=frac,
        residual_px=float(np.median(resid[mask]) if mask.any() else np.nan))


def focal_from_horizon_and_vertical(horizon, vp_vertical_px,
                                    frame_shape) -> tuple[float, dict]:
    """Фокус из горизонта и вертикальной точки схода.

    Горизонт — поляра вертикальной точки схода относительно образа абсолютной
    коники. При главной точке в центре и квадратном пикселе это даёт
    f^2 = c'*(v - c) / l, где l — коэффициенты горизонта в центрированных
    координатах. Оба уравнения (по x и по y) должны дать одно и то же f;
    их расхождение — честная диагностика, а не деталь реализации.
    """
    h, w = frame_shape
    c = np.array([w / 2.0, h / 2.0])
    line = np.asarray(horizon, dtype=np.float64)
    line = line / np.hypot(line[0], line[1])
    # Переносим горизонт в центрированные координаты.
    a, b = line[0], line[1]
    c_shift = float(line[2] + a * c[0] + b * c[1])
    v = np.asarray(vp_vertical_px, dtype=np.float64) - c

    # Вертикальная VP обязана быть перпендикулярна горизонту (как направление
    # от главной точки). Иначе допущение о квадратном пикселе не выполняется.
    perp_err = float(abs(a * v[1] - b * v[0]) / (np.hypot(*v) + 1e-9))

    # Устойчивая форма. Горизонт — поляра вертикальной VP, а для поляры
    # расстояние от полюса до главной точки, умноженное на расстояние от
    # главной точки до поляры, равно f^2:
    #
    #     f^2 = d(главная точка -> горизонт) * d(главная точка -> вертикальная VP)
    #
    # Покомпонентные уравнения f^2 = c'*v_x/a и f^2 = c'*v_y/b формально те же,
    # но у камеры без крена горизонт почти горизонтален, a близко к нулю, и
    # первое уравнение взрывается. На синтетике это давало ошибку фокуса 30 %
    # при точной вертикальной VP. Норма обоих коэффициентов такой ямы не имеет.
    d_vp = float(np.hypot(v[0], v[1]))
    d_horizon = abs(c_shift)          # прямая нормирована, значит это расстояние
    if d_vp < 1e-6 or d_horizon < 1e-6:
        raise CalibError(
            "вертикальная точка схода или горизонт проходят через главную точку: "
            "фокус не определён")
    # Из соотношения поляры: (a, b) = c' * (v_x, v_y) / f^2, а f^2 > 0, значит
    # скалярное произведение (a, b) на v имеет ТОТ ЖЕ знак, что и c'. Обратный
    # знак означает, что горизонт и вертикальная точка схода не образуют пару
    # полюс-поляра, то есть геометрия несогласована.
    if float(v[0] * a + v[1] * b) * c_shift < 0:
        raise CalibError(
            "горизонт и вертикальная точка схода не образуют пару полюс-поляра: "
            "знаки не сходятся, геометрия несогласована")
    f_sq = d_horizon * d_vp
    cands = [f_sq]
    return float(np.sqrt(f_sq)), {
        "d_horizon_px": d_horizon,
        "d_vertical_vp_px": d_vp,
        "vertical_perp_err_px": perp_err,
    }


def point_on_line_distance_px(line, point_px) -> float:
    line = np.asarray(line, dtype=np.float64)
    line = line / np.hypot(line[0], line[1])
    p = np.asarray(point_px, dtype=np.float64)
    return float(abs(line[0] * p[0] + line[1] * p[1] + line[2]))


def scale_from_street_width(h_px_to_unit, line_a_px: Sequence, line_b_px: Sequence,
                            width_m: float) -> tuple[float, dict]:
    """Масштаб метры-на-единицу по известной ширине улицы.

    Рост в подгонке НЕ участвует, поэтому дальше он становится независимой
    проверкой. В прежней схеме масштаб брался из роста, и проверять его было
    нечем — медиана роста попадала в диапазон по построению.
    """
    from looq.geometry import facade_lines_separation_m

    a = apply_h(h_px_to_unit, np.asarray(line_a_px, dtype=np.float64))
    b = apply_h(h_px_to_unit, np.asarray(line_b_px, dtype=np.float64))
    sep = facade_lines_separation_m(a[[0, -1]], b[[0, -1]])
    width_units = sep["width_mean_m"]
    if not np.isfinite(width_units) or width_units <= 0:
        raise CalibError(f"ширина улицы на плане вышла {width_units} — масштаб не определён")
    return float(width_m) / width_units, {
        "street_width_units": width_units,
        "street_width_spread_units": sep["width_spread_m"],
        "lines_angle_deg": sep["lines_angle_deg"],
    }


def implied_street_width_m(reference_width_m: float, target_height_m: float,
                           measured_median_height_m: float) -> float:
    """Обратная задача: какая ширина улицы даёт медиану роста target_height_m.

    Рост линейно зависит от масштаба, а масштаб — от заданной ширины. Значит
    ширина, при которой медиана роста становится правильной, считается одним
    делением. Число проверяемо глазами: если оно попадает в правдоподобный
    диапазон, версия о неверном эталоне подтверждена, если нет — дело в другом.
    """
    if measured_median_height_m <= 0:
        raise CalibError(f"медиана роста {measured_median_height_m} — деление невозможно")
    return float(reference_width_m) * float(target_height_m) / float(measured_median_height_m)


def street_grade_from_height_drift(slope_m_per_m: float, median_height_m: float,
                                   camera_height_m: float) -> dict:
    """Уклон улицы, объясняющий дрейф роста по глубине.

    Модель земли плоская. Если земля на самом деле идёт под уклон, человек на
    расстоянии d стоит на высоте dz = -d*tan(theta) относительно принятой
    плоскости. Камера тогда возвышается над его стопами не на H, а на H - dz,
    и рост меряется как

        h_apparent ~ h * (1 + dz / H) = h * (1 - d * tan(theta) / H)

    откуда наклон = -h * tan(theta) / H, и уклон достаётся обратным ходом.

    Это ОЦЕНКА, а не измерение: тот же дрейф может давать и систематическая
    ошибка рамки, растущая с глубиной. Число идёт в ограничения отчёта именно
    как оценка сверху на кривизну плоской модели.
    """
    if camera_height_m <= 0 or median_height_m <= 0:
        raise CalibError("высота камеры и рост должны быть положительны")
    tan_theta = -float(slope_m_per_m) * float(camera_height_m) / float(median_height_m)
    return {
        "tan_theta": tan_theta,
        "grade_percent": tan_theta * 100.0,
        "angle_deg": float(np.degrees(np.arctan(tan_theta))),
        "direction_ru": ("земля уходит ВНИЗ от камеры" if tan_theta > 0
                         else "земля идёт ВВЕРХ от камеры"),
    }
