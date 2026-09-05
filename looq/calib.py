"""Автокалибровка камеры по точкам схода и росту людей.

Ручных замеров геометрии нет. Схема:

  1. пользователь кликает 3+3 точки на основаниях фасадов и 2 на одной вертикали
     (``scripts/pick_hints.py``) — это ЗАТРАВКА, не контрольные точки;
  2. горизонтальная VP = пересечение базовых линий, уточняется RANSAC-ом по
     отрезкам LSD/Hough всего кадра; вертикальная VP — так же от клика 7-8;
  3. из двух ОРТОГОНАЛЬНЫХ направлений при главной точке в центре кадра и
     квадратном пикселе восстанавливается фокус, дальше K, R и линия горизонта;
  4. гомография плоскости земли известна с точностью до высоты камеры —
     это и есть неизвестный масштаб;
  5. масштаб берётся из роста людей: подбирается так, чтобы медиана роста
     выборки равнялась target_height_m.

Единицы. До шага 5 всё считается в ЕДИНИЦАХ ВЫСОТЫ КАМЕРЫ: камера стоит на
высоте ровно 1.0 unit над началом координат. Поэтому и координаты земли, и рост
человека выражены в одних и тех же единицах, и один множитель
``scale_m_per_unit`` переводит в метры сразу всё. Это важнее, чем кажется:
раздельные множители для плана и для роста разъезжаются, и тогда проверка
ширины улицы перестаёт что-либо проверять.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from looq.geometry import polygon_cross_signs


class CalibError(RuntimeError):
    """Калибровка невозможна. Всегда громко, без возврата бесконечностей."""


# --------------------------------------------------------------------------- #
# Прямые и точки в однородных координатах
# --------------------------------------------------------------------------- #

def fit_line_px(points_px) -> tuple[np.ndarray, float]:
    """Прямая через 2+ точек методом наименьших квадратов.

    Возвращает (l, residual_px), где l = (a, b, c), a^2 + b^2 = 1, так что
    |l . p| — это прямо расстояние в пикселях. residual — СКО точек от прямой:
    именно оно показывает, насколько криво накликал пользователь.
    """
    pts = np.asarray(points_px, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise CalibError(f"для прямой нужно >=2 точек (N,2), получено {pts.shape}")
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    c = -float(normal @ centroid)
    line = np.array([normal[0], normal[1], c], dtype=np.float64)
    residual = float(np.sqrt(np.mean((pts @ normal + c) ** 2)))
    return line, residual


def line_through(p1_px, p2_px) -> np.ndarray:
    """Прямая через две точки, нормированная так, что a^2 + b^2 = 1."""
    a = np.array([p1_px[0], p1_px[1], 1.0])
    b = np.array([p2_px[0], p2_px[1], 1.0])
    line = np.cross(a, b)
    n = np.hypot(line[0], line[1])
    if n < 1e-12:
        raise CalibError("две совпадающие точки не задают прямую")
    return line / n


def intersect_lines_px(l1, l2, *, min_sin: float = 1e-3) -> np.ndarray:
    """Пересечение двух прямых. Почти параллельные — ошибка, а не бесконечность."""
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    sin_angle = abs(l1[0] * l2[1] - l1[1] * l2[0])  # обе нормированы
    if sin_angle < min_sin:
        raise CalibError(
            f"прямые почти параллельны (sin={sin_angle:.2e} < {min_sin:.0e}): "
            f"точка схода уходит в бесконечность, гомография не определена"
        )
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-12:
        raise CalibError("точка схода на бесконечности")
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Отрезки кадра
# --------------------------------------------------------------------------- #

def detect_segments(image_bgr, *, min_len_px: float = 40.0) -> np.ndarray:
    """Отрезки кадра: LSD, при его отсутствии — HoughLinesP.

    Возвращает (N, 4): x1, y1, x2, y2.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    segs: np.ndarray | None = None
    try:
        lsd = cv2.createLineSegmentDetector()
        found = lsd.detect(gray)[0]
        if found is not None:
            segs = found.reshape(-1, 4)
    except (AttributeError, cv2.error):
        segs = None  # в некоторых сборках opencv LSD вырезан
    if segs is None:
        edges = cv2.Canny(gray, 60, 180, apertureSize=3)
        found = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80,
                                minLineLength=min_len_px, maxLineGap=6)
        if found is None:
            raise CalibError("в кадре не найдено ни одного отрезка: "
                             "ни LSD, ни Hough ничего не дали")
        segs = found.reshape(-1, 4).astype(np.float64)

    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs = segs[lengths >= min_len_px]
    if len(segs) == 0:
        raise CalibError(f"после фильтра по длине >= {min_len_px} px отрезков не осталось")
    return segs.astype(np.float64)


def _segment_lines(segs: np.ndarray) -> np.ndarray:
    """Нормированные прямые отрезков: (N, 3), a^2 + b^2 = 1."""
    out = []
    for x1, y1, x2, y2 in segs:
        out.append(line_through((x1, y1), (x2, y2)))
    return np.asarray(out)


def _segment_angles(segs: np.ndarray) -> np.ndarray:
    """Наклон отрезка к горизонтали в градусах, [0, 90]."""
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    return np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))


# --------------------------------------------------------------------------- #
# RANSAC по точке схода
# --------------------------------------------------------------------------- #

def lines_common_vp(lines: Sequence[np.ndarray],
                    *, min_sin: float = 1e-3) -> tuple[np.ndarray, dict[str, float]]:
    """Общая точка схода набора прямых методом наименьших квадратов.

    Четыре уличные линии (низ и верх обеих стен) сходятся в одной точке, потому
    что все четыре параллельны улице в мире. Пересечение по МНК обусловлено
    заметно лучше, чем пересечение двух: линии на стенах идут высоко над землёй
    и дают широкую базу.

    Второй элемент — диагностика согласия. Если попарные пересечения разбросаны,
    линии накликаны криво, и одна МНК-точка это спрячет. Разброс идёт в артефакт.
    """
    ls = [np.asarray(l, dtype=np.float64) for l in lines]
    if len(ls) < 2:
        raise CalibError(f"для общей точки схода нужно >=2 прямых, получено {len(ls)}")

    pairwise: list[np.ndarray] = []
    for i in range(len(ls)):
        for j in range(i + 1, len(ls)):
            sin_angle = abs(ls[i][0] * ls[j][1] - ls[i][1] * ls[j][0])
            if sin_angle < min_sin:
                continue  # эта пара почти параллельна, но другие могут спасти
            p = np.cross(ls[i], ls[j])
            if abs(p[2]) > 1e-12:
                pairwise.append(np.array([p[0] / p[2], p[1] / p[2]]))
    if not pairwise:
        raise CalibError(
            "все уличные линии почти параллельны между собой: точка схода уходит "
            "в бесконечность. Скорее всего линии накликаны почти одинаково"
        )

    # Затравка — ПОКООРДИНАТНАЯ МЕДИАНА попарных пересечений, а не МНК по прямым.
    # Причина найдена на реальных кликах: пользователь может обвести две линии
    # одной и той же физической грани (низ стены и её продолжение), и они почти
    # коллинеарны. Их пересечение уезжает на сотни пикселей, а МНК по прямым
    # такой выброс не отбрасывает — он его усредняет. Медиана отбрасывает.
    pts = np.asarray(pairwise)
    vp = np.median(pts, axis=0) if len(pts) >= 3 else _refine_vp(np.asarray(ls))
    spread = float(np.max(np.linalg.norm(pts - vp[None, :], axis=1)))
    return vp, {
        "n_lines": float(len(ls)),
        "n_pairs": float(len(pairwise)),
        "pairwise_spread_px": spread,
        "pairwise_median_dist_px": float(np.median(np.linalg.norm(pts - vp[None, :], axis=1))),
    }


def vp_candidates_on_line(line, segs: np.ndarray) -> list[np.ndarray]:
    """Кандидаты в точку схода: пересечения заданной прямой с прямыми отрезков.

    Для вертикали пользователь кликает ОДНУ линию, а одна прямая точку схода не
    задаёт. Зато настоящая вертикальная VP обязана лежать на этой прямой, так что
    кандидаты ищутся только на ней — это RANSAC с жёстким ограничением, а не
    свободный перебор.
    """
    l0 = np.asarray(line, dtype=np.float64)
    out: list[np.ndarray] = []
    for x1, y1, x2, y2 in segs:
        try:
            li = line_through((x1, y1), (x2, y2))
        except CalibError:
            continue
        sin_angle = abs(l0[0] * li[1] - l0[1] * li[0])
        if sin_angle < 1e-3:
            continue
        p = np.cross(l0, li)
        if abs(p[2]) < 1e-12:
            continue
        out.append(np.array([p[0] / p[2], p[1] / p[2]]))
    if not out:
        raise CalibError("ни один отрезок кадра не пересекает линию вертикали")
    return out


@dataclass
class VPResult:
    vp_px: np.ndarray
    n_inliers: int
    n_candidates: int
    holdout_residual_px: float
    n_holdout: int


def _refine_vp(lines: np.ndarray) -> np.ndarray:
    """VP как точка, минимизирующая сумму квадратов расстояний до прямых."""
    m = lines.T @ lines
    _, _, vh = np.linalg.svd(m)
    v = vh[-1]
    if abs(v[2]) < 1e-12:
        raise CalibError("уточнённая точка схода ушла на бесконечность")
    return np.array([v[0] / v[2], v[1] / v[2]], dtype=np.float64)


def ransac_vp(
    segs: np.ndarray,
    seed_vp_px,
    *,
    inlier_tol_px: float,
    min_inliers: int,
    holdout_frac: float = 0.30,
    seed: int = 20260903,
    frame_shape: tuple[int, int] | None = None,
    seed_max_deviation_deg: float | None = None,
) -> VPResult:
    """Уточняет точку схода по отрезкам кадра.

    Инлаер — отрезок, чья ПРЯМАЯ проходит ближе inlier_tol_px от кандидата VP.
    Инлаеры делятся 70/30 с фиксированным сидом: 70 % идут в финальную оценку,
    30 % удерживаются и дают невязку. Клики пользователя в удержанную выборку
    не входят — они только затравка.
    """
    lines = _segment_lines(segs)
    # seed_vp_px — либо одна точка, либо набор кандидатов (например, все
    # пересечения линии вертикали с отрезками кадра). Берём тот кандидат,
    # который собирает больше всего инлаеров.
    seeds = np.atleast_2d(np.asarray(seed_vp_px, dtype=np.float64))
    best_mask, best_n = None, -1
    # Имя candidate, а не seed: параметр seed — это сид ГСЧ для деления 70/30,
    # и переменная цикла его затирала. Ошибка нашлась только тестом.
    for candidate in seeds:
        mask = np.abs(lines @ np.array([candidate[0], candidate[1], 1.0])) < inlier_tol_px
        n = int(mask.sum())
        if n > best_n:
            best_mask, best_n = mask, n
    inlier_mask = best_mask
    n_in = best_n
    if n_in < min_inliers:
        raise CalibError(
            f"точка схода подтверждена лишь {n_in} отрезками при минимуме {min_inliers}: "
            f"затравка неверна либо в кадре нет нужной структуры"
        )

    idx = np.flatnonzero(inlier_mask)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_hold = max(1, int(round(holdout_frac * len(idx))))
    hold_idx, fit_idx = idx[:n_hold], idx[n_hold:]
    if len(fit_idx) < 2:
        raise CalibError("после разделения 70/30 на оценку осталось меньше 2 отрезков")

    vp = _refine_vp(lines[fit_idx])

    if frame_shape is not None:
        h, w = frame_shape
        # РАНЬШЕ ЗДЕСЬ БЫЛ ЗАПРЕТ на точку схода внутри кадра. Он был НЕВЕРЕН:
        # у камеры, смотрящей вдоль улицы, точка схода улицы естественно попадает
        # в кадр, и на реальных кликах Kabukicho она оказалась в (1889, 441).
        # Допущение стоило прогона. Оставлено как информация, не как ошибка.
        if 0 <= vp[0] < w and 0 <= vp[1] < h:
            _inside_frame_note(vp, w, h)
        if seed_max_deviation_deg is not None:
            dev = _seed_deviation_deg(seeds, vp, (w / 2.0, h / 2.0))
            if dev > float(seed_max_deviation_deg):
                raise CalibError(
                    f"уточнённая точка схода отклонилась от затравки на {dev:.1f} град "
                    f"при пределе {seed_max_deviation_deg}: RANSAC зацепился за другое "
                    f"семейство параллельных линий в кадре, а не за то, которое кликали"
                )

    residual = _holdout_residual_px(segs[hold_idx], vp)
    return VPResult(vp_px=vp, n_inliers=len(fit_idx), n_candidates=int(len(segs)),
                    holdout_residual_px=residual, n_holdout=len(hold_idx))


def _inside_frame_note(vp: np.ndarray, w: int, h: int) -> None:
    """Точка схода внутри кадра — это нормально, а не ошибка. См. комментарий выше."""
    import sys
    print(f"[calib] точка схода {vp.round(1).tolist()} внутри кадра {w}x{h} — "
          f"обычное дело для камеры, направленной вдоль улицы", file=sys.stderr)


def _seed_deviation_deg(seeds: np.ndarray, vp: np.ndarray,
                        principal_px: tuple[float, float]) -> float:
    """Угол между направлениями на затравку и на уточнённую VP из центра кадра.

    Мерить отклонение в ПИКСЕЛЯХ нельзя: точка схода почти горизонтальной улицы
    уходит на тысячи пикселей, и там смещение в сотню пикселей — это доли
    градуса, тогда как у близкой VP та же сотня означает совсем другое
    направление. Угол от главной точки — единственная устойчивая мера.

    Берётся минимум по кандидатам: победил тот из них, у которого больше
    инлаеров, и именно от него отсчитывается отклонение.
    """
    c = np.asarray(principal_px, dtype=np.float64)
    v = vp - c
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        return 0.0
    best = 180.0
    for seed in np.atleast_2d(seeds):
        d = seed - c
        nd = np.linalg.norm(d)
        if nd < 1e-9:
            continue
        cos = float(np.clip((v @ d) / (nv * nd), -1.0, 1.0))
        best = min(best, float(np.degrees(np.arccos(cos))))
    return best


def _holdout_residual_px(segs: np.ndarray, vp_px) -> float:
    """Медианное угловое отклонение удержанных отрезков, переведённое в пиксели.

    Для каждого отрезка берётся угол между его направлением и направлением
    от его середины на VP; в пиксели переводится как sin(угла) * длина отрезка —
    это смещение дальнего конца, если отрезок довернуть на VP.
    """
    if len(segs) == 0:
        raise CalibError("удержанная выборка пуста, невязку считать не по чему")
    vp = np.asarray(vp_px, dtype=np.float64)
    d = segs[:, 2:4] - segs[:, 0:2]
    lengths = np.hypot(d[:, 0], d[:, 1])
    mid = (segs[:, 0:2] + segs[:, 2:4]) / 2.0
    to_vp = vp[None, :] - mid
    cross = np.abs(d[:, 0] * to_vp[:, 1] - d[:, 1] * to_vp[:, 0])
    sin_angle = cross / (lengths * np.hypot(to_vp[:, 0], to_vp[:, 1]) + 1e-12)
    return float(np.median(sin_angle * lengths))


# --------------------------------------------------------------------------- #
# Камера из двух ортогональных точек схода
# --------------------------------------------------------------------------- #

@dataclass
class Camera:
    K: np.ndarray             # (3, 3)
    focal_px: float
    R: np.ndarray             # (3, 3), столбцы — мировые оси X, Y, Z в камере
    horizon_line: np.ndarray  # (3,), нормирована a^2 + b^2 = 1
    H_px_to_unit: np.ndarray  # (3, 3), пиксели -> план в единицах высоты камеры
    H_unit_to_px: np.ndarray


def camera_from_focal_and_vps(focal_px: float, vp_street_px, vp_vertical_px,
                              frame_shape) -> Camera:
    """K, R и горизонт при УЖЕ ИЗВЕСТНОМ фокусе.

    Отличие от camera_from_vps: там фокус выводился из ортогональности двух
    точек схода, здесь он приходит снаружи — из горизонта, найденного по
    пешеходам. Это важно: ортогональность двух кликнутых VP оказалась
    ненадёжной, а горизонт по тысяче пар людей — надёжен.
    """
    h, w = frame_shape
    c = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    f = float(focal_px)
    if not np.isfinite(f) or f <= 0:
        raise CalibError(f"фокус {focal_px} невозможен")
    K = np.array([[f, 0.0, c[0]], [0.0, f, c[1]], [0.0, 0.0, 1.0]])
    K_inv = np.linalg.inv(K)

    def _dir(vp_px):
        d = K_inv @ np.array([vp_px[0], vp_px[1], 1.0])
        return d / np.linalg.norm(d)

    d_x = _dir(vp_street_px)
    d_z = _dir(vp_vertical_px)
    if d_z[1] > 0:
        d_z = -d_z                       # вертикаль мира смотрит вверх
    d_x = d_x - (d_x @ d_z) * d_z
    n = np.linalg.norm(d_x)
    if n < 1e-9:
        raise CalibError("направление улицы совпало с вертикалью — план не построить")
    d_x /= n
    d_y = np.cross(d_z, d_x)
    d_y /= np.linalg.norm(d_y)
    R = np.column_stack([d_x, d_y, d_z])

    horizon = K_inv.T @ d_z
    horizon = horizon / np.hypot(horizon[0], horizon[1])
    H_unit_to_px = K @ np.column_stack([d_x, d_y, -d_z])
    if abs(np.linalg.det(H_unit_to_px)) < 1e-12:
        raise CalibError("гомография плоскости земли вырождена")
    return Camera(K=K, focal_px=f, R=R, horizon_line=horizon,
                  H_px_to_unit=np.linalg.inv(H_unit_to_px), H_unit_to_px=H_unit_to_px)


def camera_from_vps(vp_street_px, vp_vertical_px, frame_shape) -> Camera:
    """K, R и горизонт из двух ортогональных направлений.

    Допущения, которые обязаны попасть в отчёт: главная точка в центре кадра,
    квадратный пиксель, нулевой перекос. Для фиксированной уличной камеры это
    стандартно, но это ДОПУЩЕНИЯ, а не измерения.
    """
    h, w = frame_shape
    c = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    v1 = np.asarray(vp_street_px, dtype=np.float64) - c
    v2 = np.asarray(vp_vertical_px, dtype=np.float64) - c

    # Ортогональность направлений: (v1 - c) . (v2 - c) + f^2 = 0.
    f_sq = -float(v1 @ v2)
    if f_sq <= 0.0:
        raise CalibError(
            f"фокус не восстанавливается: f^2 = {f_sq:.1f} <= 0. Точки схода не "
            f"отвечают ортогональным направлениям — вероятно, перепутаны клики "
            f"вертикали и основания фасада"
        )
    f = float(np.sqrt(f_sq))
    K = np.array([[f, 0.0, c[0]], [0.0, f, c[1]], [0.0, 0.0, 1.0]])
    K_inv = np.linalg.inv(K)

    def _dir(vp_px):
        d = K_inv @ np.array([vp_px[0], vp_px[1], 1.0])
        return d / np.linalg.norm(d)

    d_x = _dir(vp_street_px)
    d_z = _dir(vp_vertical_px)
    # Вертикаль мира смотрит ВВЕРХ, а ось y изображения — вниз: если знак
    # перепутан, план окажется зеркальным, и это не заметят до самого отчёта.
    if d_z[1] > 0:
        d_z = -d_z
    d_x = d_x - (d_x @ d_z) * d_z
    d_x /= np.linalg.norm(d_x)
    d_y = np.cross(d_z, d_x)
    d_y /= np.linalg.norm(d_y)
    R = np.column_stack([d_x, d_y, d_z])

    # Вершинная линия плоскости земли: l = K^-T n, n — нормаль плоскости.
    horizon = K_inv.T @ d_z
    horizon = horizon / np.hypot(horizon[0], horizon[1])

    # Камера на высоте ровно 1.0 unit над началом координат плана.
    H_unit_to_px = K @ np.column_stack([d_x, d_y, -d_z])
    if abs(np.linalg.det(H_unit_to_px)) < 1e-12:
        raise CalibError("гомография плоскости земли вырождена")
    return Camera(K=K, focal_px=f, R=R, horizon_line=horizon,
                  H_px_to_unit=np.linalg.inv(H_unit_to_px), H_unit_to_px=H_unit_to_px)


# --------------------------------------------------------------------------- #
# Рост человека в единицах высоты камеры
# --------------------------------------------------------------------------- #

def person_height_units(cam: Camera, foot_px, head_px) -> float:
    """Рост человека в единицах высоты камеры.

    Точка ног проецируется на план, затем из положения макушки решается высота.
    Считается в тех же единицах, что и координаты плана, — один масштаб на всё.
    """
    ground = apply_h(cam.H_px_to_unit, np.asarray(foot_px, dtype=np.float64)[None, :])[0]
    x, y = float(ground[0]), float(ground[1])

    # p ~ K (x*d_x + y*d_y + (Z - 1)*d_z) = A + Z*B
    d_x, d_y, d_z = cam.R[:, 0], cam.R[:, 1], cam.R[:, 2]
    A = cam.K @ (x * d_x + y * d_y - d_z)
    B = cam.K @ d_z
    u, v = float(head_px[0]), float(head_px[1])
    # (A1 + Z B1) - u (A3 + Z B3) = 0  и то же по v: два линейных уравнения на Z.
    coeff = np.array([B[0] - u * B[2], B[1] - v * B[2]])
    rhs = np.array([u * A[2] - A[0], v * A[2] - A[1]])
    denom = float(coeff @ coeff)
    if denom < 1e-15:
        raise CalibError("высота не определяется: макушка на линии горизонта")
    return float((coeff @ rhs) / denom)


def apply_h(h_matrix, pts) -> np.ndarray:
    """Однородное преобразование набора точек (N, 2) -> (N, 2)."""
    pts = np.asarray(pts, dtype=np.float64)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    out = homo @ np.asarray(h_matrix, dtype=np.float64).T
    wcol = out[:, 2:3]
    bad = np.abs(wcol[:, 0]) < 1e-12
    if np.any(bad):
        raise CalibError(f"{int(bad.sum())} точек проецируются на линию горизонта")
    return out[:, :2] / wcol


#: Знак якобиана отображения кадр -> план у канонического плана. Взят от пути
#: через точки схода (looq.calib.camera_from_vps), который проверен тестами.
#: Любой другой способ построить план обязан приводиться к этому знаку, иначе
#: план окажется зеркальным и все углы развернутся.
CANONICAL_PLANE_SIGN = -1.0


def _plane_orientation_sign(h_px_to_unit, probe_px, eps: float = 1.0) -> float:
    """Знак якобиана отображения кадр -> план в окрестности точки."""
    p = np.asarray(probe_px, dtype=np.float64)
    o = apply_h(h_px_to_unit, [p])[0]
    dx = apply_h(h_px_to_unit, [p + [eps, 0.0]])[0] - o
    dy = apply_h(h_px_to_unit, [p + [0.0, eps]])[0] - o
    return float(np.sign(dx[0] * dy[1] - dx[1] * dy[0]))


def homography_from_ground_rect(rect_px, aspect_w_over_h: float) -> tuple[np.ndarray, dict]:
    """Плоскость земли по четырём углам прямоугольного участка мостовой.

    Запасной путь калибровки, когда точки схода не даются: на ночной сцене
    линии фасадов кликаются с разбросом в десятки пикселей, и VP уезжает.
    Четыре угла прямоугольника с ИЗВЕСТНЫМИ пропорциями задают плоскость земли
    однозначно и с точностью до масштаба.

    rect_px — углы В ПОРЯДКЕ TL, TR, BR, BL, как в scripts/pick_zones.py.
    aspect_w_over_h — отношение стороны TL->TR к стороне TL->BL.

    Оси плана: +x вдоль TL->TR, +y вдоль TL->BL, сторона TL->BL принята за 1.0
    условной единицы. Значит МЕТРОВ ЗДЕСЬ НЕТ: длины в условных единицах,
    и любое число, названное метром, было бы враньём.

    Что этого хватает для: углы на плоскости (луч ориентации), принадлежность
    к зонам, ОТНОШЕНИЯ длин и скоростей. Чего не хватает для: абсолютных
    метров, роста, порога остановки в м/с.
    """
    pts = np.asarray(rect_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape != (4, 2):
        raise CalibError(f"нужны ровно 4 угла (4, 2), получено {pts.shape}")
    a = float(aspect_w_over_h)
    if not (a > 0) or not np.isfinite(a):
        raise CalibError(f"пропорция должна быть положительной, получено {aspect_w_over_h}")

    # Порядок углов проверяется так же, как в обводке зон: BR и BL ниже TL и TR.
    if (pts[2][1] + pts[3][1]) / 2.0 <= (pts[0][1] + pts[1][1]) / 2.0:
        raise CalibError(
            "углы участка перепутаны: точки 3-4 (BR, BL) не ниже точек 1-2 (TL, TR). "
            "Порядок обхода — TL, TR, BR, BL по часовой стрелке")
    signs = polygon_cross_signs(pts)
    if not (np.all(signs > 0) or np.all(signs < 0)):
        raise CalibError("участок невыпуклый или самопересекается — это не прямоугольник")

    dst = np.array([[0.0, 0.0], [a, 0.0], [a, 1.0], [0.0, 1.0]], dtype=np.float64)
    h_px_to_unit = cv2.getPerspectiveTransform(pts.astype(np.float32),
                                               dst.astype(np.float32))
    if abs(np.linalg.det(h_px_to_unit)) < 1e-12:
        raise CalibError("гомография вырождена: углы участка почти на одной прямой")

    # Ориентация плана приводится к канону. Прямоугольник можно обвести так, что
    # план окажется ЗЕРКАЛЬНЫМ относительно того, который даёт камера из точек
    # схода. В зеркальном плане поворот «левое плечо -> правое» на +90 против
    # часовой указывает НАЗАД, и каждая ориентация молча развернулась бы на 180.
    # Ошибка выглядела бы правдоподобно и не поймалась бы ничем, кроме разметки.
    mirrored = _plane_orientation_sign(h_px_to_unit, pts.mean(axis=0)) != CANONICAL_PLANE_SIGN
    if mirrored:
        h_px_to_unit = np.diag([1.0, -1.0, 1.0]) @ h_px_to_unit
        dst = dst * np.array([1.0, -1.0])

    # Контроль: прогоняем углы обратно и смотрим, куда они легли. Ошибка здесь
    # это чистая арифметика решателя, а не качество кликов, поэтому она должна
    # быть около нуля; если нет — что-то не так с точками.
    back = apply_h(h_px_to_unit, pts)
    residual = float(np.max(np.linalg.norm(back - dst, axis=1)))
    diag = {
        "aspect_w_over_h": a,
        "corner_residual_units": residual,
        "unit_side": "TL->BL",
        "plane_y_flipped": bool(mirrored),
        "horizon_line": horizon_from_h(h_px_to_unit).tolist(),
    }
    return h_px_to_unit, diag


def horizon_from_h(h_px_to_unit) -> np.ndarray:
    """Линия горизонта плоскости земли прямо из гомографии.

    Точки плоскости на бесконечности имеют вид (x, y, 0); их образы лежат на
    прямой, третья строка H_px_to_unit и есть эта прямая. То есть горизонт
    достаётся бесплатно, без вертикальной точки схода, — именно поэтому
    запасной путь калибровки способен считать ориентацию.
    """
    h = np.asarray(h_px_to_unit, dtype=np.float64)
    line = h[2, :].astype(np.float64).copy()
    n = np.hypot(line[0], line[1])
    if n < 1e-12:
        raise CalibError("горизонт не определён: третья строка гомографии вырождена")
    return line / n


def yaw_from_pair_via_horizon(h_px_to_unit, left_px, right_px) -> float:
    """Направление корпуса по паре точек через плоскость земли. Без высоты и без K, R.

    Отрезок плеч в мире горизонтален, значит его направление на плане полностью
    определяется ПРЯМОЙ, на которой он лежит в кадре, — высота плеч не нужна.
    Гомография переводит эту прямую в прямую на плане, и её направление и есть
    направление плеч.

    Важно: h(L) и h(R) — НЕ положения плеч на плане (плечи не на земле), это
    просто две точки на нужной прямой. Берётся только направление между ними.

    Чем это лучше обратной проекции на высоту плеч: не нужны ни высота, ни K, R.
    Работает и с полной камерой, и когда есть только плоскость земли (запасной
    путь калибровки). Заодно исчезают два неоткалиброванных параметра.

    Чем это лучше пересечения с горизонтом: у точки схода в однородных
    координатах знак произволен, и восстановить по ней ЗНАК направления нельзя
    без отдельной возни. Разность двух образов даёт знак сразу и правильно.

    Неоднозначность 180 градусов снимается анатомией пары: «левое -> правое»,
    поворот на +90 против часовой стрелки.
    """
    h = np.asarray(h_px_to_unit, dtype=np.float64)
    left = np.asarray(left_px, dtype=np.float64)
    right = np.asarray(right_px, dtype=np.float64)
    if float(np.hypot(*(right - left))) < 1e-6:
        return float("nan")

    pl = h @ np.array([left[0], left[1], 1.0])
    pr = h @ np.array([right[0], right[1], 1.0])
    # Разные знаки w означают, что отрезок пересекает горизонт: тогда порядок
    # точек на плане переворачивается, и направление посчиталось бы задом наперёд.
    if abs(pl[2]) < 1e-12 or abs(pr[2]) < 1e-12 or np.sign(pl[2]) != np.sign(pr[2]):
        return float("nan")

    v = pr[:2] / pr[2] - pl[:2] / pl[2]
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-12:
        return float("nan")
    v = v / n
    facing = np.array([-v[1], v[0]])          # поворот на +90 против часовой
    return float(np.degrees(np.arctan2(facing[1], facing[0])) % 360.0)


def backproject_to_height(cam: Camera, pts_px, z_units: float) -> np.ndarray:
    """Точки кадра на ГОРИЗОНТАЛЬНУЮ плоскость высоты z_units над землёй.

    Обычная гомография переводит только на землю (z = 0). Плечи и уши лежат
    выше, и проецировать их на землю нельзя: получится точка, где человек не
    стоит, а ориентация уедет тем сильнее, чем дальше человек от камеры.

    z_units — в единицах высоты камеры, как и всё до подбора масштаба.
    Камера стоит в (0, 0, 1), поэтому плоскость z = 1 — это уровень камеры,
    и лучи туда не приходят: такой случай отбрасывается.
    """
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    d_cam = homo @ np.linalg.inv(cam.K).T          # (N, 3) в координатах камеры
    d_world = d_cam @ cam.R                        # R^T @ d, записанное построчно

    wz = d_world[:, 2]
    denom = np.where(np.abs(wz) < 1e-12, np.nan, wz)
    t = (float(z_units) - 1.0) / denom             # камера в (0, 0, 1)
    bad = ~np.isfinite(t) | (t <= 0)               # луч уходит за камеру или в горизонт
    t = np.where(bad, np.nan, t)
    return np.stack([t * d_world[:, 0], t * d_world[:, 1]], axis=1)


def yaw_from_pair_deg(cam: Camera, left_px, right_px, z_units: float) -> float:
    """Направление корпуса по паре ЛЕВАЯ-ПРАВАЯ точка (плечи или уши), градусы.

    Пара разрешает неоднозначность 180 градусов сама: модель размечает плечи
    анатомически, левое и правое, а не «первое и второе». Человек, стоящий
    лицом в направлении f, держит левое плечо слева от f. Значит поворот
    вектора «левое -> правое» на +90 градусов против часовой стрелки и есть
    направление взгляда корпуса.

    Без этого пришлось бы гадать по лицу или по движению, а гадание по движению
    ломает всю затею: тогда ориентация перестаёт быть независимой от траектории.

    Результат — в конвенции проекта: 0 градусов = +x плана, против часовой,
    диапазон [0, 360). Возвращает nan, если точка не проецируется.
    """
    pts = backproject_to_height(cam, [left_px, right_px], z_units)
    if not np.all(np.isfinite(pts)):
        return float("nan")
    v = pts[1] - pts[0]                            # левая -> правая на плане
    if float(np.hypot(v[0], v[1])) < 1e-9:
        return float("nan")
    facing = np.array([-v[1], v[0]])               # поворот на +90 против часовой
    return float(np.degrees(np.arctan2(facing[1], facing[0])) % 360.0)


def scale_from_heights(heights_units: Sequence[float], target_height_m: float,
                       trim_frac: float = 0.10) -> dict:
    """Масштаб метры-на-единицу: медиана роста выборки = target_height_m.

    Хвосты по trim_frac с каждой стороны отбрасываются ДО подгонки: обрезанные
    краем кадра и склеенные детекции дают выбросы в обе стороны.
    """
    h = np.asarray(list(heights_units), dtype=np.float64)
    h = h[np.isfinite(h) & (h > 0)]
    if h.size < 2:
        raise CalibError(f"для подбора масштаба нужно >=2 оценок роста, есть {h.size}")
    lo, hi = np.quantile(h, [trim_frac, 1.0 - trim_frac])
    trimmed = h[(h >= lo) & (h <= hi)]
    if trimmed.size < 2:
        raise CalibError("после отбрасывания хвостов оценок роста не осталось")
    median_units = float(np.median(trimmed))
    if median_units <= 0:
        raise CalibError(f"медианный рост в единицах = {median_units}, масштаб не определён")
    return {
        "scale_m_per_unit": float(target_height_m / median_units),
        "median_units": median_units,
        "n_total": int(h.size),
        "n_after_trim": int(trimmed.size),
        "trim_frac": float(trim_frac),
    }


def height_depth_slope(heights_m, depths_m) -> dict:
    """Линейная регрессия роста по глубине: наклон и его 95 % интервал.

    Проверка добавлена по предложению исполнителя (см. docs/DECISIONS.md).
    Ненулевой наклон означает, что гомография систематически врёт с глубиной —
    именно этот дефект пуловый IQR прячет, потому что смешивает ближний и
    дальний план.
    """
    y = np.asarray(list(heights_m), dtype=np.float64)
    x = np.asarray(list(depths_m), dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 10:
        raise CalibError(f"для регрессии роста по глубине нужно >=10 точек, есть {n}")
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx < 1e-9:
        raise CalibError("все точки на одной глубине: наклон не определён")

    slope = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (intercept + slope * x)
    dof = n - 2
    se = float(np.sqrt((resid ** 2).sum() / dof / sxx))
    # t(0.975) при большом dof -> 1.96; при малом берём поправку через scipy,
    # но зависимость от scipy тут не нужна: dof >= 8 уже даёт 2.31, а мы
    # требуем n >= 10. Используем консервативные 2.31 при dof < 30.
    t_crit = 1.96 if dof >= 30 else 2.31
    return {
        "slope_m_per_m": slope,
        "slope_se": se,
        "ci95_low": slope - t_crit * se,
        "ci95_high": slope + t_crit * se,
        "covers_zero": bool((slope - t_crit * se) <= 0.0 <= (slope + t_crit * se)),
        "intercept_m": intercept,
        "n": int(n),
    }
