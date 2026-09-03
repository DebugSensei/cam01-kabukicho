"""Тесты математического ядра автокалибровки.

Клики пользователя получить в тесте нельзя, поэтому проверяем иначе: строим
синтетическую камеру с ИЗВЕСТНЫМИ параметрами, проецируем ей людей известного
роста и требуем, чтобы калибровка восстановила и высоту камеры, и рост, и
ширину улицы. Если математика врёт, это видно здесь, а не через час на клипе.
"""

from __future__ import annotations

import numpy as np
import pytest

from looq.calib import (
    CalibError,
    apply_h,
    camera_from_vps,
    fit_line_px,
    height_depth_slope,
    intersect_lines_px,
    line_through,
    person_height_units,
    ransac_vp,
    scale_from_heights,
)

FRAME_W, FRAME_H = 1920, 1080
FRAME_SHAPE = (FRAME_H, FRAME_W)

TRUE_FOCAL = 1400.0
TRUE_TILT_RAD = np.deg2rad(22.0)   # наклон камеры вниз
TRUE_CAM_HEIGHT_M = 6.4            # высота камеры над мостовой
TRUE_STREET_WIDTH_M = 6.06
TARGET_HEIGHT_M = 1.65


def _true_camera():
    """Мир: +X вдоль улицы, +Y поперёк, +Z вверх. Камера на (0, 0, H)."""
    t = TRUE_TILT_RAD
    x_cam_w = np.array([0.0, -1.0, 0.0])
    y_cam_w = np.array([-np.sin(t), 0.0, -np.cos(t)])
    z_cam_w = np.array([np.cos(t), 0.0, -np.sin(t)])
    R = np.vstack([x_cam_w, y_cam_w, z_cam_w])          # мир -> камера
    K = np.array([[TRUE_FOCAL, 0.0, FRAME_W / 2.0],
                  [0.0, TRUE_FOCAL, FRAME_H / 2.0],
                  [0.0, 0.0, 1.0]])
    C = np.array([0.0, 0.0, TRUE_CAM_HEIGHT_M])
    return K, R, C


def _project(points_world):
    K, R, C = _true_camera()
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    cam = (R @ (pts - C).T).T
    img = (K @ cam.T).T
    return img[:, :2] / img[:, 2:3]


def _project_direction(direction_world):
    """Точка схода мирового направления."""
    K, R, _ = _true_camera()
    v = K @ R @ np.asarray(direction_world, dtype=np.float64)
    return np.array([v[0] / v[2], v[1] / v[2]])


# --------------------------------------------------------------------------- #
# Прямые и пересечения
# --------------------------------------------------------------------------- #

def test_fit_line_residual_reports_sloppy_clicks():
    """Разброс кликов относительно прямой попадает в невязку, а не теряется."""
    clean = [(100.0, 500.0), (400.0, 500.0), (700.0, 500.0)]
    line, res = fit_line_px(clean)
    assert res < 1e-9

    sloppy = [(100.0, 500.0), (400.0, 512.0), (700.0, 494.0)]
    _, res2 = fit_line_px(sloppy)
    assert res2 > 4.0, f"кривые клики дали невязку {res2:.2f}, ожидался заметный разброс"


def test_parallel_lines_fail_loudly():
    """Почти параллельные базовые линии — ошибка, а не бесконечность (правило 8)."""
    l1 = line_through((0.0, 500.0), (1000.0, 500.0))
    l2 = line_through((0.0, 700.0), (1000.0, 700.0))
    with pytest.raises(CalibError, match="параллельны"):
        intersect_lines_px(l1, l2)


def test_street_base_lines_give_street_vp():
    """Пересечение оснований фасадов совпадает с истинной точкой схода улицы."""
    half = TRUE_STREET_WIDTH_M / 2.0
    left = _project([[5.0, -half, 0.0], [40.0, -half, 0.0]])
    right = _project([[5.0, half, 0.0], [40.0, half, 0.0]])
    vp = intersect_lines_px(fit_line_px(left)[0], fit_line_px(right)[0])
    assert np.allclose(vp, _project_direction([1.0, 0.0, 0.0]), atol=1.0)


# --------------------------------------------------------------------------- #
# Восстановление камеры
# --------------------------------------------------------------------------- #

def test_camera_from_vps_recovers_focal():
    cam = camera_from_vps(_project_direction([1.0, 0.0, 0.0]),
                          _project_direction([0.0, 0.0, 1.0]), FRAME_SHAPE)
    assert cam.focal_px == pytest.approx(TRUE_FOCAL, rel=0.02)


def test_vp_inside_frame_is_allowed():
    """VP внутри кадра — НОРМА, а не ошибка.

    Раньше код это отвергал. Допущение было неверным: у камеры, направленной
    вдоль улицы, точка схода улицы естественно попадает в кадр. На реальных
    кликах Kabukicho она оказалась в (1889, 441), и запрет стоил прогона.
    Тест закрепляет исправленное поведение.
    """
    segs = np.array([[0.0, 0.0, 100.0, 100.0], [0.0, 100.0, 100.0, 0.0],
                     [10.0, 0.0, 90.0, 100.0], [0.0, 90.0, 100.0, 10.0]])
    res = ransac_vp(segs, (50.0, 50.0), inlier_tol_px=50.0, min_inliers=3,
                    frame_shape=FRAME_SHAPE)
    assert 0 <= res.vp_px[0] < FRAME_W and 0 <= res.vp_px[1] < FRAME_H


def test_swapped_vps_rejected():
    """Клики вертикали и фасада перепутаны -> f^2 <= 0, падаем с объяснением."""
    with pytest.raises(CalibError, match="фокус не восстанавливается"):
        camera_from_vps(_project_direction([1.0, 0.0, 0.0]),
                        _project_direction([1.0, 0.05, 0.0]), FRAME_SHAPE)


# --------------------------------------------------------------------------- #
# Главное: масштаб из роста
# --------------------------------------------------------------------------- #

def _synthetic_people(n=300, seed=0):
    """Люди роста N(1.65, 0.07) на разной глубине и поперёк улицы."""
    rng = np.random.default_rng(seed)
    heights = rng.normal(TARGET_HEIGHT_M, 0.07, n)
    xs = rng.uniform(6.0, 30.0, n)
    ys = rng.uniform(-TRUE_STREET_WIDTH_M / 2 + 0.4, TRUE_STREET_WIDTH_M / 2 - 0.4, n)
    feet = _project(np.column_stack([xs, ys, np.zeros(n)]))
    heads = _project(np.column_stack([xs, ys, heights]))
    return heights, feet, heads


def test_scale_recovers_camera_height_and_heights():
    """Восстановленный масштаб = истинной высоте камеры; рост восстановлен."""
    cam = camera_from_vps(_project_direction([1.0, 0.0, 0.0]),
                          _project_direction([0.0, 0.0, 1.0]), FRAME_SHAPE)
    true_h, feet, heads = _synthetic_people()
    units = [person_height_units(cam, f, hd) for f, hd in zip(feet, heads)]

    fit = scale_from_heights(units, TARGET_HEIGHT_M)
    scale = fit["scale_m_per_unit"]
    # Единица длины — высота камеры, поэтому множитель обязан ей равняться.
    assert scale == pytest.approx(TRUE_CAM_HEIGHT_M, rel=0.02), (
        f"масштаб {scale:.3f} против истинной высоты камеры {TRUE_CAM_HEIGHT_M}")

    est_m = np.array(units) * scale
    assert np.abs(np.median(est_m) - np.median(true_h)) < 0.02
    assert np.abs(est_m - true_h).max() < 0.05


def test_street_width_recovered_from_ground_plane():
    """Ширина улицы на восстановленном плане совпадает с истинной."""
    cam = camera_from_vps(_project_direction([1.0, 0.0, 0.0]),
                          _project_direction([0.0, 0.0, 1.0]), FRAME_SHAPE)
    _, feet, heads = _synthetic_people()
    units = [person_height_units(cam, f, hd) for f, hd in zip(feet, heads)]
    scale = scale_from_heights(units, TARGET_HEIGHT_M)["scale_m_per_unit"]

    half = TRUE_STREET_WIDTH_M / 2.0
    left_px = _project([[5.0, -half, 0.0], [40.0, -half, 0.0]])
    right_px = _project([[5.0, half, 0.0], [40.0, half, 0.0]])
    left_m = apply_h(cam.H_px_to_unit, left_px) * scale
    right_m = apply_h(cam.H_px_to_unit, right_px) * scale

    from looq.geometry import facade_lines_separation_m
    sep = facade_lines_separation_m(left_m, right_m)
    assert sep["width_mean_m"] == pytest.approx(TRUE_STREET_WIDTH_M, abs=0.15)


def test_depth_slope_flat_for_correct_calibration():
    """При верной калибровке наклон роста по глубине накрывает ноль."""
    cam = camera_from_vps(_project_direction([1.0, 0.0, 0.0]),
                          _project_direction([0.0, 0.0, 1.0]), FRAME_SHAPE)
    _, feet, heads = _synthetic_people()
    units = [person_height_units(cam, f, hd) for f, hd in zip(feet, heads)]
    scale = scale_from_heights(units, TARGET_HEIGHT_M)["scale_m_per_unit"]
    ground = apply_h(cam.H_px_to_unit, feet) * scale

    reg = height_depth_slope(np.array(units) * scale, ground[:, 0])
    assert reg["covers_zero"], f"наклон {reg['slope_m_per_m']:.4f} не накрывает ноль"


def test_depth_slope_catches_wrong_focal():
    """Неверный фокус -> рост плывёт с глубиной, наклон ноль не накрывает.

    Это тот дефект, который пуловый IQR прячет: он смешивает ближний и дальний
    план, а регрессия смотрит именно на зависимость от глубины.
    """
    wrong = camera_from_vps(_project_direction([1.0, 0.0, 0.0]),
                            _project_direction([0.0, 0.0, 1.0]), (FRAME_H, FRAME_W))
    # Портим фокус на 25 %: так выглядит ошибка в точке схода.
    wrong.K[0, 0] *= 1.25
    wrong.K[1, 1] *= 1.25
    K_inv = np.linalg.inv(wrong.K)
    d_x = K_inv @ np.append(_project_direction([1.0, 0.0, 0.0]), 1.0)
    d_z = K_inv @ np.append(_project_direction([0.0, 0.0, 1.0]), 1.0)
    d_x /= np.linalg.norm(d_x)
    d_z /= np.linalg.norm(d_z)
    if d_z[1] > 0:
        d_z = -d_z
    d_x = d_x - (d_x @ d_z) * d_z
    d_x /= np.linalg.norm(d_x)
    d_y = np.cross(d_z, d_x)
    wrong.R = np.column_stack([d_x, d_y, d_z])
    wrong.H_unit_to_px = wrong.K @ np.column_stack([d_x, d_y, -d_z])
    wrong.H_px_to_unit = np.linalg.inv(wrong.H_unit_to_px)

    _, feet, heads = _synthetic_people()
    units = [person_height_units(wrong, f, hd) for f, hd in zip(feet, heads)]
    scale = scale_from_heights(units, TARGET_HEIGHT_M)["scale_m_per_unit"]
    ground = apply_h(wrong.H_px_to_unit, feet) * scale

    reg = height_depth_slope(np.array(units) * scale, ground[:, 0])
    assert not reg["covers_zero"], (
        f"кривой фокус не пойман: наклон {reg['slope_m_per_m']:.4f}, "
        f"CI [{reg['ci95_low']:.4f}, {reg['ci95_high']:.4f}]")


def test_height_sample_too_small_fails():
    with pytest.raises(CalibError):
        scale_from_heights([1.0], TARGET_HEIGHT_M)
    with pytest.raises(CalibError):
        height_depth_slope([1.6] * 5, [1.0] * 5)


# --------------------------------------------------------------------------- #
# Четыре уличные линии вместо двух (14 кликов)
# --------------------------------------------------------------------------- #

def _street_line_px(y_m: float, z_m: float, noise_px: float = 0.0, seed: int = 0):
    """Три точки на линии, параллельной улице, на высоте z_m над землёй."""
    rng = np.random.default_rng(seed)
    pts = _project([[5.0, y_m, z_m], [20.0, y_m, z_m], [40.0, y_m, z_m]])
    if noise_px:
        pts = pts + rng.normal(0, noise_px, pts.shape)
    return pts


def test_four_street_lines_share_one_vp():
    """L1..L4 параллельны улице в мире, значит сходятся в одной точке."""
    from looq.calib import lines_common_vp

    half = TRUE_STREET_WIDTH_M / 2
    lines = [fit_line_px(_street_line_px(y, z))[0]
             for y, z in ((-half, 0.0), (-half, 3.5), (half, 0.0), (half, 4.2))]
    vp, diag = lines_common_vp(lines)
    assert np.allclose(vp, _project_direction([1.0, 0.0, 0.0]), atol=0.5)
    assert diag["n_lines"] == 4 and diag["n_pairs"] == 6
    assert diag["pairwise_spread_px"] < 0.5


def test_four_lines_beat_two_under_click_noise():
    """Линии на стенах дают широкую базу: ошибка VP падает примерно вдвое."""
    from looq.calib import lines_common_vp

    half = TRUE_STREET_WIDTH_M / 2
    true_vp = _project_direction([1.0, 0.0, 0.0])
    err4, err2 = [], []
    for k in range(120):
        lines = [fit_line_px(_street_line_px(y, z, noise_px=2.0, seed=100 * k + i))[0]
                 for i, (y, z) in enumerate(((-half, 0.0), (half, 0.0),
                                             (-half, 3.5), (half, 4.2)))]
        err4.append(np.linalg.norm(lines_common_vp(lines)[0] - true_vp))
        err2.append(np.linalg.norm(lines_common_vp(lines[:2])[0] - true_vp))
    assert np.median(err4) < np.median(err2) * 0.75, (
        f"четыре линии не дали выигрыша: {np.median(err4):.1f} против {np.median(err2):.1f}")


def test_degenerate_street_lines_fail_loudly():
    """Все линии накликаны почти одинаково — VP не определена, падаем."""
    from looq.calib import lines_common_vp

    same = [fit_line_px(_street_line_px(0.0, 0.0))[0] for _ in range(4)]
    with pytest.raises(CalibError, match="параллельны"):
        lines_common_vp(same)


def test_vertical_vp_from_candidates_on_v1_line():
    """Одна кликнутая вертикаль задаёт прямую, VP ищется кандидатами на ней."""
    from looq.calib import vp_candidates_on_line

    true_vp = _project_direction([0.0, 0.0, 1.0])
    v1 = fit_line_px(_project([[12.0, 0.0, 0.0], [12.0, 0.0, 3.0]]))[0]
    segs = np.array([np.concatenate(_project([[x, y, 0.0], [x, y, 3.0]]))
                     for x, y in ((8.0, -2.0), (16.0, 2.0), (22.0, -1.0), (30.0, 1.5))])
    res = ransac_vp(segs, np.asarray(vp_candidates_on_line(v1, segs)),
                    inlier_tol_px=4.0, min_inliers=3, frame_shape=FRAME_SHAPE,
                    seed_max_deviation_deg=25.0)
    assert np.allclose(res.vp_px, true_vp, atol=2.0)


def test_seed_deviation_guard_catches_wrong_family():
    """RANSAC зацепился за другое семейство линий — ловим по углу, не по пикселям."""
    rng = np.random.default_rng(0)
    vp = np.array([3000.0, -200.0])
    segs = []
    for _ in range(80):
        p0 = rng.uniform([100, 300], [1800, 1000])
        d = vp - p0
        d /= np.linalg.norm(d)
        segs.append([*p0, *(p0 + d * rng.uniform(60, 200))])
    segs = np.array(segs)

    ok = ransac_vp(segs, vp, inlier_tol_px=4.0, min_inliers=20,
                   frame_shape=FRAME_SHAPE, seed_max_deviation_deg=25.0)
    assert np.allclose(ok.vp_px, vp, atol=1.0)

    with pytest.raises(CalibError, match="отклонилась от затравки"):
        ransac_vp(segs, (-6000.0, 4000.0), inlier_tol_px=1e6, min_inliers=20,
                  frame_shape=FRAME_SHAPE, seed_max_deviation_deg=25.0)


# --------------------------------------------------------------------------- #
# S5: обратная проекция на высоту и угол по анатомической паре
# --------------------------------------------------------------------------- #

SHOULDER_M, HALF_SHOULDER_M = 1.40, 0.19


def _cam():
    return camera_from_vps(_project_direction([1.0, 0.0, 0.0]),
                           _project_direction([0.0, 0.0, 1.0]), FRAME_SHAPE)


def test_backproject_at_ground_matches_homography():
    """На земле обратная проекция обязана совпасть с гомографией до нуля."""
    from looq.calib import backproject_to_height

    cam = _cam()
    foot = _project([[15.0, 1.0, 0.0], [30.0, -2.0, 0.0]])
    a = backproject_to_height(cam, foot, 0.0)
    b = apply_h(cam.H_px_to_unit, foot)
    assert np.allclose(a, b, atol=1e-9), f"расхождение {np.abs(a - b).max()}"


def test_backproject_at_height_hits_true_point():
    """Точка на высоте 1.40 м возвращается в своё место на плане."""
    from looq.calib import backproject_to_height

    cam = _cam()
    x, y = 18.0, 1.5
    px = _project([[x, y, SHOULDER_M]])
    got = backproject_to_height(cam, px, SHOULDER_M / TRUE_CAM_HEIGHT_M)[0] * TRUE_CAM_HEIGHT_M
    assert np.allclose(got, [x, y], atol=0.01), f"получено {got}, ожидалось {[x, y]}"


def test_projecting_shoulders_to_ground_is_wrong():
    """Контроль: спроецировать плечи на землю нельзя — ошибка метры, не сантиметры.

    Это и есть причина, по которой заведена backproject_to_height.
    """
    from looq.calib import backproject_to_height

    cam = _cam()
    x, y = 18.0, 1.5
    px = _project([[x, y, SHOULDER_M]])
    wrong = apply_h(cam.H_px_to_unit, px)[0] * TRUE_CAM_HEIGHT_M
    assert np.linalg.norm(wrong - np.array([x, y])) > 1.0, (
        "проекция плеч на землю оказалась точной — тест потерял смысл")


def _shoulders_px(yaw_deg: float, x=15.0, y=1.0):
    """Плечи человека, стоящего в (x, y) лицом в направлении yaw_deg."""
    f = np.deg2rad(yaw_deg)
    left_dir = np.array([-np.sin(f), np.cos(f)])   # левое плечо слева от взгляда
    left = [x + left_dir[0] * HALF_SHOULDER_M, y + left_dir[1] * HALF_SHOULDER_M, SHOULDER_M]
    right = [x - left_dir[0] * HALF_SHOULDER_M, y - left_dir[1] * HALF_SHOULDER_M, SHOULDER_M]
    return _project([left])[0], _project([right])[0]


@pytest.mark.parametrize("true_yaw", [0.0, 45.0, 90.0, 170.0, 250.0, 330.0])
def test_yaw_from_shoulders_exact(true_yaw):
    """Угол корпуса восстанавливается точно во всём диапазоне [0, 360)."""
    from looq.calib import yaw_from_pair_deg
    from looq.geometry import yaw_diff_deg

    cam = _cam()
    left_px, right_px = _shoulders_px(true_yaw)
    got = yaw_from_pair_deg(cam, left_px, right_px, SHOULDER_M / TRUE_CAM_HEIGHT_M)
    assert 0.0 <= got < 360.0
    assert abs(float(yaw_diff_deg(got, true_yaw))) < 0.5


def test_yaw_resolves_front_back_ambiguity():
    """Развернувшись на 180 градусов, человек даёт угол, отличный на 180.

    Пара «левое-правое» анатомическая, поэтому неоднозначность снимается сама,
    без гадания по движению.
    """
    from looq.calib import yaw_from_pair_deg
    from looq.geometry import yaw_diff_deg

    cam = _cam()
    z = SHOULDER_M / TRUE_CAM_HEIGHT_M
    a = yaw_from_pair_deg(cam, *_shoulders_px(40.0), z)
    b = yaw_from_pair_deg(cam, *_shoulders_px(220.0), z)
    assert abs(abs(float(yaw_diff_deg(a, b))) - 180.0) < 0.5


def test_yaw_nan_for_degenerate_pair():
    """Плечи слились в точку — угла нет, а не случайное число."""
    from looq.calib import yaw_from_pair_deg

    cam = _cam()
    p = _project([[15.0, 1.0, SHOULDER_M]])[0]
    assert not np.isfinite(yaw_from_pair_deg(cam, p, p, SHOULDER_M / TRUE_CAM_HEIGHT_M))


# --------------------------------------------------------------------------- #
# Запасной путь: плоскость земли по прямоугольнику мостовой
# --------------------------------------------------------------------------- #

def _ground_rect_px(w_m=4.0, h_m=2.0, x0=10.0, y0=-1.0):
    """Прямоугольный участок мостовой, углы в порядке TL, TR, BR, BL."""
    pts = _project([[x0, y0, 0.0], [x0 + w_m, y0, 0.0],
                    [x0 + w_m, y0 + h_m, 0.0], [x0, y0 + h_m, 0.0]])
    order = np.argsort(pts[:, 1])
    top, bot = pts[order[:2]], pts[order[2:]]
    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(-bot[:, 0])]
    return np.vstack([top, bot])


def test_ground_rect_homography_is_exact():
    from looq.calib import homography_from_ground_rect

    _, diag = homography_from_ground_rect(_ground_rect_px(), 0.5)
    assert diag["corner_residual_units"] < 1e-6


def test_ground_rect_rejects_bad_corner_order():
    """Углы снизу вверх — не прямоугольник в нужном порядке."""
    from looq.calib import homography_from_ground_rect

    rect = _ground_rect_px()
    flipped = rect[[3, 2, 1, 0]]
    with pytest.raises(CalibError, match="перепутаны"):
        homography_from_ground_rect(flipped, 0.5)


def test_ground_rect_rejects_bad_aspect():
    from looq.calib import homography_from_ground_rect

    with pytest.raises(CalibError, match="пропорция"):
        homography_from_ground_rect(_ground_rect_px(), 0.0)


def test_ground_plane_is_canonically_oriented():
    """План приводится к той же ориентации, что даёт камера из точек схода.

    Без этого план мог оказаться зеркальным, и КАЖДАЯ ориентация развернулась бы
    на 180 градусов. Ошибка выглядела бы правдоподобно и не поймалась бы ничем,
    кроме ручной разметки.
    """
    from looq.calib import (CANONICAL_PLANE_SIGN, _plane_orientation_sign,
                            homography_from_ground_rect)

    cam = _cam()
    probe = (FRAME_W / 2.0, FRAME_H - 130.0)
    assert _plane_orientation_sign(cam.H_px_to_unit, probe) == CANONICAL_PLANE_SIGN

    h, diag = homography_from_ground_rect(_ground_rect_px(), 0.5)
    assert _plane_orientation_sign(h, probe) == CANONICAL_PLANE_SIGN
    assert diag["plane_y_flipped"] is True, "в этой раскладке план обязан быть развёрнут"


def test_yaw_via_plane_matches_camera_method_exactly():
    """Два независимых способа получить угол обязаны совпадать на одном плане.

    Через горизонт плоскости — без K, R и без высоты плеч. Через камеру —
    с обратной проекцией на высоту. Расхождение означало бы ошибку в одном из них.
    """
    from looq.calib import yaw_from_pair_deg, yaw_from_pair_via_horizon
    from looq.geometry import yaw_diff_deg

    cam = _cam()
    z = SHOULDER_M / TRUE_CAM_HEIGHT_M
    for true_yaw in range(0, 360, 15):
        left, right = _shoulders_px(float(true_yaw))
        a = yaw_from_pair_via_horizon(cam.H_px_to_unit, left, right)
        b = yaw_from_pair_deg(cam, left, right, z)
        assert abs(float(yaw_diff_deg(a, b))) < 0.01, f"yaw={true_yaw}: {a} против {b}"


def test_yaw_on_ground_rect_plane_differs_by_pure_rotation():
    """На плане от прямоугольника углы отличаются ПОСТОЯННЫМ поворотом.

    Постоянство и есть проверка: зеркальность или сбой разрешения 180 градусов
    дали бы разброс, а не сдвиг.
    """
    from looq.calib import homography_from_ground_rect, yaw_from_pair_via_horizon
    from looq.geometry import yaw_diff_deg

    cam = _cam()
    h, _ = homography_from_ground_rect(_ground_rect_px(), 0.5)
    diffs = []
    for true_yaw in range(0, 360, 15):
        left, right = _shoulders_px(float(true_yaw))
        a = yaw_from_pair_via_horizon(h, left, right)
        b = yaw_from_pair_via_horizon(cam.H_px_to_unit, left, right)
        diffs.append(float(yaw_diff_deg(a, b)))
    assert float(np.ptp(diffs)) < 0.05, f"разброс разницы {np.ptp(diffs):.3f} град"


def test_horizon_from_h_matches_camera_horizon():
    from looq.calib import horizon_from_h

    cam = _cam()
    a = horizon_from_h(cam.H_px_to_unit)
    b = cam.horizon_line / np.hypot(cam.horizon_line[0], cam.horizon_line[1])
    if np.dot(a, b) < 0:
        b = -b
    assert np.allclose(a, b, atol=1e-6), f"{a} против {b}"
