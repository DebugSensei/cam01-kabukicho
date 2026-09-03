"""Тесты самокалибровки по пешеходам.

Реальных людей в тесте нет, поэтому проверяем на синтетической камере
с известными параметрами: она проецирует людей известного роста, а калибровка
обязана восстановить фокус. Если математика врёт, это видно здесь, а не через
час на клипе.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_calib import (FRAME_SHAPE, TRUE_FOCAL, _project,  # noqa: E402
                        _project_direction)

from looq.calib import CalibError  # noqa: E402
from looq.selfcalib import (focal_from_horizon_and_vertical,  # noqa: E402
                            horizon_from_pedestrian_pairs,
                            point_on_line_distance_px,
                            vertical_vp_from_pedestrians)


def _crowd(n_frames=120, per_frame=6, sigma=0.07, seed=0):
    """Люди роста N(1.68, sigma) в разных местах кадра."""
    rng = np.random.default_rng(seed)
    feet, heads, fidx = [], [], []
    for f in range(n_frames):
        for _ in range(per_frame):
            h = rng.normal(1.68, sigma)
            x = rng.uniform(6.0, 34.0)
            y = rng.uniform(-2.6, 2.6)
            feet.append(_project([[x, y, 0.0]])[0])
            heads.append(_project([[x, y, h]])[0])
            fidx.append(f)
    return np.asarray(feet), np.asarray(heads), np.asarray(fidx)


#: Измеренная точность горизонта: ошибка растёт ЛИНЕЙНО с удалением точки схода
#: от центра кадра. Синтетика, разброс роста 0.07 м:
#:     |VP-центр|  566 px -> ошибка   4.7 px
#:                 943 px ->         22.4 px
#:                1612 px ->         40.1 px
#:                4565 px ->        110.9 px
#: То есть примерно 0.024 * расстояние. Фиксированный порог в пикселях здесь
#: бессмыслен: для близкой точки схода он слишком мягок, для далёкой запретителен.
HORIZON_ERR_PER_PX = 0.024


def test_horizon_error_scales_with_vp_distance():
    """Горизонт проходит через точки схода наземных направлений.

    Допуск масштабируется с удалением точки схода: это измеренное свойство
    метода, а не догадка. См. HORIZON_ERR_PER_PX.
    """
    feet, heads, fidx = _crowd()
    hor = horizon_from_pedestrian_pairs(feet, heads, fidx)
    centre = np.array([FRAME_SHAPE[1] / 2.0, FRAME_SHAPE[0] / 2.0])
    for direction in ([1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0]):
        vp = _project_direction(direction)
        if not np.all(np.isfinite(vp)):
            continue
        d = point_on_line_distance_px(hor.line, vp)
        budget = 1.5 * HORIZON_ERR_PER_PX * float(np.linalg.norm(vp - centre))
        assert d < budget, (f"направление {direction}: {d:.1f} px при бюджете "
                            f"{budget:.1f} px")


def test_horizon_needs_depth_separation():
    """Пары на одной глубине вырождены и должны отбрасываться."""
    feet, heads, fidx = _crowd()
    with pytest.raises(CalibError):
        # Требуем разнесения больше кадра — годных пар не останется.
        horizon_from_pedestrian_pairs(feet, heads, fidx, min_depth_sep_px=5000.0)


def test_vertical_vp_recovered():
    """Вертикальная точка схода восстанавливается по отрезкам стопа-макушка."""
    feet, heads, _ = _crowd()
    res = vertical_vp_from_pedestrians(feet, heads)
    true_vp = _project_direction([0.0, 0.0, 1.0])
    err = float(np.linalg.norm(res.vp_px - true_vp))
    assert err < 0.05 * abs(true_vp[1]), f"ошибка {err:.0f} px при VP {true_vp.round(0)}"


def test_vertical_vp_rejects_bbox_style_points():
    """Точки из ЦЕНТРА РАМКИ вертикальны по построению — метод обязан падать.

    Это не гипотетический случай: на реальных детекциях так и было, и RANSAC
    не находил ни одного кандидата. Ошибка должна называть причину.
    """
    feet, heads, _ = _crowd()
    heads_bbox = heads.copy()
    heads_bbox[:, 0] = feet[:, 0]      # тот же x, как у bbox
    with pytest.raises(CalibError, match="ЦЕНТРА РАМКИ"):
        vertical_vp_from_pedestrians(feet, heads_bbox)


def test_focal_recovered_from_horizon_and_vertical():
    """Фокус из соотношения полюс-поляра. Реалистичный разброс роста."""
    feet, heads, fidx = _crowd(sigma=0.07)
    hor = horizon_from_pedestrian_pairs(feet, heads, fidx)
    vert = vertical_vp_from_pedestrians(feet, heads)
    f, diag = focal_from_horizon_and_vertical(hor.line, vert.vp_px, FRAME_SHAPE)
    assert abs(f / TRUE_FOCAL - 1.0) < 0.05, f"фокус {f:.1f} против {TRUE_FOCAL}"
    assert diag["d_horizon_px"] > 0 and diag["d_vertical_vp_px"] > 0


def test_focal_exact_when_heights_identical():
    """При одинаковом росте допущение метода выполняется точно."""
    feet, heads, fidx = _crowd(sigma=0.0)
    hor = horizon_from_pedestrian_pairs(feet, heads, fidx)
    vert = vertical_vp_from_pedestrians(feet, heads)
    f, _ = focal_from_horizon_and_vertical(hor.line, vert.vp_px, FRAME_SHAPE)
    assert abs(f / TRUE_FOCAL - 1.0) < 0.01


def test_focal_rejects_inconsistent_pair():
    """Горизонт и вертикаль, не образующие полюс-поляру, отвергаются.

    Раньше здесь стояла покомпонентная формула, и она такую пару молча
    принимала, выдавая правдоподобный, но неверный фокус.
    """
    feet, heads, _ = _crowd()
    vert = vertical_vp_from_pedestrians(feet, heads)
    # Горизонт по ту же сторону от главной точки, что и вертикальная VP.
    bad = np.array([0.0, -1.0, FRAME_SHAPE[0] / 2.0 + 300.0])
    with pytest.raises(CalibError):
        focal_from_horizon_and_vertical(bad, vert.vp_px, FRAME_SHAPE)


def test_focal_formula_is_well_conditioned_for_level_camera():
    """У камеры без крена горизонт почти горизонтален.

    Покомпонентная формула f^2 = c'*v_x/a там взрывается: a близко к нулю.
    Замер до исправления давал ошибку фокуса 30 % при ТОЧНОЙ вертикальной VP.
    Тест держит исправленную форму через произведение расстояний.
    """
    vert_vp = _project_direction([0.0, 0.0, 1.0])
    true_horizon_vp = _project_direction([1.0, 0.0, 0.0])
    # Строго горизонтальная линия через истинную точку схода улицы.
    horizon = np.array([0.0, -1.0, float(true_horizon_vp[1])])
    f, _ = focal_from_horizon_and_vertical(horizon, vert_vp, FRAME_SHAPE)
    assert abs(f / TRUE_FOCAL - 1.0) < 0.02, f"фокус {f:.1f} против {TRUE_FOCAL}"
