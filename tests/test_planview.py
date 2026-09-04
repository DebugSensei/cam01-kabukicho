"""План не должен зеркалить сцену.

Регресс от 2026-09-04: PlanView._swap делал p[:, ::-1] — транспозицию, то
есть отражение относительно диагонали (определитель -1), а не поворот.
Витрины уезжали направо, хотя в кадре они слева, и заказчик это увидел.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.fixture(scope="module")
def PlanView():
    pytest.importorskip("cv2")
    from render_overlay import PlanView as PV
    return PV


def test_rotate_is_a_rotation_not_a_reflection(PlanView):
    """Ориентация плоскости обязана сохраняться."""
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    pv = PlanView(400, 400, pts, pad=0.0, rotate=True)
    o, ex, ey = pv.to_px([0, 0])[0], pv.to_px([1, 0])[0], pv.to_px([0, 1])[0]

    # В координатах зрителя ось y экрана смотрит ВНИЗ, поэтому знак площади
    # переворачивается: правая тройка на плане обязана дать ОТРИЦАТЕЛЬНУЮ
    # ориентированную площадь в пикселях.
    dx, dy = (ex - o).astype(float), (ey - o).astype(float)
    cross = dx[0] * dy[1] - dx[1] * dy[0]
    assert cross < 0, ("план отражён: поворот от +x к +y на экране идёт "
                       "в обратную сторону")


def test_plus_y_goes_left_when_street_runs_up(PlanView):
    """+y плана — это сторона витрин, и на канве она обязана быть слева.

    Замер по сохранённой H: шаг +y_m уводит в кадре влево на 95.6 px.
    Значит и на плане большая y должна оказаться левее.
    """
    pts = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 10.0], [30.0, 10.0]])
    pv = PlanView(400, 900, pts, pad=0.0, rotate=True)
    near_road = pv.to_px([15.0, 1.0])[0]
    near_shops = pv.to_px([15.0, 9.0])[0]
    assert near_shops[0] < near_road[0], "витрины нарисованы справа от дороги"


def test_street_runs_along_the_tall_axis(PlanView):
    """Улица длиной 35 м обязана лечь вдоль высоты, а не в ленту по ширине."""
    pts = np.array([[0.0, 0.0], [35.0, 0.0], [0.0, 12.0], [35.0, 12.0]])
    pv = PlanView(430, 1080, pts, pad=0.0, rotate=True)
    a, b = pv.to_px([0.0, 6.0])[0], pv.to_px([35.0, 6.0])[0]
    assert abs(b[1] - a[1]) > abs(b[0] - a[0])


def test_no_rotate_preserves_axes(PlanView):
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    pv = PlanView(400, 400, pts, pad=0.0, rotate=False)
    o, ex = pv.to_px([0, 0])[0], pv.to_px([1, 0])[0]
    assert ex[0] > o[0], "без разворота +x обязан идти вправо"
