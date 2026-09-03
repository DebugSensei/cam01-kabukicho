"""Тесты проверок обводки зон.

Клики получить в тесте нельзя, но все проверки геометрии — чистые функции,
и именно они решают, попадёт ли кривая обводка в пайплайн.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "pick_zones", Path(__file__).resolve().parent.parent / "scripts" / "pick_zones.py")
pick_zones = importlib.util.module_from_spec(_spec)
sys.modules["pick_zones"] = pick_zones
_spec.loader.exec_module(pick_zones)

#: Порядок углов: TL, TR, BR, BL — по часовой стрелке при оси y вниз.
GOOD = [[100.0, 200.0], [300.0, 200.0], [300.0, 400.0], [100.0, 400.0]]
ROI = [[0.0, 100.0], [1900.0, 100.0], [1900.0, 1000.0], [0.0, 1000.0]]


def test_good_quad_passes():
    assert pick_zones.check_polygon(GOOD, "M1") == []
    assert pick_zones.check_ground_segment(GOOD, "M1") == []
    assert pick_zones.check_inside_roi(GOOD, ROI, "M1") == []


def test_selfintersecting_bowtie_rejected():
    """Перепутаны BR и BL — полигон складывается бабочкой."""
    bowtie = [[100.0, 200.0], [300.0, 200.0], [100.0, 400.0], [300.0, 400.0]]
    problems = pick_zones.check_polygon(bowtie, "M2")
    assert problems and "невыпуклый или самопересекается" in problems[0]


def test_concave_quad_rejected():
    concave = [[100.0, 200.0], [300.0, 200.0], [150.0, 250.0], [100.0, 400.0]]
    assert pick_zones.check_polygon(concave, "M3")


def test_degenerate_collinear_rejected():
    flat = [[100.0, 200.0], [200.0, 200.0], [300.0, 200.0], [100.0, 400.0]]
    problems = pick_zones.check_polygon(flat, "M4")
    assert problems and "вырожден" in problems[0]


def test_upside_down_order_rejected():
    """Обвели снизу вверх: нижнее ребро оказалось бы верхним."""
    flipped = [[100.0, 400.0], [300.0, 400.0], [300.0, 200.0], [100.0, 200.0]]
    problems = pick_zones.check_polygon(flipped, "M1")
    assert any("Порядок углов перепутан" in p for p in problems)


def test_short_ground_edge_rejected():
    """Нижнее ребро короче 12 px — направление фасада по нему не определить."""
    narrow = [[100.0, 200.0], [108.0, 200.0], [108.0, 400.0], [100.0, 400.0]]
    problems = pick_zones.check_ground_segment(narrow, "M4")
    assert problems and "короче" in problems[0]


def test_ground_edge_is_bottom_not_top():
    """След на земле — именно нижнее ребро BL->BR, а не верхнее."""
    poly = [[100.0, 200.0], [500.0, 200.0], [400.0, 400.0], [120.0, 400.0]]
    bl, br = poly[3], poly[2]
    assert bl[1] > poly[0][1] and br[1] > poly[1][1]
    assert pick_zones.check_ground_segment(poly, "M1") == []


def test_zone_outside_roi_horizontally_rejected():
    far_right = [[1950.0, 200.0], [2150.0, 200.0], [2150.0, 400.0], [1950.0, 400.0]]
    problems = pick_zones.check_inside_roi(far_right, ROI, "M2")
    assert problems and "вне ROI" in problems[0]


def test_overlapping_zones_rejected():
    a = {"id": "M1", "polygon_px": GOOD}
    b = {"id": "M2", "polygon_px": [[150.0, 200.0], [350.0, 200.0],
                                    [350.0, 400.0], [150.0, 400.0]]}
    problems = pick_zones.check_overlaps([a, b])
    assert problems and "пересекаются" in problems[0]


def test_touching_zones_allowed():
    """Соседние витрины стоят вплотную — это норма, а не наложение."""
    a = {"id": "M1", "polygon_px": GOOD}
    b = {"id": "M2", "polygon_px": [[300.0, 200.0], [500.0, 200.0],
                                    [500.0, 400.0], [300.0, 400.0]]}
    assert pick_zones.check_overlaps([a, b]) == []


def test_validate_reports_all_problems_at_once():
    """Перекликивать по одной проблеме долго — выдаём весь список сразу."""
    zones = [
        {"id": "M1", "polygon_px": GOOD},
        {"id": "M2", "polygon_px": [[100.0, 200.0], [108.0, 200.0],
                                    [108.0, 400.0], [100.0, 400.0]]},
        {"id": "M3", "polygon_px": [[1950.0, 200.0], [2150.0, 200.0],
                                    [2150.0, 400.0], [1950.0, 400.0]]},
    ]
    with pytest.raises(SystemExit) as exc:
        pick_zones.validate(zones, ROI)
    text = str(exc.value)
    assert "M2" in text and "M3" in text, f"часть проблем потерялась: {text}"


@pytest.mark.parametrize("name,expected", [
    ("M1 角煮/げんかつ", "M1"),
    ("M2 入口/らーめん", "M2"),
    ("芝浦ホルモン", "Z9"),
    ("", "Z9"),
])
def test_ascii_label(name, expected):
    """cv2.putText кириллицу и кандзи не рисует — на экран идёт ASCII-токен."""
    assert pick_zones.ascii_label(name, "Z9") == expected
