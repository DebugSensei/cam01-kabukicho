# -*- coding: utf-8 -*-
"""Ссылки вида `файл.py:строка` в README обязаны указывать на то, о чём текст.

Правило 1 CLAUDE.md: «я должен уметь показать пальцем строчку, которая его
считает». Ссылка, уехавшая на пустую строку, ломает не оформление, а само
правило: показать пальцем становится нечего.

Проверено, что это не гипотеза. Аудит 2026-09-06 нашёл три уехавшие ссылки
из девяти: `render_overlay.py:122` вместо 183 (`PlanView._swap`),
`s4_track.py:164` вместо 165 (`_speed_series`) и `make_dashboard.py:928`
вместо 934 (`b64_img`) — последняя уехала от правок этого же дня.

Тест не проверяет смысл, он **закрепляет якорь**: для каждой ссылки записан
кусок строки, на которую она обязана указывать. Код сдвинулся — тест падает,
и человек решает, куда ссылка должна вести. Это дешевле, чем перечитывать
README после каждой правки, и надёжнее, чем надеяться.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

#: file:line -> кусок строки, на которую ссылка обязана указывать.
#: Обновляется ОСОЗНАННО вместе со ссылкой в README, а не автоматически:
#: автообновление превратило бы гейт в переписывание ожидания под факт.
ANCHORS = {
    "looq/stages/s1_calib.py:211": "Наклон НЕ пересчитывается",
    "looq/stages/s1_calib.py:309": '_pedestrian_boxes("det/frames.parquet"',
    "looq/stages/s4_track.py:165": "def _speed_series",
    "looq/stages/s6_attn.py:91": "фасады вышли по 0.5 м",
    "looq/stages/s6_attn.py:168": "Единицы зон и треков обязаны совпадать",
    "looq/stages/s6_attn.py:290": "делилось на ВСЕ кадры трека",
    "scripts/make_replay.py:549": "hits_by_frame",
    "scripts/make_dashboard.py:955": "def b64_img",
    "scripts/render_overlay.py:183": "def _swap",
}


def _refs() -> list[tuple[str, int]]:
    """Все `путь.py:N` из README, включая форму `путь.py:N,M`."""
    out = []
    for m in re.finditer(r"`([\w/]+\.py):([\d,]+)`", README.read_text(encoding="utf-8")):
        for ln in m.group(2).split(","):
            out.append((m.group(1), int(ln)))
    return out


def test_every_line_reference_exists():
    """Ни одна ссылка не указывает за конец файла или в несуществующий файл."""
    bad = []
    for f, ln in _refs():
        p = ROOT / f
        if not p.is_file():
            bad.append(f"{f}:{ln} — файла нет")
            continue
        n = len(p.read_text(encoding="utf-8").splitlines())
        if ln > n:
            bad.append(f"{f}:{ln} — за концом файла ({n} строк)")
    assert not bad, "битые ссылки в README: " + "; ".join(bad)


def test_every_line_reference_still_points_at_its_anchor():
    """Ссылка указывает на ту строку, ради которой она написана.

    Существование строки ничего не доказывает: `render_overlay.py:122`
    существовала и указывала на текст ошибки о весах позы, а абзац был про
    отражение плана.
    """
    drifted = []
    for f, ln in _refs():
        key = f"{f}:{ln}"
        if key not in ANCHORS:
            drifted.append(f"{key} — новая ссылка без якоря в ANCHORS")
            continue
        line = (ROOT / f).read_text(encoding="utf-8").splitlines()[ln - 1]
        if ANCHORS[key] not in line:
            drifted.append(f"{key} — ожидалось {ANCHORS[key]!r}, там {line.strip()[:60]!r}")
    assert not drifted, (
        "ссылки в README уехали от кода, на который ссылаются:\n  "
        + "\n  ".join(drifted))


def test_anchors_cover_every_reference():
    """В ANCHORS нет мусора: каждый якорь соответствует живой ссылке."""
    refs = {f"{f}:{ln}" for f, ln in _refs()}
    stale = sorted(set(ANCHORS) - refs)
    assert not stale, f"якоря без ссылки в README (ссылку убрали, якорь забыли): {stale}"
