"""Артефакт несёт sha своих входов, и рассогласование обязано быть провалом.

Дефект, ради которого это написано: S4 и S5 читали calib/homography.json и
нигде не фиксировали, КАКУЮ калибровку прочитали. После перекалибровки метры
в track/ и pose/ оставались от старой гомографии, а числа выглядели
нормальными. Молчаливая порча хуже падения (правило 8).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.stages._base import (Col, check_inputs_sha,  # noqa: E402
                               read_inputs_sha, write_parquet)

COLS = [Col("a", "int64", False, "-", "-")]
ROWS = [{"a": 1}, {"a": 2}]


def _artifact(tmp_path: Path, inputs):
    """Пишет артефакт из tmp_path, потому что sha считается по ОТНОСИТЕЛЬНЫМ
    путям входов — как это делают стадии."""
    out = tmp_path / "out.parquet"
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_parquet(out, COLS, ROWS, "test_stage", "ok", inputs=inputs)
    finally:
        os.chdir(cwd)
    return out


def test_artifact_records_sha_of_every_input(tmp_path):
    src = tmp_path / "src.json"
    src.write_text('{"H": 1}', encoding="utf-8")
    out = _artifact(tmp_path, ["src.json"])

    rec = read_inputs_sha(out)
    assert rec is not None, "артефакт обязан нести sha входов"
    assert set(rec) == {"src.json"}
    assert isinstance(rec["src.json"], str) and len(rec["src.json"]) == 64


def test_matching_inputs_pass(tmp_path):
    src = tmp_path / "src.json"
    src.write_text('{"H": 1}', encoding="utf-8")
    out = _artifact(tmp_path, ["src.json"])

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ok, problems = check_inputs_sha(out)
    finally:
        os.chdir(cwd)
    assert ok, problems


def test_changed_input_is_a_failure(tmp_path):
    """Тот самый случай: перекалибровали, а артефакт остался от старой."""
    src = tmp_path / "src.json"
    src.write_text('{"H": 1}', encoding="utf-8")
    out = _artifact(tmp_path, ["src.json"])

    src.write_text('{"H": 2}', encoding="utf-8")      # перекалибровка

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ok, problems = check_inputs_sha(out)
    finally:
        os.chdir(cwd)
    assert not ok, "изменившийся вход обязан валить проверку"
    assert any("изменился после прогона" in q for q in problems), problems


def test_missing_input_is_a_failure(tmp_path):
    src = tmp_path / "src.json"
    src.write_text('{"H": 1}', encoding="utf-8")
    out = _artifact(tmp_path, ["src.json"])
    src.unlink()

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ok, problems = check_inputs_sha(out)
    finally:
        os.chdir(cwd)
    assert not ok
    assert any("больше нет на диске" in q for q in problems), problems


def test_artifact_without_recorded_sha_is_a_failure(tmp_path):
    """Молчание — не «ок». Артефакт без записанных входов не подтверждён."""
    out = tmp_path / "bare.parquet"
    write_parquet(out, COLS, ROWS, "test_stage", "ok")     # inputs не переданы

    assert read_inputs_sha(out) is None
    ok, problems = check_inputs_sha(out)
    assert not ok
    assert any("не записан" in q for q in problems), problems


@pytest.mark.parametrize("module,artifact", [
    ("looq.stages.s4_track", "track/tracks.parquet"),
    ("looq.stages.s5_orient", "pose/orient.parquet"),
    ("looq.stages.s6_attn", "attn/events.parquet"),
    ("looq.stages.s7_attrs", "attr/tracks_attr.parquet"),
])
def test_stage_declares_its_inputs(module, artifact):
    """Каждая стадия, читающая чужой артефакт, обязана объявить INPUTS.

    Без объявления sha записать не из чего, и проверка выше становится
    бессмысленной: она бы молча проходила на пустом списке.
    """
    import importlib

    mod = importlib.import_module(module)
    inputs = getattr(mod, "INPUTS", None)
    assert inputs, f"{module} не объявляет INPUTS"
    assert all(isinstance(q, str) and q for q in inputs)
