"""Тесты провенанса прогонов.

Манифест обязан честно отражать состояние: запись, оставшаяся в running после
убитого прогона, утверждает, что этап выполняется, и это хуже отсутствия
манифеста — такой провенанс нельзя предъявлять.
"""

from __future__ import annotations

from looq.io import RunManifest, read_json

CFG = {"_config_path": "configs/x.yaml", "_config_sha256": "abc", "model": None}


def test_manifest_records_start_and_finish(tmp_path):
    path = tmp_path / "run_manifest.json"
    m = RunManifest("s3_detect", CFG, path)
    m.start()
    assert read_json(path)["stages"]["s3_detect"]["status"] == "running"
    m.note("throughput_fps", 16.6)
    m.finish("ok")
    entry = read_json(path)["stages"]["s3_detect"]
    assert entry["status"] == "ok"
    assert entry["finished_at"] is not None
    assert entry["notes"]["throughput_fps"] == 16.6


def test_killed_run_is_marked_aborted_not_running(tmp_path):
    """Убитый прогон не должен вечно выглядеть выполняющимся."""
    path = tmp_path / "run_manifest.json"
    RunManifest("s3_detect", CFG, path).start()      # прогон "убит": finish не вызван
    assert read_json(path)["stages"]["s3_detect"]["status"] == "running"

    RunManifest("s3_detect", CFG, path).start()      # следующий запуск этапа
    doc = read_json(path)
    assert doc["stages"]["s3_detect"]["status"] == "running"   # это уже новый прогон
    assert len(doc["aborted_runs"]) == 1, "оборванный прогон не зафиксирован"
    assert doc["aborted_runs"][0]["stage"] == "s3_detect"


def test_other_stages_untouched_by_abort_detection(tmp_path):
    """Пометка оборванного прогона трогает только свой этап."""
    path = tmp_path / "run_manifest.json"
    other = RunManifest("s1_calib", CFG, path)
    other.start()
    other.finish("ok")
    RunManifest("s3_detect", CFG, path).start()
    RunManifest("s3_detect", CFG, path).start()
    doc = read_json(path)
    assert doc["stages"]["s1_calib"]["status"] == "ok"
    assert all(a["stage"] == "s3_detect" for a in doc["aborted_runs"])


def test_finish_records_error(tmp_path):
    path = tmp_path / "run_manifest.json"
    m = RunManifest("s4_track", CFG, path)
    m.start()
    m.finish("failed", error="нет входного артефакта")
    entry = read_json(path)["stages"]["s4_track"]
    assert entry["status"] == "failed"
    assert "нет входного артефакта" in entry["error"]
