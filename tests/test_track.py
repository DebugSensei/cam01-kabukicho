"""Тесты трекинга S4 и записи parquet по контракту.

ByteTrack применяется к СОХРАНЁННЫМ детекциям S3, а не к кадрам, поэтому его
можно проверить синтетикой без видео и без модели.
"""

from __future__ import annotations

import numpy as np
import pytest

from looq.stages._base import Col, StageError, write_parquet
from looq.stages.s4_track import _bytetrack_over_detections, _speed_series

TRACK_CFG = {"track_high_thresh": 0.5, "track_low_thresh": 0.1,
             "new_track_thresh": 0.6, "track_buffer_frames": 30, "match_thresh": 0.8}


def _two_people(n=40):
    """Двое идут навстречу друг другу, бокс 60x160."""
    return {f: np.array([[200 + 8 * f, 500, 260 + 8 * f, 660, 0.90, 0],
                         [1400 - 8 * f, 520, 1460 - 8 * f, 680, 0.85, 0]],
                        dtype=np.float64) for f in range(n)}


def test_bytetrack_keeps_two_ids():
    det = _two_people()
    tracked = _bytetrack_over_detections(det, list(range(40)), (1080, 1920), TRACK_CFG)
    ids = sorted({b["track_id"] for v in tracked.values() for b in v})
    assert len(ids) == 2, f"ожидалось два трека, получено {ids}"
    for tid in ids:
        n = sum(1 for v in tracked.values() for b in v if b["track_id"] == tid)
        assert n == 40, f"трек {tid} прожил {n} кадров вместо 40"


def test_bytetrack_survives_empty_frames():
    """Кадр без детекций не должен ронять трекер и не создаёт треков."""
    det = _two_people()
    det[20] = np.zeros((0, 6))
    tracked = _bytetrack_over_detections(det, list(range(40)), (1080, 1920), TRACK_CFG)
    assert 20 not in tracked or not tracked[20]
    assert len({b["track_id"] for v in tracked.values() for b in v}) == 2


def test_speed_is_least_squares_not_neighbour_diff():
    """МНК даёт точную скорость; на краях окна — null, а не выдуманное число."""
    frames = np.arange(40, dtype=float)
    xs, ys = 1.3 * frames / 30.0, np.zeros(40)
    sp = _speed_series(frames, xs, ys, 30.0, window=9)
    assert np.nanmedian(sp) == pytest.approx(1.3, abs=1e-6)
    assert int(np.isnan(sp).sum()) == 8, "на краях окна должно быть ровно 8 null"


def test_speed_null_when_ground_point_missing():
    """Дыра в проекции не заполняется интерполяцией — окно даёт null."""
    frames = np.arange(20, dtype=float)
    xs, ys = 1.3 * frames / 30.0, np.zeros(20)
    xs[10] = np.nan
    sp = _speed_series(frames, xs, ys, 30.0, window=9)
    assert np.isnan(sp[10])


# --------------------------------------------------------------------------- #
# write_parquet: схема контракта, а не то, что оказалось в данных
# --------------------------------------------------------------------------- #

COLS = [Col("a", "int64", False, "-", "-"), Col("b", "string", True, "-", "-")]


def test_write_parquet_roundtrip(tmp_path):
    import pyarrow.parquet as pq

    p = write_parquet(tmp_path / "x.parquet", COLS,
                      [{"a": 1, "b": "x"}, {"a": 2, "b": None}], "s3_detect", "ok")
    table = pq.read_table(p)
    assert table.column_names == ["a", "b"]
    assert table.num_rows == 2
    assert pq.read_schema(p).metadata[b"status"] == b"ok"


@pytest.mark.parametrize("rows,why", [
    ([{"a": 1}], "нет обязательной колонки"),
    ([{"a": 1, "b": "x", "c": 9}], "ключ вне схемы"),
    ([{"a": None, "b": "x"}], "None в not-null колонке"),
])
def test_write_parquet_rejects_schema_drift(tmp_path, rows, why):
    """Молча дописать или подставить null значило бы разойтись с контрактом."""
    with pytest.raises(StageError):
        write_parquet(tmp_path / "bad.parquet", COLS, rows, "s3_detect", "ok")
