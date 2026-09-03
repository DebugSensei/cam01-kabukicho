"""Тесты модуля пруфов.
from pathlib import Path

Проверяем ровно три вещи, на которых модуль стоит:
  1. обезличивание реально применилось и неразмытое на диск не попало;
  2. индекс пишется и совпадает с контрактом;
  3. отбор стратифицированный, а не top-N.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from looq.evidence import (
    DEFAULT_PIXELATE_FACTOR,
    INDEX_COLUMNS,
    EvidenceError,
    EvidenceSampler,
    EvidenceWriter,
    blur_face_region,
)

CROP_H, CROP_W = 200, 80
TOP_FRAC = 0.30


def _noisy_crop(seed: int = 0) -> np.ndarray:
    """Кроп с высокочастотным шумом сверху и гладкой заливкой снизу.

    Шум сверху — потому что размытие видно именно по падению высоких частот.
    Гладкий низ — контроль: там ничего меняться не должно.
    """
    rng = np.random.default_rng(seed)
    crop = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
    head_h = int(round(TOP_FRAC * CROP_H))
    crop[:head_h] = rng.integers(0, 256, size=(head_h, CROP_W, 3), dtype=np.uint8)
    crop[head_h:] = 128
    return crop


def _hf_energy(img: np.ndarray) -> float:
    """Энергия высоких частот: средний квадрат градиента по обеим осям."""
    a = img.astype(np.float64)
    gy = np.diff(a, axis=0)
    gx = np.diff(a, axis=1)
    return float((gy ** 2).mean() + (gx ** 2).mean())


def _block_residual(img: np.ndarray, factor: int = DEFAULT_PIXELATE_FACTOR) -> float:
    """Сколько информации в изображении лежит МЕЛЬЧЕ блока пикселизации.

    Для пикселизованной картинки остаток близок к нулю: внутри блока всё уже
    однородно. Для исходного шума остаток большой.
    """
    import cv2

    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)),
                       interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return float(np.abs(img.astype(np.float64) - back.astype(np.float64)).mean())


def _writer(tmp_path, **kw) -> EvidenceWriter:
    kw.setdefault("stage", "s3_detect")
    kw.setdefault("model_name", "test-model")
    return EvidenceWriter(root=tmp_path / "evidence", **kw)


# --------------------------------------------------------------------------- #
# 1. Обезличивание
# --------------------------------------------------------------------------- #

def test_blur_applied_to_top_region(tmp_path):
    """Верхние 22% изменились, остальное — нет."""
    import cv2

    crop = _noisy_crop()
    head_h = int(round(TOP_FRAC * CROP_H))

    w = _writer(tmp_path)
    rel = w.add("claim.test", track_id=7, frame_idx=42, ts=1.5,
                crop_bgr=crop, value=0.9, confidence=0.8)

    assert rel.endswith("claim.test/7_42.jpg"), rel
    written = cv2.imread(rel)
    assert written is not None, f"jpeg не читается с диска: {rel}"
    assert written.shape == crop.shape

    before = _hf_energy(crop[:head_h])
    after = _hf_energy(written[:head_h])
    assert after < before / 5.0, (
        f"верхняя область почти не изменилась: было {before:.1f}, стало {after:.1f}"
    )

    # Низ кропа не тронут (с допуском на артефакты jpeg).
    assert np.abs(written[head_h:].astype(np.int16)
                  - crop[head_h:].astype(np.int16)).max() <= 12


def test_blur_does_not_mutate_caller_array(tmp_path):
    """Массив вызывающей стороны остаётся прежним."""
    crop = _noisy_crop(seed=1)
    original = crop.copy()
    _writer(tmp_path).add("claim.test", 1, 1, 0.0, crop, 1.0, 0.5)
    assert np.array_equal(crop, original)


def test_blur_kernel_scales_with_crop_size():
    """Ядро пропорционально размеру кропа, а не константа."""
    _, small = blur_face_region(np.full((40, 20, 3), 100, np.uint8))
    _, large = blur_face_region(np.full((400, 200, 3), 100, np.uint8))
    assert large["blur_kernel_px"] > small["blur_kernel_px"]
    assert small["blur_kernel_px"] % 2 == 1 and large["blur_kernel_px"] % 2 == 1
    assert large["blur_head_h_px"] == round(TOP_FRAC * 400)


def test_pixelation_destroys_subblock_detail(tmp_path):
    """Пикселизация, а не только гаусс: информации мельче блока не остаётся.

    Чистая гауссиана — обратимая свёртка, деблюр по ней восстанавливает лицо.
    Проверяем именно необратимую часть: остаток относительно сетки блоков.
    """
    import cv2

    crop = _noisy_crop(seed=3)
    head_h = int(round(TOP_FRAC * CROP_H))
    rel = _writer(tmp_path).add("claim.test", 3, 3, 0.0, crop, 1.0, 0.5)
    written = cv2.imread(rel)

    before = _block_residual(crop[:head_h])
    after = _block_residual(written[:head_h])
    assert before > 20.0, f"контроль: в исходном шуме должен быть остаток, {before:.1f}"
    assert after < before / 10.0, (
        f"под сеткой блоков осталась структура: было {before:.1f}, стало {after:.1f} — "
        f"похоже, применён только гаусс"
    )


def test_pixelate_factor_must_be_meaningful():
    """Отключить пикселизацию нельзя: остался бы только обратимый гаусс."""
    with pytest.raises(EvidenceError, match="pixelate_factor"):
        blur_face_region(np.full((100, 50, 3), 7, np.uint8), pixelate_factor=1)


def test_blur_params_recorded_in_extra(tmp_path):
    """Параметры обезличивания попадают в индекс — их можно проверить, а не поверить."""
    w = _writer(tmp_path)
    w.add("claim.test", 1, 1, 0.0, _noisy_crop(), 1.0, 0.5, extra={"note": "x"})
    payload = json.loads(w.rows[0]["extra_json"])
    assert payload["note"] == "x"
    assert payload["blur_top_frac"] == pytest.approx(TOP_FRAC)
    assert payload["blur_kernel_px"] >= 5
    assert payload["blur_pixelate_factor"] == DEFAULT_PIXELATE_FACTOR
    assert len(payload["blur_pixel_block_px"]) == 2


def test_bad_crop_rejected(tmp_path):
    """Не-BGR кроп на диск не попадает."""
    w = _writer(tmp_path)
    with pytest.raises(EvidenceError):
        w.add("claim.test", 1, 1, 0.0, np.zeros((10, 10), np.uint8), 1.0, 0.5)
    assert not list((tmp_path / "evidence").rglob("*.jpg"))


def test_duplicate_key_rejected(tmp_path):
    """Повторный пруф не перетирает предыдущий молча."""
    w = _writer(tmp_path)
    w.add("claim.test", 1, 1, 0.0, _noisy_crop(), 1.0, 0.5)
    with pytest.raises(EvidenceError):
        w.add("claim.test", 1, 1, 0.0, _noisy_crop(), 1.0, 0.5)


# --------------------------------------------------------------------------- #
# 2. Индекс
# --------------------------------------------------------------------------- #

def test_index_written(tmp_path):
    import pandas as pd

    w = _writer(tmp_path)
    for i in range(5):
        w.add("claim.test", track_id=i, frame_idx=100 + i, ts=float(i),
              crop_bgr=_noisy_crop(seed=i), value=0.5 * i, confidence=0.1 * i)
    path = w.finalize()

    assert path.is_file()
    df = pd.read_parquet(path)
    assert list(df.columns) == list(INDEX_COLUMNS)
    assert len(df) == 5
    assert set(df["stage"]) == {"s3_detect"}
    assert set(df["model_name"]) == {"test-model"}
    for rel in df["path"]:
        assert (tmp_path.parent / rel).is_file() or (tmp_path / rel).is_file() \
            or (tmp_path / "evidence").joinpath(*rel.split("/")[1:]).is_file()


def test_empty_index_is_not_success(tmp_path):
    """Пустой индекс — ошибка, а не успех (правило 8)."""
    with pytest.raises(EvidenceError):
        _writer(tmp_path).finalize()
    assert _writer(tmp_path).finalize(allow_empty=True).is_file()


# --------------------------------------------------------------------------- #
# 3. Стратификация
# --------------------------------------------------------------------------- #

def _fill_sampler(tmp_path, n_cand=90, seed=20260903):
    w = _writer(tmp_path)
    s = EvidenceSampler(w, n_per_claim=12, n_strata=3, seed=seed)
    s.declare(["claim.strat"])
    confs = np.linspace(0.01, 0.99, n_cand)
    for i, c in enumerate(confs):
        s.offer("claim.strat", track_id=i, frame_idx=i, ts=float(i),
                crop_bgr=_noisy_crop(seed=i % 7), value=float(c), confidence=float(c))
    return w, s, confs


def test_stratified_selection_three_levels(tmp_path):
    import pandas as pd

    w, s, confs = _fill_sampler(tmp_path)
    path = s.finalize(n=12)
    df = pd.read_parquet(path)

    assert len(df) == 12
    assert sorted(df["stratum"].tolist()) == [0] * 4 + [1] * 4 + [2] * 4

    lo = df.loc[df["stratum"] == 0, "confidence"]
    mid = df.loc[df["stratum"] == 1, "confidence"]
    hi = df.loc[df["stratum"] == 2, "confidence"]

    # Три РАЗНЫХ уровня уверенности, страты не пересекаются.
    assert lo.max() < mid.min() < mid.max() < hi.min()
    assert lo.mean() < mid.mean() < hi.mean()

    # Выборка накрывает весь диапазон, а не только верх.
    assert df["confidence"].min() < 0.34
    assert df["confidence"].max() > 0.66


def test_selection_is_not_top_n(tmp_path):
    """Ключевое: это НЕ top-N."""
    import pandas as pd

    w, s, confs = _fill_sampler(tmp_path)
    df = pd.read_parquet(s.finalize(n=12))

    top12 = set(np.round(sorted(confs)[-12:], 9))
    got = set(np.round(sorted(df["confidence"]), 9))
    assert got != top12, "отбор совпал с top-12 — стратификация не работает"
    assert len(got & top12) <= 4, "в выборке слишком много верхушки"


def test_sampling_stats_report_shortfall(tmp_path):
    """Недобор не проглатывается (правило 7)."""
    w = _writer(tmp_path)
    s = EvidenceSampler(w, n_per_claim=12, n_strata=3)
    s.declare(["claim.few"])
    for i in range(5):
        s.offer("claim.few", i, i, float(i), _noisy_crop(seed=i), 1.0, 0.1 * (i + 1))
    s.finalize(n=12)
    st = s.stats()["claim.few"]
    assert st["n_selected"] == 5
    assert st["shortfall"] == 7
    assert st["n_offered"] == 5


def test_declared_claim_without_evidence_fails(tmp_path):
    """Объявленный claim без пруфов роняет этап, а не тихо исчезает."""
    s = EvidenceSampler(_writer(tmp_path))
    s.declare(["claim.promised"])
    with pytest.raises(EvidenceError, match="claim.promised"):
        s.finalize()


def test_selection_is_deterministic(tmp_path):
    import pandas as pd

    _, s1, _ = _fill_sampler(tmp_path / "a", seed=777)
    _, s2, _ = _fill_sampler(tmp_path / "b", seed=777)
    a = pd.read_parquet(s1.finalize(n=12))["frame_idx"].tolist()
    b = pd.read_parquet(s2.finalize(n=12))["frame_idx"].tolist()
    assert a == b


def test_index_merges_claims_of_other_stages(tmp_path, monkeypatch):
    """Прогон одного этапа не должен стирать из индекса пруфы другого.

    Регресс от 2026-09-04: S7 перезаписывал evidence/index.parquet целиком,
    и witness-ссылки S4 и S6 исчезали, хотя jpg на диске оставались. Дашборд
    молча оставался без доказательств — ровно то, что запрещает правило 1.
    """
    import numpy as np
    import pandas as pd

    from looq.evidence import EvidenceWriter

    monkeypatch.chdir(tmp_path)
    crop = np.full((80, 40, 3), 120, np.uint8)

    older = EvidenceWriter(root="evidence", stage="s4_track", model_name="m")
    older.add("claim.track.x", track_id=1, frame_idx=10, ts=1.0, crop_bgr=crop,
              value=0.5, confidence=0.5)
    older.finalize()

    newer = EvidenceWriter(root="evidence", stage="s7_attrs", model_name="m")
    newer.add("claim.attrs.y", track_id=2, frame_idx=20, ts=2.0, crop_bgr=crop,
              value=0.7, confidence=0.7)
    newer.finalize()

    idx = pd.read_parquet("evidence/index.parquet")
    assert set(idx["claim_id"]) == {"claim.track.x", "claim.attrs.y"}, \
        "пруфы предыдущего этапа пропали из индекса"

    # Повторный прогон этапа заменяет ВСЁ, что этап писал под своим именем,
    # включая claim, который он перестал выпускать: иначе устаревший пруф
    # противоречил бы новому числу на той же странице.
    again = EvidenceWriter(root="evidence", stage="s7_attrs", model_name="m")
    again.add("claim.attrs.z", track_id=3, frame_idx=30, ts=3.0, crop_bgr=crop,
              value=0.9, confidence=0.9)
    again.finalize()
    idx = pd.read_parquet("evidence/index.parquet")
    assert set(idx["claim_id"]) == {"claim.track.x", "claim.attrs.z"},         "исчезнувший claim этапа остался в индексе"
    assert set(idx["track_id"]) == {1, 3}


def test_index_drops_rows_whose_file_vanished(tmp_path, monkeypatch):
    """Ссылка без файла — обещание пруфа, которого нет: такие строки выбрасываются."""
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from looq.evidence import EvidenceWriter

    monkeypatch.chdir(tmp_path)
    crop = np.full((80, 40, 3), 120, np.uint8)
    w = EvidenceWriter(root="evidence", stage="s4_track", model_name="m")
    p = w.add("claim.track.x", track_id=1, frame_idx=10, ts=1.0, crop_bgr=crop,
              value=0.5, confidence=0.5)
    w.finalize()
    Path(p).unlink()

    w2 = EvidenceWriter(root="evidence", stage="s7_attrs", model_name="m")
    w2.add("claim.attrs.y", track_id=2, frame_idx=20, ts=2.0, crop_bgr=crop,
           value=0.7, confidence=0.7)
    w2.finalize()
    idx = pd.read_parquet("evidence/index.parquet")
    assert set(idx["claim_id"]) == {"claim.attrs.y"}
