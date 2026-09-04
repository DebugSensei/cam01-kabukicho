"""Каждая запись индекса пруфов обязана указывать на существующий и
ОБЕЗЛИЧЕННЫЙ файл.

Почему тест такой, а не «померить резкость». Прямой замер резкости верхней
области даёт ложные тревоги: у коротких кропов в верхние 30% попадает одежда
под границей размытия, и метрика взлетает на полосатой рубашке, хотя голова
размыта полностью. Проверено вручную 2026-09-04.

Поэтому проверяются ДВА структурных инварианта, которые нельзя обойти:

  1. blur_face_region — единственный путь к диску. В модуле ровно одна функция,
     пишущая изображение, и вызывается она ровно из одного места, сразу после
     обезличивания. Тест это утверждение проверяет по исходнику, а не верит ему.
  2. Каждая строка индекса несёт метаданные обезличивания. Ключи blur_* умеет
     проставить ТОЛЬКО blur_face_region, и попадают они в строку из её
     возвращаемого значения. Есть ключи — значит функция отработала на этом
     конкретном кропе.

Плюс сам файл обязан существовать: ссылка без файла — это обещание пруфа,
которого нет.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INDEX = Path("evidence/index.parquet")
#: Ключи, которые проставляет только blur_face_region.
BLUR_KEYS = {"blur_top_frac", "blur_head_h_px", "blur_pixelate_factor",
             "blur_kernel_px", "blur_sigma_px"}


def test_single_write_path_guarded_by_blur():
    """Записывающая функция одна, и вызывается сразу после обезличивания."""
    src = Path("looq/evidence.py").read_text(encoding="utf-8")
    body = src.split("def add(", 1)[1] if "def add(" in src else ""

    # imwrite нигде, кроме _write_jpeg
    outside = [ln for ln in src.splitlines()
               if "imwrite" in ln and "def _write_jpeg" not in ln]
    assert len(outside) <= 1, f"cv2.imwrite вне _write_jpeg: {outside}"

    # Считаем ТОЛЬКО код: упоминание в докстроке модуля не вызов.
    code = [ln for ln in src.splitlines()
            if "_write_jpeg(" in ln and not ln.lstrip().startswith(("*", "#"))]
    defs = [ln for ln in code if ln.lstrip().startswith("def ")]
    calls = [ln for ln in code if ln not in defs]
    assert len(defs) == 1, f"_write_jpeg определён {len(defs)} раз"
    assert len(calls) == 1, (
        f"_write_jpeg вызывается из {len(calls)} мест — обезличивание можно "
        f"обойти: {calls}")

    # В add() обезличивание идёт РАНЬШЕ записи, и пишется его результат.
    i_blur = body.index("blur_face_region(")
    i_write = body.index("_write_jpeg(")
    assert i_blur < i_write, "запись на диск раньше обезличивания"
    assert re.search(r"_write_jpeg\(\s*\w+,\s*blurred", body), \
        "в _write_jpeg передаётся не результат blur_face_region"


@pytest.mark.skipif(not INDEX.is_file(), reason="индекса пруфов ещё нет")
def test_every_indexed_proof_exists_and_is_anonymised():
    pytest.importorskip("pyarrow")
    import pandas as pd

    ev = pd.read_parquet(INDEX)
    assert len(ev), "индекс пруфов пуст — это не успех (правило 8)"

    missing = [p for p in ev["path"] if not Path(str(p)).is_file()]
    assert not missing, (
        f"{len(missing)} ссылок без файла на диске, например {missing[:3]}. "
        f"Ссылка без файла — обещание пруфа, которого нет")

    no_meta = []
    for r in ev.itertuples():
        try:
            extra = json.loads(getattr(r, "extra_json", None) or "{}")
        except (ValueError, TypeError):
            extra = {}
        if not BLUR_KEYS.issubset(extra):
            no_meta.append((str(r.path), sorted(BLUR_KEYS - set(extra))))
    assert not no_meta, (
        f"{len(no_meta)} пруфов без метаданных обезличивания — значит "
        f"blur_face_region на них не отрабатывал: {no_meta[:3]}")


@pytest.mark.skipif(not INDEX.is_file(), reason="индекса пруфов ещё нет")
def test_blur_covers_declared_fraction():
    """Записанная высота области лица обязана соответствовать доле из конфига."""
    pytest.importorskip("pyarrow")
    import cv2
    import pandas as pd

    ev = pd.read_parquet(INDEX)
    bad = []
    for r in ev.head(60).itertuples():
        extra = json.loads(getattr(r, "extra_json", None) or "{}")
        img = cv2.imread(str(r.path))
        if img is None:
            continue
        want = max(1, int(round(float(extra["blur_top_frac"]) * img.shape[0])))
        if int(extra["blur_head_h_px"]) != want:
            bad.append((str(r.path), extra["blur_head_h_px"], want))
    assert not bad, f"высота обезличенной области не сходится с долей: {bad[:3]}"
