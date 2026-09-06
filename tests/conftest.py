# -*- coding: utf-8 -*-
"""Общая обвязка тестов.

Единственное, что здесь есть, — заглушка обезличивателя для машин без весов.

Зачем. С тех пор как запись изображения идёт только через
``looq.anonymise``, ``looq.evidence`` не может записать кроп, не спросив у
детектора, где на нём головы. В рабочем прогоне это правильно: S3, S5 и S7
всё равно грузят те же веса, и упасть без них громко — правило 8. Но на
чистом клоне весов нет, а двенадцать тестов пруфов пишут синтетические
шумовые кропы и проверяют полосу размытия и индекс — детектор им не нужен,
и людей на их кропах нет.

Заглушка ставится ТОЛЬКО когда весов действительно нет. Если они есть,
тесты идут через настоящие модели, и подмены не происходит. Это не
выключатель обезличивания: подменяется, ЧТО ищется, а не ищется ли вообще,
и путь записи остаётся тем же самым. Вызов ``_install_models_for_tests`` из
рабочего кода запрещает ``test_single_image_write.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _NoPeople:
    """Модель, которая честно ничего не находит.

    Ровно то, что вернула бы настоящая сеть на шумовом кропе из теста. Не
    «пропусти обезличивание», а «людей здесь нет».
    """

    class _Res:
        boxes = None
        keypoints = None

    def predict(self, *a, **kw):
        return [self._Res()]


@pytest.fixture(scope="session", autouse=True)
def _anonymiser_models():
    """Настоящие модели, если веса есть; иначе честная пустая заглушка."""
    from looq import anonymise as A

    if A.models_available():
        yield "real"
        return
    A._install_models_for_tests((_NoPeople(),))
    yield "stub"
