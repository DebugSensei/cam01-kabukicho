"""Гейт S2 — s2_zones.

Требование гейта (CLAUDE.md): >=30% детекций попадают хотя бы в одну зону на 100 случайных кадрах; ROI-граница обоснована кривой recall по глубине

СТАТУС: НЕ РЕАЛИЗОВАН. Возвращает 1.
Правило 3: этап не считается сделанным, пока гейт не вернул 0. Порог не подкручивать —
сначала объяснить причину провала в docs/DECISIONS.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq import STATUS_SKELETON                       # noqa: E402
from looq.stages._base import read_artifact_status     # noqa: E402

STAGE = "s2_zones"
ARTIFACT = "zones/zones.geojson"
REQUIREMENT = ">=30% детекций попадают хотя бы в одну зону на 100 случайных кадрах; ROI-граница обоснована кривой recall по глубине"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unmeasured", action="store_true",
                    help="считать НЕ ИЗМЕРЕНО предупреждением, а не провалом. "
                         "Реально ПРОВАЛЕННЫЙ порог этот флаг НЕ прощает")
    args = ap.parse_args(argv)

    print(f"[S2] гейт: {REQUIREMENT}")

    status = read_artifact_status(ARTIFACT)
    # Правило 8: пустой артефакт каркаса не проходит гейт НИ ПРИ КАКИХ
    # флагах. --allow-unmeasured понижает НЕИЗМЕРЕННУЮ метрику до
    # предупреждения; это другое. Неизмеренная метрика означает «данные
    # есть, разметки нет», каркас — «данных нет вообще». Раньше флаг
    # прощал и то и другое, и [S2] возвращал 0 на пустом артефакте.
    if status is None:
        print(f"[S2] ПРОВАЛ: артефакт {ARTIFACT} отсутствует или без статуса")
        return 1
    if status == STATUS_SKELETON:
        print(f"[S2] ПРОВАЛ: артефакт {ARTIFACT} помечен "
              f"status={STATUS_SKELETON} — данных нет, порог тут ни при чём")
        return 1

    print(f"[S2] НЕ РЕАЛИЗОВАН: метрика гейта не считается, результат не подтверждён")
    if args.allow_unmeasured:
        # Флаг понижает НЕИЗМЕРЕННОЕ до предупреждения. Провал реального порога
        # он не прощает и не может: это разные вещи, и смешать их значило бы
        # получить зелёный гейт на плохих числах.
        print(f"[S2] --allow-unmeasured: метрика не измерена, гейт пропущен")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
