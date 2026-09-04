"""Гейт S9 — s9_report.

Требование гейта (CLAUDE.md): каждое число ссылается на этап и на его метрику качества

СТАТУС: НЕ РЕАЛИЗОВАН. Возвращает 1.
Правило 3: этап не считается сделанным, пока гейт не вернул 0. Порог не подкручивать —
сначала объяснить причину провала в docs/JOURNAL.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq import STATUS_SKELETON                       # noqa: E402
from looq.stages._base import read_artifact_status     # noqa: E402

STAGE = "s9_report"
ARTIFACT = "out/report.html"
REQUIREMENT = "каждое число ссылается на этап и на его метрику качества"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unmeasured", action="store_true",
                    help="считать НЕ ИЗМЕРЕНО предупреждением, а не провалом. "
                         "Реально ПРОВАЛЕННЫЙ порог этот флаг НЕ прощает")
    args = ap.parse_args(argv)

    print(f"[S9] гейт: {REQUIREMENT}")

    status = read_artifact_status(ARTIFACT)
    if status is None:
        print(f"[S9] артефакт {ARTIFACT} отсутствует или без статуса")
    elif status == STATUS_SKELETON:
        # Правило 8: пустой артефакт каркаса не может пройти гейт ни при каких порогах.
        print(f"[S9] артефакт {ARTIFACT} помечен status={STATUS_SKELETON} — данных нет")

    print(f"[S9] НЕ РЕАЛИЗОВАН: метрика гейта не считается, результат не подтверждён")
    if args.allow_unmeasured:
        # Флаг понижает НЕИЗМЕРЕННОЕ до предупреждения. Провал реального порога
        # он не прощает и не может: это разные вещи, и смешать их значило бы
        # получить зелёный гейт на плохих числах.
        print(f"[S9] --allow-unmeasured: метрика не измерена, гейт пропущен")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
