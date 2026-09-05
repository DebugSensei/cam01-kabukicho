"""Гейт S4 — s4_track.

Требование гейта (CLAUDE.md): IDF1 и ID-switch на 3 минутах, отдельно на разреженном и плотном куске

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
from looq.stages._base import (check_inputs_sha,       # noqa: E402
                               read_artifact_status)

STAGE = "s4_track"
ARTIFACT = "track/tracks.parquet"
REQUIREMENT = "IDF1 и ID-switch на 3 минутах, отдельно на разреженном и плотном куске"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unmeasured", action="store_true",
                    help="считать НЕ ИЗМЕРЕНО предупреждением, а не провалом. "
                         "Реально ПРОВАЛЕННЫЙ порог этот флаг НЕ прощает")
    args = ap.parse_args(argv)

    print(f"[S4] гейт: {REQUIREMENT}")

    status = read_artifact_status(ARTIFACT)
    if status is None:
        print(f"[S4] артефакт {ARTIFACT} отсутствует или без статуса")
    elif status == STATUS_SKELETON:
        # Правило 8: пустой артефакт каркаса не может пройти гейт ни при каких порогах.
        print(f"[S4] артефакт {ARTIFACT} помечен status={STATUS_SKELETON} — данных нет")

    # Единственная настоящая проверка этого гейта: входы артефакта обязаны
    # совпадать с тем, что лежит на диске сейчас. Иначе после перекалибровки
    # метры остаются от старой гомографии, а числа выглядят нормальными —
    # молчаливая порча хуже падения (правило 8).
    ok_sha, sha_problems = check_inputs_sha(ARTIFACT)
    print(f"[S4] sha входных артефактов: {'ok' if ok_sha else 'ПРОВАЛ'}")
    for q in sha_problems:
        print(f"[S4]      {q}")

    print(f"[S4] НЕ РЕАЛИЗОВАН: метрика гейта не считается, результат не подтверждён")
    if args.allow_unmeasured:
        # Флаг понижает НЕИЗМЕРЕННОЕ до предупреждения. Провал реального порога
        # он не прощает и не может: это разные вещи, и смешать их значило бы
        # получить зелёный гейт на плохих числах.
        print(f"[S4] --allow-unmeasured: метрика не измерена, гейт пропущен")
        # Флаг прощает НЕИЗМЕРЕННОЕ, но не рассогласование входов:
        # это не «не посчитали», это «посчитали не по тем данным».
        return 0 if ok_sha else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
