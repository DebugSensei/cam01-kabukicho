"""S0 ingest — Запись часа с YouTube live в raw/*.ts и манифест сегментов.

Каркас. CV-логики нет: этап отрабатывает контракт и падает с кодом 1.
Схема артефакта — docs/CONTRACTS.md, раздел S0 ingest.

    python -m looq.stages.s0_ingest --config configs/s0_ingest.yaml
"""

from __future__ import annotations

from looq.stages._base import Col, stage_main

STAGE = "s0_ingest"
INPUTS = []
OUTPUT = "raw/manifest.json"
OUTPUT_KIND = "json"

OUTPUT_COLS: list[Col] = [

]


def main(argv=None) -> int:
    return stage_main(STAGE, INPUTS, OUTPUT, OUTPUT_KIND, OUTPUT_COLS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
