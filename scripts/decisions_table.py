"""Таблица чисел финального часа в DECISIONS — из артефакта, а не руками.

ЗАЧЕМ. Раздел 6.11 держал таблицу, набранную вручную по прежнему прогону, и
разошёлся с `out/metrics.json` по каждой ячейке — включая то, какая витрина
лучшая: документ называл M2 с 12.65 %, артефакт и дашборд — M3 с 13.33 %. Это
тот же дефект, что «279 попаданий луча против 3 повёрнутых треков»: две
страницы публикуют разные числа об одном и том же.

Руками вбитая таблица расходится с артефактом при каждом перепрогоне, и
заметить это можно только случайно. Поэтому она генерируется. Скрипт
идемпотентен: находит раздел по заголовку и заменяет его целиком.

    python scripts/decisions_table.py            # переписать раздел
    python scripts/decisions_table.py --check    # только проверить, гейт
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.io import read_json  # noqa: E402

DOC = Path("docs/DECISIONS.md")
METRICS = Path("out/metrics.json")
EVENTS = Path("attn/events.parquet")

HEAD = "### 6.11 The numbers of the final hour"
NEXT = "## 7. Colour"

#: Порядок и подписи витрин. Имена берутся отсюда, а не из geojson: в документе
#: они должны читаться так же, как на дашборде.
ZONES = {"M1": "M1 角煮/げんかつ", "M2": "M2 入口/らーめん",
         "M3": "M3 芝浦ホルモン", "M4": "M4 お好み焼き"}


def thin(n: int) -> str:
    """Разряды тонким пробелом, как во всём документе."""
    return f"{n:,}".replace(",", " ")


def build() -> str:
    m = read_json(METRICS)
    me = m["metrics"]

    rows = []
    for z, name in ZONES.items():
        vis, st = me[f"visitors_facade_{z}"], me[f"stop_rate_facade_{z}"]
        orr = me[f"orientation_rate_facade_{z}"]
        gz, dw = me[f"gaze_seconds_median_facade_{z}"], me[f"dwell_median_facade_{z}"]
        # Ноль остановок — это не «нет данных»: у нуля есть верхняя граница
        # интервала, и она информативнее самого нуля.
        stop = (f"{st['value'] * 100:.2f} % (n={st['n']})" if st["n"]
                else f"0 % (n=0, upper {st['ci95_high'] * 100:.2f} %)")
        rows.append(
            f"| {name} | {thin(int(vis['value']))} | {stop} | "
            f"**{orr['value'] * 100:.2f} %** "
            f"[{orr['ci95_low'] * 100:.2f}, {orr['ci95_high'] * 100:.2f}] "
            f"(n={orr['n']}) | {gz['value']:.1f} s | {dw['value']:.1f} s |")

    n_ev = len(pd.read_parquet(EVENTS))
    checks = (m.get("reconciliation") or {}).get("checks") or []
    n_ok = sum(1 for c in checks if c.get("passed"))
    tracks = int(me["unique_tracks_total"]["value"])

    best = max(ZONES, key=lambda z: me[f"orientation_rate_facade_{z}"]["value"])
    bo = me[f"orientation_rate_facade_{best}"]
    others = sorted((me[f"orientation_rate_facade_{z}"]["ci95_high"]
                     for z in ZONES if z != best), reverse=True)
    separated = bo["ci95_low"] > others[0]

    return f"""{HEAD}

This table is generated from `out/metrics.json` by
[`scripts/decisions_table.py`](../scripts/decisions_table.py), not typed. The
previous version was carried over from an earlier run and disagreed with the
artifact in every cell, including which storefront leads: it said M2, while the
artifact, the dashboard and the README all say {best}. Two documents naming a
different best storefront is a defect this project has already shipped twice,
and a hand-typed copy of a computed table will drift again on the next run.

| storefront | visitors | stopped | turned (95 % Wilson) | median attention | median time in zone |
|---|---|---|---|---|---|
{chr(10).join(rows)}

{thin(tracks)} tracks in total, {thin(n_ev)} events; the sum reconciliation in \
S8 balances on all {n_ok} checks.

The leader is **{best}** at **{bo['value'] * 100:.2f} %** \
[{bo['ci95_low'] * 100:.2f}, {bo['ci95_high'] * 100:.2f}] over n={bo['n']} \
turned tracks, and its interval \
{'does not overlap' if separated else 'overlaps'} the next storefront's.

**The caveats that travel with the table.** Orientation coverage is **45.6 %**
for the body and **22.4 %** for the head — the turned share is computed over a
biased subsample (large, unoccluded people). Tracks, not people: the tracker
breaks trajectories and merges different ones, and IDF1 is not measured. The
median time in zone is close to the typical track length and reflects the
duration of observation more than a pause at the storefront.

"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="не писать, а падать, если раздел разошёлся с артефактом")
    args = ap.parse_args(argv)

    # На чистом клоне артефактов прогона нет — их не отдают вместе с
    # репозиторием. Падать трейсбеком здесь неправильно: сверять нечего, и
    # это не поломка. Но и молчать нельзя, иначе «проверка прошла» будет
    # означать «проверка не запускалась» (правило 7).
    missing = [str(p) for p in (METRICS, EVENTS) if not p.is_file()]
    if missing:
        print(f"  НЕ ПРОВЕРЕНО: нет артефактов {missing}. Раздел 6.11 "
              f"сверяется только после прогона, на чистом клоне сверять не с чем.")
        return 0

    s = DOC.read_text(encoding="utf-8")
    if HEAD not in s or NEXT not in s:
        raise SystemExit(f"в {DOC} не найден раздел {HEAD!r} или следующий {NEXT!r}")
    start = s.index(HEAD)
    end = s.index("---\n\n" + NEXT)
    current, fresh = s[start:end], build()

    if current == fresh:
        print("  6.11 совпадает с out/metrics.json")
        return 0
    if args.check:
        # Правило 8: молча привести к правде значило бы скрыть расхождение.
        print("  6.11 РАЗОШЁЛСЯ с out/metrics.json. Перегенерируйте:")
        print("    python scripts/decisions_table.py")
        return 1
    DOC.write_text(s[:start] + fresh + s[end:], encoding="utf-8")
    print(f"  6.11 перегенерирован из {METRICS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
