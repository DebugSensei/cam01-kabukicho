"""Прогон всех гейтов подряд.

Печатает таблицу и возвращает 1, если хоть один гейт не вернул 0.
Ни один этап не считается сделанным, пока его гейт не зелёный (правило 3).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["s0_ingest", "s1_calib", "s2_zones", "s3_detect", "s4_track",
          "s5_orient", "s6_attn", "s7_attrs", "s8_aggregate", "s9_report"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unmeasured", action="store_true",
                    help="пробрасывается в каждый гейт: НЕ ИЗМЕРЕНО становится "
                         "предупреждением, провал порога — нет")
    args = ap.parse_args(argv)
    extra = ["--allow-unmeasured"] if args.allow_unmeasured else []

    results: list[tuple[int, str, int]] = []
    for n, stage in enumerate(STAGES):
        script = HERE / f"verify_s{n}.py"
        if not script.is_file():
            print(f"[S{n}] нет файла {script}", file=sys.stderr)
            results.append((n, stage, 127))
            continue
        # encoding задан явно: консоль Windows отдаёт cp1251/cp866 и роняет
        # чтение русских сообщений гейтов.
        proc = subprocess.run([sys.executable, str(script)] + extra,
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        for line in ((proc.stdout or "") + (proc.stderr or "")).splitlines():
            if line.strip():
                print("   " + line)
        results.append((n, stage, proc.returncode))

    print()
    print("ЭТАП".ljust(17) + "ГЕЙТ".ljust(10) + "КОД")
    print("-" * 32)
    failed = 0
    for n, stage, code in results:
        mark = "ok" if code == 0 else "ПРОВАЛ"
        if code != 0:
            failed += 1
        print(f"S{n} {stage:<13}{mark:<10}{code}")
    print("-" * 32)
    print(f"провалено гейтов: {failed} из {len(results)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
