"""Гейт S5 — s5_orient.

Требование гейта (CLAUDE.md): MAE угла на 200 размеченных вручную людях.

СТАТУС: инварианты и ПОКРЫТИЕ считаются, MAE — нет: ручной разметки ещё не
существует. Гейт возвращает 1 и печатает, чего именно не хватает.

Покрытие считается независимо от разметки и обязано попадать в отчёт при любом
исходе гейта (правило 7): доля треков без угла — это не «не смотрел», а
«не измерено», и подменять одно другим нельзя.

Правило 3: этап не считается сделанным, пока гейт не вернул 0. Порог не подкручивать —
сначала объяснить причину провала в docs/DECISIONS.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq import STATUS_SKELETON                       # noqa: E402
from looq.io import load_config                        # noqa: E402
from looq.stages._base import (check_inputs_sha,       # noqa: E402
                               read_artifact_status)

STAGE = "s5_orient"
ARTIFACT = "pose/orient.parquet"
CONFIG = "configs/s5_orient.yaml"
#: Файл разметки. Имя с числом кропов не хардкодится: сколько удалось
#: разметить — столько и есть, а требование CLAUDE.md (200) проверяется
#: отдельно и печатается как недобор, а не прячется.
LABELS_GLOB = "s5_orient_*.jsonl"
LABELS_DIR = Path("labels")
N_REQUIRED = 200          # из CLAUDE.md


def angular_error_deg(pred: float, true: float) -> float:
    """Ошибка угла с учётом цикличности: 350 и 10 различаются на 20, не на 340."""
    d = abs(float(pred) - float(true)) % 360.0
    return min(d, 360.0 - d)


def measure_mae(path: Path) -> dict:
    """MAE по файлу разметки. Предсказание лежит В ФАЙЛЕ, поэтому метрика
    считается даже после того, как артефакт перезаписал следующий прогон."""
    import json

    import numpy as np

    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    hdr, rows = rows[0], rows[1:]
    used = [r for r in rows
            if r.get("label") is not None and r.get("predicted_yaw_deg") is not None]
    skipped = len(rows) - len(used)
    if not used:
        return {"n": 0, "skipped": skipped, "header": hdr}
    err = np.array([angular_error_deg(r["predicted_yaw_deg"], r["label"])
                    for r in used])
    # Интервал бутстрэпом, а не по формуле: распределение ошибки угла не
    # нормальное и ограничено снизу нулём.
    rng = np.random.default_rng(20260904)
    bs = np.array([rng.choice(err, len(err), replace=True).mean()
                   for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {
        "n": len(used), "skipped": skipped, "header": hdr,
        "mae": float(err.mean()), "median": float(np.median(err)),
        "ci95": [float(lo), float(hi)],
        "p90": float(np.percentile(err, 90)),
        "frac_over_90": float((err > 90).mean()),
        "frac_over_135": float((err > 135).mean()),
    }


def check_invariants(df) -> tuple[bool, list[str]]:
    """Инварианты контракта: диапазоны углов и связь null с непокрытием."""
    problems: list[str] = []
    for col in ("body_yaw_deg", "head_yaw_deg"):
        vals = df[col].dropna()
        if len(vals) and not ((vals >= 0.0).all() and (vals < 360.0).all()):
            problems.append(f"{col}: есть значения вне [0, 360) — конвенция углов нарушена")
    n_kp = df["n_kpts_valid"]
    if not ((n_kp >= 0).all() and (n_kp <= 17).all()):
        problems.append("n_kpts_valid вне [0, 17]")
    both_null = df["body_yaw_deg"].isna() & df["head_yaw_deg"].isna()
    conf_set = both_null & df["yaw_conf"].notna()
    if conf_set.any():
        problems.append(f"{int(conf_set.sum())} строк без обоих углов, но с yaw_conf: "
                        f"уверенность не может относиться к отсутствующему углу")
    return not problems, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unmeasured", action="store_true")
    args = ap.parse_args(argv)

    print("[S5] гейт: MAE угла на 200 размеченных вручную людях")

    status = read_artifact_status(ARTIFACT)
    if status is None:
        print(f"[S5] артефакт {ARTIFACT} отсутствует или без статуса")
    elif status == STATUS_SKELETON:
        # Правило 8: пустой артефакт каркаса не проходит гейт ни при каких порогах.
        print(f"[S5] артефакт {ARTIFACT} помечен status={STATUS_SKELETON} — данных нет")

    # Конфиг нужен и в ветке «артефакта нет» — порог гейта читается ниже
    # безусловно. Раньше он определялся только в else, и на свежем клоне
    # гейт падал с UnboundLocalError вместо честного отчёта.
    cfg = load_config(CONFIG)
    failed = 0

    if status in (None, STATUS_SKELETON):
        print("[S5] 1. инварианты контракта: пропущена, нет артефакта")
        print("[S5] 2. покрытие ориентации: пропущена, нет артефакта")
        failed += 2
    else:
        import pandas as pd
        df = pd.read_parquet(ARTIFACT)

        ok, problems = check_invariants(df)
        print(f"[S5] 1. инварианты контракта: {'ok' if ok else 'ПРОВАЛ'}")
        for p in problems:
            print(f"[S5]      {p}")
        if not ok:
            failed += 1

        n = len(df)
        body = float(df["body_yaw_deg"].notna().mean()) if n else 0.0
        head = float(df["head_yaw_deg"].notna().mean()) if n else 0.0
        need = float((cfg.get("gates") or {}).get("min_pose_coverage_frac", 0.0))
        print(f"[S5] 2. покрытие ориентации: корпус {body:.1%}, голова {head:.1%} "
              f"по {n} строкам (минимум {need:.0%})")
        print(f"[S5]      непокрытие {1 - body:.1%} идёт в отчёт отдельным числом — "
              f"это НЕ ИЗМЕРЕНО, а не «не смотрел» (правило 7)")
        if body < need:
            failed += 1

    # Главная метрика гейта. Считать её не по чему, и подменять её покрытием нельзя.
    found = sorted(LABELS_DIR.glob(LABELS_GLOB)) if LABELS_DIR.is_dir() else []
    if found:
        best = max(found, key=lambda q: q.stat().st_size)
        m = measure_mae(best)
        thr = float((cfg.get("gates") or {}).get("max_yaw_mae_deg", 20.0))
        if m["n"] == 0:
            print(f"[S5] 3. MAE угла: НЕ ИЗМЕРЕН — в {best} нет пригодных строк")
            failed += 1
        else:
            print(f"[S5] 3. MAE угла по {best.name}: {m['mae']:.1f} град "
                  f"(95% бутстрэп [{m['ci95'][0]:.1f}, {m['ci95'][1]:.1f}]), "
                  f"медиана {m['median']:.1f}, p90 {m['p90']:.1f}, n={m['n']}")
            print(f"[S5]    грубых ошибок: >90 град {m['frac_over_90']:.1%}, "
                  f">135 град {m['frac_over_135']:.1%} "
                  f"(ноль означает, что знак и система координат верны)")
            print(f"[S5]    разметка снята на {m['header'].get('video')}")
            if m["n"] < N_REQUIRED:
                # Правило 7: недобор объёма — это не «прошло», это отдельное число.
                print(f"[S5]    НЕДОБОР ОБЪЁМА: размечено {m['n']} из {N_REQUIRED} "
                      f"по CLAUDE.md. Интервал шире, чем требует гейт.")
            # Решение принимается по ВЕРХНЕЙ границе интервала: точечная оценка
            # ниже порога при интервале, накрывающем порог, — это не «прошло».
            if m["ci95"][1] <= thr:
                print(f"[S5]    ok: верхняя граница {m['ci95'][1]:.1f} <= {thr}")
            elif m["mae"] <= thr:
                print(f"[S5]    НЕ ПОДТВЕРЖДЕНО: точечная оценка {m['mae']:.1f} <= "
                      f"{thr}, но интервал накрывает порог")
                failed += 1
            else:
                print(f"[S5]    ПРОВАЛ: MAE {m['mae']:.1f} > порога {thr}")
                failed += 1
    else:
        print(f"[S5] 3. MAE угла: НЕ ИЗМЕРЕН — нет файла разметки {LABELS_DIR}/{LABELS_GLOB}. "
              f"Нужны 200 вручную размеченных людей; без них качество угла "
              f"не подтверждено ничем")
        # Штраф за отсутствие разметки начисляется ТОЛЬКО здесь. Раньше этот
        # блок стоял на уровне функции и выполнялся всегда: гейт не мог
        # вернуть 0 ни при каком MAE, а счётчик показывал на единицу больше
        # реально проваленных проверок.
        if args.allow_unmeasured:
            print("[S5] --allow-unmeasured: MAE не измерен, понижено до предупреждения")
        else:
            failed += 1

    ok_sha, sha_problems = check_inputs_sha(ARTIFACT)
    print(f"[S5] 4. sha входных артефактов: {'ok' if ok_sha else 'ПРОВАЛ'}")
    for q in sha_problems:
        print(f"[S5]      {q}")
    if not ok_sha:
        failed += 1

    print(f"[S5] провалено проверок: {failed} из 4")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
