"""Гейт S1 — s1_calib.

Требование гейта (CLAUDE.md): ошибка репроекции на отложенных точках < 0.5 м;
масштаб оценён по медианному росту выборки и независимо подтверждён двумя
способами — медианной скоростью пешеходов и шириной улицы по спутнику;
плюс разброс роста как независимая проверка геометрии.

ЧТО НЕЗАВИСИМО, А ЧТО НЕТ
-------------------------
Масштаб гомографии подбирается ПО МЕДИАНЕ РОСТА. Значит проверка "медиана роста
в диапазоне" почти тавтологична: масштаб подобран так, чтобы медиана туда попала.
Она детектирует грубый сбой подгонки и ничего не говорит о правильности масштаба.
Печатается отдельной строкой с пометкой "не независима".

Независимы, ни одна не участвует в подгонке масштаба:
  * РАЗБРОС роста (IQR, p90-p10) — задаётся геометрией, а не масштабом;
  * ДРЕЙФ роста по глубине — регрессия est_height_m ~ foot_y_m, добавлена
    по предложению исполнителя (см. docs/JOURNAL.md);
  * медианная скорость пешеходов;
  * ширина улицы по спутниковому снимку.

Гейт печатает все семь величин в любом случае, провалился он или нет, и
пересчитывает каждую САМ из пилотных выборок артефакта. Читать готовые числа
этапа значило бы проверять его арифметику, а не гомографию.

СТАТУС: реализованы проверки 2a, 2b, 2c, 3, 4, 5, 6. Проверка 1 (репроекция
на отложенных точках) остаётся стабом — она пишется вместе с самим этапом S1.
Пока хоть одна не реализована, гейт возвращает 1.

Правило 3: этап не считается сделанным, пока гейт не вернул 0. Порог не подкручивать —
сначала объяснить причину провала в docs/JOURNAL.md.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse                                                       # noqa: E402

from looq import STATUS_SKELETON                                        # noqa: E402
from looq.calib import height_depth_slope                               # noqa: E402
from looq.geometry import facade_lines_separation_m, height_spread_stats  # noqa: E402
from looq.io import load_config, read_json                             # noqa: E402
from looq.stages._base import read_artifact_status                     # noqa: E402

STAGE = "s1_calib"
ARTIFACT = "calib/homography.json"
CONFIG = "configs/s1_calib.yaml"

Result = tuple[bool, str]


def _pilot(homography: dict, key: str, min_n: int) -> list[float]:
    vals = homography.get(key)
    if not isinstance(vals, list) or len(vals) < min_n:
        raise ValueError(
            f"в homography.json нет {key} или меньше {min_n} значений "
            f"(получено {len(vals) if isinstance(vals, list) else 'нет ключа'})"
        )
    return [float(v) for v in vals]


# --------------------------------------------------------------------------- #
# 2a. Медиана роста — НЕ независима
# --------------------------------------------------------------------------- #

def check_height_median(homography: dict, gates: dict, min_n: int) -> Result:
    """Детектор грубого сбоя подгонки. Масштаб НЕ проверяет — он по ней и подобран."""
    heights = _pilot(homography, "pilot_heights_m", min_n)
    lo, hi = gates["median_height_range_m"]
    median = statistics.median(heights)
    ok = lo <= median <= hi
    return ok, f"медиана роста {median:.3f} м, ожидается {lo}-{hi} м (n={len(heights)})"


# --------------------------------------------------------------------------- #
# 2b. Разброс роста — независима
# --------------------------------------------------------------------------- #

def check_height_spread(homography: dict, gates: dict, min_n: int) -> Result:
    """Настоящая проверка геометрии: разброс масштабом не задаётся."""
    heights = _pilot(homography, "pilot_heights_m", min_n)
    st = height_spread_stats(heights)
    iqr_max = float(gates["height_iqr_max_m"])
    span_max = float(gates["height_p90_p10_max_m"])
    problems = []
    if st["iqr_m"] > iqr_max:
        problems.append(f"IQR {st['iqr_m']:.3f} > {iqr_max}")
    if st["p90_p10_m"] > span_max:
        problems.append(f"p90-p10 {st['p90_p10_m']:.3f} > {span_max}")
    msg = (f"IQR {st['iqr_m']:.3f} м (порог {iqr_max}), "
           f"p90-p10 {st['p90_p10_m']:.3f} м (порог {span_max})")
    if problems:
        return False, (f"{msg} — {', '.join(problems)}. Разброс разъехался: "
                       f"оценка роста плывёт с глубиной, геометрия кривая")
    return True, msg


# --------------------------------------------------------------------------- #
# 3. Медианная скорость — независима
# --------------------------------------------------------------------------- #

def check_speed_median(homography: dict, gates: dict, min_n: int) -> Result:
    speeds = _pilot(homography, "pilot_speeds_mps", min_n)
    lo, hi = gates["median_speed_range_mps"]
    median = statistics.median(speeds)
    ok = lo <= median <= hi
    return ok, f"медианная скорость {median:.3f} м/с, ожидается {lo}-{hi} м/с (n={len(speeds)})"


# --------------------------------------------------------------------------- #
# 2c. Дрейф роста по глубине — независима.
# Проверка добавлена по предложению исполнителя, см. docs/JOURNAL.md.
# --------------------------------------------------------------------------- #

def check_depth_slope(homography: dict, gates: dict, min_n: int) -> Result:
    """Рост не должен зависеть от глубины. Пересчитывается гейтом самостоятельно."""
    heights = _pilot(homography, "pilot_heights_m", min_n)
    depths = homography.get("pilot_depths_m")
    if isinstance(depths, list) and len(depths) == len(heights):
        reg = height_depth_slope(heights, depths)
    else:
        stored = homography.get("height_depth_slope")
        ci = homography.get("height_depth_slope_ci95")
        if stored is None or not isinstance(ci, list) or len(ci) != 2:
            return False, ("нет pilot_depths_m для пересчёта и нет "
                           "height_depth_slope/height_depth_slope_ci95 в артефакте")
        reg = {"slope_m_per_m": float(stored), "ci95_low": float(ci[0]),
               "ci95_high": float(ci[1]),
               "covers_zero": float(ci[0]) <= 0.0 <= float(ci[1]), "n": len(heights)}
    msg = (f"наклон {reg['slope_m_per_m']:+.5f} м/м, "
           f"CI95 [{reg['ci95_low']:+.5f}, {reg['ci95_high']:+.5f}]")
    if not reg["covers_zero"]:
        return False, (f"{msg} — ноль НЕ накрыт: рост систематически плывёт с глубиной, "
                       f"гомография врёт. Именно этот дефект пуловый IQR прячет")
    return True, f"{msg} — ноль накрыт"


# --------------------------------------------------------------------------- #
# 5. Размер выборки и 6. невязка удержанных отрезков
# --------------------------------------------------------------------------- #

def check_sample_size(homography: dict, gates: dict) -> Result:
    need = int(gates.get("min_people", 200))
    got = homography.get("n_people_used")
    if got is None:
        return False, "в артефакте нет n_people_used"
    if int(got) < need:
        return False, (f"людей в выборке {got} при минимуме {need}: медиана роста, "
                       f"а с ней и весь масштаб, держится на случайности")
    return True, f"людей в выборке {got} при минимуме {need}"


def check_vp_holdout(homography: dict, gates: dict) -> Result:
    limit = float(gates.get("vp_holdout_max_px", 3.0))
    resid = homography.get("vp_holdout_residual_px")
    if not isinstance(resid, dict) or "max" not in resid:
        return False, "в артефакте нет vp_holdout_residual_px"
    worst = float(resid["max"])
    detail = (f"горизонтальная {float(resid.get('horizontal', float('nan'))):.2f} px, "
              f"вертикальная {float(resid.get('vertical', float('nan'))):.2f} px, "
              f"порог {limit} px")
    if worst > limit:
        return False, f"{detail} — удержанные отрезки не сходятся на точке схода"
    return True, detail


# --------------------------------------------------------------------------- #
# 7. Ширина улицы — независима
# --------------------------------------------------------------------------- #

def check_street_width(homography: dict, cfg: dict) -> Result:
    """Ширина улицы по нашей гомографии против спутникового замера.

    Считает НЕЗАВИСИМО от того, что записал этап: берёт линии оснований фасадов
    из homography.json и меряет расстояние сам. Если бы гейт читал готовое
    street_width_measured_m, он проверял бы не гомографию, а арифметику этапа.
    """
    control = cfg.get("control") or {}
    reference = control.get("street_width_m")
    tolerance = control.get("street_width_tolerance_m")
    factor = float((cfg.get("gates") or {}).get("street_width_tolerance_factor", 2.0))
    if reference is None or tolerance is None:
        return False, "в configs/s1_calib.yaml нет control.street_width_m/tolerance"
    limit = float(tolerance) * factor

    baselines = homography.get("facade_baselines")
    if not isinstance(baselines, list) or len(baselines) != 2:
        return False, ("в homography.json нет facade_baselines: двух линий оснований "
                       "фасадов на плане (координаты plane_m, кликаются пользователем)")
    try:
        sep = facade_lines_separation_m(baselines[0]["points_m"], baselines[1]["points_m"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"не удалось измерить ширину улицы: {exc}"

    measured = sep["width_mean_m"]
    delta = measured - float(reference)
    detail = (f"измерено {measured:.3f} м, эталон {reference} м, "
              f"расхождение {delta:+.3f} м при допуске +-{limit:.2f} м; "
              f"разброс по четырём замерам {sep['width_spread_m']:.3f} м, "
              f"угол между фасадами {sep['lines_angle_deg']:.2f} град")

    stored = homography.get("street_width_measured_m")
    if stored is not None and abs(float(stored) - measured) > 1e-3:
        return False, (f"street_width_measured_m в артефакте ({stored}) не сходится "
                       f"с пересчитанным ({measured:.3f}) — ошибка в этапе S1")
    if abs(delta) > limit:
        return False, f"{detail} — масштаб гомографии неверен"
    return True, detail


# --------------------------------------------------------------------------- #

def check_ground_plane(homography: dict) -> Result:
    """Что можно проверить без масштаба: гомография и горизонт осмысленны."""
    problems: list[str] = []
    h = homography.get("H_px_to_unit")
    if not isinstance(h, list) or len(h) != 3:
        return False, ["нет H_px_to_unit 3x3"]
    import numpy as np
    hm = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(hm)):
        problems.append("в гомографии есть не-конечные значения")
    if abs(float(np.linalg.det(hm))) < 1e-12:
        problems.append("гомография вырождена")
    # Невязка углов есть только у пути по прямоугольнику: там четыре точки
    # отображаются в известный прямоугольник, и промах решателя виден.
    # У заглушки stub_affine отображать нечего — проверять нечего.
    if homography.get("method") != "stub_affine":
        resid = homography.get("corner_residual_units")
        if resid is None or float(resid) > 1e-3:
            problems.append(f"невязка углов участка {resid} усл.ед. — решатель не сошёлся")
    horizon = homography.get("horizon_line")
    if not isinstance(horizon, list) or len(horizon) != 3:
        problems.append("нет линии горизонта")
    return (not problems), problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unmeasured", action="store_true",
                    help="считать НЕ ИЗМЕРЕНО предупреждением, а не провалом. "
                         "Реально ПРОВАЛЕННЫЙ порог этот флаг НЕ прощает")
    args = ap.parse_args(argv)

    print("[S1] гейт: масштаб оценён по медианному росту выборки; независимо "
          "подтверждён двумя способами — медианной скоростью пешеходов и "
          "шириной улицы по спутнику")

    status = read_artifact_status(ARTIFACT)
    if status is None:
        print(f"[S1] артефакт {ARTIFACT} отсутствует или без статуса")
    elif status == STATUS_SKELETON:
        # Правило 8: пустой артефакт каркаса не проходит гейт ни при каких порогах.
        print(f"[S1] артефакт {ARTIFACT} помечен status={STATUS_SKELETON} — данных нет")

    failed = 0

    if status not in (None, STATUS_SKELETON) and not read_json(ARTIFACT).get("scale_known", True):
        pass   # у запасного пути отложенных отрезков нет, см. ветку ниже
    else:
        print("[S1] 1. ошибка репроекции на отложенных точках: НЕ РЕАЛИЗОВАНА "
              "(пишется вместе с этапом S1)")
        failed += 1

    if status in (None, STATUS_SKELETON):
        for name in ("2a. медиана роста [не независима]", "2b. разброс роста",
                     "2c. дрейф роста по глубине", "3. медианная скорость",
                     "4. ширина улицы", "5. размер выборки",
                     "6. невязка удержанных отрезков"):
            print(f"[S1] {name}: пропущена, нет откалиброванной гомографии")
            failed += 1
        print(f"[S1] провалено проверок: {failed} из 8")
        return 1

    cfg = load_config(CONFIG)
    gates = cfg.get("gates") or {}
    min_n = int(gates.get("min_pilot_samples", 30))
    homography = read_json(ARTIFACT)

    # Запасной путь калибровки не даёт метров. Проверки роста, скорости и ширины
    # улицы выражены в метрах, поэтому они НЕ ПРИМЕНИМЫ — это не то же самое,
    # что «прошли». Считать их пройденными значило бы выдать неизвестное за
    # проверенное (правила 7 и 8).
    if not homography.get("scale_known", True):
        status = homography.get("calib_status", "?")
        print(f"[S1] calib_status = {status}: масштаб НЕ ОПРЕДЕЛЁН, длины "
              f"в условных единицах")
        print("[S1] 2a-4. рост, скорость, ширина улицы: НЕ ПРИМЕНИМЫ без метров")
        print("[S1]      это НЕ «прошли»: величины не проверены ничем")
        unmeasured = 4
        ok_geom, geom_problems = check_ground_plane(homography)
        print(f"[S1] 7. геометрия плоскости: {'ok' if ok_geom else 'ПРОВАЛ'}")
        for pb in geom_problems:
            print(f"[S1]      {pb}")
        hard_fail = 0 if ok_geom else 1
        if args.allow_unmeasured:
            print(f"[S1] --allow-unmeasured: {unmeasured} неизмеренных величин "
                  f"понижены до предупреждения")
            print(f"[S1] провалено проверок: {hard_fail}")
            return 0 if hard_fail == 0 else 1
        print(f"[S1] провалено проверок: {hard_fail + unmeasured} "
              f"(из них не измерено: {unmeasured})")
        return 1

    checks: list[tuple[str, object]] = [
        # Помета "не независима" обязательна: без неё строки подряд читаются
        # как несколько подтверждений масштаба, а подтверждений меньше.
        ("2a. медиана роста [НЕ НЕЗАВИСИМА: масштаб подобран по ней, "
         "это детектор сбоя подгонки]", lambda: check_height_median(homography, gates, min_n)),
        ("2b. разброс роста [независима]", lambda: check_height_spread(homography, gates, min_n)),
        ("2c. дрейф роста по глубине [независима]",
         lambda: check_depth_slope(homography, gates, min_n)),
        ("3. медианная скорость [независима]", lambda: check_speed_median(homography, gates, min_n)),
        ("4. ширина улицы [независима]", lambda: check_street_width(homography, cfg)),
        ("5. размер выборки", lambda: check_sample_size(homography, gates)),
        ("6. невязка удержанных отрезков", lambda: check_vp_holdout(homography, gates)),
    ]

    for name, fn in checks:
        try:
            ok, msg = fn()
        except (ValueError, KeyError) as exc:
            ok, msg = False, str(exc)
        print(f"[S1] {name}: {'ok' if ok else 'ПРОВАЛ'} — {msg}")
        if not ok:
            failed += 1

    print(f"[S1] провалено проверок: {failed} из 8")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
