"""out/benchmark.html — что измерено, что только покрыто, что не измерено.

ГЛАВНОЕ РАЗДЕЛЕНИЕ. Таблица 1 меряет ПРАВИЛЬНОСТЬ: у каждой строки есть
эталон или прямой геометрический замер. Таблица 2 меряет ОБЪЁМ и
СТАБИЛЬНОСТЬ: сколько данных и насколько они согласованы — это НЕ
правильность. Таблица 3 — чего нет и что нужно, чтобы появилось.

Числа без эталона в таблицу 1 не попадают. Если метрики нет, она идёт
в таблицу 3, а не в первую с оговоркой.

Страница ничего не пересчитывает сверх готовых артефактов.

    python scripts/make_benchmark.py
    make benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_dashboard import (CSS, JS, esc, header_html, set_public_build,  # noqa: E402
                            i18n_payload, register_i18n, t)

from looq.io import atomic_write_text, read_json  # noqa: E402

OUT = Path("out/benchmark.html")


def mae_from_labels():
    # ВНИМАНИЕ: возвращает и "source" — имя файла, из которого взято число.
    # Раньше имя было вписано в таблицу руками и разошлось с реальностью:
    # печаталось n=50 рядом со ссылкой на файл с 24 строками.
    d = Path("labels")
    files = sorted(d.glob("s5_orient_*.jsonl")) if d.is_dir() else []
    if not files:
        return None
    rows = [json.loads(x) for x in
            max(files, key=lambda q: q.stat().st_size).read_text(
                encoding="utf-8").splitlines() if x]
    hdr, rows = rows[0], rows[1:]
    used = [r for r in rows if r.get("label") is not None
            and r.get("predicted_yaw_deg") is not None]
    if not used:
        return None
    e = []
    for r in used:
        dd = abs(float(r["predicted_yaw_deg"]) - float(r["label"])) % 360.0
        e.append(min(dd, 360.0 - dd))
    e = np.array(e)
    rng = np.random.default_rng(20260904)
    bs = [rng.choice(e, len(e), replace=True).mean() for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mae": e.mean(), "lo": lo, "hi": hi, "med": np.median(e),
            "n": len(e), "gross": float((e > 90).mean()),
            "video": hdr.get("video", "?"),
            "source": str(max(files, key=lambda q: q.stat().st_size)).replace("\\", "/")}


#: Счётчик ключей строк: имя и пояснение каждой строки переводимы, а путь
#: к источнику — нет, это имя файла.
_ROW_N = [0]


def _drift_slope(hom: dict) -> float:
    """Наклон рост-от-глубины ПО МАССИВАМ, а не из поля height_depth_slope.

    S1 домножал это поле на scale_rescale_factor, хотя м/м к масштабу
    инвариантен, и артефакт до перепрогона хранит заниженное значение.
    Гейт verify_s1 ловит расхождение; здесь считаем сами, чтобы бенчмарк
    и README не печатали разные числа.
    """
    h = np.asarray(hom.get("pilot_heights_m") or [], dtype=float)
    d = np.asarray(hom.get("pilot_depths_m") or [], dtype=float)
    if h.size < 30 or h.size != d.size:
        return float(hom.get("height_depth_slope", float("nan")))
    ok = (h > 0.8) & (h < 2.6) & np.isfinite(d) & (d > 0)
    if int(ok.sum()) < 30:
        return float(hom.get("height_depth_slope", float("nan")))
    return float(np.polyfit(d[ok], h[ok], 1)[0])


def row(name, value, source, note=("", "", "")):
    """Строка таблицы. name и note — тройки (en, ru, ja).

    Переводы регистрируются здесь же: если объявить строку в разметке и
    забыть в словаре, она молча останется на одном языке — ровно то, что
    случилось со всей этой страницей.
    """
    _ROW_N[0] += 1
    k = f"bm.r{_ROW_N[0]}"
    if isinstance(name, str):
        name = (name, name, name)
    if isinstance(note, str):
        note = (note, note, note)
    register_i18n({k + ".n": name, k + ".c": note})
    return (f"<tr><td><b>{t(k + '.n')}</b></td><td class='num'>{value}</td>"
            f"<td><code>{esc(source)}</code></td><td>{t(k + '.c')}</td></tr>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--public", action="store_true",
                    help="публичная сборка: шапка ведёт только на "
                         "опубликованные страницы, видео — на YouTube")
    ap.add_argument("--hide-unmeasured", action="store_true",
                    help="скрыть таблицу 3 на демонстрации. Раздел НЕ удаляется "
                         "из кода: он возвращается запуском без этого флага")
    args = ap.parse_args(argv)
    set_public_build(args.public)

    # Регистрация ДО первого t(): f-строка с t("bm.t3") вычисляется в
    # момент сборки sec3, и если словарь пополнить после неё, страница
    # падает с KeyError. Ровно это и происходило.
    register_i18n({
        "bm.eyebrow": ("Benchmark", "Бенчмарк", "ベンチマーク"),
        "bm.h1": ("What is measured, what is only covered, what is missing",
                  "Что измерено, что только покрыто, чего нет",
                  "何が計測され、何が範囲だけで、何が欠けているか"),
        "bm.warn": ("Table 1 measures correctness, table 2 measures only volume "
                    "and stability. They must not be mixed.",
                    "Таблица 1 меряет правильность, таблица 2 — только объём и "
                    "стабильность. Смешивать их нельзя.",
                    "表1は正しさを、表2は量と安定性のみを測ります。混同は禁物です。"),
        "bm.warn2": ("A number without a reference does not enter the first table.",
                     "Число без эталона в первую таблицу не попадает.",
                     "基準のない数値は表1に入りません。"),
        "bm.t1": ("1. Measured — a reference or a direct measurement exists",
                  "1. Измерено — есть эталон или прямой замер",
                  "1. 計測済み — 基準または直接計測あり"),
        "bm.t1sub": ("Every row rests on manual labelling, on the geometry of two "
                     "vanishing points, or on a direct measurement over frames.",
                     "Каждая строка опирается на ручную разметку, на геометрию "
                     "двух точек схода или на прямой замер по кадрам.",
                     "各行は手動ラベル、2消失点の幾何、またはフレームの直接計測に "
                     "基づきます。"),
        "bm.t2": ("2. Coverage and stability — measures volume, NOT correctness",
                  "2. Покрытие и стабильность — меряет объём, НЕ правильность",
                  "2. カバレッジと安定性 — 量であり正しさではない"),
        "bm.t2sub": ("These numbers say how much data was collected and how "
                     "consistent it is. A model agreeing with itself is not "
                     "evidence that it is right.",
                     "Эти числа говорят, сколько данных набрано и насколько они "
                     "согласованы между собой. Согласованность модели с самой "
                     "собой не является доказательством того, что она права.",
                     "これらはデータ量と内的整合性を示します。モデルの自己整合性は "
                     "正しさの証明ではありません。"),
        "bm.bins": ("Median box height by depth", "Медианная высота рамки по глубине",
                    "奥行き別の枠高中央値"),
        "bm.binsub": ("A direct measurement over matched tracks. The cutoff "
                      "threshold is derived from it.",
                      "Прямой замер по сопоставленным трекам. Отсюда выведен "
                      "порог отсечки.",
                      "対応付けた追跡の直接計測。ここから打ち切り閾値を導出。"),
        "bm.th.q": ("Quantity", "Величина", "項目"),
        "bm.th.v": ("Value", "Значение", "値"),
        "bm.th.s": ("Source", "Источник", "出典"),
        "bm.th.h": ("How it was obtained", "Как получено", "取得方法"),
        "bm.th.m": ("What it means", "Что это значит", "意味"),
        "bm.th.depth": ("Depth", "Глубина", "奥行き"),
        "bm.th.rows": ("Rows", "Строк", "行数"),
        "bm.th.med": ("Median height", "Медиана высоты", "高さ中央値"),
        "bm.t3": ("3. Not measured — and what it would take",
                  "3. Не измерено — и что нужно, чтобы измерить",
                  "3. 未計測 — 計測に必要なもの"),
        "bm.th.miss": ("What is missing", "Чего нет", "欠けているもの"),
        "bm.th.need": ("What it would take", "Что нужно", "必要なもの"),
    })


    import pandas as pd

    hom = read_json("calib/homography.json")
    m = read_json("out/metrics.json")
    met, scope = m["metrics"], m["scope"]
    dc = read_json("out/depth_cutoff.json")
    tracks = pd.read_parquet("track/tracks.parquet")
    orient = pd.read_parquet("pose/orient.parquet")
    attr = pd.read_parquet("attr/tracks_attr.parquet")
    events = pd.read_parquet("attn/events.parquet")
    zf = pd.read_parquet("attn/track_zone_frames.parquet")
    mae = mae_from_labels()

    # ---- таблица 1: измерено, есть эталон или прямой замер ----------------- #
    t1 = []
    if mae:
        ci = f"[{mae['lo']:.1f}, {mae['hi']:.1f}]"
        t1.append(row(
            ("Body orientation MAE", "MAE ориентации корпуса", "体の方位MAE"),
            f"{mae['mae']:.1f}\u00b0",
            mae.get("source", "labels/"),
            (f"95% bootstrap {ci}, median {mae['med']:.1f}, n={mae['n']} labelled "
             f"by hand, gross errors >90 deg {mae['gross']:.0%}",
             f"95% бутстрэп {ci}, медиана {mae['med']:.1f}, n={mae['n']} размечено "
             f"вручную, грубых ошибок >90 град {mae['gross']:.0%}",
             f"95%ブートストラップ {ci}、中央値 {mae['med']:.1f}、手動ラベル "
             f"n={mae['n']}、90度超の粗大誤差 {mae['gross']:.0%}")))
    t1 += [
        row(("Camera height", "Высота камеры", "カメラ高"),
            f"{hom['camera_height_m']:.2f} m", "calib/homography.json",
            ("from two vanishing points, pole-polar relation",
             "из двух точек схода, соотношение полюс-поляра",
             "2つの消失点、極・極線の関係より")),
        row(("Focal length", "Фокусное расстояние", "焦点距離"),
            f"{hom['focal_px']:.0f} px", "calib/homography.json",
            (f"{hom['focal_over_diagonal']:.3f} of the frame diagonal",
             f"{hom['focal_over_diagonal']:.3f} диагонали кадра",
             f"画面対角の{hom['focal_over_diagonal']:.3f}倍")),
        row(("Median height of the sample", "Медианный рост выборки",
             "サンプルの身長中央値"),
            f"{hom['height_median_m']:.2f} m", "calib/homography.json",
            ("SET as the source of scale, therefore NOT a check",
             "ЗАДАН как источник масштаба, поэтому НЕ является проверкой",
             "スケールの根拠として設定。したがって検証ではありません")),
        row(("Implied L1\u2194L3 width", "Подразумеваемая ширина L1\u2194L3",
             "含意されるL1\u2194L3幅"),
            "5.28 m", "docs/DECISIONS.md",
            ("against 6.06 m from satellite: a different cross-section, "
             "curb offset ~0.8 m",
             "против спутниковых 6.06 м: иное сечение, отступ бордюра ~0.8 м",
             "衛星の6.06mに対して。断面が異なり、縁石オフセット約0.8m")),
        row(("Height drift with depth", "Дрейф роста по глубине",
             "奥行きに伴う身長ドリフト"),
            f"{_drift_slope(hom):.4f} m/m", "calib/homography.json",
            (f"equivalent to a {hom.get('street_grade', {}).get('grade_percent', 0):.1f}% "
             f"grade; calling it a street gradient would be a guess",
             f"эквивалент уклона {hom.get('street_grade', {}).get('grade_percent', 0):.1f}%; "
             f"называть это уклоном улицы нельзя — не проверено",
             f"勾配{hom.get('street_grade', {}).get('grade_percent', 0):.1f}%相当。"
             f"通りの勾配と断定はできません")),
        row(("Camera shift over a day", "Сдвиг камеры за сутки",
             "1日でのカメラ移動"),
            "< 0.25 px", "phase correlation, 3 files",
            ("against clip_debug_2030JST.ts: the camera is fixed, so the "
             "calibration holds for another hour",
             "против clip_debug_2030JST.ts: камера неподвижна, калибровка "
             "действительна для другого часа",
             "clip_debug_2030JST.tsとの比較。カメラは固定で、別時間帯でも較正は有効")),
        row(("Box-height cutoff threshold", "Порог отсечки по высоте рамки",
             "枠高による打ち切り閾値"),
            f"{dc['min_box_height_px']:.0f} px", "configs/s3_detect.yaml",
            ("derived from the focal length: 1030 x 1.68 / 25 = 69.2",
             "выведен из фокуса: 1030 x 1.68 / 25 = 69.2",
             "焦点距離より算出：1030 x 1.68 / 25 = 69.2")),
        row(("Depth beyond which nobody remains",
             "Глубина, дальше которой людей нет", "人が残らない奥行き"),
            f"{tracks['foot_x_m'].max():.1f} m", "track/tracks.parquet",
            ("after the 70 px threshold was applied in S3",
             "после применения порога 70 px в S3",
             "S3で70pxの閾値を適用した後")),
    ]

    bins = "".join(
        f"<tr><td>{b['x_lo_m']:.0f}&ndash;{b['x_hi_m']:.0f} m</td>"
        f"<td class='num'>{b['n_rows']}</td>"
        f"<td class='num'>{b['median_box_h_px']:.0f} px</td></tr>"
        for b in dc["bins"])

    # ---- таблица 2: покрытие и стабильность -------------------------------- #
    tl = tracks.groupby("track_id").size()
    yaw_ok = orient[np.isfinite(orient["body_yaw_deg"])]
    # Разброс yaw внутри трека: устойчивость оценки, НЕ её правильность.
    spread = (yaw_ok.groupby("track_id")["body_yaw_deg"]
              .apply(lambda v: float(np.percentile(v, 75) - np.percentile(v, 25))))
    n_ev = len(events)
    low = int(events["low_confidence"].sum())
    graz = int(zf["grazing"].sum())
    ind = float((tracks["foot_source"] != "ankle").mean())

    t2 = [
        row(("Frames processed", "Кадров обработано", "処理フレーム"),
            f"{scope['n_frames_processed']} / {scope['n_frames_total']}",
            "det/frames_index.parquet",
            (f"{scope['n_frames_processed'] / scope['n_frames_total']:.0%} at "
             f"frame_stride 3",
              f"{scope['n_frames_processed'] / scope['n_frames_total']:.0%} при "
              f"frame_stride 3",
              f"frame_stride 3 で{scope['n_frames_processed'] / scope['n_frames_total']:.0%}")),
        row(("Tracks in total", "Треков всего", "追跡総数"),
            f"{int(tracks['track_id'].nunique())}", "track/tracks.parquet",
            ("after the box-height cutoff in S3",
             "после отсечки по высоте рамки в S3", "S3の枠高打ち切り後")),
        row(("Share of indirect foot points", "Доля косвенных опорных точек",
             "間接的な足元点の割合"),
            f"{ind:.0%}", "track/tracks.parquet",
            ("taken from the bottom of the box, not from the ankles",
             "опора взята от низа рамки, а не от голеностопа",
             "足首ではなく枠の下端から取得")),
        row(("Body orientation coverage", "Покрытие ориентации корпуса",
             "体の方位カバレッジ"),
            f"{yaw_ok['track_id'].nunique() / max(1, tracks['track_id'].nunique()):.0%}",
            "pose/orient.parquet",
            ("share of tracks for which yaw was estimated at all",
             "доля треков, у которых yaw вообще оценён",
             "yawが推定できた追跡の割合")),
        row(("Upper-garment class coverage", "Покрытие класса верха",
             "上衣クラスのカバレッジ"),
            f"{(attr['top_color_status'] == 'ok').mean():.0%}",
            "attr/tracks_attr.parquet",
            (f"{int((attr['top_color_status'] == 'ok').sum())} of {len(attr)} tracks",
             f"{int((attr['top_color_status'] == 'ok').sum())} треков из {len(attr)}",
             f"{len(attr)}件中{int((attr['top_color_status'] == 'ok').sum())}件")),
        row(("Track length, median", "Длина трека, медиана", "追跡長の中央値"),
            f"{tl.median():.0f}",
            "track/tracks.parquet",
            (f"frames; p10 {np.percentile(tl, 10):.0f}, p90 {np.percentile(tl, 90):.0f}",
             f"кадров; p10 {np.percentile(tl, 10):.0f}, p90 {np.percentile(tl, 90):.0f}",
             f"フレーム。p10 {np.percentile(tl, 10):.0f}、p90 {np.percentile(tl, 90):.0f}")),
        row(("Yaw spread within a track, median",
             "Разброс yaw внутри трека, медиана", "追跡内yawのばらつき中央値"),
            f"{spread.median():.0f}&deg;" if len(spread) else "n/a",
            "pose/orient.parquet",
            ("interquartile range. Stability of the estimate, NOT its correctness",
             "межквартильный размах. Стабильность оценки, НЕ правильность",
             "四分位範囲。推定の安定性であり正しさではありません")),
        row(("Events flagged low_confidence", "Событий помечено low_confidence",
             "low_confidenceの事象"),
            f"{low} / {n_ev} ({low / max(1, n_ev):.0%})", "attn/events.parquet",
            ("grazing angle, or orientation not measured; excluded from the aggregate",
             "скользящий угол или ориентация не измерена; в агрегат не идут",
             "斜入射または方位未計測。集計から除外")),
        row(("Frames at a grazing angle", "Кадров со скользящим углом",
             "斜入射のフレーム"),
            f"{graz}", "attn/track_zone_frames.parquet",
            ("the facade is seen edge-on, the intersection is unreliable",
             "фасад виден с ребра, пересечение ненадёжно",
             "ファサードを真横から見ており交差の信頼性が低い")),
    ]
    for z in ("M1", "M2", "M3", "M4"):
        e = met.get(f"orientation_rate_facade_{z}", {})
        lo, hi = e.get("ci95_low"), e.get("ci95_high")
        cis = (f"[{lo * 100:.1f}, {hi * 100:.1f}]" if lo is not None else "")
        t2.append(row(
            (f"Turned toward {z}", f"Доля повёрнутых, {z}", f"{z}を向いた割合"),
            f"{(e.get('value') or 0) * 100:.1f}%", "out/metrics.json",
            ((f"95% CI {cis}, Wilson, resampled by track",
              f"95% CI {cis} по Уилсону, пересэмплирование по трекам",
              f"95%CI {cis}、Wilson、追跡単位でリサンプリング")
             if lo is not None else
             ("no interval", "интервала нет", "区間なし"))))

    # ---- таблица 3: не измерено ------------------------------------------- #
    t3 = [
        ("Recall детекции по глубине",
         "Нужна ручная разметка ~200 боксов на разных глубинах. Отсечка по "
         "высоте рамки — это ПРОКСИ, а не recall."),
        ("MAE поворота головы",
         "Размечен только корпус. Голова оценивается отдельной колонкой и "
         "не проверена ничем."),
        ("Точность класса верха",
         "Нужна размеченная выборка 50+ кропов. Баланс белого применён, но "
         "его верность не подтверждена: коэффициенты считались из того же фона."),
        ("Независимая проверка масштаба",
         "Нужен замер рулеткой известного отрезка в кадре. Сейчас масштаб "
         "задан ростом, а рост поэтому не является проверкой."),
        ("MAE на плотной сцене",
         f"Разметка снята на {mae['video'] if mae else 'утреннем клипе'}, а метрики "
         f"считаны по вечернему часу. В плотной сцене люди перекрывают друг "
         f"друга сильнее, и MAE там может быть хуже."),
        ("Precision событий внимания",
         "Нужна ручная проверка ~100 событий: действительно ли человек был "
         "повёрнут к витрине в засчитанный момент."),
    ]
    t3_html = "".join(f"<tr><td><b>{esc(a)}</b></td><td>{esc(b)}</td></tr>"
                      for a, b in t3)

    hint3 = ("Число без эталона в первую таблицу не попадает."
             if args.hide_unmeasured else
             "Число без эталона в первую таблицу не попадает. Если метрики нет, "
             "она в третьей таблице, а не в первой с оговоркой.")
    sec3 = "" if args.hide_unmeasured else f"""<section>
  <h2>{t("bm.t3")}</h2>
  <div class="card t3"><div class="tbl"><table>
  <tr><th>{t("bm.th.miss")}</th><th>{t("bm.th.need")}</th></tr>
  {t3_html}</table></div></div>
</section>"""


    page = f"""<!doctype html><html lang="en" data-theme="light"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAM-01 benchmark</title><style>{CSS}
td.num{{text-align:right;font-variant-numeric:tabular-nums;font-weight:700;
  white-space:nowrap}}
table td{{font-size:13px}}
.t1{{border-left:4px solid #22c55e}} .t2{{border-left:4px solid #f59e0b}}
.t3{{border-left:4px solid #ef4444}}
</style>
{header_html("benchmark.html", [("nav.camera", "CAM-01 Kabukicho"), ("nav.window", f"{scope[chr(39)+chr(39)] if False else scope['duration_s']/60:.0f} min")])}

<div class="wrap">
<div class="eyebrow">{t("bm.eyebrow")}</div>
<h1>{t("bm.h1")}</h1>
<div class="warn" style="margin:0 0 26px"><b>{t("bm.warn")}</b>
<div>{t("bm.warn2")}</div></div>

<section>
  <h2>{t("bm.t1")}</h2>
  <p class="sub">{t("bm.t1sub")}</p>
  <div class="card t1"><div class="tbl"><table>
  <tr><th>{t("bm.th.q")}</th><th>{t("bm.th.v")}</th><th>{t("bm.th.s")}</th><th>{t("bm.th.h")}</th></tr>
  {''.join(t1)}</table></div></div>

  <div class="card t1" style="margin-top:16px">
    <div class="zname">{t("bm.bins")}</div>
    <p class="sub" style="margin:8px 0 12px">{t("bm.binsub")}</p>
    <div class="tbl"><table><tr><th>{t("bm.th.depth")}</th><th>{t("bm.th.rows")}</th><th>{t("bm.th.med")}</th></tr>
    {bins}</table></div></div>
</section>

<section>
  <h2>{t("bm.t2")}</h2>
  <p class="sub">{t("bm.t2sub")}</p>
  <div class="card t2"><div class="tbl"><table>
  <tr><th>{t("bm.th.q")}</th><th>{t("bm.th.v")}</th><th>{t("bm.th.s")}</th><th>{t("bm.th.m")}</th></tr>
  {''.join(t2)}</table></div></div>
</section>

{sec3}
</div>
<div id="lb"><img id="lbimg" alt=""><div class="meta" id="lbmeta"></div></div>
<script>window.__I18N__ = {i18n_payload()};</script>
<script>{JS}</script></html>"""

    atomic_write_text(args.out, page)
    print(f"готово: {args.out} ({args.out.stat().st_size / 1e3:.0f} КБ)")
    print(f"  таблица 1: {len(t1)} строк, таблица 2: {len(t2)}, "
          f"таблица 3: {'СКРЫТА' if args.hide_unmeasured else len(t3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
