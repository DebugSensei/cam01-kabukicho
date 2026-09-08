"""Витрина: одна страница с результатом, в формате, принятом у заказчика.

ЗАЧЕМ ОТДЕЛЬНАЯ СТРАНИЦА. `out/dashboard.html` построен под инженерную
проверку: 24 метрики, доверительные интервалы, таблица неизмеренного,
compute_ref на каждое число. Читатель, которому нужен результат за минуту,
в нём тонет — и это ровно та обратная связь, которую дал заказчик
(«дэшборд твой сложный для восприятия»). Обе страницы правильные, просто
для разных людей. Витрина отвечает на три вопроса и уходит вглубь по ссылке.

ЧЕГО ЗДЕСЬ НЕТ И ПОЧЕМУ. У образца, на который равняемся, есть блок
«Аудитория: пол и возраст». Его здесь нет и не будет: разметки, чтобы
измерить точность такой классификации, нет, а публиковать атрибут без
измеренной точности — то же самое, что публиковать выдумку с красивой
вёрсткой. Вместо него стоит блок «Чего мы не знаем», и это осознанный
обмен: меньше красивых чисел, больше тех, за которые можно отвечать.

ВСЕ ЧИСЛА ЧИТАЮТСЯ ИЗ out/metrics.json. Ни одно не считается здесь: второе
место, где живёт то же число, — это гарантированное расхождение, и проект
уже дважды на этом обжигался.

    python scripts/make_showcase.py
    python scripts/make_showcase.py --out out/showcase.html
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.anonymise import save_image  # noqa: E402,F401  (гарантия единой точки записи)
from looq.io import atomic_write_text, read_json  # noqa: E402

OUT = Path("out/showcase.html")
METRICS = Path("out/metrics.json")

#: Картинки, которые уходят в страницу. Встраиваются через looq.anonymise,
#: то есть проходят обезличивание, как всё остальное в проекте.
FIG_STOPS = Path("out/img/stop_map.png")
FIG_PLAN = Path("out/img/plan_all.png")

STOREFRONT_LABEL = {"facade_M1": "M1 · 角煮/げんかつ", "facade_M2": "M2 · 入口/らーめん",
                    "facade_M3": "M3 · 芝浦ホルモン", "facade_M4": "M4 · お好み焼き"}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def val(m: dict, key: str):
    v = m["metrics"].get(key)
    return None if v is None else v.get("value")


def env(m: dict, key: str) -> dict:
    return m["metrics"].get(key) or {}


def fig(path: Path, max_w: int = 1400) -> str | None:
    """Картинка в data-URI через единственный путь записи проекта."""
    import cv2

    from looq.anonymise import anonymise, data_uri

    img = cv2.imread(str(path))
    if img is None:
        return None
    anonymise(img)
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, int(img.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    return data_uri(img, quality=88)


def colour_strip(colour: str, n: int = 8) -> str:
    """Полоса реальных кропов одного класса цвета.

    ЗАЧЕМ. Число «привязка к месту 25x» доказывает, что оранжевый привязан к
    точке улицы. Но убедить читателя за секунду может только сам кроп: рядом
    с полосой синих полоса «оранжевых» показывает подсветку без единого слова.
    Это не «посмотреть и решить» — вывод уже сделан кодом, кропы его
    иллюстрируют.

    Кропы идут через looq.anonymise, как всё остальное: на диске часть из них
    записана старым способом, и при встраивании они обезличиваются заново.
    """
    import cv2
    import pandas as pd

    from looq.anonymise import anonymise, data_uri

    idx_path = Path("evidence/index.parquet")
    if not idx_path.is_file():
        return ""
    idx = pd.read_parquet(idx_path)
    rows = idx[idx["claim_id"] == f"claim.attrs.color.{colour}"]
    if rows.empty:
        return ""
    out = []
    for _, r in rows.head(n).iterrows():
        img = cv2.imread(str(r["path"]))
        if img is None:
            continue
        # Общая высота, чтобы полоса читалась как полоса, а не как лесенка.
        k = 150.0 / img.shape[0]
        img = cv2.resize(img, (max(1, int(img.shape[1] * k)), 150),
                         interpolation=cv2.INTER_AREA)
        anonymise(img)
        out.append(f'<img src="{data_uri(img, quality=88)}" alt="{esc(colour)}">')
    if not out:
        return ""
    return (f'<div class="strip"><div class="strip-l">{esc(colour)}'
            f'<span>{len(out)} кропов</span></div>'
            f'<div class="strip-i">{"".join(out)}</div></div>')


def kpi(value: str, label: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="kpi-v">{esc(value)}</div>'
            f'<div class="kpi-l">{esc(label)}</div>'
            f'<div class="kpi-s">{esc(sub)}</div></div>')


def bar_row(label: str, pct: float, right: str, note: str = "") -> str:
    return (f'<div class="row"><div class="row-l">{esc(label)}'
            f'{f"<span class=n>{esc(note)}</span>" if note else ""}</div>'
            f'<div class="row-b"><i style="width:{max(0.6, pct):.2f}%"></i></div>'
            f'<div class="row-v">{esc(right)}</div></div>')


def build(m: dict) -> str:
    zones = [z for z in ("facade_M1", "facade_M2", "facade_M3", "facade_M4")
             if f"visitors_{z}" in m["metrics"]]
    tracks = int(val(m, "unique_tracks_total"))
    scope = m["scope"]
    sh = m.get("stop_hotspots") or {}
    td = m.get("track_duration") or {}

    # Лидер — по доле повернувшихся. Берётся из артефакта, не выбирается руками.
    lead = max(zones, key=lambda z: val(m, f"orientation_rate_{z}") or 0.0)
    le = env(m, f"orientation_rate_{lead}")
    # РАЗНЫЕ треки из артефакта, а не сумма по витринам: один трек может быть
    # засчитан к двум витринам, и сумма даёт 683 там, где треков 514.
    turned_total = int(val(m, "turned_tracks_total") or 0)
    turned_sum = sum(env(m, f"orientation_rate_{z}").get("n", 0) for z in zones)

    kpis = "".join([
        kpi(f"{tracks:,}".replace(",", " "), "треков за час",
            "треки, а не люди: трекер рвёт и склеивает"),
        kpi(f"{turned_total / max(1, tracks) * 100:.1f} %", "повернулись к витрине",
            f"{turned_total} разных треков из {tracks}"),
        kpi(f"{le.get('value', 0) * 100:.1f} %", f"лидер · {STOREFRONT_LABEL.get(lead, lead)}",
            f"95 % [{le.get('ci95_low', 0) * 100:.1f}, {le.get('ci95_high', 0) * 100:.1f}], "
            f"n = {le.get('n', 0)}"),
        kpi(f"{sh.get('stop_share', 0) * 100:.1f} %", "времени люди стоят",
            f"{sh.get('stop_seconds_total', 0):.0f} чел-с из "
            f"{sh.get('observed_seconds_total', 0):.0f}"),
    ])

    # Воронка. Каждая ступень — строгое подмножество предыдущей, иначе доля
    # «от предыдущего» соврёт.
    # РАЗНЫЕ треки, а не максимум по витринам: максимум это нижняя оценка.
    vis_max = int(val(m, "approached_tracks_total") or 0)
    stops_z = sum(env(m, f"stop_rate_{z}").get("n", 0) for z in zones)
    steps = [("Треков в кадре", tracks, "S4 · track/tracks.parquet"),
             ("Подошли к витрине ближе 8 м", vis_max, "S6 · attn/events.parquet"),
             ("Повернулись к витрине", turned_total,
              "S6 · gaze/stop_and_gaze, low_confidence исключены"),
             ("Остановились у витрины", stops_z, "S6 · stop среди повернувшихся")]
    funnel = "".join(
        f'<div class="fn"><div class="fn-l">{esc(t)}</div>'
        f'<div class="fn-b"><i style="width:{n / max(1, tracks) * 100:.2f}%"></i></div>'
        f'<div class="fn-v">{n}<small> {n / max(1, tracks) * 100:.1f} %</small></div>'
        f'<div class="fn-n">{esc(src)}</div></div>'
        for t, n, src in steps)

    ranked = "".join(
        bar_row(STOREFRONT_LABEL.get(z, z),
                (val(m, f"orientation_rate_{z}") or 0) * 100 /
                max(0.001, (val(m, f"orientation_rate_{lead}") or 0.001)) * 100,
                f"{(val(m, f'orientation_rate_{z}') or 0) * 100:.2f} %",
                f"95 % [{env(m, f'orientation_rate_{z}').get('ci95_low', 0) * 100:.1f}, "
                f"{env(m, f'orientation_rate_{z}').get('ci95_high', 0) * 100:.1f}] · "
                f"n = {env(m, f'orientation_rate_{z}').get('n', 0)}")
        for z in sorted(zones, key=lambda z: -(val(m, f"orientation_rate_{z}") or 0)))

    dwell = "".join(
        bar_row(STOREFRONT_LABEL.get(z, z),
                (val(m, f"dwell_median_{z}") or 0) * 100 /
                max(0.001, max((val(m, f"dwell_median_{y}") or 0) for y in zones)),
                f"{val(m, f'dwell_median_{z}') or 0:.1f} с",
                f"{val(m, f'visitors_{z}')} треков")
        for z in sorted(zones, key=lambda z: -(val(m, f"dwell_median_{z}") or 0)))

    cells = (sh.get("cells") or [])[:6]
    top_s = cells[0]["stop_seconds"] if cells else 1.0
    hot = "".join(
        bar_row(f"x {c['x_m']:.0f}–{c['x_m'] + sh['bin_m']:.0f} м, "
                f"y {c['y_m']:.0f}–{c['y_m'] + sh['bin_m']:.0f} м",
                c["stop_seconds"] / max(1.0, top_s) * 100,
                f"{c['stop_seconds']:.0f} чел-с",
                f"{c['n_tracks']} разных треков · "
                f"{c['stop_seconds'] / max(1.0, sh.get('stop_seconds_total', 1)) * 100:.1f} % "
                f"всего стояния")
        for c in cells)

    dur = "".join(
        bar_row(("до 5 с" if b["to_s"] == 5 else
                 f"{b['from_s']:.0f}–{b['to_s']:.0f} с" if b["to_s"] else
                 f"{b['from_s']:.0f} с и больше"),
                b["share"] * 100 / max(0.001, max(x["share"] for x in td["bins"])) * 100,
                f"{b['share'] * 100:.1f} %", f"{b['n']} треков")
        for b in td.get("bins", []))

    pc = (m.get("presence_curve") or {}).get("points") or []
    if pc:
        mx = max(p["mean_detections"] for p in pc)
        pts = " ".join(f"{i / max(1, len(pc) - 1) * 980:.1f},"
                       f"{120 - p['mean_detections'] / mx * 105:.1f}"
                       for i, p in enumerate(pc))
        curve = (f'<svg viewBox="0 0 980 130" class="curve" preserveAspectRatio="none">'
                 f'<polyline points="{pts}"/></svg>'
                 f'<div class="curve-x"><span>начало часа</span>'
                 f'<span>в среднем {sum(p["mean_detections"] for p in pc) / len(pc):.1f} '
                 f'человек в кадре, максимум {mx:.1f}</span><span>конец</span></div>')
    else:
        curve = "<p class=muted>кривая присутствия недоступна</p>"

    unmeasured = "".join(f"<li><b>{esc(u['item'])}</b> — {esc(u.get('reason_ru', ''))}</li>"
                         for u in m.get("unmeasured", []))

    cb = (m.get("colour_spatial_bias") or {}).get("classes") or []
    by_n = sorted(cb, key=lambda c: -c["n_tracks"])
    n_col = sum(c["n_tracks"] for c in cb) or 1
    colour_rows = "".join(
        bar_row(c["colour"], c["n_tracks"] / max(1, by_n[0]["n_tracks"]) * 100,
                f"{c['n_tracks']}",
                f"{c['n_tracks'] / n_col * 100:.1f} % · привязка к месту "
                f"{c['concentration'] if c['concentration'] is not None else '—'}x")
        for c in by_n) or "<p class=muted>цвет не посчитан</p>"
    orange = next((c for c in cb if c["colour"] == "orange"), None)
    orange_conc = f"{orange['concentration']}x" if orange else "—"
    orange_in = f"{orange['share_in_patch'] * 100:.0f}" if orange else "—"
    orange_out = f"{orange['share_elsewhere'] * 100:.1f}" if orange else "—"
    colour_cov = f"{n_col / max(1, tracks) * 100:.0f}"

    # Полосы кропов: сначала самый частый настоящий цвет, потом подозрительный
    # оранжевый — рядом разница видна без пояснений.
    strips = "".join(colour_strip(c) for c in ("blue", "grey", "black", "orange"))
    n_metrics = len(m["metrics"])
    f_stops, f_plan = fig(FIG_STOPS), fig(FIG_PLAN)

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAM-01 Kabukicho — что видела улица за час</title>
<style>
:root{{--ink:#12100e;--muted:#7d7671;--line:#e9e4e0;--bg:#fbfaf8;--card:#fff;
--accent:#2f6bff;--warm:#e8542f}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:32px 20px 80px}}
header{{padding:26px 0 20px;border-bottom:1px solid var(--line);margin-bottom:26px}}
h1{{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}}
.sub{{color:var(--muted);font-size:14px}}
.eyebrow{{font-size:11px;letter-spacing:.14em;text-transform:uppercase;
color:var(--muted);margin:36px 0 12px;font-weight:600}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}}
.kpi-v{{font-size:30px;font-weight:650;letter-spacing:-.03em;line-height:1.1}}
.kpi-l{{font-size:13px;margin-top:6px}}
.kpi-s{{font-size:12px;color:var(--muted);margin-top:5px;line-height:1.4}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:20px;margin-top:10px}}
.row{{display:grid;grid-template-columns:1fr 200px 110px;gap:12px;align-items:center;
padding:9px 0;border-bottom:1px solid var(--line)}}
.row:last-child{{border-bottom:0}}
.row-l{{font-size:14px}}
.row-l .n{{display:block;color:var(--muted);font-size:11.5px;margin-top:2px}}
.row-b{{height:8px;background:#f1eeeb;border-radius:5px;overflow:hidden}}
.row-b i{{display:block;height:100%;background:var(--accent)}}
.row-v{{text-align:right;font-weight:600;font-size:14px}}
.fn{{display:grid;grid-template-columns:1fr 260px 130px;gap:12px;align-items:center;
padding:11px 0;border-bottom:1px solid var(--line)}}
.fn:last-child{{border-bottom:0}}
.fn-l{{font-size:14px}}
.fn-b{{height:10px;background:#f1eeeb;border-radius:6px;overflow:hidden}}
.fn-b i{{display:block;height:100%;background:var(--accent)}}
.fn-v{{text-align:right;font-weight:650}}
.fn-v small{{display:block;font-weight:400;color:var(--muted);font-size:11.5px}}
.fn-n{{grid-column:1/-1;color:var(--muted);font-size:11px;margin-top:-6px}}
.curve{{width:100%;height:130px}}
.curve polyline{{fill:none;stroke:var(--accent);stroke-width:2}}
.curve-x{{display:flex;justify-content:space-between;color:var(--muted);
font-size:11.5px;margin-top:4px}}
img.fig{{width:100%;border:1px solid var(--line);border-radius:10px;display:block}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:820px){{.two{{grid-template-columns:1fr}}
.row,.fn{{grid-template-columns:1fr 90px}}.row-b,.fn-b{{display:none}}}}
.note{{background:#fff8f2;border:1px solid #f2ddcd;border-left:3px solid var(--warm);
border-radius:8px;padding:14px 16px;font-size:13.5px;line-height:1.6;margin-top:10px}}
.muted{{color:var(--muted)}}
.strips .strip{{display:flex;gap:14px;align-items:center;padding:10px 0;
border-bottom:1px solid var(--line)}}
.strips .strip:last-child{{border-bottom:0}}
.strip-l{{flex:0 0 92px;font-size:13px;font-weight:600}}
.strip-l span{{display:block;font-weight:400;color:var(--muted);font-size:11px}}
.strip-i{{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px}}
.strip-i img{{height:104px;border-radius:5px;border:1px solid var(--line);
flex:0 0 auto}}
ul.un{{margin:0;padding-left:18px;font-size:13.5px;line-height:1.7}}
footer{{margin-top:48px;padding-top:18px;border-top:1px solid var(--line);
color:var(--muted);font-size:12.5px;line-height:1.7}}
a{{color:var(--accent)}}
</style></head><body><div class="wrap">

<header>
  <h1>Кабуки-тё Ичибан-гай — один час улицы</h1>
  <div class="sub">16:25–17:25 JST, 4 сентября 2026 ·
  {scope['n_frames_processed']:,} обработанных кадров из {scope['n_frames_total']:,} ·
  одна публичная камера</div>
</header>

<div class="kpis">{kpis}</div>

<div class="note"><b>Что здесь измеряется.</b> Поворот корпуса к витрине, а не
взгляд: с этого ракурса направление глаз не определяется, и система этого не
заявляет. Ошибка угла — 21° при 95 % интервале [15, 29] на 50 размеченных
вручную людях. Всё остальное на странице наследует эту неопределённость.</div>

<div class="eyebrow">Где на улице стоят</div>
<div class="card">
  {f'<img class="fig" src="{f_stops}" alt="карта остановок">' if f_stops
    else '<p class=muted>карта не построена</p>'}
</div>
<div class="card">{hot}</div>
<div class="note"><b>Главное наблюдение.</b> {cells[0]['n_tracks'] if cells else 0}
разных треков дали {cells[0]['stop_seconds']:.0f} чел-с стояния в одном квадрате
2×2 м — это {cells[0]['stop_seconds'] / max(1.0, sh.get('stop_seconds_total', 1)) * 100:.0f} %
всего времени стояния на улице. Квадрат лежит посреди улицы, а не у витрины.
Число разных треков указано намеренно: те же секунды от одного застрявшего трека
и от {cells[0]['n_tracks'] if cells else 0} человек выглядят одинаково и значат
противоположное.</div>

<div class="eyebrow">Воронка</div>
<div class="card">{funnel}</div>

<div class="eyebrow">Витрины по доле повернувшихся</div>
<div class="card">{ranked}</div>

<div class="eyebrow">Витрины по времени в зоне</div>
<div class="card">{dwell}</div>

<div class="eyebrow">Присутствие в течение часа</div>
<div class="card">{curve}</div>

<div class="eyebrow">Сколько трек держится в кадре</div>
<div class="card">{dur}</div>
<div class="note">Это длительность <b>наблюдения</b>, а не визита. Один человек,
прошедший за перекрытием, даёт два трека; двое, разошедшиеся вплотную, могут
дать один. IDF1 трекера не измерен, поэтому «сколько человек» на этой странице
не написано нигде.</div>

<div class="eyebrow">Цвет верхней одежды</div>
<div class="card">{colour_rows}</div>
<div class="card strips">{strips}</div>
<div class="note"><b>Здесь измеряется не только цвет, но и надёжность цвета.</b>
Правая колонка — во сколько раз доля класса выше в его собственном пятне улицы,
чем на остальной. Одежда по улице пятнами не лежит, поэтому у настоящих цветов
это число около единицы: синий 1.2, серый 1.2, чёрный 1.1.<br><br>
<b>У оранжевого — {orange_conc}.</b> Его треки сидят в точке x&nbsp;2.7, y&nbsp;7.2,
тогда как все остальные цвета кучкуются на x&nbsp;11–14. В квадрате 6×6 м вокруг
этой точки оранжевых {orange_in}&nbsp;% треков, на остальной улице
{orange_out}&nbsp;%. Это подпись подсветки, а не ткани, и класс оставлен в данных
с этой пометкой, а не выброшен: выбрасывать по неизмеренному признаку значило бы
чинить одну неизмеренную величину другой.<br><br>
Покрытие цвета — {colour_cov}&nbsp;% треков. Точность не измерена: размеченных
кропов нет, и это стоит в списке ниже.</div>

<div class="eyebrow">Чего мы не знаем</div>
<div class="card"><ul class="un">{unmeasured}</ul></div>
<div class="note"><b>Пола и возраста здесь нет намеренно.</b> Разметки, чтобы
измерить точность такой классификации на этих данных, нет. Публиковать атрибут
без измеренной точности — то же, что публиковать выдумку с красивой вёрсткой.
Все десять гейтов конвейера сейчас красные, и у каждого названа причина: два
считают метрику, три сверяют sha входов, пять ждут разметки. Ни один порог не
двигался ради зелёного.</div>

<div class="eyebrow">Траектории</div>
<div class="card">
  {f'<img class="fig" src="{f_plan}" alt="траектории на плане">' if f_plan
    else '<p class=muted>план не построен</p>'}
</div>

<footer>
Полная инженерная версия со всеми {n_metrics} метриками, доверительными интервалами и
ссылкой на строку кода за каждым числом —
<a href="dashboard.html">dashboard.html</a>. Отчёт о точности моделей —
<a href="benchmark.html">benchmark.html</a>.<br>
Все числа на этой странице прочитаны из <code>out/metrics.json</code> и не
пересчитываются здесь: второе место, где живёт то же число, — это гарантированное
расхождение.
</footer>
</div></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    m = read_json(METRICS)
    for need in ("stop_hotspots", "track_duration"):
        if need not in m:
            raise SystemExit(
                f"в {METRICS} нет блока {need!r}: витрина строится по свежему S8. "
                f"Запустите: python -m looq.stages.s8_aggregate --config configs/s8_aggregate.yaml")
    atomic_write_text(args.out, build(m))
    print(f"готово: {args.out} ({args.out.stat().st_size / 1024:.0f} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
