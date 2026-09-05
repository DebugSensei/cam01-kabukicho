"""S9 report — одна HTML-страница отчёта о достоверности.

Каждое число берётся из out/metrics.json и несёт при себе этап-источник и
метрику качества этого этапа. Числа, посчитать которые не по чему, выводятся
как НЕ ИЗМЕРЕНО с причиной, а не как ноль и не прочерком.

Страница self-contained: картинки вшиты как data:URI, внешних запросов нет.
Реплея нет — только агрегаты и контактные листы пруф-кадров.

Слова «взгляд», «смотрит» в русском тексте запрещены: мы меряем поворот
корпуса и головы, а не направление взгляда.

    python -m looq.stages.s9_report --config configs/s9_report.yaml
"""

from __future__ import annotations

import argparse
import base64
import html
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from looq import STATUS_OK, STATUS_SKELETON
from looq.geometry import ORIENTATION_DISCLAIMER
from looq.evidence import EvidenceError, blur_face_region
from looq.io import ConfigError, RunManifest, atomic_write_text, load_config, read_json, require
from looq.stages._base import StageError, read_artifact_status

STAGE = "s9_report"
OUTPUT = "out/report.html"

CSS = """
:root{--bg:#12141a;--card:#1b1e27;--ink:#e9ecf3;--dim:#98a0b3;--line:#2b3040;
--ok:#5ad18e;--warn:#ffb454;--bad:#ff6b6b;--accent:#6aa9ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:18px;margin:34px 0 12px}
.sub{color:var(--dim);margin:0 0 22px}
.banner{background:#3a1010;border:1px solid var(--bad);border-radius:10px;
padding:16px 18px;margin:0 0 24px}
.banner b{color:var(--bad);font-size:16px;display:block;margin-bottom:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px 18px;margin:0 0 14px}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);
vertical-align:top}
th{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.big{font-size:22px;font-weight:600}
.unmeasured{color:var(--warn)}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
border:1px solid var(--line);color:var(--dim)}
.pill.bad{border-color:var(--bad);color:var(--bad)}
.src{color:var(--dim);font-size:12px}
.bar{height:10px;background:#232838;border-radius:5px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent)}
.sheet img{max-width:100%;border:1px solid var(--line);border-radius:8px}
.foot{color:var(--dim);font-size:12px;margin-top:36px;border-top:1px solid var(--line);
padding-top:14px}
code{background:#0e1017;padding:1px 5px;border-radius:4px;font-size:12px}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt(v, unit: str) -> str:
    if v is None:
        return '<span class="unmeasured">не измерено</span>'
    if unit == "frac":
        return f"{v * 100:.1f}%"
    if unit == "s":
        return f"{v:.1f} с"
    if unit in ("tracks", "count"):
        return f"{int(v)}"
    return f"{v:.3f}"


def _metric_row(key: str, m: dict) -> str:
    q = m.get("quality") or {}
    qtxt = (f"{q.get('metric')} — не измерена" if not q.get("measured")
            else f"{q.get('metric')} = {q.get('value')}")
    ci = ""
    if m.get("ci95_low") is not None:
        ci = (f'<div class="src">ДИ 95%: {m["ci95_low"] * 100:.1f}–'
              f'{m["ci95_high"] * 100:.1f}% ({m.get("ci_method")})</div>')
    ref = m.get("compute_ref") or {}
    caveats = "".join(f'<div class="src">• {_esc(c)}</div>'
                      for c in (m.get("caveats_ru") or []))
    reason = (f'<div class="src">причина: {_esc(m["reason_ru"])}</div>'
              if m.get("reason_ru") else "")
    return f"""<tr>
<td>{_esc(key)}</td>
<td class="num big">{_fmt(m.get("value"), m.get("unit", ""))}</td>
<td class="num">{m.get("n", "")}</td>
<td><span class="pill">{_esc(m.get("source_stage"))}</span>
<div class="src">{_esc(qtxt)}</div>
<div class="src">код: <code>{_esc(ref.get("file"))}:{_esc(ref.get("line"))}
{_esc(ref.get("function"))}</code></div>{ci}{reason}{caveats}</td></tr>"""


def _presence_svg(points: list[dict], bin_s: float) -> str:
    if not points:
        return "<p class='unmeasured'>кривая присутствия не построена</p>"
    xs = [p["t_start_s"] for p in points]
    ys = [p["mean_detections"] for p in points]
    w, h, pad = 1000, 220, 34
    ymax = max(ys) * 1.15 or 1.0
    xspan = (max(xs) - min(xs)) or 1.0

    def px(x):
        return pad + (x - min(xs)) / xspan * (w - 2 * pad)

    def py(y):
        return h - pad - (y / ymax) * (h - 2 * pad)

    poly = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    area = f"{px(xs[0]):.1f},{h - pad} " + poly + f" {px(xs[-1]):.1f},{h - pad}"
    ticks = ""
    for i in range(0, len(xs), max(1, len(xs) // 8)):
        ticks += (f'<text x="{px(xs[i]):.0f}" y="{h - 10}" fill="#98a0b3" '
                  f'font-size="11" text-anchor="middle">{xs[i]:.0f}с</text>')
    for frac in (0.0, 0.5, 1.0):
        y = py(ymax * frac)
        ticks += (f'<line x1="{pad}" y1="{y:.1f}" x2="{w - pad}" y2="{y:.1f}" '
                  f'stroke="#2b3040"/>'
                  f'<text x="6" y="{y + 4:.1f}" fill="#98a0b3" font-size="11">'
                  f'{ymax * frac:.1f}</text>')
    return f"""<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto">
{ticks}<polygon points="{area}" fill="#6aa9ff22"/>
<polyline points="{poly}" fill="none" stroke="#6aa9ff" stroke-width="2"/></svg>"""


def _contact_sheet(index_path: Path, claim_id: str, max_n: int,
                   cell=(120, 200), cols=6) -> str | None:
    """Контактный лист пруф-кадров одного утверждения, вшитый в страницу."""
    import cv2
    import pandas as pd

    if not index_path.is_file():
        return None
    df = pd.read_parquet(index_path)
    df = df[df["claim_id"] == claim_id].sort_values("stratum").head(max_n)
    if df.empty:
        return None
    cw, ch = cell
    cells = []
    # Кропы на диске писались разными прогонами и несут разные пороги
    # обезличивания. Страница обязана показывать ТЕКУЩИЙ порог независимо от
    # того, когда кроп записан: три из четырёх листов иначе показывали 0.22,
    # который владелец забраковал. Повторное размытие уже размытого безвредно.
    priv = load_config("configs/evidence.yaml").get("privacy") or {}
    need = ("face_blur_top_frac", "blur_kernel_frac", "blur_sigma_frac",
            "pixelate_factor")
    missing = [k for k in need if k not in priv]
    if missing:
        raise StageError(
            f"в configs/evidence.yaml нет ключей приватности {missing}: "
            f"вшивать кропы, не зная порога обезличивания, нельзя (правило 9)")

    for _, r in df.iterrows():
        img = cv2.imread(str(r["path"]))
        if img is None:
            continue
        try:
            img, _ = blur_face_region(
                img,
                top_frac=float(priv["face_blur_top_frac"]),
                kernel_frac=float(priv["blur_kernel_frac"]),
                sigma_frac=float(priv["blur_sigma_frac"]),
                pixelate_factor=int(priv["pixelate_factor"]))
        except EvidenceError:
            pass          # область уже однородна: кроп обезличен своей стадией
        hgt, wid = img.shape[:2]
        sc = min(cw / wid, ch / hgt)
        res = cv2.resize(img, (max(1, int(wid * sc)), max(1, int(hgt * sc))),
                         interpolation=cv2.INTER_NEAREST)
        cell_img = np.full((ch, cw, 3), 27, np.uint8)
        y0, x0 = (ch - res.shape[0]) // 2, (cw - res.shape[1]) // 2
        cell_img[y0:y0 + res.shape[0], x0:x0 + res.shape[1]] = res
        cv2.putText(cell_img, f"s{int(r['stratum'])} c{r['confidence']:.2f}",
                    (4, ch - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cells.append(cell_img)
    if not cells:
        return None
    rows = []
    for i in range(0, len(cells), cols):
        chunk = cells[i:i + cols]
        while len(chunk) < cols:
            chunk.append(np.full_like(cells[0], 27))
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    ok, buf = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def run(cfg: dict[str, Any], manifest: RunManifest) -> str:
    if read_artifact_status("out/metrics.json") == STATUS_SKELETON:
        raise StageError("out/metrics.json помечен status=skeleton: S8 не отработал")
    m = read_json("out/metrics.json")
    metrics = m["metrics"]
    scope = m["scope"]
    scale_known = bool(m.get("scale_known", True))

    zones = read_json("zones/zones.geojson")
    facades = [f["properties"] for f in zones["features"]
               if f["properties"]["zone_type"] == "facade"]

    banner = ""
    if not scale_known:
        reasons = read_json("calib/homography.json").get("stub_reasons_ru", [])
        lst = "".join(f"<div>• {_esc(r)}</div>" for r in reasons)
        banner = (f'<div class="banner"><b>КАЛИБРОВКА НЕ ПРОЙДЕНА '
                  f'({_esc(m.get("calib_status"))})</b>'
                  f'Метры не показаны нигде: длины и скорости в УСЛОВНЫХ единицах, '
                  f'углы приближённые. Числа годятся для сравнения витрин между '
                  f'собой и ни для чего больше.{lst}</div>')

    # Витрины.
    rows_z = []
    vmax = max([metrics[f"visitors_{f['zone_id']}"]["value"] or 0
                for f in facades] or [1])
    for f in facades:
        zid = f["zone_id"]
        v = metrics[f"visitors_{zid}"]
        st = metrics[f"stop_rate_{zid}"]
        orr = metrics[f"orientation_rate_{zid}"]
        dw = metrics[f"dwell_median_{zid}"]
        width = (v["value"] or 0) / vmax * 100
        rows_z.append(f"""<tr><td>{_esc(f['name_ru'])}<div class="src">{_esc(zid)}</div></td>
<td class="num big">{_fmt(v['value'], 'tracks')}
<div class="bar"><i style="width:{width:.0f}%"></i></div></td>
<td class="num">{_fmt(st['value'], 'frac')}</td>
<td class="num">{_fmt(orr['value'], 'frac')}</td>
<td class="num">{_fmt(dw['value'], 's')}</td></tr>""")

    sheets = ""
    idx = Path("evidence/index.parquet")
    max_n = int((cfg.get("report") or {}).get("min_evidence_per_claim", 6)) * 2
    for claim in (cfg.get("evidence_claims") or []) + [
            "claim.detect.near_half", "claim.detect.far_half",
            "claim.orient.body_yaw", "claim.track.foot_source_indirect"]:
        b64 = _contact_sheet(idx, claim, max_n)
        if b64:
            sheets += (f'<div class="card sheet"><b>{_esc(claim)}</b>'
                       f'<div class="src">кадры отобраны стратифицированно по '
                       f'уверенности: s0 нижняя страта, s2 верхняя. Не top-N.</div>'
                       f'<img src="data:image/jpeg;base64,{b64}"></div>')
    if not sheets:
        sheets = ('<div class="card unmeasured">пруф-кадры не собраны — '
                  'ни одно утверждение не подкреплено кадрами</div>')

    lim = m.get("limitations", [])
    limitations = "".join(
        f"<tr><td>{_esc(x['item'])}</td><td>{_esc(x['text_ru'])}</td>"
        f"<td>{_esc(x['consequence_ru'])}</td></tr>" for x in lim) or         "<tr><td colspan=3>ограничений не зафиксировано</td></tr>"

    unmeasured = "".join(
        f"<tr><td>{_esc(u['item'])}</td><td>{_esc(u['reason_ru'])}</td></tr>"
        for u in m.get("unmeasured", []))

    rec = m.get("reconciliation", {})
    rec_rows = "".join(
        f"<tr><td>{_esc(c['check_id'])}</td>"
        f"<td>{'ok' if c['passed'] else 'ПРОВАЛ'}</td>"
        f"<td class='src'>{_esc(c['detail'])}</td></tr>"
        for c in rec.get("checks", []))

    other = "".join(_metric_row(k, v) for k, v in metrics.items()
                    if not any(k.startswith(p) for p in
                               ("visitors_", "stop_rate_", "orientation_rate_",
                                "dwell_median_")))

    # Строка возврата и английская врезка. Страница публикуется на Pages, и
    # читатель попадает сюда с англоязычного дашборда: без ссылок назад это
    # тупик, а без врезки — русская страница без объяснения, что это такое.
    # Ссылки — простой разметкой: тянуть шапку из scripts/ в стадию нельзя,
    # этапы не зависят от инструментов отрисовки.
    nav = ('<div class="src" style="margin-bottom:18px">'
           '<a href="dashboard.html" style="color:#7aa2ff">&#8592; Dashboard</a>'
           ' &middot; <a href="benchmark.html" style="color:#7aa2ff">Benchmark</a>'
           '</div>')
    lede = ('<div class="card"><b>What this page is</b>'
            '<div class="src">The trustworthiness report: every published number '
            'with the stage that produced it, that stage quality metric — or an '
            'explicit "not measured" — and the file, function and line that '
            'computed it. The dashboard shows the numbers; this page shows where '
            'each one comes from.<br><br>'
            'The body below is in Russian. It is the working engineering report, '
            'written in the language the project was built in, and translating it '
            'would put a second source of truth next to the first. The dashboard '
            'and the benchmark page are in English, Russian and Japanese.'
            '</div></div>')

    doc = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<title>CAM-01 Kabukicho — provenance report</title><style>{CSS}</style>
<div class="wrap">
{nav}
<h1>CAM-01 Kabukicho</h1>
<p class="sub">Отчёт о достоверности. Обработано кадров
{scope['n_frames_processed']} из {scope['n_frames_total']}
({scope['processed_frac'] * 100:.1f}%), длительность {scope['duration_s']:.0f} с.</p>
{banner}
{lede}

<div class="card"><b>{_esc(ORIENTATION_DISCLAIMER)}</b>
<div class="src">Везде ниже «повёрнут к витрине» означает поворот корпуса или
головы. Направление взгляда не измеряется и не оценивается.</div></div>

<h2>Присутствие в кадре</h2>
<div class="card">{_presence_svg(m['presence_curve']['points'],
                                 m['presence_curve']['bin_s'])}
<div class="src">Среднее число детекций на обработанный кадр, корзина
{m['presence_curve']['bin_s']:.0f} с. Знаменатель — только обработанные кадры:
необработанные не превращаются в «людей не было».</div></div>

<h2>Витрины</h2>
<div class="card"><table>
<tr><th>витрина</th><th>треков рядом</th><th>остановились</th>
<th>повёрнуты к витрине</th><th>медиана времени</th></tr>
{''.join(rows_z)}</table>
<div class="src">Доли считаются от числа треков, побывавших у витрины.
Трек — не человек: трекер рвёт траектории и склеивает разных.</div></div>

<h2>Остальные метрики</h2>
<div class="card"><table>
<tr><th>метрика</th><th>значение</th><th>n</th><th>источник и качество</th></tr>
{other}</table></div>

<h2>Сходимость</h2>
<div class="card"><table><tr><th>проверка</th><th>итог</th><th>детали</th></tr>
{rec_rows}</table></div>

<h2>Ограничения</h2>
<div class="card"><table><tr><th>что</th><th>суть</th><th>следствие</th></tr>
{limitations}</table>
<div class="src">Эти оговорки относятся ко ВСЕМ числам выше и не отменяются
ни одной из них.</div></div>

<h2>Что не измерено</h2>
<div class="card"><table><tr><th>величина</th><th>почему</th></tr>
{unmeasured}</table>
<div class="src">Ни одно из этих значений не заменено нулём или прочерком.</div></div>

<h2>Пруф-кадры</h2>
{sheets}

<div class="foot">Каждое число посчитано кодом из артефакта на диске; в колонке
«источник» указан файл и строка. Лица на кропах обезличены до записи на диск
(пикселизация и гаусс). Реплея в отчёте нет.</div>
</div></html>"""
    manifest.note("n_metrics_rendered", len(metrics))
    manifest.note("scale_known", scale_known)
    print(f"[{STAGE}] метрик на странице {len(metrics)}, витрин {len(facades)}")
    return doc


def main(argv=None) -> int:
    _t0 = time.time()
    parser = argparse.ArgumentParser(prog=f"python -m looq.stages.{STAGE}")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest: RunManifest | None = None
    try:
        cfg = load_config(args.config)
        if cfg.get("stage") != STAGE:
            raise ConfigError(f"конфиг {args.config} объявляет stage={cfg.get('stage')!r}")
        manifest = RunManifest(STAGE, cfg)
        manifest.start()
        doc = run(cfg, manifest)
        atomic_write_text(OUTPUT, doc)
        manifest.note("output_artifact", OUTPUT)
        manifest.note("elapsed_s", round(time.time() - _t0, 1))
        manifest.finish(STATUS_OK)
        print(f"[{STAGE}] записано: {OUTPUT} ({len(doc) / 1024:.0f} КБ)")
        return 0
    except (StageError, EvidenceError, ConfigError, OSError, ValueError, KeyError) as exc:
        if manifest is not None:
            manifest.finish("failed", error=str(exc))
        print(f"[{STAGE}] ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
