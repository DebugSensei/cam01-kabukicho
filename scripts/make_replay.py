"""out/replay.html — видео слева, синхронный вид сверху справа, шкала событий снизу.

Чистый HTML+JS, без сервера: данные вшиты в страницу как JSON, видео берётся
относительным путём. Открывается двойным щелчком по файлу.

Синхронизация времени точная, без подгонки. Оверлей рендерится из ОБРАБОТАННЫХ
кадров с частотой fps/stride, а обработан каждый stride-й кадр, поэтому
кадр k видео = кадр k*stride источника, и время видео совпадает с ts источника
секунда в секунду.

Вид сверху рисуется из тех же артефактов, что и метрики: положения на плане
из track/tracks.parquet, зоны из zones/zones.geojson, попадания луча из
attn/track_zone_frames.parquet. Ничего не пересчитывается.

    python scripts/make_replay.py
    make replay
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_dashboard import header_html, i18n_payload  # noqa: E402

from looq.geometry import (ORIENTATION_DISCLAIMER_EN,  # noqa: E402
                           point_in_polygon_m)
from looq.io import atomic_write_text, load_config, read_json, require  # noqa: E402

OUT = Path("out/replay.html")

PAGE = """<!doctype html><html lang="en" data-theme="light"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAM-01 Kabukicho — replay</title>
<style>
:root{
  --bg:#f5f6f8; --panel:#fff; --ink:#0f1622; --muted:#6b7482; --line:#e6e9ef;
  --accent:#2f6bff; --plan-bg:#fbfcfe; --grid:#e9edf4; --zone:#7a8397;
  --hit:#e8890c; --stop:#17a35b; --apron:rgba(120,130,150,.13); --idink:#4a5568;
  --grid-bold:#cdd5e0;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 8px 24px rgba(16,24,40,.05);
}
:root[data-theme="dark"]{
  --bg:#0c1017; --panel:#151b26; --ink:#e8ecf3; --muted:#93a0b4; --line:#232c3b;
  --accent:#5b8bff; --plan-bg:#0b0f16; --grid:#1c2434; --zone:#7a8397;
  --hit:#ffb454; --stop:#5ad18e; --apron:rgba(120,130,150,.16); --idink:#cfd6e4;
  --grid-bold:#2b3648;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:13.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Arial,sans-serif}
a{color:var(--accent)}
header{position:sticky;top:0;z-index:40;background:var(--panel);
  border-bottom:1px solid var(--line)}
.hd{max-width:1600px;margin:0 auto;padding:12px 18px;display:flex;
  align-items:center;gap:10px;flex-wrap:nowrap}
.hd .chip{flex:0 1 auto;min-width:0}
.hd .chip .vv{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
@media(max-width:1180px){.hd .chip{display:none}}
@media(max-width:720px){
  .hd{padding:10px 14px}
  .logo{font-size:16px}
}
.logo{font-weight:800;letter-spacing:.22em;font-size:18px}
.chip{border:1px solid var(--line);border-radius:12px;padding:6px 13px;line-height:1.2}
.chip .kk{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.chip .vv{font-weight:650;font-size:13px}
.sp{flex:1}
.navgrp{display:flex;align-items:center;gap:9px;flex:none;margin-left:auto}
.langs{display:inline-flex;border:1px solid var(--line);border-radius:999px;
  overflow:hidden}
.lang{border:0;background:var(--panel);color:var(--muted);font:inherit;
  font-weight:700;font-size:12px;padding:8px 12px;cursor:pointer}
.lang.on{background:var(--accent);color:#fff}
.icobtn{width:38px;height:38px;padding:0;border-radius:50%;display:inline-flex;
  align-items:center;justify-content:center;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);cursor:pointer;flex:none}
.icobtn:hover{border-color:var(--accent);color:var(--accent)}
.icobtn svg{width:19px;height:19px;display:block}
.icobtn .moon{display:none}
:root[data-theme="dark"] .icobtn .sun{display:none}
:root[data-theme="dark"] .icobtn .moon{display:block}
.btn{border:1px solid var(--line);background:var(--panel);color:var(--ink);
  border-radius:999px;padding:7px 15px;font:inherit;font-weight:600;font-size:12.5px;
  cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;gap:6px}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.wrap{max-width:1600px;margin:0 auto;padding:18px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}
.sub{color:var(--muted);margin:0 0 16px;font-size:12.5px;max-width:92ch}
.main{display:grid;grid-template-columns:1fr 300px 216px;gap:14px;align-items:start}
@media(max-width:1300px){.main{grid-template-columns:1fr 1fr}}
@media(max-width:900px){
  .main{grid-template-columns:1fr}
  .wrap{padding:14px}
  h1{font-size:22px}
  canvas{max-height:70vh;object-fit:contain}
}
@media(max-width:560px){
  .wrap{padding:10px}
  h1{font-size:19px}
  .v{font-size:21px}
  #tl{height:96px}
  .card{padding:10px;border-radius:12px}
}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:12px;box-shadow:var(--shadow)}
video{width:100%;border-radius:9px;display:block;background:#000}
canvas{width:100%;border-radius:9px;background:var(--plan-bg);display:block}
.k{color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.06em}
.v{font-size:25px;font-weight:700;font-variant-numeric:tabular-nums}
.row{padding:8px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0}
#tl{position:relative;height:112px;margin-top:14px;background:var(--panel);
  border:1px solid var(--line);border-radius:14px;overflow:hidden;cursor:pointer;
  box-shadow:var(--shadow)}
.ev{position:absolute;height:16px;border-radius:3px;opacity:.9}
.ev:hover{opacity:1;outline:1px solid var(--ink)}
.lane{position:absolute;left:0;right:0;height:1px;background:var(--line)}
.lab{position:absolute;left:8px;font-size:10px;color:var(--muted)}
#cur{position:absolute;top:0;bottom:0;width:2px;background:var(--accent);
  pointer-events:none}
.legend{display:flex;gap:14px;font-size:11px;color:var(--muted);margin-top:8px;
  flex-wrap:wrap}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
.note{color:var(--muted);font-size:11px;margin-top:8px}
/* ---- бургер ------------------------------------------------------------ */
/* На телефоне восемь кнопок в строку не помещаются никак: горизонтальная
   прокрутка внутри шапки прятала половину из них за краем, и найти язык или
   тему было нельзя. Всё уезжает в меню. */
.burger{display:none;width:38px;height:38px;padding:0;border-radius:11px;
  align-items:center;justify-content:center;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);cursor:pointer;flex:none;
  margin-left:auto}
.burger svg{width:20px;height:20px;display:block}
.burger .x{display:none}
.burger[aria-expanded="true"] .x{display:block}
.burger[aria-expanded="true"] .bars{display:none}
.burger[aria-expanded="true"]{border-color:var(--accent);color:var(--accent)}
@media(max-width:720px){
  .hd{position:relative}
  .burger{display:inline-flex}
  .navgrp{display:none;position:absolute;top:100%;left:0;right:0;
    flex-direction:column;align-items:stretch;gap:8px;
    background:var(--panel);border-bottom:1px solid var(--line);
    padding:12px 14px 16px;box-shadow:var(--shadow);z-index:50;
    margin-left:0;width:auto;overflow:visible}
  .navgrp.open{display:flex}
  .navgrp .btn{justify-content:center;padding:11px 16px;font-size:14px}
  .navgrp .lbl{display:inline}
  .langs{align-self:stretch}
  .lang{flex:1;padding:11px 0;font-size:13px}
  .icobtn{width:auto;height:auto;border-radius:999px;padding:11px 16px;
    gap:9px;font:inherit;font-weight:600;font-size:14px}
  .icobtn::after{content:attr(data-label)}
}

</style>
__HEADER__
<div class="wrap">
<h1>Replay: frame and ground plane, in sync</h1>
<p class="sub">__SUB__</p>
<div class="main">
  <div class="card"><video id="v" src="__VIDEO__" controls preload="auto"></video>
  <div class="note">__DISCLAIMER__</div></div>
  <div class="card"><canvas id="plan" width="__CANVW__" height="__CANVH__"></canvas>
  <div class="legend">
    <span><i class="sw" id="sw1"></i>track</span>
    <span><i class="sw" id="sw2"></i>ray crosses the facade</span>
    <span><i class="sw" id="sw3"></i>slow, inside apron</span>
    <span><i class="sw" id="sw4"></i>facade / apron</span>
  </div></div>
  <div class="card" id="counters"></div>
</div>
<div id="tl"><div id="cur"></div></div>
<div class="note">Event timeline: click to seek. One lane per storefront. A dashed
outline marks an event that S6 flagged as low_confidence.</div>
</div>
<script>window.__I18N__ = __I18NJSON__;</script>
<script>
const D = __DATA__;
const v = document.getElementById('v'), cv = document.getElementById('plan');
// Канва не наследует CSS-переменные, поэтому цвета читаются из них явно и
// перечитываются при смене темы: иначе тёмный план остался бы на светлой
// странице.
const ROOT = document.documentElement;
let C = {};
function readTheme(){
  const g = getComputedStyle(ROOT);
  C = {planBg:g.getPropertyValue('--plan-bg').trim(),
       grid:g.getPropertyValue('--grid').trim(),
       zone:g.getPropertyValue('--zone').trim(),
       hit:g.getPropertyValue('--hit').trim(),
       stop:g.getPropertyValue('--stop').trim(),
       apron:g.getPropertyValue('--apron').trim(),
       muted:g.getPropertyValue('--muted').trim(),
       idink:g.getPropertyValue('--idink').trim(),
       gridBold:g.getPropertyValue('--grid-bold').trim(),
       ink:g.getPropertyValue('--ink').trim(),
       accent:g.getPropertyValue('--accent').trim()};
  ['sw1','sw2','sw3','sw4'].forEach(function(id,i){
    const el = document.getElementById(id);
    if (el) el.style.background = [C.accent, C.hit, C.stop, C.zone][i];
  });
}
readTheme();
try{var st=localStorage.getItem('looq-theme');
    if(st){ROOT.setAttribute('data-theme',st); readTheme();}}catch(e){}

  var burger=document.getElementById('burger'), nav=document.querySelector('.navgrp');
  if (burger && nav){
    burger.onclick=function(e){
      e.stopPropagation();
      var open=nav.classList.toggle('open');
      burger.setAttribute('aria-expanded', open?'true':'false');
    };
    // Клик по пункту закрывает меню: иначе после выбора языка панель
    // остаётся раскрытой и закрывает собой саму страницу.
    nav.addEventListener('click',function(e){
      if (e.target.closest('a')) close();
    });
    document.addEventListener('click',function(e){
      if (nav.classList.contains('open') && !nav.contains(e.target)
          && e.target!==burger) close();
    });
    document.addEventListener('keydown',function(e){
      if (e.key==='Escape') close();
    });
  }
  function close(){
    if (!nav) return;
    nav.classList.remove('open');
    if (burger) burger.setAttribute('aria-expanded','false');
  }

var I18N = window.__I18N__ || {};
function setLang(code){
  document.querySelectorAll('[data-i18n]').forEach(function(el){
    var row = I18N[el.getAttribute('data-i18n')];
    if (row && row[code]) el.textContent = row[code];
  });
  document.querySelectorAll('.lang').forEach(function(b){
    b.classList.toggle('on', b.dataset.lang===code); });
  document.documentElement.lang = code;
  try{localStorage.setItem('looq-lang', code);}catch(e){}
}
document.querySelectorAll('.lang').forEach(function(b){
  b.onclick = function(){ setLang(b.dataset.lang); }; });
try{var L=localStorage.getItem('looq-lang'); if(L) setLang(L);}catch(e){}
document.getElementById('theme').onclick = function(){
  const d = ROOT.getAttribute('data-theme')==='dark';
  ROOT.setAttribute('data-theme', d?'light':'dark');
  try{localStorage.setItem('looq-theme', d?'light':'dark');}catch(e){}
  readTheme();
  // Перерисовываем СРАЗУ, а не ждём следующего кадра: в скрытой или
  // неактивной вкладке requestAnimationFrame не тикает, и план оставался бы
  // в старой теме до возвращения фокуса.
  try{ drawPlan(frameAt(v.currentTime)); }catch(e){}
};
const ctx = cv.getContext('2d'), tl = document.getElementById('tl');
const cur = document.getElementById('cur'), counters = document.getElementById('counters');

// Холст подогнан под пропорции данных при генерации, поэтому масштаб по обеим
// осям выходит одинаковым и план заполняет карточку, а не её верхнюю треть.
// ТА ЖЕ проекция, что у статического плана: поворот на 90 градусов, чтобы
// улица шла вдоль высоты, а не сплющивалась в ленту. Поворот, а не
// транспозиция: у транспозиции определитель -1, и витрины уезжают направо,
// хотя в кадре они слева.
const B = D.bounds, PAD = 30;
const U0 = -B.y1, U1 = -B.y0, V0 = B.x0, V1 = B.x1;
const S = Math.min((cv.width - 2*PAD) / (U1 - U0),
                   (cv.height - 2*PAD) / (V1 - V0));
const OX = PAD + ((cv.width - 2*PAD) - (U1 - U0) * S) / 2;
const OY = PAD + ((cv.height - 2*PAD) - (V1 - V0) * S) / 2;
const PX = (x, y) => [OX + (-y - U0) * S, cv.height - OY - (x - V0) * S];
const X = (x, y) => PX(x, y)[0], Y = (x, y) => PX(x, y)[1];

// Кадры хранятся разреженно: индекс = позиция в D.frames, не номер кадра.
const times = D.frames.map(f => f.t);
function frameAt(t){
  let lo = 0, hi = times.length - 1;
  if (t <= times[0]) return 0;
  if (t >= times[hi]) return hi;
  while (lo < hi){ const m = (lo+hi)>>1; if (times[m] < t) lo = m+1; else hi = m; }
  return lo;
}

function trackColor(id){ return `hsl(${(id*47)%360} 70% 62%)`; }

// След копится, пока идёт воспроизведение. Ограничен сверху, чтобы канва не
// начала тормозить на часовой записи.
const SEEN = [], SEEN_MAX = 4000;
let lastFi = -1;

function drawPlan(fi){
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.fillStyle = C.planBg; ctx.fillRect(0,0,cv.width,cv.height);

  // Сетка каждый метр, каждые 5 м жирнее: без неё «прямая улица» —
  // впечатление, а не проверка.
  for (let x = Math.ceil(B.x0); x <= B.x1; x += 1){
    ctx.strokeStyle = (x % 5 === 0) ? C.gridBold : C.grid;
    ctx.lineWidth = (x % 5 === 0) ? 1.4 : 1;
    const a = PX(x, B.y0), b = PX(x, B.y1);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  for (let y = Math.ceil(B.y0); y <= B.y1; y += 1){
    ctx.strokeStyle = (y % 5 === 0) ? C.gridBold : C.grid;
    ctx.lineWidth = (y % 5 === 0) ? 1.4 : 1;
    const a = PX(B.x0, y), b = PX(B.x1, y);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }

  const f = D.frames[fi] || {p:[], hits:[]};
  const litZones = new Set(f.hits.map(h => h[0]));

  // Контур области достоверности: всё вне него не считается и не рисуется.
  if (D.roi){
    ctx.strokeStyle = C.zone; ctx.lineWidth = 1.5; ctx.setLineDash([6,4]);
    ctx.beginPath();
    D.roi.forEach((q,i) => { const w = PX(q[0],q[1]);
      i ? ctx.lineTo(w[0],w[1]) : ctx.moveTo(w[0],w[1]); });
    ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
  }

  // Накопительный след: план заполняется по ходу воспроизведения, как на
  // статической картинке. Без него в каждый момент видно 8 точек в пустоте,
  // и проверить «улица прямая, люди идут вдоль неё» невозможно.
  if (SEEN.length){
    ctx.globalAlpha = .30; ctx.lineWidth = 1;
    for (const seg of SEEN){
      ctx.strokeStyle = seg.c; ctx.beginPath();
      seg.p.forEach((q,i) => i ? ctx.lineTo(q[0],q[1]) : ctx.moveTo(q[0],q[1]));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // зоны
  D.zones.forEach(z => {
    if (z.kind === 'apron'){
      ctx.fillStyle = C.apron;
      ctx.beginPath(); z.poly.forEach((p,i) => { const q = PX(p[0],p[1]);
        i ? ctx.lineTo(q[0],q[1]) : ctx.moveTo(q[0],q[1]); });
      ctx.closePath(); ctx.fill();
    } else if (z.kind === 'facade'){
      const on = litZones.has(z.id);
      const s0 = PX(z.seg[0][0], z.seg[0][1]), s1 = PX(z.seg[1][0], z.seg[1][1]);
      // Фасад — жирный отрезок одного стиля, как на статическом плане.
      ctx.strokeStyle = on ? C.hit : z.color || C.zone;
      ctx.lineWidth = on ? 9 : 6; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(s0[0], s0[1]); ctx.lineTo(s1[0], s1[1]); ctx.stroke();
      ctx.lineCap = 'butt';
      ctx.fillStyle = on ? C.hit : C.ink; ctx.font = '600 12px system-ui';
      ctx.fillText(z.short || z.name, (s0[0]+s1[0])/2 + 12, (s0[1]+s1[1])/2 + 4);
    }
  });

  // хвосты и люди
  f.p.forEach(p => {
    const [id, x, y, yaw, slow, inA] = p;
    const tail = D.tails[id];
    if (tail){
      ctx.strokeStyle = trackColor(id); ctx.lineWidth = 1.6; ctx.globalAlpha = .5;
      ctx.beginPath(); let started = false;
      for (const [tt, tx, ty] of tail){
        if (tt > f.t || tt < f.t - 3) continue;
        const q = PX(tx, ty);
        started ? ctx.lineTo(q[0],q[1]) : (ctx.moveTo(q[0],q[1]), started = true);
      }
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    if (tail && fi !== lastFi && SEEN.length < SEEN_MAX){
      const pts = [];
      for (const [tt, tx, ty] of tail){
        if (tt > f.t) break;
        if (tt < f.t - 3) continue;
        pts.push(PX(tx, ty));
      }
      if (pts.length > 1) SEEN.push({c: trackColor(id), p: pts});
    }
    if (yaw !== null){
      const r = yaw * Math.PI/180, L = 1.6;
      ctx.strokeStyle = trackColor(id); ctx.lineWidth = 2;
      const a0 = PX(x, y), a1 = PX(x + L*Math.cos(r), y + L*Math.sin(r));
      ctx.beginPath(); ctx.moveTo(a0[0],a0[1]); ctx.lineTo(a1[0],a1[1]); ctx.stroke();
    }
    ctx.fillStyle = (slow && inA) ? C.stop : trackColor(id);
    const d0 = PX(x, y);
    ctx.beginPath(); ctx.arc(d0[0],d0[1], 5, 0, 6.283); ctx.fill();
    ctx.fillStyle = C.idink; ctx.font = '10px system-ui';
    ctx.fillText('#'+id, d0[0]+7, d0[1]-6);
  });

  // масштабная линейка
  ctx.strokeStyle = C.muted; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(PAD, cv.height-14);
  ctx.lineTo(PAD + 5*S, cv.height-14); ctx.stroke();
  ctx.fillStyle = C.muted; ctx.font = '11px system-ui';
  ctx.fillText('5 ' + D.unit, PAD + 5*S + 6, cv.height-10);
}

function drawCounters(fi){
  const f = D.frames[fi] || {p:[], hits:[]};
  const seen = D.cum[fi] || {tracks:0, hits:0, lookers:0};
  const rows = [
    ['time', (f.t||0).toFixed(1) + ' s'],
    ['people in frame', f.p.length],
    ['tracks so far', seen.tracks],
    ['storefronts lit NOW', new Set(f.hits.map(h=>h[0])).size],
    ['people turned so far', seen.lookers],
    ['ray-hit frames so far', seen.hits],
  ];
  counters.innerHTML = rows.map(r =>
    `<div class="row"><div class="k">${r[0]}</div><div class="v">${r[1]}</div></div>`).join('');
}

// шкала событий
const T = D.duration;
const lanes = D.zone_order;
lanes.forEach((z,i) => {
  const y = 18 + i*22;
  const l = document.createElement('div'); l.className='lane'; l.style.top=(y+16)+'px'; tl.appendChild(l);
  const lb = document.createElement('div'); lb.className='lab'; lb.style.top=(y+2)+'px';
  lb.textContent = z.replace('facade_',''); tl.appendChild(lb);
});
D.events.forEach(e => {
  const i = lanes.indexOf(e.zone); if (i < 0) return;
  const d = document.createElement('div'); d.className='ev';
  d.style.left = (e.t0/T*100)+'%';
  d.style.width = Math.max(0.4, (e.t1-e.t0)/T*100)+'%';
  d.style.top = (18 + i*22)+'px';
  d.style.background = e.type.includes('stop') ? C.stop
                     : e.type.includes('gaze') ? C.hit : C.zone;
  d.title = `#${e.track} ${e.type} ${e.t0.toFixed(1)}-${e.t1.toFixed(1)} s`
          + (e.low ? ' (low_confidence)' : '');
  if (e.low) d.style.outline = '1px dashed #ff6b6b';
  d.onclick = ev => { ev.stopPropagation(); v.currentTime = e.t0; };
  tl.appendChild(d);
});
tl.onclick = e => { const r = tl.getBoundingClientRect();
  v.currentTime = (e.clientX - r.left)/r.width * T; };

function tick(){
  const fi = frameAt(v.currentTime);
  drawPlan(fi); drawCounters(fi); lastFi = fi;
  cur.style.left = (v.currentTime/T*100)+'%';
  requestAnimationFrame(tick);
}
v.addEventListener('loadedmetadata', () => { drawPlan(0); drawCounters(0); });
drawPlan(0); drawCounters(0); requestAnimationFrame(tick);
</script></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/s3_detect.yaml")
    ap.add_argument("--video", default="overlay.mp4",
                    help="относительный путь к видео от out/")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    import pandas as pd

    cfg = load_config(args.config)
    hom = read_json("calib/homography.json")
    # Страница английская: единица тоже. Кириллическая "м" выглядела бы
    # как латинская и ломала бы любое сравнение unit == "m".
    unit = "m" if hom.get("scale_known") else "conv.unit"
    zones_doc = read_json("zones/zones.geojson")
    metrics = read_json("out/metrics.json")

    tracks = pd.read_parquet("track/tracks.parquet")
    orient = pd.read_parquet("pose/orient.parquet")
    events = pd.read_parquet("attn/events.parquet")
    zf = pd.read_parquet("attn/track_zone_frames.parquet")

    df = tracks.merge(orient[["track_id", "frame_idx", "body_yaw_deg", "head_yaw_deg",
                              "foot_x_m_refined", "foot_y_m_refined"]],
                      on=["track_id", "frame_idx"], how="left")
    df["yaw"] = df["head_yaw_deg"].fillna(df["body_yaw_deg"])
    df["x"] = df["foot_x_m_refined"].fillna(df["foot_x_m"])
    df["y"] = df["foot_y_m_refined"].fillna(df["foot_y_m"])
    df = df[df["x"].notna() & df["y"].notna()]

    # Рамка плана — по ОБЛАСТИ ДОСТОВЕРНОСТИ, а не по всем позициям часа.
    # Иначе редкие дальние точки растягивают вид, и восемь человек в кадре
    # оказываются пятном в углу пустого поля.
    _roi_poly, _roi_bounds = None, None
    for f in zones_doc["features"]:
        if f["properties"]["zone_type"] == "roi":
            _roi_poly = f["geometry"]["coordinates"][0]
            rx = [q[0] for q in _roi_poly]
            ry = [q[1] for q in _roi_poly]
            _roi_bounds = {"x0": min(rx) - 1.0, "x1": max(rx) + 1.0,
                           "y0": min(ry) - 1.0, "y1": max(ry) + 1.0}


    # ВСЁ ВНЕ ROI НЕ РИСУЕТСЯ И НЕ СЧИТАЕТСЯ. Статический план это уже делал,
    # а реплей рисовал дальние треки: они вылетали веером за пунктирный контур
    # и заслоняли то, что внутри области достоверности.
    if _roi_poly is not None:
        _roi_arr = np.asarray(_roi_poly, dtype=np.float64)
        _n_all = int(df["track_id"].nunique())
        _keep = np.array([point_in_polygon_m((float(a), float(b)), _roi_arr)
                          for a, b in df[["x", "y"]].to_numpy()])
        df = df[_keep]
        print(f"[replay] ROI: остаётся {_keep.mean():.1%} строк, "
              f"{int(df['track_id'].nunique())} треков из {_n_all}")

    zones, zone_order = [], []
    for f in zones_doc["features"]:
        pr = f["properties"]
        if pr["zone_type"] == "facade":
            # Цвет и короткий id — те же, что на статическом плане и в
            # оверлее: одна витрина обязана быть одного цвета везде.
            _hex = ["#22c55e", "#f59e0b", "#3b82f6", "#ec4899"][len(zone_order) % 4]
            zones.append({"kind": "facade", "id": pr["zone_id"], "name": pr["name_ru"],
                          "short": pr["zone_id"].replace("facade_", ""),
                          "color": _hex,
                          "seg": f["geometry"]["coordinates"]})
            zone_order.append(pr["zone_id"])
        elif pr["zone_type"] == "apron":
            zones.append({"kind": "apron", "id": pr["zone_id"],
                          "poly": f["geometry"]["coordinates"][0]})

    apron_flag = {(int(r.frame_idx), int(r.track_id)): bool(r.in_apron)
                  for r in zf.itertuples()}
    slow_flag = {(int(r.frame_idx), int(r.track_id)): bool(r.is_slow)
                 for r in zf.itertuples()}
    # ТО ЖЕ множество, что стоит за долей повёрнутых в S8 и в дашборде.
    # Раньше сюда шли сырые gaze_hit без фильтров: страница печатала 279
    # попаданий, пока дашборд по тем же данным давал 3 повёрнутых трека.
    # Цепочка замерена: 279 сырых -> 243 без скользящих углов -> 62 кадра
    # и 3 трека после привязки к засчитанным событиям S6.
    _ok = events[events["event_type"].isin(["gaze", "stop_and_gaze"])
                 & (~events["low_confidence"])]
    _pairs = set(zip(_ok["track_id"], _ok["zone_id"]))
    _hits = zf[zf["gaze_hit"] & (~zf["grazing"])]
    _hits = _hits[[p in _pairs for p in zip(_hits["track_id"], _hits["zone_id"])]]
    hits_by_frame: dict[int, list] = {}
    for r in _hits.itertuples():
        hits_by_frame.setdefault(int(r.frame_idx), []).append(
            [r.zone_id, int(r.track_id)])
    print(f"[replay] попаданий луча: сырых {int(zf['gaze_hit'].sum())}, "
          f"после фильтров {len(_hits)} у {_hits['track_id'].nunique()} треков")

    tails: dict[int, list] = {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame_idx")
        tails[int(tid)] = [[round(float(t), 2), round(float(x), 2), round(float(y), 2)]
                           for t, x, y in zip(g["ts"], g["x"], g["y"])]

    frames, cum = [], []
    seen: set[int] = set()
    seen_lookers: set[int] = set()
    n_hits = 0
    for fi, g in df.groupby("frame_idx"):
        fi = int(fi)
        pts = []
        for r in g.itertuples():
            tid = int(r.track_id)
            seen.add(tid)
            yaw = None if not np.isfinite(r.yaw) else round(float(r.yaw), 1)
            pts.append([tid, round(float(r.x), 2), round(float(r.y), 2), yaw,
                        slow_flag.get((fi, tid), False),
                        apron_flag.get((fi, tid), False)])
        hits = hits_by_frame.get(fi, [])
        n_hits += len(hits)
        seen_lookers.update(int(t) for _, t in hits)
        frames.append({"t": round(float(g["ts"].iloc[0]), 2), "p": pts, "hits": hits})
        cum.append({"tracks": len(seen), "hits": n_hits,
                    "lookers": len(seen_lookers)})

    ev = [{"track": int(r.track_id), "zone": r.zone_id, "type": r.event_type,
           "t0": round(float(r.t_start), 2), "t1": round(float(r.t_end), 2),
           "low": bool(r.low_confidence)} for r in events.itertuples()]

    xs = df["x"].to_numpy(); ys = df["y"].to_numpy()
    seg = np.array([p for z in zones if z["kind"] == "facade" for p in z["seg"]])
    allx = np.concatenate([xs, seg[:, 0]]) if len(seg) else xs
    ally = np.concatenate([ys, seg[:, 1]]) if len(seg) else ys
    pad = 2.0
    # Высота холста из пропорций плана: иначе min(sx, sy) прижимает картинку
    # к верху и половина карточки пустует.
    span_x = float(allx.max() - allx.min()) + 2 * pad
    span_y = float(ally.max() - ally.min()) + 2 * pad
    # После поворота ширина канвы задаётся поперечником улицы, а высота —
    # её длиной. Прежняя формула считала наоборот и сплющивала план в ленту.
    canvas_w = 520
    canvas_h = int(np.clip(round(canvas_w * span_x / max(span_y, 1e-6)), 400, 1400))
    # Какое окно реально лежит в видео. Нет паспорта — считаем, что видео
    # покрывает всё, и честно говорим об этом в подписи: молчаливое
    # предположение здесь уже один раз развело план с картинкой.
    win_path = Path(args.video).with_suffix(".window.json")
    win = read_json(win_path) if win_path.is_file() else None

    data = {
        "unit": unit,
        "video_window": win,
        "duration": float(df["ts"].max()),
        "bounds": _roi_bounds if _roi_bounds else {
            "x0": float(allx.min()) - pad, "x1": float(allx.max()) + pad,
            "y0": float(ally.min()) - pad, "y1": float(ally.max()) + pad},
        "roi": _roi_poly,
        "zones": zones, "zone_order": zone_order,
        "frames": frames, "cum": cum, "tails": tails, "events": ev,
    }

    if win and win.get("src_frame_first") is not None:
        win_note = (f"Video covers source frames {win['src_frame_first']}"
                    f"-{win['src_frame_last']}, not the whole run. ")
    else:
        win_note = ("No window descriptor next to the video: the plan clock "
                    "assumes the video covers every processed frame. ")

    sub = (f"{len(frames)} frames, {len(seen)} tracks, {len(ev)} events "
           f"(inside the ROI only, narrower than the full run). {win_note}"
           f"calib_status = {hom.get('calib_status')}, unit: {unit}. "
           f"{metrics['scope']['n_frames_processed']} of "
           f"{metrics['scope']['n_frames_total']} source frames processed. "
           f"Video and plan share one clock: the overlay is rendered from the "
           f"processed frames at fps/stride, so video frame k is source frame "
           f"k*stride and no time fudging is applied.")
    # "</script>" внутри данных закрыл бы тег и вывалил остаток JSON в разметку.
    # Данные приходят из артефактов, но экранирование здесь стоит одной строки.
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = (PAGE.replace("__DATA__", data_json)
                .replace("__CANVH__", str(canvas_h))
                .replace("__CANVW__", str(canvas_w))
                .replace("__VIDEO__", args.video)
                .replace("__HEADER__", header_html("replay.html", [("nav.camera", "CAM-01 Kabukicho")]))
                .replace("__I18NJSON__", i18n_payload())
                .replace("__SUB__", sub)
                .replace("__DISCLAIMER__", ORIENTATION_DISCLAIMER_EN))
    atomic_write_text(args.out, page)
    print(f"готово: {args.out} ({args.out.stat().st_size / 1e6:.1f} МБ)")
    print(f"  кадров {len(frames)}, треков {len(seen)}, событий {len(ev)}, "
          f"зон {len(zone_order)}")
    print(f"  видео берётся относительным путём: {args.video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
