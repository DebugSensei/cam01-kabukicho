"""out/frames.html — галерея скриншотов оверлея.

Кадры НЕ вшиваются в страницу: их сотни по полмегабайта, base64 раздул бы
файл до сотен мегабайт. Ссылки относительные, файлы отдаёт тот же сервер.

    python scripts/make_frames_page.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_dashboard import (CSS, JS, header_html,  # noqa: E402
                            i18n_payload, register_i18n, t)

from looq.io import atomic_write_text  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("out/overlay_frames"))
    ap.add_argument("--out", type=Path, default=Path("out/frames.html"))
    args = ap.parse_args(argv)

    EXTRA = {
        "fr.title": ("Overlay frames", "Кадры оверлея", "オーバーレイのフレーム"),
        "fr.sub": ("Every 30th processed frame of the densest window of the peak "
                   "hour. On each frame: box and id, body-turn arrow, 3-second "
                   "trajectory tail, foot point, storefront outlines and the "
                   "highlight when a ray crosses one. Click to open in place.",
                   "Каждый 30-й обработанный кадр самого плотного окна часа пик. "
                   "На кадре: рамка и id, стрелка поворота корпуса, хвост "
                   "траектории за 3 секунды, точка ног, контуры витрин и "
                   "подсветка при попадании луча. Клик открывает на месте.",
                   "ピーク時間の最も混雑した区間の30フレームごと。枠とID、体の "
                   "向き矢印、3秒の軌跡、足元点、店舗の輪郭、レイ交差時の強調表示。"
                   "クリックでその場に拡大表示。"),
    }
    register_i18n(EXTRA)
    files = sorted(args.dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"нет кадров в {args.dir}")
    rel = args.dir.name
    # Открываем в лайтбоксе на месте, а не в новой вкладке: новая вкладка
    # выкидывает из галереи, и вернуться можно только назад в истории.
    cells = "".join(
        f'<figure><img class="fig" loading="lazy" src="{rel}/{f.name}" '
        f'alt="{f.stem}" data-meta="{f.stem}"><span>{f.stem}</span></figure>'
        for f in files)

    page = f"""<!doctype html><html lang="en" data-theme="light"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAM-01 screenshots</title><style>{CSS}
.gal{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.gal figure{{margin:0;background:var(--panel);
  border:1px solid var(--line);border-radius:12px;overflow:hidden;
  box-shadow:var(--shadow)}}
.gal img{{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;
  background:var(--bg);cursor:zoom-in}}
.gal figure span{{display:block;padding:7px 10px;font-size:11px;color:var(--muted);
  font-family:ui-monospace,Consolas,monospace}}
</style>
{header_html("frames.html", [("nav.frames", f"{len(files)}")])}
<div class="wrap">
<div class="eyebrow">{t("nav.frames")}</div>
<h1>{t("fr.title")}</h1>
<p class="sub">{t("fr.sub")}</p>
<div class="gal">{cells}</div>
</div>
<div id="lb"><img id="lbimg" alt=""><div class="meta" id="lbmeta"></div></div>
<script>window.__I18N__ = {i18n_payload(EXTRA)};</script>
<script>{JS}</script></html>"""
    atomic_write_text(args.out, page)
    print(f"готово: {args.out} ({len(files)} кадров, "
          f"{args.out.stat().st_size / 1e3:.0f} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
