"""Плоскость земли по прямоугольному участку мостовой. Четыре клика.

Запасной путь калибровки, когда точки схода не даются. На ночной сцене линии
фасадов кликаются с разбросом в десятки пикселей, и точка схода уезжает;
четыре угла прямоугольника с ИЗВЕСТНЫМИ пропорциями задают плоскость земли
однозначно и с точностью до масштаба.

Порядок углов тот же, что в обводке зон: **TL, TR, BR, BL** по часовой стрелке.
Сторона TL->TR становится осью +x плана, сторона TL->BL — осью +y и принимается
за 1.0 условной единицы. Пропорция задаётся в configs/s1_calib.yaml,
ключ ground_rect.aspect_tl_tr_over_tl_bl.

ЧЕГО ЭТОТ ПУТЬ НЕ ДАЁТ. Метров. Масштаб остаётся неизвестным, длины выражены
в условных единицах, и любое число, названное метром, было бы враньём.
Хватает для: углов на плоскости (луч ориентации), принадлежности к зонам,
ОТНОШЕНИЙ длин и скоростей. Не хватает для: абсолютных метров, роста,
порога остановки в м/с.

Управление: ЛКМ поставить, ПКМ отменить последнюю, BACKSPACE начать заново,
ENTER сохранить, ESC выйти. Лупа x4 как в обводке зон.

    python scripts/pick_ground.py --config configs/s1_calib.yaml
    make ground
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ui import draw_magnifier, grab_frame, header        # noqa: E402

from looq.calib import CalibError, homography_from_ground_rect  # noqa: E402
from looq.io import load_config, sha256_file, write_json        # noqa: E402

WINDOW = "S1 ground plane: 4 corners of a rectangular pavement patch"
OUT = Path("calib/ground_rect.json")
CORNERS = ["TL", "TR", "BR", "BL"]
COLOR = (90, 230, 90)


def _render(base, pts, cursor, aspect):
    img = base.copy()
    for i, (x, y) in enumerate(pts):
        cv2.drawMarker(img, (int(x), int(y)), COLOR, cv2.MARKER_CROSS, 22, 2)
        cv2.putText(img, CORNERS[i], (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2, cv2.LINE_AA)
    if len(pts) >= 2:
        closed = len(pts) == 4
        cv2.polylines(img, [np.asarray(pts, dtype=np.int32)], closed, COLOR, 2, cv2.LINE_AA)
    if len(pts) == 4:
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.asarray(pts, dtype=np.int32)], COLOR)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, dst=img)

    nxt = CORNERS[len(pts)] if len(pts) < 4 else "ENTER to save"
    header(img,
           f"ground rectangle  aspect TL-TR : TL-BL = {aspect}   next: {nxt}",
           "LMB add  RMB undo  BACKSPACE restart  ENTER save  ESC quit")
    draw_magnifier(img, base, cursor)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/s1_calib.yaml")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    rect_cfg = cfg.get("ground_rect") or {}
    aspect = rect_cfg.get("aspect_tl_tr_over_tl_bl")
    if aspect is None:
        raise SystemExit(
            f"в {args.config} нет ground_rect.aspect_tl_tr_over_tl_bl — пропорция "
            f"участка. Без неё плоскость земли не определена (правило 5)")
    aspect = float(aspect)
    clip = Path(args.clip) if args.clip else Path(cfg["input"]["clip"])
    base = grab_frame(clip, args.frame)

    pts: list[tuple[float, float]] = []
    cursor = [base.shape[1] // 2, base.shape[0] // 2]

    def on_mouse(event, x, y, _flags, _param):
        nonlocal cursor
        cursor = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1600, base.shape[1]), min(900, base.shape[0]))
    cv2.setMouseCallback(WINDOW, on_mouse)

    print(f"участок мостовой по кадру {args.frame} клипа {clip}")
    print(f"пропорция TL-TR : TL-BL = {aspect}")
    while True:
        cv2.imshow(WINDOW, _render(base, pts, cursor, aspect))
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            print("отменено, файл не записан")
            return 1
        if key == 8:
            pts.clear()
        elif key in (13, 10):
            if len(pts) != 4:
                print(f"поставлено {len(pts)} из 4 углов")
                continue
            try:
                _, diag = homography_from_ground_rect(pts, aspect)
            except CalibError as exc:
                # Ловим здесь, а не при чтении артефакта: перекликать четыре
                # точки дешевле, чем узнать о проблеме на прогоне этапа.
                print(f"участок отклонён: {exc}")
                print("нажмите BACKSPACE и обведите заново")
                continue
            break
    cv2.destroyAllWindows()

    write_json(OUT, {
        "schema_version": "1",
        "method": "manual_ground_plane",
        "clip": str(clip).replace("\\", "/"),
        "clip_sha256": sha256_file(clip),
        "frame_idx": int(args.frame),
        "coordinate_frame": "frame_px",
        "corner_order": CORNERS,
        "aspect_tl_tr_over_tl_bl": aspect,
        "rect_px": [[float(x), float(y)] for x, y in pts],
        "note_ru": ("Углы участка мостовой В ПИКСЕЛЯХ. Масштаб отсюда НЕ следует: "
                    "плоскость земли определяется с точностью до множителя, длины "
                    "в условных единицах."),
        "diagnostics": diag,
    })
    print(f"записано: {OUT}")
    print(f"невязка углов {diag['corner_residual_units']:.2e} усл.ед., "
          f"план развёрнут: {diag['plane_y_flipped']}")
    print("МЕТРОВ ЗДЕСЬ НЕТ: длины в условных единицах, "
          "calib_status будет angles_ok_scale_unverified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
