"""Ручная обводка зон для S2. Ровно 20 кликов.

    Этап A: четыре витрины по 4 угла   16 кликов
    Этап B: пол, 4 угла области ROI     4 клика

Углы КАЖДОГО полигона ставятся строго по часовой стрелке начиная с верхнего
левого: TL, TR, BR, BL. Порядок фиксирован не для красоты: нижнее ребро BL->BR —
это след витрины на земле, и только оно идёт в расчёт взгляда. Поставь углы
в другом порядке — след будет неверным, а ошибка вылезет только в S6.

Имена витрин берутся из configs/s2_zones.yaml, ключ zone_names. Скрипт ничего
не спрашивает с клавиатуры.

ЧТО ЭТО ЗА АРТЕФАКТ. zones/zones.json — обводка В ПИКСЕЛЯХ КАДРА, то есть
затравка, а не артефакт этапа. Артефакт S2 по контракту — zones/zones.geojson
в МЕТРАХ ПЛАНА; проекцию делает сам этап S2 через гомографию S1. Смешивать их
нельзя: полигон в пикселях и полигон в метрах — разные вещи, и путаница между
ними ровно тот класс ошибок, против которого заведены суффиксы _px и _m.

Управление:
    ЛКМ        поставить точку
    ПКМ        отменить последнюю точку
    BACKSPACE  отменить весь текущий полигон целиком
    ENTER      принять полигон и перейти к следующему
    ESC        выход без сохранения

Лупа: врезка 200x200 с четырёхкратным увеличением вокруг курсора, в дальнем от
курсора углу окна, с перекрестием по центру. Она же служит проверкой попадания:
если перекрестие стоит не на том, куда вы целитесь, значит координаты окна и
кадра разъехались — тогда уменьшите окно или запустите с --no-resize.

    python scripts/pick_zones.py --config configs/s2_zones.yaml
    make zones
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ui import draw_magnifier, grab_frame  # noqa: E402

from looq.geometry import polygon_cross_signs  # noqa: E402
from looq.io import load_config, sha256_file, write_json  # noqa: E402

WINDOW = "S2 zones: 4 storefronts x 4 corners + ROI"
OUT = Path("zones/zones.json")
PREV = Path("zones/zones_prev.json")

CORNERS = ["TL", "TR", "BR", "BL"]
N_ZONES = 4
N_CORNERS = 4
N_ROI = 4
TOTAL_CLICKS = N_ZONES * N_CORNERS + N_ROI      # 20

#: Фолбэк порогов. Рабочие значения приходят из configs/s2_zones.yaml, секция
#: picking: правило 5 запрещает молчаливые дефолты в коде.
MIN_GROUND_EDGE_PX = 12.0
MAX_ZONE_OVERLAP_FRAC = 0.20

ZONE_COLORS = [(90, 230, 90), (60, 190, 255), (255, 160, 80), (230, 120, 255)]
ROI_COLOR = (200, 200, 200)

def ascii_label(name: str, fallback: str) -> str:
    """cv2.putText кириллицу и кандзи не рисует — на экран идёт только ASCII.

    Берём первый пробельный токен, если он целиком ASCII: у имён вида
    "M1 角煮/げんかつ" это ровно идентификатор M1. Полное имя печатается
    в терминал и уходит в json.
    """
    head = name.strip().split()[0] if name.strip() else ""
    if head and all(ord(c) < 128 for c in head):
        return head
    return fallback


# --------------------------------------------------------------------------- #
# Проверки геометрии
# --------------------------------------------------------------------------- #

def check_polygon(poly_px, name: str) -> list[str]:
    """Выпуклость, отсутствие самопересечений и правильный порядок обхода."""
    poly = np.asarray(poly_px, dtype=np.float64)
    problems: list[str] = []
    if len(poly) != 4:
        return [f"{name}: {len(poly)} точек вместо 4"]

    signs = polygon_cross_signs(poly)
    if np.any(np.abs(signs) < 1e-9):
        problems.append(f"{name}: три угла на одной прямой — полигон вырожден")
    elif not (np.all(signs > 0) or np.all(signs < 0)):
        problems.append(
            f"{name}: полигон невыпуклый или самопересекается. Углы обходятся "
            f"не по кругу — проверьте порядок TL, TR, BR, BL")

    # Порядок обхода: BR и BL обязаны лежать НИЖЕ, чем TL и TR (ось y вниз).
    # Иначе "нижнее ребро" окажется верхним, и след витрины на земле уедет.
    top_y = (poly[0][1] + poly[1][1]) / 2.0
    bot_y = (poly[2][1] + poly[3][1]) / 2.0
    if bot_y <= top_y:
        problems.append(
            f"{name}: точки 3-4 (BR, BL) не ниже точек 1-2 (TL, TR). Порядок углов "
            f"перепутан, нижнее ребро окажется верхним")
    return problems


def check_ground_segment(poly_px, name: str,
                         min_len_px: float = MIN_GROUND_EDGE_PX) -> list[str]:
    poly = np.asarray(poly_px, dtype=np.float64)
    bl, br = poly[3], poly[2]
    length = float(np.hypot(br[0] - bl[0], br[1] - bl[1]))
    if length < min_len_px:
        return [f"{name}: нижнее ребро {length:.1f} px короче {min_len_px} px — "
                f"след витрины на земле слишком короткий, направление фасада по нему "
                f"не определить"]
    return []


def check_inside_roi(poly_px, roi_px, name: str) -> list[str]:
    poly = np.asarray(poly_px, dtype=np.float64)
    roi = np.asarray(roi_px, dtype=np.float64)
    if poly[:, 0].max() < roi[:, 0].min() or poly[:, 0].min() > roi[:, 0].max():
        return [f"{name}: зона целиком вне ROI по горизонтали "
                f"(зона x {poly[:, 0].min():.0f}..{poly[:, 0].max():.0f}, "
                f"ROI x {roi[:, 0].min():.0f}..{roi[:, 0].max():.0f})"]
    return []


def _poly_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def check_overlaps(zones: list[dict],
                   max_frac: float = MAX_ZONE_OVERLAP_FRAC) -> list[str]:
    """Пересечение выпуклых четырёхугольников через cv2.intersectConvexConvex."""
    problems: list[str] = []
    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):
            a = np.asarray(zones[i]["polygon_px"], dtype=np.float32)
            b = np.asarray(zones[j]["polygon_px"], dtype=np.float32)
            inter_area, _ = cv2.intersectConvexConvex(a, b)
            smaller = min(_poly_area(a.astype(np.float64)), _poly_area(b.astype(np.float64)))
            if smaller <= 0:
                continue
            frac = float(inter_area) / smaller
            if frac > max_frac:
                problems.append(
                    f"{zones[i]['id']} и {zones[j]['id']} пересекаются на {frac:.0%} "
                    f"площади меньшей из них при пределе "
                    f"{max_frac:.0%} — витрины обведены внахлёст")
    return problems


def validate(zones: list[dict], roi_px, picking: dict | None = None) -> None:
    """Все проблемы разом, а не первая попавшаяся: перекликивать по одной долго."""
    picking = picking or {}
    min_edge = float(picking.get("min_ground_edge_px", MIN_GROUND_EDGE_PX))
    max_frac = float(picking.get("max_zone_overlap_frac", MAX_ZONE_OVERLAP_FRAC))
    problems: list[str] = []
    problems += check_polygon(roi_px, "ROI")
    for z in zones:
        problems += check_polygon(z["polygon_px"], z["id"])
        problems += check_ground_segment(z["polygon_px"], z["id"], min_edge)
        problems += check_inside_roi(z["polygon_px"], roi_px, z["id"])
    problems += check_overlaps(zones, max_frac)
    if problems:
        raise SystemExit("обводка не прошла проверки:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------- #
# Отрисовка
# --------------------------------------------------------------------------- #

def _draw_polygon(img, poly, color, label: str, filled: bool = True) -> None:
    pts = np.asarray(poly, dtype=np.int32)
    if filled:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    # Нижнее ребро BL->BR белым и толстым: пользователь обязан видеть, что
    # именно пойдёт в расчёт взгляда.
    cv2.line(img, tuple(pts[3]), tuple(pts[2]), (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, label, (int(pts[:, 0].min()), int(pts[:, 1].min()) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _render(base, done_polys, current, cursor, header: str, sub: str):
    img = base.copy()
    for poly, color, label in done_polys:
        _draw_polygon(img, poly, color, label)

    for i, (x, y) in enumerate(current):
        cv2.drawMarker(img, (int(x), int(y)), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(img, CORNERS[i], (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if len(current) >= 2:
        cv2.polylines(img, [np.asarray(current, dtype=np.int32)], False,
                      (0, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (img.shape[1], 64), (20, 20, 20), -1)
    cv2.putText(img, header, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sub, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)
    draw_magnifier(img, base, cursor)
    return img


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/s2_zones.yaml")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--no-resize", action="store_true",
                    help="не масштабировать окно: координаты гарантированно 1:1")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    names = list(cfg.get("zone_names") or [])
    if len(names) != N_ZONES:
        raise SystemExit(
            f"в {args.config} ключ zone_names должен содержать ровно {N_ZONES} имён, "
            f"получено {len(names)}")
    clip = Path(args.clip) if args.clip else Path(cfg["input"]["clip"])
    base = grab_frame(clip, args.frame)

    # План обхода: сначала витрины, потом пол.
    plan = [(ascii_label(n, f"Z{i + 1}"), n, ZONE_COLORS[i]) for i, n in enumerate(names)]
    plan.append(("ROI", "floor / ROI", ROI_COLOR))

    cursor: list | None = [base.shape[1] // 2, base.shape[0] // 2]
    current: list[tuple[float, float]] = []
    done: list[tuple[list, tuple, str]] = []
    results: list[list[tuple[float, float]]] = []

    def on_mouse(event, x, y, _flags, _param):
        nonlocal cursor
        cursor = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN and len(current) < N_CORNERS:
            current.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE if args.no_resize else cv2.WINDOW_NORMAL)
    if not args.no_resize:
        cv2.resizeWindow(WINDOW, min(1600, base.shape[1]), min(900, base.shape[0]))
    cv2.setMouseCallback(WINDOW, on_mouse)

    idx = 0
    print(f"обводка по кадру {args.frame} клипа {clip}")
    print(f"сейчас: {plan[0][1]}")
    while idx < len(plan):
        label, full_name, color = plan[idx]
        stage = "A storefront" if idx < N_ZONES else "B floor / ROI"
        remaining = N_CORNERS - len(current)
        expect = CORNERS[len(current)] if len(current) < N_CORNERS else "ENTER to accept"
        header = f"[{stage}]  {label}  ({idx + 1}/{len(plan)})   next corner: {expect}"
        sub = (f"clicks left in this shape: {remaining}   |   "
               f"LMB add  RMB undo  BACKSPACE clear shape  ENTER accept  ESC quit")

        cv2.imshow(WINDOW, _render(base, done, current, cursor, header, sub))
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            print("отменено, файл не записан")
            return 1
        if key == 8:                       # BACKSPACE
            current.clear()
        elif key in (13, 10):              # ENTER
            if len(current) != N_CORNERS:
                print(f"{label}: поставлено {len(current)} из {N_CORNERS} углов")
                continue
            poly = list(current)
            problems = check_polygon(poly, label)
            if problems:
                # Ловим сразу, а не при сохранении: перерисовать один полигон
                # дешевле, чем начинать двадцать кликов заново.
                for p in problems:
                    print("  " + p)
                print(f"{label}: полигон отклонён, нажмите BACKSPACE и обведите заново")
                continue
            results.append(poly)
            done.append((poly, color, label))
            current.clear()
            idx += 1
            if idx < len(plan):
                print(f"сейчас: {plan[idx][1]}")
    cv2.destroyAllWindows()

    zones = []
    for i, (poly, name) in enumerate(zip(results[:N_ZONES], names)):
        zones.append({
            "id": ascii_label(name, f"Z{i + 1}"),
            "name": name,
            "polygon_px": [[float(x), float(y)] for x, y in poly],
            # Нижнее ребро вычисляется, а не спрашивается отдельно: спрошенное
            # можно перепутать с обведённым, вычисленное — нет.
            "ground_segment_px": [[float(poly[3][0]), float(poly[3][1])],
                                  [float(poly[2][0]), float(poly[2][1])]],
        })
    roi_px = [[float(x), float(y)] for x, y in results[N_ZONES]]

    validate(zones, roi_px, cfg.get("picking"))

    if OUT.is_file():
        PREV.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT, PREV)
        print(f"предыдущая обводка сохранена в {PREV}")

    write_json(OUT, {
        "schema_version": "1",
        "clip": str(clip).replace("\\", "/"),
        "clip_sha256": sha256_file(clip),
        "frame_idx": int(args.frame),
        "coordinate_frame": "frame_px",
        "note_ru": ("Обводка в ПИКСЕЛЯХ КАДРА. Это затравка, а не артефакт этапа: "
                    "артефакт S2 — zones/zones.geojson в МЕТРАХ ПЛАНА, проекцию "
                    "делает сам этап S2 через гомографию S1."),
        "corner_order": CORNERS,
        "picking_thresholds": dict(cfg.get("picking") or {}),
        "ground_segment_note_ru": "нижнее ребро BL->BR, вычислено из polygon_px",
        "roi_px": roi_px,
        "zones": zones,
    })
    print(f"записано: {OUT}")
    for z in zones:
        seg = np.asarray(z["ground_segment_px"])
        length = float(np.hypot(*(seg[1] - seg[0])))
        print(f"  {z['id']:4s} {z['name']}   нижнее ребро {length:.0f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
