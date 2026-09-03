"""Клики-затравки для автокалибровки S1.

Четырнадцать точек в строгом порядке на опорном кадре клипа: пять линий.

    L1  street_ground  левая сторона   низ левой стены (стык с мостовой)   3 точки
    L2  street         левая сторона   карниз / нижний край ряда вывесок   3 точки
    L3  street_ground  правая сторона  правый бордюр                       3 точки
    L4  street         правая сторона  верхняя линия правых фасадов        3 точки
    V1  vertical                       две точки на одной вертикали        2 точки

ground_pair = [L1, L3] — только эти две лежат на плоскости земли, и только по ним
меряется ширина улицы. L2 и L4 идут высоко по стенам: расстояние между ними
шириной улицы не является.

ЗАЧЕМ ЧЕТЫРЕ УЛИЧНЫЕ ЛИНИИ, А НЕ ДВЕ. Все четыре параллельны улице в мире, значит
сходятся в одной точке. Линии на стенах идут высоко над землёй и дают широкую базу
для пересечения, поэтому МНК по четырём обусловлен заметно лучше. Измерено на
синтетике при шуме кликов 2 px: медианная ошибка точки схода 1.5 px по четырём
линиям против 3.1 px по двум наземным.

Три точки на линию, а не две, — чтобы была избыточность: разброс точек
относительно подогнанной прямой уходит в артефакт как hint_line_residual_px
и показывает, насколько криво накликано.

Клики — ТОЛЬКО ЗАТРАВКА RANSAC. В удержанную выборку, по которой считается
невязка, они не входят: иначе гейт проверял бы качество кликов, а не калибровки.

    python scripts/pick_hints.py --config configs/s1_calib.yaml
    python scripts/pick_hints.py --clip raw/clip_debug_2030JST.ts --frame 900

Управление: левая кнопка — поставить точку, правая — отменить последнюю,
Enter — сохранить, Esc — выйти без сохранения.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.calib import CalibError, fit_line_px  # noqa: E402
from looq.io import load_config, sha256_file    # noqa: E402

WINDOW = "S1 hints: L1 L2 L3 L4 V1"

#: Порядок ввода жёсткий. (line_id, role, side, подпись на экране, сколько точек, цвет BGR)
LINES = [
    ("L1", "street_ground", "left",  "ЛЕВАЯ сторона, низ стены (стык с мостовой)", 3, (90, 230, 90)),
    ("L2", "street",        "left",  "ЛЕВАЯ сторона, карниз / низ ряда вывесок",   3, (60, 190, 255)),
    ("L3", "street_ground", "right", "ПРАВАЯ сторона, бордюр",                     3, (255, 160, 80)),
    ("L4", "street",        "right", "ПРАВАЯ сторона, верх фасадов",               3, (230, 120, 255)),
    ("V1", "vertical",      None,    "ВЕРТИКАЛЬ: низ, затем верх",                 2, (80, 80, 255)),
]
GROUND_PAIR = ["L1", "L3"]
TOTAL = sum(n for *_, n, _ in LINES)   # 14

OUT = Path("configs/calib_hints.yaml")


def _plan() -> list[tuple[str, str, tuple[int, int, int]]]:
    """Развёрнутый план кликов: на каждый клик своя подпись."""
    steps = []
    for line_id, _role, _side, label, n, color in LINES:
        for k in range(n):
            steps.append((line_id, f"{line_id} — {label}: точка {k + 1}/{n}", color))
    return steps


STEPS = _plan()


def grab_frame(clip: Path, frame_idx: int):
    """Кадр по номеру. Читаем последовательно: у .ts перемотка врёт."""
    if not clip.is_file():
        raise SystemExit(f"нет клипа: {clip}")
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"cv2 не открыл {clip}")
    frame = None
    for i in range(frame_idx + 1):
        ok, f = cap.read()
        if not ok:
            break
        if i == frame_idx:
            frame = f
    cap.release()
    if frame is None:
        raise SystemExit(f"в {clip} нет кадра {frame_idx}")
    return frame


def _draw(base, points):
    img = base.copy()
    for i, (x, y) in enumerate(points):
        _, _, color = STEPS[i]
        cv2.drawMarker(img, (int(x), int(y)), color, cv2.MARKER_CROSS, 22, 2)
        cv2.putText(img, f"{STEPS[i][0]}.{i}", (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Соединяем точки уже завершённых линий, чтобы кривизна была видна сразу.
    start = 0
    for line_id, _role, _side, _label, n, color in LINES:
        got = points[start:start + n]
        if len(got) >= 2:
            cv2.line(img, tuple(map(int, got[0])), tuple(map(int, got[-1])),
                     color, 1, cv2.LINE_AA)
        start += n

    done = len(points)
    prompt = STEPS[done][1] if done < TOTAL else "ГОТОВО — Enter сохранить"
    cv2.rectangle(img, (0, 0), (img.shape[1], 64), (20, 20, 20), -1)
    cv2.putText(img, f"{done}/{TOTAL}  {prompt}", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "ЛКМ поставить  ПКМ отменить  Enter сохранить  Esc выйти",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/s1_calib.yaml")
    ap.add_argument("--clip", default=None, help="переопределить клип из конфига")
    ap.add_argument("--frame", type=int, default=0, help="номер опорного кадра")
    args = ap.parse_args(argv)

    clip = Path(args.clip) if args.clip else Path(load_config(args.config)["input"]["clip"])
    base = grab_frame(clip, args.frame)

    points: list[tuple[float, float]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < TOTAL:
            points.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1600, base.shape[1]), min(900, base.shape[0]))
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        cv2.imshow(WINDOW, _draw(base, points))
        key = cv2.waitKey(20) & 0xFF
        if key == 27:                      # Esc
            cv2.destroyAllWindows()
            print("отменено, файл не записан")
            return 1
        if key in (13, 10):                # Enter
            if len(points) != TOTAL:
                print(f"поставлено {len(points)} из {TOTAL} точек — не сохраняю")
                continue
            break
    cv2.destroyAllWindows()

    lines_doc: dict[str, dict] = {}
    residuals: dict[str, float] = {}
    start = 0
    for line_id, role, side, label, n, _color in LINES:
        pts = points[start:start + n]
        start += n
        try:
            _, res = fit_line_px(pts)
        except CalibError as exc:
            raise SystemExit(f"{line_id}: клики не образуют прямую: {exc}")
        residuals[line_id] = round(res, 3)
        lines_doc[line_id] = {
            "role": role,
            "side": side,
            "name_ru": label,
            "points_px": [[float(x), float(y)] for x, y in pts],
        }

    doc = {
        "schema_version": "2",   # 14 кликов и именованные линии вместо 8 кликов
        "clip": str(clip).replace("\\", "/"),
        "clip_sha256": sha256_file(clip),
        "frame_idx": int(args.frame),
        "coordinate_frame": "frame_px",
        "note_ru": ("Клики — ЗАТРАВКА RANSAC, а не контрольные точки. В удержанную "
                    "выборку для невязки они не входят. ground_pair — единственные "
                    "линии на плоскости земли, только по ним меряется ширина улицы."),
        "lines": lines_doc,
        "ground_pair": GROUND_PAIR,
        "hint_line_residual_px": {**residuals, "max": max(residuals.values())},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")

    print(f"записано: {OUT}")
    print("разброс кликов относительно прямой, px:")
    for line_id, res in residuals.items():
        flag = "  <-- криво" if res > 5.0 else ""
        print(f"  {line_id}  {res:6.2f}{flag}")
    if max(residuals.values()) > 5.0:
        print("ВНИМАНИЕ: разброс больше 5 px хотя бы на одной линии — точка схода "
              "уедет. Стоит перекликать эту линию.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
