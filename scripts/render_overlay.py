"""out/overlay.mp4 — видео с наложением всего, что посчитали этапы.

Ничего не пересчитывает: читает det/track/pose/attn/attr и рисует. Если число
не лежит в артефакте, оно не рисуется — пририсовать трек там, где его не
считали, значило бы показать несуществующее измерение.

ЧТО НА КАДРЕ
  * рамка человека и его track_id;
  * стрелка ПОВОРОТА КОРПУСА из центра человека, длина фиксированная в пикселях;
  * хвост траектории за последние 3 секунды;
  * точка ног на земле, цвет — по foot_source (прямое измерение или косвенное);
  * контуры четырёх витрин, каждая своим цветом;
  * витрина загорается, когда в её нижнее ребро попал луч, рядом — кто смотрит;
  * подпись у человека: класс верха и время в кадре;
  * счётчик в углу.

ВТОРОЕ ОКНО (--plan). Справа план земли: те же люди точками, витрины отрезками,
следы траекторий. Это визуальная проверка калибровки: если улица на плане прямая
и люди идут вдоль неё, гомография верна. Изображение НЕ разворачивается в вид
сверху — рисуется схема плана по координатам, потому что варп картинки на
вертикальные поверхности врёт, а к геометрии ничего не добавляет.

ДЛИНА СТРЕЛКИ ЗАДАЁТСЯ В ПИКСЕЛЯХ, а не в метрах на плане. Отрезок 1.6 м,
спроецированный целиком, у человека лицом к камере уходит через весь экран:
дальний конец оказывается ближе камеры. С плана берётся только НАПРАВЛЕНИЕ.

    python scripts/render_overlay.py
    python scripts/render_overlay.py --plan --out out/overlay_plan.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.geometry import point_in_polygon_m  # noqa: E402
from looq.evidence import EvidenceError, blur_face_region  # noqa: E402
from looq.io import load_config, read_json, require, write_json  # noqa: E402
from looq.pilot import infer_params  # noqa: E402

OUT = Path("out/overlay.mp4")
FRAME_DIR = Path("out/overlay_frames")

TAIL_SEC = 3.0
ARROW_PX = 60              # длина стрелки в пикселях, по требованию
ARROW_PROBE_M = 1.0        # пробный отрезок на плане: нужен ТОЛЬКО для направления
JPEG_EVERY = 30            # каждый 30-й обработанный кадр уходит в jpg
#: Допуск сопоставления трека с его детекцией — ДОЛЯ ВЫСОТЫ РАМКИ, не пиксели.
#: Трекер сглаживает состояние, поэтому опорная точка трека не совпадает с
#: детекцией бит в бит. Абсолютный порог здесь неверен как критерий: 15 px для
#: рамки 40 px и для рамки 204 px — это разные вопросы.
#: ЗАМЕРЕНО на этом клипе: относительное расстояние до ближайшей детекции имеет
#: медиану 0.006, p95 = 0.024, p99 = 0.044 высоты рамки. Порог 0.08 лежит почти
#: вдвое выше p99 и покрывает 99.81% строк.
BOX_MATCH_TOL_FRAC = 0.08
BOX_MATCH_TOL_MIN_PX = 4.0     # для совсем мелких рамок доля вырождается
PLAN_W = 430

#: Порог детектора для ОБЕЗЛИЧИВАНИЯ. Намеренно много ниже боевого: цена
#: лишнего размытия — размытый столб, цена пропуска — опубликованное лицо.
#: Полнота здесь важнее точности, и это не тот компромисс, который стоит
#: подкручивать ради красоты кадра.
ANON_CONF = 0.05

#: Ниже этой высоты рамки полоса головы вырождается: при top_frac 0.3 у рамки
#: в 20 px это 6 px, меньше min_kernel_px из конфига.
ANON_MIN_BOX_H = 20

EVIDENCE_CONFIG = "configs/evidence.yaml"

ZONE_COLORS = [(90, 230, 90), (60, 190, 255), (255, 160, 80), (230, 120, 255)]
#: Цвет точки ног по источнику опорной точки. Прямое измерение и косвенная
#: подстановка обязаны различаться глазом: иначе картинка внушает точность,
#: которой нет.
FOOT_COLORS = {"ankle": (80, 255, 80), "ankles": (80, 255, 80),
               "bbox_bottom": (60, 130, 255)}
FOOT_FALLBACK = (160, 160, 160)


def track_color(track_id: int) -> tuple[int, int, int]:
    """Устойчивый цвет трека: один и тот же id всегда одного цвета."""
    h = (int(track_id) * 47) % 180
    b, g, r = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def load_anonymiser(detect_cfg: dict):
    """Детектор и параметры обезличивания. Без них рендер не начинается.

    Отдельный проход детектором, а НЕ сохранённые детекции из
    det/frames.parquet: те отфильтрованы боевым порогом уверенности и
    отсечкой по высоте рамки в 70 px, то есть людей мельче и неувереннее
    там просто нет. Для рисования боксов это правильно, для обезличивания —
    нет: не попавший в паркет детекций всё равно остаётся человеком
    на кадре.
    """
    from ultralytics import YOLO

    priv = load_config(EVIDENCE_CONFIG).get("privacy") or {}
    need = ("face_blur_top_frac", "blur_kernel_frac", "blur_sigma_frac",
            "pixelate_factor")
    missing = [k for k in need if k not in priv]
    if missing:
        raise SystemExit(
            f"в {EVIDENCE_CONFIG} нет ключей приватности {missing}. Рендер "
            f"оверлея без обезличивания не запускается (правило 9)")

    weights = (detect_cfg.get("model") or {}).get("weights")
    if not weights or not Path(weights).is_file():
        raise SystemExit(
            f"нет весов детектора {weights!r}. Обезличивание — часть рендера, "
            f"а не опция: без детектора кадр не отрисовывается")

    # ВТОРАЯ модель, поз. Замер на восьми кадрах: детектор на пороге 0.05
    # пропустил 11 человек, чьи лица модель позы нашла уверенно. Одна сеть
    # ошибается там, где другая справляется, и для обезличивания дешевле
    # объединить их находки, чем спорить, какая права.
    pose_w = Path("models/yolo11m-pose.pt")
    if not pose_w.is_file():
        raise SystemExit(
            f"нет весов позы {pose_w}. Они нужны не для красоты: детектор "
            f"в одиночку пропускает людей, и это измерено")
    return (YOLO(weights), YOLO(str(pose_w))), priv


#: Кейпоинты лица в COCO: нос, глаза, уши.
FACE_KP = (0, 1, 2, 3, 4)

#: Насколько раздуть рамку вокруг найденных кейпоинтов лица, в долях от её
#: собственного размера. Кейпоинты отмечают точки, а закрыть надо всю голову:
#: волосы, подбородок, уши по краям.
HEAD_PAD = 0.9


def _head_boxes_from_pose(res) -> list[tuple[float, float, float, float]]:
    """Рамки вокруг найденных кейпоинтов лица.

    Полоса в верхних 30% рамки человека — приближение, и оно ломается на
    наклонённой голове: замер на кадре f060369 показал лицо, у которого нос
    внутри полосы, а подбородок под её краем, с резкой границей ровно по
    лицу. Кейпоинты говорят, где голова НА САМОМ ДЕЛЕ, и размывать надо там.
    """
    if res.keypoints is None or res.keypoints.conf is None:
        return []
    xy = np.asarray(res.keypoints.xy.cpu())
    cf = np.asarray(res.keypoints.conf.cpu())
    out = []
    for pi in range(cf.shape[0]):
        pts = [xy[pi, j] for j in FACE_KP if cf[pi, j] >= 0.20]
        if not pts:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        # у профиля видны один глаз и ухо: рамка вырождается в точку, и
        # раздувать её от нулевого размера нечего. Берём запас от роста.
        span = max(x1 - x0, y1 - y0, 12.0)
        pad = span * HEAD_PAD
        out.append((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    return out


def _boxes_union(models, frame, params: dict) -> np.ndarray:
    """Рамки для обезличивания: люди от обеих моделей плюс головы по позе.

    Дубликаты не убираются: размыть одну и ту же голову дважды безвредно,
    а выкидывать пересечения значило бы рисковать ради экономии, которой
    здесь нет.
    """
    out = []
    for m in models:
        r = m.predict(frame, imgsz=params.get("imgsz", 1280), conf=ANON_CONF,
                      device=params.get("device", "cpu"),
                      half=bool(params.get("half", False)),
                      classes=[0], verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            out.append(np.asarray(r.boxes.xyxy.cpu(), dtype=np.float64))
        heads = _head_boxes_from_pose(r)
        if heads:
            out.append(np.asarray(heads, dtype=np.float64))
    return np.vstack(out) if out else np.empty((0, 4))


def anonymise_frame(frame, model, params: dict, priv: dict) -> tuple[int, int]:
    """Размывает голову каждому найденному человеку. Правит кадр на месте.

    Вызывается ДО отрисовки боксов и стрелок: иначе размытие затёрло бы
    разметку, а не лицо. Возвращает (размыто, отброшено).
    """
    models = model if isinstance(model, (tuple, list)) else (model,)
    done = skipped = 0
    boxes = _boxes_union(models, frame, params)
    # Рамка человека вытянута вертикально (h/w около 2.5), рамка головы близка
    # к квадрату. У головы размывается ВСЯ площадь, а не верхняя доля: она и
    # есть голова, доли внутри неё брать неоткуда.
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 4 or y2 - y1 < ANON_MIN_BOX_H:
            skipped += 1
            continue
        h, w = y2 - y1, x2 - x1
        top = 1.0 if h < 1.7 * w else float(priv["face_blur_top_frac"])
        try:
            blurred, _ = blur_face_region(
                frame[y1:y2, x1:x2].copy(),
                top_frac=top,
                kernel_frac=float(priv["blur_kernel_frac"]),
                sigma_frac=float(priv["blur_sigma_frac"]),
                pixelate_factor=int(priv["pixelate_factor"]))
        except EvidenceError:
            skipped += 1          # область уже однородна, лица там нет
            continue
        frame[y1:y2, x1:x2] = blurred
        done += 1
    return done, skipped


def open_encoder(path: Path, size: tuple[int, int], fps: float):
    """ffmpeg на stdin. Проверяем, что он есть, ДО начала работы."""
    w, h = size
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit(
            "ffmpeg не найден в PATH. Он нужен для H.264: OpenCV на этой машине "
            "H.264 не пишет (нет OpenH264), fourcc открывается, а файл выходит пустой")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
           "-pix_fmt", "yuv420p",           # иначе не откроется в части плееров
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


class PlanView:
    """Схема плана земли. Метры -> пиксели канвы, масштаб общий по обеим осям."""

    def __init__(self, w: int, h: int, pts_m: np.ndarray, pad: float = 0.08,
                 rotate: bool = True):
        # Улица тянется на 35 м вдоль оси +x и всего на 12 м поперёк. Без
        # разворота она ложится в горизонтальную ленту, а канва вертикальная,
        # и план вырождается в полоску. Разворот на 90 градусов — это ТОЛЬКО
        # смена осей отображения: масштаб остаётся общим, прямая остаётся
        # прямой, и проверка калибровки глазом сохраняет силу.
        self.w, self.h, self.rotate = w, h, rotate
        pts_m = self._swap(pts_m)
        lo, hi = pts_m.min(axis=0), pts_m.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        lo, hi = lo - span * pad, hi + span * pad
        span = hi - lo
        # Один масштаб на обе оси, иначе прямая улица выглядит кривой и
        # «проверка калибровки глазом» перестаёт быть проверкой.
        self.s = float(min(w / span[0], h / span[1]))
        self.lo = lo
        self.off = np.array([(w - span[0] * self.s) / 2.0,
                             (h - span[1] * self.s) / 2.0])

    def _swap(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if not self.rotate:
            return p
        # ПОВОРОТ, а не транспозиция. Прежнее p[:, ::-1] меняло оси местами,
        # то есть отражало план относительно диагонали (определитель -1), и
        # витрины уезжали направо, хотя в кадре они слева. Замер по сохранённой
        # H: шаг +y_m уводит в кадре ВЛЕВО на 95.6 px, значит гомография
        # согласна с камерой, а врала именно отрисовка.
        return np.stack([-p[:, 1], p[:, 0]], axis=1)

    def to_px(self, xy_m) -> np.ndarray:
        p = (self._swap(xy_m) - self.lo) * self.s
        p += self.off
        p[:, 1] = self.h - p[:, 1]      # ось y вверх, как на карте
        return p.astype(np.int32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/s3_detect.yaml")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--frame-dir", type=Path, default=FRAME_DIR)
    ap.add_argument("--plan", action="store_true", help="второе окно: вид сверху")
    ap.add_argument("--arrow-px", type=int, default=ARROW_PX)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--densest-sec", type=float, default=None,
                    help="вырезать самое ПЛОТНОЕ окно такой длины вместо начала "
                         "записи. Час целиком — это 36000 кадров, минут десять "
                         "рендера и файл, который не открывается")
    args = ap.parse_args(argv)

    import pandas as pd

    cfg = load_config(args.config)
    video = Path(require(cfg, "input", "video"))
    hom = read_json("calib/homography.json")
    h_px_to_m = np.asarray(hom["H"], dtype=np.float64)
    # Обратное направление в контракте не хранится намеренно, чтобы нельзя было
    # перепутать сторону. Здесь нужно ТОЛЬКО для рисования и считается локально.
    h_m_to_px = np.linalg.inv(h_px_to_m)
    unit = "m" if hom.get("scale_known") else "conv.unit"

    det = pd.read_parquet("det/frames.parquet")
    tracks = pd.read_parquet("track/tracks.parquet")
    orient = pd.read_parquet("pose/orient.parquet")
    zone_frames = pd.read_parquet("attn/track_zone_frames.parquet")
    zones = read_json("zones/zones.geojson")
    try:
        attr = pd.read_parquet("attr/tracks_attr.parquet")
        top_color = dict(zip(attr["track_id"], attr["top_color_name"].fillna("?")))
    except (OSError, FileNotFoundError):
        top_color = {}

    facades: dict[str, dict] = {}
    roi_m = None
    for f in zones["features"]:
        pr, geom = f["properties"], f["geometry"]
        if pr["zone_type"] == "roi":
            roi_m = np.asarray(geom["coordinates"][0], dtype=np.float64)
        elif pr["zone_type"] == "facade":
            facades[pr["zone_id"]] = {
                "name": pr["name_ru"],
                "color": ZONE_COLORS[len(facades) % len(ZONE_COLORS)],
                "poly": np.asarray(pr["polygon_px"], dtype=np.int32),
                "seg_m": np.asarray(geom["coordinates"], dtype=np.float64),
            }
    if not facades:
        raise SystemExit("в zones/zones.geojson нет ни одного фасада")

    # Всё вне ROI не рисуется и не считается: дальний конец улицы за отсечкой
    # в аналитику не идёт, и показывать его как результат нельзя.
    if roi_m is not None:
        n_all = tracks["track_id"].nunique()
        keep = np.array([point_in_polygon_m((float(x), float(y)), roi_m)
                         for x, y in tracks[["foot_x_m", "foot_y_m"]].to_numpy()])
        tracks = tracks[keep]
        print(f"[overlay] ROI: остаётся {keep.mean():.1%} строк, "
              f"{tracks['track_id'].nunique()} треков из {n_all}")

    trk = tracks.merge(orient[["track_id", "frame_idx", "body_yaw_deg", "head_yaw_deg"]],
                       on=["track_id", "frame_idx"], how="left")
    trk["yaw"] = trk["head_yaw_deg"].fillna(trk["body_yaw_deg"])
    by_frame = {int(f): g for f, g in trk.groupby("frame_idx")}
    # Массивы, а не срезы датафрейма: поиск рамки идёт на каждого человека
    # на каждом кадре, и pandas-маска здесь стоит минуты.
    det_by_frame = {}
    for f, g in det.groupby("frame_idx"):
        b = g[["x1_px", "y1_px", "x2_px", "y2_px"]].to_numpy(np.float64)
        det_by_frame[int(f)] = (b, (b[:, 0] + b[:, 2]) / 2.0, b[:, 3])

    # Хвосты и время в кадре — по предпосчитанным массивам на трек. Фильтровать
    # весь датафрейм на каждого человека на каждом кадре — это O(n^2) и минуты
    # лишнего времени на трёхминутном куске.
    hist: dict[int, dict] = {}
    for tid, g in trk.sort_values("frame_idx").groupby("track_id"):
        hist[int(tid)] = {
            "f": g["frame_idx"].to_numpy(),
            "px": g[["foot_x_px", "foot_y_px"]].to_numpy(np.float64),
            "m": g[["foot_x_m", "foot_y_m"]].to_numpy(np.float64),
            "t0": float(g["ts"].iloc[0]),
        }

    hits_by_frame: dict[int, list] = {}
    for _, r in zone_frames[zone_frames["gaze_hit"]].iterrows():
        hits_by_frame.setdefault(int(r["frame_idx"]), []).append(
            (r["zone_id"], int(r["track_id"])))

    frames = sorted(by_frame)
    if args.densest_sec:
        # Самое плотное окно по числу людей в кадре, скользящей суммой.
        # Шаг между обработанными кадрами известен из конфига: stride/fps.
        n_per = np.array([len(by_frame[f]) for f in frames], dtype=np.float64)
        stride_cfg = int((cfg.get("detect") or {}).get("frame_stride", 1))
        sec_per_frame = stride_cfg / 30.0
        win = max(1, min(len(frames), int(round(args.densest_sec / sec_per_frame))))
        c = np.convolve(n_per, np.ones(win), mode="valid")
        i0 = int(np.argmax(c))
        frames = frames[i0:i0 + win]
        print(f"[overlay] самое плотное окно: кадры {frames[0]}..{frames[-1]}, "
              f"{win} кадров, в среднем {c[i0] / win:.1f} человека в кадре")
    if args.max_frames:
        frames = frames[:args.max_frames]
    if not frames:
        raise SystemExit("в track/tracks.parquet нет кадров")

    # Модель обезличивания грузится ДО открытия видео: если весов или ключей
    # приватности нет, рендер обязан не начаться вовсе, а не остановиться на
    # середине с половиной необезличенных кадров на диске.
    anon_model, anon_priv = load_anonymiser(cfg)
    params = infer_params(cfg)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cv2 не открыл {video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = int((cfg.get("detect") or {}).get("frame_stride", 1))
    out_fps = src_fps / stride

    plan = None
    if args.plan:
        pts = [roi_m] if roi_m is not None else []
        pts += [fac["seg_m"] for fac in facades.values()]
        pts.append(trk[["foot_x_m", "foot_y_m"]].dropna().to_numpy(np.float64))
        plan = PlanView(PLAN_W, h, np.concatenate(pts))
    out_w = w + (PLAN_W if plan else 0)

    # Старые кадры обязаны исчезнуть, иначе в папке окажется смесь двух
    # записей. Но удаляются они ПОСЛЕ успешного рендера, а не до: раньше
    # упавший прогон оставлял пустой каталог, то есть терял и старое, и новое.
    args.frame_dir.mkdir(parents=True, exist_ok=True)
    stale = {f.name for f in args.frame_dir.glob("*.jpg")}
    fresh: set[str] = set()
    enc = open_encoder(args.out, (out_w, h), out_fps)
    print(f"{video} -> {args.out}: {len(frames)} кадров, {out_fps:.1f} fps, "
          f"{out_w}x{h}{' (кадр + план)' if plan else ''}")

    tail_frames = int(TAIL_SEC * src_fps)
    seen_tracks: set[int] = set()
    n_gaze_events = 0
    n_predicted = 0
    trail_plan: list[tuple[np.ndarray, tuple]] = []
    wanted, idx, written = set(frames), 0, 0
    anon_done = anon_skipped = 0

    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if idx not in wanted:
            idx += 1
            continue
        wanted.discard(idx)

        # ОБЕЗЛИЧИВАНИЕ — ПЕРВОЕ, ЧТО ПРОИСХОДИТ С КАДРОМ, и оно
        # безусловно. Флага нет намеренно: опция, которую можно забыть
        # выставить, рано или поздно окажется невыставленной, а цена
        # забывчивости здесь — опубликованное лицо. До отрисовки, а не
        # после: иначе размытие затёрло бы боксы и стрелки вместо лиц.
        n_anon, n_skip = anonymise_frame(frame, anon_model, params, anon_priv)
        anon_done += n_anon
        anon_skipped += n_skip

        g = by_frame[idx]
        lit = hits_by_frame.get(idx, [])
        lit_zones = {z for z, _ in lit}
        n_gaze_events += len(lit)

        # --- витрины на кадре ------------------------------------------------ #
        for zid, fac in facades.items():
            on = zid in lit_zones
            if on:
                ov = frame.copy()
                cv2.fillPoly(ov, [fac["poly"]], fac["color"])
                cv2.addWeighted(ov, 0.32, frame, 0.68, 0, dst=frame)
            cv2.polylines(frame, [fac["poly"]], True, fac["color"],
                          4 if on else 2, cv2.LINE_AA)
            x0, y0 = int(fac["poly"][:, 0].min()), int(fac["poly"][:, 1].min())
            label = zid.replace("facade_", "")
            if on:
                label += " <- " + ",".join(f"#{t}" for z, t in lit if z == zid)
            cv2.putText(frame, label, (x0, max(18, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fac["color"],
                        3 if on else 2, cv2.LINE_AA)

        # --- люди ------------------------------------------------------------ #
        looking_now = {t for _, t in lit}
        dets = det_by_frame.get(idx)
        for _, r in g.iterrows():
            tid = int(r["track_id"])
            seen_tracks.add(tid)
            col = track_color(tid)
            fx, fy = float(r["foot_x_px"]), float(r["foot_y_px"])
            hh = hist[tid]
            cx_box, cy_box, bbox_h = fx, fy - 40.0, 0.0

            dwell = float(r["ts"]) - hh["t0"]
            box = None
            if dets is not None:
                boxes, cxs, y2s = dets
                j = int(np.argmin(np.hypot(cxs - fx, y2s - fy)))
                tol = max(BOX_MATCH_TOL_MIN_PX,
                          BOX_MATCH_TOL_FRAC * (boxes[j][3] - boxes[j][1]))
                if np.hypot(cxs[j] - fx, y2s[j] - fy) <= tol:
                    box = boxes[j]
            if box is not None:
                x1, y1, x2, y2 = box
                bbox_h = float(y2 - y1)
                cx_box, cy_box = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col,
                              3 if tid in looking_now else 2)
                cap_txt = f"#{tid} {top_color.get(tid, '?')} {dwell:.1f}s"
                cv2.putText(frame, cap_txt, (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            else:
                # Детекции на этом кадре нет: трек экстраполирован трекером.
                # Рисовать рамку было бы показом несуществующего измерения.
                n_predicted += 1
                cv2.putText(frame, f"#{tid} pred {dwell:.1f}s",
                            (int(fx) - 30, int(fy) - 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

            # --- хвост траектории 3 с --------------------------------------- #
            sel = (hh["f"] <= idx) & (hh["f"] > idx - tail_frames)
            if sel.sum() > 1:
                cv2.polylines(frame, [hh["px"][sel].astype(np.int32)],
                              False, col, 2, cv2.LINE_AA)

            # --- точка ног, цвет по источнику ------------------------------- #
            fcol = FOOT_COLORS.get(str(r.get("foot_source")), FOOT_FALLBACK)
            cv2.circle(frame, (int(fx), int(fy)), 5, fcol, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(fx), int(fy)), 5, (20, 20, 20), 1, cv2.LINE_AA)

            # --- стрелка поворота корпуса из ЦЕНТРА человека ---------------- #
            yaw = r.get("yaw")
            if yaw is not None and np.isfinite(yaw) and np.isfinite(r["foot_x_m"]):
                rad = np.deg2rad(float(yaw))
                tip_m = np.array([float(r["foot_x_m"]) + ARROW_PROBE_M * np.cos(rad),
                                  float(r["foot_y_m"]) + ARROW_PROBE_M * np.sin(rad), 1.0])
                p = h_m_to_px @ tip_m
                if abs(p[2]) > 1e-9:
                    dx, dy = p[0] / p[2] - fx, p[1] / p[2] - fy
                    n = float(np.hypot(dx, dy))
                    if n > 1e-6:
                        acol = (60, 255, 255) if tid in looking_now else col
                        tip = (int(cx_box + dx / n * args.arrow_px),
                               int(cy_box + dy / n * args.arrow_px))
                        cv2.arrowedLine(frame, (int(cx_box), int(cy_box)), tip,
                                        acol, 3, cv2.LINE_AA, tipLength=0.3)

        # --- счётчик ---------------------------------------------------------- #
        ts = float(g["ts"].iloc[0])
        frame[0:144, 0:500] = (frame[0:144, 0:500] * 0.25).astype(np.uint8)
        for i, line in enumerate([
                f"t = {ts:6.1f} s   frame {idx}",
                f"in frame: {len(g)}   looking now: {len(looking_now)}",
                f"tracks total: {len(seen_tracks)}   hits total: {n_gaze_events}",
                f"no-detection (predicted): {n_predicted}",
                f"units: {unit}   calib: {hom.get('calib_status', '?')}"]):
            cv2.putText(frame, line, (12, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "arrow = body/head turn, NOT gaze direction",
                    (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 2, cv2.LINE_AA)

        # --- второе окно: план ------------------------------------------------ #
        if plan is not None:
            canvas = np.full((h, PLAN_W, 3), 22, np.uint8)
            if roi_m is not None:
                cv2.polylines(canvas, [plan.to_px(roi_m)], True, (70, 70, 70), 1,
                              cv2.LINE_AA)
            for old_pts, ocol in trail_plan[-4000:]:
                cv2.polylines(canvas, [old_pts], False,
                              tuple(int(c * 0.28) for c in ocol), 1, cv2.LINE_AA)
            for zid, fac in facades.items():
                seg = plan.to_px(fac["seg_m"])
                on = zid in lit_zones
                cv2.line(canvas, tuple(seg[0]), tuple(seg[1]), fac["color"],
                         7 if on else 3, cv2.LINE_AA)
            for _, r in g.iterrows():
                if not np.isfinite(r["foot_x_m"]):
                    continue
                tid = int(r["track_id"])
                col = track_color(tid)
                p = plan.to_px([r["foot_x_m"], r["foot_y_m"]])[0]
                hh = hist[tid]
                sel = (hh["f"] <= idx) & (hh["f"] > idx - tail_frames)
                if sel.sum() > 1:
                    seg = plan.to_px(hh["m"][sel])
                    cv2.polylines(canvas, [seg], False, col, 1, cv2.LINE_AA)
                    trail_plan.append((seg, col))
                cv2.circle(canvas, tuple(p), 4, col, -1, cv2.LINE_AA)
                yaw = r.get("yaw")
                if yaw is not None and np.isfinite(yaw):
                    rad = np.deg2rad(float(yaw))
                    tip = plan.to_px([r["foot_x_m"] + 1.6 * np.cos(rad),
                                      r["foot_y_m"] + 1.6 * np.sin(rad)])[0]
                    acol = (60, 255, 255) if tid in looking_now else col
                    cv2.arrowedLine(canvas, tuple(p), tuple(tip), acol, 2,
                                    cv2.LINE_AA, tipLength=0.35)
            cv2.putText(canvas, f"PLAN VIEW ({unit})", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2, cv2.LINE_AA)
            cv2.putText(canvas, "street straight + people along it = calib ok",
                        (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (150, 150, 150), 1, cv2.LINE_AA)
            frame = np.hstack([frame, canvas])

        enc.stdin.write(frame.tobytes())
        if written % JPEG_EVERY == 0:
            fresh.add(f"f{idx:06d}.jpg")
            cv2.imwrite(str(args.frame_dir / f"f{idx:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        written += 1
        if written % 200 == 0:
            print(f"  кадров записано {written}/{len(frames)}")
        idx += 1

    cap.release()
    enc.stdin.close()
    rc = enc.wait()
    if rc != 0:
        raise SystemExit(
            f"ffmpeg вернул {rc}: видео {args.out} обрезано или не записано. "
            f"Молча считать это успехом нельзя — дальше страница сошлётся "
            f"на битый файл")
    # Рендер дошёл до конца — только теперь убираем кадры прошлого прогона.
    dropped = 0
    for name in stale - fresh:
        try:
            (args.frame_dir / name).unlink()
            dropped += 1
        except OSError:
            pass
    if dropped:
        print(f"[overlay] удалено кадров прошлого прогона: {dropped}")
    # Паспорт видео рядом с самим видео: какие исходные кадры в него попали.
    # Без него страница реплея не может знать, весь ли это час или вырезка,
    # и её часы расходятся с картинкой.
    side = args.out.with_suffix(".window.json")
    write_json(side, {
        "video": str(args.out).replace("\\", "/"),
        "src_frame_first": int(frames[0]) if frames else None,
        "src_frame_last": int(frames[-1]) if frames else None,
        "n_frames": int(written),
        "fps": float(out_fps),
        "note_ru": "исходные кадры, попавшие в видео. Реплей сдвигает часы "
                   "по src_frame_first, иначе план разъедется с картинкой",
    })
    print(f"[overlay] окно записано: {side}")

    print(f"[overlay] обезличено рамок {anon_done}, отброшено вырожденных "
          f"{anon_skipped} (порог детектора {ANON_CONF}, полоса "
          f"{anon_priv['face_blur_top_frac']})")
    n_jpg = len(list(args.frame_dir.glob("*.jpg")))
    size_mb = args.out.stat().st_size / 1e6 if args.out.is_file() else 0
    print(f"готово: {args.out} ({size_mb:.1f} МБ), кадров {written}, "
          f"треков {len(seen_tracks)}, попаданий луча {n_gaze_events}, "
          f"jpg {n_jpg} в {args.frame_dir}")
    if size_mb < 0.05:
        raise SystemExit("видео пустое: ffmpeg ничего не записал")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
