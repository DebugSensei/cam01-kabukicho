"""Общий код детекции и трекинга.

Два потребителя: внутренний пилот S1 (подбор масштаба по росту) и полноценные
этапы S3/S4. Параметры инференса читаются из configs/s3_detect.yaml, чтобы
калибровка и пайплайн считали ОДНИХ И ТЕХ ЖЕ людей: разойдись пороги, и масштаб
был бы подобран по одной популяции, а метрики посчитаны по другой.

Пилотные результаты S1 никогда не пишутся в det/ и track/. Они живут внутри
calib/pilot/ и в полях pilot_* артефакта S1 — иначе пилот однажды подменит
настоящий прогон.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


class PilotError(RuntimeError):
    """Пилотный прогон невозможен."""


@dataclass
class Detection:
    frame_idx: int
    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float
    conf: float

    @property
    def height_px(self) -> float:
        return self.y2_px - self.y1_px

    @property
    def foot_px(self) -> tuple[float, float]:
        """Низ бокса по центру — точка ног."""
        return ((self.x1_px + self.x2_px) / 2.0, self.y2_px)

    @property
    def head_px(self) -> tuple[float, float]:
        """Верх бокса по центру — макушка."""
        return ((self.x1_px + self.x2_px) / 2.0, self.y1_px)


def infer_params(detect_cfg: dict[str, Any]) -> dict[str, Any]:
    """Параметры инференса из configs/s3_detect.yaml. Одни на калибровку и пайплайн."""
    model = detect_cfg.get("model") or {}
    detect = detect_cfg.get("detect") or {}
    missing = [k for k in ("imgsz", "device", "precision") if model.get(k) is None]
    if missing:
        raise PilotError(f"в configs/s3_detect.yaml не заданы model.{', model.'.join(missing)}")
    return {
        "weights": model.get("weights"),
        "imgsz": int(model["imgsz"]),
        "device": model["device"],
        "half": str(model["precision"]).lower() == "fp16",
        "classes": list(model.get("classes") or [0]),
        "conf": float(detect.get("conf_thr", 0.25)),
        "iou": float(detect.get("iou_nms", 0.7)),
    }


def sample_burst_indices(n_frames: int, n_bursts: int,
                         burst_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Непрерывные пачки кадров, равномерно разнесённые по длине клипа.

    Возвращает (indices, burst_id) — номера кадров и номер пачки для каждого.

    Почему пачки, а не одиночные кадры. Одиночная выборка 300 кадров из 5400
    даёт шаг 0.6 с между наблюдениями. ByteTrack в толпе на таком шаге
    связывает ненадёжно, и медианная скорость выходит смещённой. Внутри пачки
    кадры идут подряд, связывание честное, а покрытие по времени то же самое:
    пачки разнесены по всей длине клипа.

    Номер пачки нужен дальше: скорость считается ТОЛЬКО внутри пачки. Между
    пачками разрыв в минуты, и разность координат через него ничего не значит.
    """
    if n_frames < 1:
        raise PilotError(f"в клипе {n_frames} кадров")
    n_bursts = int(n_bursts)
    burst_frames = int(burst_frames)
    if n_bursts < 1 or burst_frames < 1:
        raise PilotError(f"нужны n_bursts >= 1 и burst_frames >= 1, "
                         f"получено {n_bursts} и {burst_frames}")
    total = n_bursts * burst_frames
    if total > n_frames:
        raise PilotError(
            f"запрошено {n_bursts} x {burst_frames} = {total} кадров, "
            f"а в клипе всего {n_frames}"
        )
    starts = np.unique(
        np.linspace(0, n_frames - burst_frames, n_bursts).round().astype(np.int64))
    idx, bid = [], []
    for b, s0 in enumerate(starts):
        for k in range(burst_frames):
            idx.append(int(s0) + k)
            bid.append(b)
    indices = np.asarray(idx, dtype=np.int64)
    burst_id = np.asarray(bid, dtype=np.int64)
    order = np.argsort(indices)
    return indices[order], burst_id[order]


def iter_frames(video_path: str | Path, indices: np.ndarray,
                progress=None) -> Iterator[tuple[int, np.ndarray]]:
    """Последовательное чтение с отдачей только нужных кадров.

    Не перемотка: у .ts из HLS-сегментов CAP_PROP_POS_FRAMES врёт на границах
    сегментов, и номера кадров разошлись бы с реальными. Поэтому идём подряд.

    Ненужные кадры берутся grab() — он демультиплексирует, но НЕ декодирует.
    Декодирование остаётся только для запрошенных. На часовом .ts, где нужные
    кадры разбросаны до 108 000-го, это разница между «читаем весь час
    полностью» и «пробегаем его».

    progress — необязательный вызываемый объект progress(idx, n_left). Без него
    длинный проход выглядит как зависание: за минуты не печатается ничего.
    """
    path = Path(video_path)
    if not path.is_file():
        raise PilotError(f"нет видео: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise PilotError(f"cv2 не открыл {path}")
    wanted = set(int(i) for i in indices)
    idx = 0
    try:
        while wanted:
            if not cap.grab():
                break
            if idx in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                wanted.discard(idx)
                yield idx, frame
            if progress is not None and idx % 2000 == 0:
                progress(idx, len(wanted))
            idx += 1
    finally:
        cap.release()


def detect_people(video_path: str | Path, indices: np.ndarray,
                  params: dict[str, Any]) -> tuple[list[Detection], tuple[int, int]]:
    """Детекция людей на заданных кадрах. Возвращает детекции и (h, w) кадра."""
    from ultralytics import YOLO

    model = YOLO(params["weights"])
    dets: list[Detection] = []
    shape: tuple[int, int] | None = None
    for frame_idx, frame in iter_frames(video_path, indices):
        if shape is None:
            shape = (frame.shape[0], frame.shape[1])
        res = model.predict(frame, imgsz=params["imgsz"], classes=params["classes"],
                            conf=params["conf"], iou=params["iou"],
                            device=params["device"], half=params["half"], verbose=False)
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            dets.append(Detection(frame_idx, float(x1), float(y1),
                                  float(x2), float(y2), float(c)))
    if shape is None:
        raise PilotError(f"из {video_path} не декодировано ни одного кадра")
    if not dets:
        raise PilotError("детектор не нашёл ни одного человека — калибровать нечем")
    return dets, shape


def _pairwise_max_iou(boxes: np.ndarray) -> np.ndarray:
    """Максимальный IoU каждого бокса с остальными в том же кадре."""
    n = len(boxes)
    if n < 2:
        return np.zeros(n)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = area[:, None] + area[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(iou, 0.0)
    return iou.max(axis=1)


def filter_height_samples(dets: list[Detection], frame_shape: tuple[int, int],
                          cfg: dict[str, Any]) -> tuple[list[Detection], dict[str, int]]:
    """Отбор детекций, по которым МОЖНО мерить рост.

    Каждый фильтр отсекает конкретный источник мусорного роста, и число
    отсеянных записывается: доля отброшенного идёт в артефакт (правило 7).
    """
    h, w = frame_shape
    min_conf = float(cfg.get("min_conf", 0.5))
    max_iou = float(cfg.get("max_neighbour_iou", 0.2))
    min_h = float(cfg.get("min_bbox_h_px", 60.0))
    edge = float(cfg.get("edge_margin_px", 2.0))

    stats = {"total": len(dets), "low_conf": 0, "touches_edge": 0,
             "crowded": 0, "too_small": 0, "kept": 0}

    by_frame: dict[int, list[Detection]] = {}
    for d in dets:
        by_frame.setdefault(d.frame_idx, []).append(d)

    kept: list[Detection] = []
    for frame_idx, group in by_frame.items():
        boxes = np.array([[d.x1_px, d.y1_px, d.x2_px, d.y2_px] for d in group])
        max_iou_per_box = _pairwise_max_iou(boxes)
        for d, iou in zip(group, max_iou_per_box):
            if d.conf <= min_conf:
                stats["low_conf"] += 1
                continue
            # Обрезанный краем человек: низ бокса — не стопа, верх — не макушка.
            if (d.x1_px <= edge or d.y1_px <= edge
                    or d.x2_px >= w - edge or d.y2_px >= h - edge):
                stats["touches_edge"] += 1
                continue
            # В толпе низ бокса лежит на чужой спине.
            if iou >= max_iou:
                stats["crowded"] += 1
                continue
            if d.height_px <= min_h:
                stats["too_small"] += 1
                continue
            kept.append(d)
    stats["kept"] = len(kept)
    return kept, stats


def track_speeds(video_path: str | Path, indices: np.ndarray, burst_id: np.ndarray,
                 params: dict[str, Any], h_px_to_m, fps: float,
                 min_track_s: float, burst_frames: int) -> dict[str, Any]:
    """Скорости пилотных треков на плане, м/с. Трекинг ВНУТРИ пачек.

    ByteTrack сбрасывается на первом кадре каждой пачки (persist=False): между
    пачками разрыв в минуты, связывать через него нельзя — трекер склеил бы
    разных людей. Треки разных пачек разведены ключом (burst_id, track_id).

    Скорость считается только между соседними кадрами одной пачки.
    """
    from ultralytics import YOLO
    from looq.calib import apply_h

    burst_duration_s = burst_frames / fps
    if min_track_s > burst_duration_s:
        # Правило 8: молча вернуть ноль треков было бы хуже всего — гейт получил
        # бы пустую выборку и не понял, что дело в арифметике, а не в людях.
        raise PilotError(
            f"min_track_s={min_track_s} с больше длительности пачки: "
            f"{burst_frames} кадров / {fps:.2f} fps = {burst_duration_s:.2f} с. "
            f"Ни один трек не может быть длиннее пачки. Либо удлините "
            f"pilot.burst_frames, либо уменьшите pilot.min_track_s"
        )

    model = YOLO(params["weights"])
    frame_to_burst = dict(zip(indices.tolist(), burst_id.tolist()))
    tracks: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    seen_bursts: set[int] = set()

    for frame_idx, frame in iter_frames(video_path, indices):
        b = int(frame_to_burst[int(frame_idx)])
        first_in_burst = b not in seen_bursts
        seen_bursts.add(b)
        res = model.track(frame, imgsz=params["imgsz"], classes=params["classes"],
                          conf=params["conf"], iou=params["iou"],
                          device=params["device"], half=params["half"],
                          tracker="bytetrack.yaml", persist=not first_in_burst,
                          verbose=False)
        boxes = res[0].boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            foot = np.array([[(x1 + x2) / 2.0, y2]])
            try:
                m = apply_h(h_px_to_m, foot)[0]
            except Exception:
                continue  # точка за горизонтом, для скорости не годится
            tracks.setdefault((b, int(tid)), []).append(
                (int(frame_idx), float(m[0]), float(m[1])))

    speeds: list[float] = []
    n_long = 0
    n_over_2s = 0
    for obs in tracks.values():
        if len(obs) < 3:
            continue
        obs.sort()
        duration_s = (obs[-1][0] - obs[0][0]) / fps
        if duration_s >= 2.0:
            n_over_2s += 1
        if duration_s < min_track_s:
            continue
        n_long += 1
        v = _track_speed_mps(obs, fps)
        if v is not None:
            speeds.append(v)

    return {
        "speeds_mps": speeds,
        "n_tracks_total": len(tracks),
        "n_tracks_long": n_long,
        # Доля треков длиннее 2 с остаётся в артефакте по требованию владельца.
        # ВНИМАНИЕ: при пачке короче 2 с она равна нулю ПО ПОСТРОЕНИЮ, а не
        # потому, что людей нет. Рядом лежит burst_duration_s, чтобы это было видно.
        "n_tracks_over_2s": n_over_2s,
        "frac_tracks_over_2s": (n_over_2s / len(tracks)) if tracks else None,
        "min_track_s": min_track_s,
        "burst_frames": burst_frames,
        "burst_duration_s": round(burst_duration_s, 3),
    }


def _track_speed_mps(obs: list[tuple[int, float, float]], fps: float) -> float | None:
    """Скорость трека как наклон МНК по всей пачке, а не разность соседних кадров.

    Почему не разность соседей. За один кадр при 30 fps пешеход проходит около
    4 см, а дрожание низа рамки на плане — порядка 10-20 см. Скорость
    неотрицательна, поэтому шум НЕ сокращается при усреднении: он сдвигает
    медиану вверх. Наклон прямой по всем точкам пачки шум усредняет честно,
    потому что оценивается смещение, а не его модуль.
    """
    t = np.array([f for f, _, _ in obs], dtype=np.float64) / fps
    x = np.array([px for _, px, _ in obs], dtype=np.float64)
    y = np.array([py for _, _, py in obs], dtype=np.float64)
    tc = t - t.mean()
    stt = float((tc ** 2).sum())
    if stt < 1e-12:
        return None
    vx = float((tc * (x - x.mean())).sum() / stt)
    vy = float((tc * (y - y.mean())).sum() / stt)
    return float(np.hypot(vx, vy))
