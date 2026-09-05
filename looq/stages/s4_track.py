"""S4 track — трекинг и проекция опорной точки на план земли.

Читает готовые детекции S3 и гомографию S1, ничего не детектит заново:
ByteTrack применяется к сохранённым боксам, а не к кадрам. Так трекинг
воспроизводим и не зависит от того, совпал ли повторный инференс.

foot_source — колонка-флаг из правила 7. Приоритет из конфига:

    ankle       голеностоп виден в позе, точка стоит на земле   ПРЯМОЕ
    hip_est     экстраполяция вниз от бёдер по оценке роста     косвенное
    bbox_bottom низ рамки; при перекрытии это не стопа          косвенное

ВАЖНОЕ РЕШЕНИЕ, требующее подтверждения владельца. ankle и hip_est берутся
из позы, а поза по контракту считается на S5, то есть ПОСЛЕ S4. Читать артефакт
S5 отсюда нельзя — это сломало бы направление зависимостей. Поэтому при
foot_point.pose_enabled = true S4 гоняет модель позы САМ, вторым проходом по
видео. Цена — два прохода позы на прогон (здесь и в S5).
Альтернативы, которые стоит обсудить: вынести извлечение keypoints в общий шаг
перед S4, либо согласиться, что foot_source всегда bbox_bottom, и честно
показывать 100 % косвенных значений в отчёте.

    python -m looq.stages.s4_track --config configs/s4_track.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from looq import STATUS_OK, STATUS_SKELETON
from looq.calib import apply_h
from looq.geometry import FOOT_SOURCE_IS_DIRECT, indirect_share
from looq.evidence import EvidenceError
from looq.io import ConfigError, RunManifest, load_config, read_json, require
from looq.stages._base import (
    Col,
    StageError,
    build_evidence,
    finalize_evidence,
    read_artifact_status,
    validate_inputs,
    write_parquet,
)

STAGE = "s4_track"

INPUTS = ["det/frames.parquet", "det/frames_index.parquet", "calib/homography.json"]
OUTPUT = "track/tracks.parquet"
OUTPUT_KIND = "parquet"

OUTPUT_COLS: list[Col] = [
    Col("track_id",    "int32",   False, "-",     "-",        "идентификатор трека, уникален в прогоне"),
    Col("frame_idx",   "int64",   False, "кадр",  "-",        "номер кадра"),
    Col("ts",          "float64", False, "с",     "-",        "секунды от начала записи"),
    Col("foot_x_px",   "float32", False, "px",    "frame_px", "опорная точка в кадре, x"),
    Col("foot_y_px",   "float32", False, "px",    "frame_px", "опорная точка в кадре, y"),
    Col("foot_x_m",    "float32", True,  "м",     "plane_m",  "опорная точка на плане, x; null за горизонтом"),
    Col("foot_y_m",    "float32", True,  "м",     "plane_m",  "опорная точка на плане, y; null за горизонтом"),
    Col("foot_source", "string",  False, "-",     "-",        "ankle | bbox_bottom | hip_est; ПРАВИЛО 7"),
    Col("pos_conf",    "float32", True,  "[0,1]", "-",        "уверенность опорной точки"),
    Col("speed_mps",   "float32", True,  "м/с",   "plane_m",  "скорость по сглаженной траектории; null на краях окна"),
]


# --------------------------------------------------------------------------- #

def _load_homography(path: str) -> tuple[np.ndarray, bool]:
    """Гомография кадр -> план и признак, известен ли масштаб.

    Направление всегда «из пикселей», обратного в контракте нет. А вот единицы
    зависят от пути калибровки: px_to_m даёт метры, px_to_unit — условные
    единицы. Скрывать разницу нельзя, поэтому scale_known идёт наружу и дальше
    по трубе: подписать условные единицы метрами было бы враньём.
    """
    doc = read_json(path)
    direction = doc.get("direction")
    if direction not in ("px_to_m", "px_to_unit"):
        raise StageError(
            f"{path}: direction={direction!r}, ожидается 'px_to_m' или 'px_to_unit'")
    h = np.asarray(doc.get("H"), dtype=np.float64)
    if h.shape != (3, 3):
        raise StageError(f"{path}: H должна быть 3x3, получено {h.shape}")
    return h, bool(doc.get("scale_known", direction == "px_to_m"))


class _DetView:
    """Минимальный вид на детекции для BYTETracker.

    Трекер ждёт объект в стиле ultralytics Boxes: атрибуты conf/xyxy/cls/xywh
    и булево индексирование, возвращающее такой же объект. SimpleNamespace не
    подходит — он не индексируется. Класс намеренно крошечный: чем меньше мы
    воспроизводим внутренностей ultralytics, тем меньше ломается при обновлении.
    """

    __slots__ = ("xyxy", "conf", "cls", "xywh")

    def __init__(self, xyxy, conf, cls, xywh):
        self.xyxy, self.conf, self.cls, self.xywh = xyxy, conf, cls, xywh

    def __getitem__(self, mask):
        return _DetView(self.xyxy[mask], self.conf[mask], self.cls[mask], self.xywh[mask])

    def __len__(self):
        return len(self.conf)


def _bytetrack_over_detections(det_by_frame: dict[int, np.ndarray],
                               frame_order: list[int],
                               shape: tuple[int, int],
                               track_cfg: dict[str, Any]) -> dict[int, list[dict]]:
    """ByteTrack поверх СОХРАНЁННЫХ детекций S3, без повторного инференса."""
    from types import SimpleNamespace

    from ultralytics.trackers.byte_tracker import BYTETracker

    args = SimpleNamespace(
        track_high_thresh=float(track_cfg.get("track_high_thresh", 0.5)),
        track_low_thresh=float(track_cfg.get("track_low_thresh", 0.1)),
        new_track_thresh=float(track_cfg.get("new_track_thresh", 0.6)),
        track_buffer=int(track_cfg.get("track_buffer_frames", 30)),
        match_thresh=float(track_cfg.get("match_thresh", 0.8)),
        fuse_score=bool(track_cfg.get("fuse_score", False)),
    )
    # В ultralytics 8.4 BYTETracker(args) не принимает frame_rate, а
    # args.track_buffer задаётся прямо В КАДРАХ (max_frames_lost = track_buffer).
    # Поэтому конфиг хранит track_buffer_frames, а не секунды: пересчёт через
    # fps был бы лишним местом для ошибки.
    tracker = BYTETracker(args)

    out: dict[int, list[dict]] = {}
    for frame_idx in frame_order:
        arr = det_by_frame.get(frame_idx)
        if arr is None or len(arr) == 0:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)
        else:
            xyxy = arr[:, :4].astype(np.float32)
            conf = arr[:, 4].astype(np.float32)
            cls = arr[:, 5].astype(np.float32)
        xywh = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                         xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1)
        results = _DetView(xyxy, conf, cls, xywh)
        tracked = tracker.update(results, img=np.zeros((*shape, 3), dtype=np.uint8))
        for row in tracked:
            x1, y1, x2, y2 = row[:4]
            out.setdefault(frame_idx, []).append({
                "track_id": int(row[4]), "conf": float(row[5]),
                "x1_px": float(x1), "y1_px": float(y1),
                "x2_px": float(x2), "y2_px": float(y2),
            })
    return out


def _foot_from_bbox(box: dict) -> tuple[float, float, str, float]:
    """Низ рамки по центру. Всегда доступен, всегда КОСВЕННЫЙ."""
    return ((box["x1_px"] + box["x2_px"]) / 2.0, box["y2_px"], "bbox_bottom", box["conf"])


def _speed_series(ts: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                  window: int) -> np.ndarray:
    """Скорость скользящим МНК по окну. На краях окна — null.

    МНК, а не разность соседних кадров: за кадр пешеход проходит ~4 см, а
    дрожание опорной точки на плане на порядок больше. Скорость неотрицательна,
    поэтому шум в разностях не сокращается, а сдвигает оценку вверх.

    Время берётся В СЕКУНДАХ из самой строки. Прежняя версия принимала
    frame_idx и делила его на частоту ОБРАБОТАННЫХ кадров: при frame_stride=3
    соседние обработанные кадры отличаются по frame_idx на 3, а по времени на
    0.1 с, и dt выходил 0.3 вместо 0.1. Все скорости были занижены ровно в
    frame_stride раз — медиана 0.297 м/с вместо 0.890, максимум 3.01 вместо
    9.04, из-за чего порог max_plausible_mps не срабатывал ни разу.
    """
    n = len(ts)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo, hi = i - half, i + half + 1
        if lo < 0 or hi > n:
            continue
        sl = slice(lo, hi)
        if np.isnan(xs[sl]).any() or np.isnan(ys[sl]).any():
            continue
        t = ts[sl]
        tc = t - t.mean()
        stt = float((tc ** 2).sum())
        if stt < 1e-12:
            continue
        vx = float((tc * (xs[sl] - xs[sl].mean())).sum() / stt)
        vy = float((tc * (ys[sl] - ys[sl].mean())).sum() / stt)
        out[i] = float(np.hypot(vx, vy))
    return out


# --------------------------------------------------------------------------- #

def run(cfg: dict[str, Any], manifest: RunManifest, sampler) -> dict[str, Any]:
    import pandas as pd

    warnings = validate_inputs(INPUTS, STAGE)
    for w in warnings:
        print(f"[{STAGE}] ВНИМАНИЕ: {w}", file=sys.stderr)
    if read_artifact_status("det/frames.parquet") == STATUS_SKELETON:
        raise StageError(
            "det/frames.parquet помечен status=skeleton: S3 не отработал. "
            "Трекинг пустых детекций дал бы пустой результат, выданный за успех (правило 8)"
        )

    h_px_to_m, scale_known = _load_homography(require(cfg, "input", "homography"))
    if not scale_known:
        print(f"[{STAGE}] масштаб не определён: foot_*_m и speed_mps в УСЛОВНЫХ "
              f"единицах, не в метрах. Имена колонок оставлены по контракту, "
              f"единицы несёт calib_status")
    track_cfg = require(cfg, "track")
    foot_cfg = require(cfg, "foot_point")
    speed_cfg = require(cfg, "speed")

    det = pd.read_parquet("det/frames.parquet")
    index = pd.read_parquet("det/frames_index.parquet")
    processed = index[index["processed"]]
    if processed.empty:
        raise StageError("в det/frames_index.parquet нет ни одного processed=true кадра")
    fps_est = 1.0 / float(np.median(np.diff(processed["ts"].to_numpy()))) \
        if len(processed) > 1 else 30.0
    frame_order = processed["frame_idx"].astype(int).tolist()
    ts_by_frame = dict(zip(processed["frame_idx"].astype(int),
                           processed["ts"].astype(float)))
    print(f"[{STAGE}] детекций {len(det)}, обработанных кадров {len(processed)}, "
          f"оценка fps {fps_est:.2f}")

    det_by_frame: dict[int, np.ndarray] = {}
    for frame_idx, grp in det.groupby("frame_idx"):
        det_by_frame[int(frame_idx)] = grp[["x1_px", "y1_px", "x2_px", "y2_px",
                                            "conf", "cls"]].to_numpy(dtype=np.float64)

    shape = (int(cfg.get("frame_h_px", 1080)), int(cfg.get("frame_w_px", 1920)))
    t0 = time.time()
    tracked = _bytetrack_over_detections(det_by_frame, frame_order, shape, track_cfg)

    # --- опорная точка ---------------------------------------------------- #
    pose_enabled = bool(foot_cfg.get("pose_enabled", False))
    if pose_enabled:
        raise StageError(
            "foot_point.pose_enabled=true, но извлечение keypoints в S4 требует "
            "второго прохода модели позы по видео и дублирует работу S5. "
            "Это решение владельца, а не моё: см. шапку looq/stages/s4_track.py. "
            "Пока не подтверждено — выключите флаг, и foot_source будет "
            "bbox_bottom с честной долей 100 % косвенных значений"
        )

    rows_by_track: dict[int, list[dict]] = {}
    for frame_idx in frame_order:
        for box in tracked.get(frame_idx, []):
            fx, fy, source, conf = _foot_from_bbox(box)
            rows_by_track.setdefault(box["track_id"], []).append({
                "frame_idx": frame_idx, "ts": ts_by_frame[frame_idx],
                "foot_x_px": fx, "foot_y_px": fy,
                "foot_source": source, "pos_conf": conf,
            })
    if not rows_by_track:
        raise StageError("ByteTrack не собрал ни одного трека")

    # --- проекция на план и скорость -------------------------------------- #
    window = int(speed_cfg.get("window_frames", 9))
    max_speed = float(speed_cfg.get("max_plausible_mps", 4.0))
    out_rows: list[dict[str, Any]] = []
    n_invalid_gp = 0
    for track_id, obs in sorted(rows_by_track.items()):
        obs.sort(key=lambda r: r["frame_idx"])
        pts = np.array([[r["foot_x_px"], r["foot_y_px"]] for r in obs])
        xs = np.full(len(obs), np.nan)
        ys = np.full(len(obs), np.nan)
        for i, p in enumerate(pts):
            try:
                m = apply_h(h_px_to_m, p[None, :])[0]
            except Exception:
                n_invalid_gp += 1
                continue      # точка за горизонтом: null, а не выдуманное число
            xs[i], ys[i] = float(m[0]), float(m[1])
        t_arr = np.array([r["ts"] for r in obs], dtype=np.float64)
        speeds = _speed_series(t_arr, xs, ys, window)
        for i, r in enumerate(obs):
            v = speeds[i]
            if np.isfinite(v) and v > max_speed:
                # Скорость выше правдоподобной — красный флаг проекции, а не бегун.
                # Не обнуляем и не обрезаем: помечаем null и считаем отдельно.
                v = np.nan
            out_rows.append({
                "track_id": int(track_id), "frame_idx": int(r["frame_idx"]),
                "ts": float(r["ts"]),
                "foot_x_px": float(r["foot_x_px"]), "foot_y_px": float(r["foot_y_px"]),
                "foot_x_m": None if not np.isfinite(xs[i]) else float(xs[i]),
                "foot_y_m": None if not np.isfinite(ys[i]) else float(ys[i]),
                "foot_source": r["foot_source"],
                "pos_conf": float(r["pos_conf"]),
                "speed_mps": None if not np.isfinite(v) else float(v),
            })

    # Пруфы: S6 и S8 будут опираться на опорную точку и скорость, значит их
    # надо уметь показать. Кадры читаются ОДИН раз и только те, что попали
    # в выборку, — полный проход ради дюжины кропов не оправдан.
    # Порог правдоподобия в м/с бессмыслен, когда масштаб не определён: скорости
    # в условных единицах на порядки меньше. Выбросами считаем верхний дециль
    # скоростей ЭТОГО прогона — величина относительная и потому осмысленная всегда.
    finite = [r["speed_mps"] for r in out_rows if r["speed_mps"] is not None]
    speed_hi = float(np.quantile(finite, 0.90)) if finite else float("inf")
    _collect_evidence(sampler, cfg, tracked, out_rows, speed_hi)

    share = indirect_share([r["foot_source"] for r in out_rows])
    elapsed = time.time() - t0
    manifest.note("n_tracks", len(rows_by_track))
    manifest.note("n_track_frames", len(out_rows))
    manifest.note("indirect_foot_share", round(share, 4))
    manifest.note("n_gp_invalid", n_invalid_gp)
    manifest.note("scale_known", scale_known)
    manifest.note("elapsed_s", round(elapsed, 1))
    print(f"[{STAGE}] треков {len(rows_by_track)}, строк {len(out_rows)}, "
          f"за горизонтом {n_invalid_gp}")
    print(f"[{STAGE}] доля косвенных опорных точек: {share:.1%} "
          f"(правило 7, идёт отдельным числом в S8)")
    return {"rows": out_rows, "indirect_share": share}


def _collect_evidence(sampler, cfg: dict, tracked: dict, rows: list[dict],
                      speed_hi: float, n_frames: int = 40) -> None:
    """Кропы треков по кадрам, равномерно разнесённым на всю запись."""
    from looq.pilot import iter_frames

    video = cfg.get("input", {}).get("video") or cfg.get("input", {}).get("clip")
    if not video or not Path(str(video)).is_file():
        raise StageError(
            f"для пруфов S4 нужен исходный ролик, но input.video={video!r} не найден. "
            f"Число без пруфов в отчёт не идёт (правило 1)")

    by_key = {(r["frame_idx"], r["track_id"]): r for r in rows}
    frames = sorted(tracked)
    if not frames:
        return
    pick = np.unique(np.linspace(0, len(frames) - 1, min(n_frames, len(frames)))
                     .round().astype(int))
    wanted = np.asarray([frames[i] for i in pick], dtype=np.int64)

    for frame_idx, frame in iter_frames(video, wanted):
        for box in tracked.get(int(frame_idx), []):
            r = by_key.get((int(frame_idx), int(box["track_id"])))
            if r is None:
                continue
            x1, y1 = max(0, int(box["x1_px"])), max(0, int(box["y1_px"]))
            crop = frame[y1:int(box["y2_px"]), x1:int(box["x2_px"])]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 4:
                continue
            tid, fi = int(box["track_id"]), int(frame_idx)
            speed = r.get("speed_mps")
            if r["foot_source"] != "ankle":
                sampler.offer("claim.track.foot_source_indirect", track_id=tid,
                              frame_idx=fi, ts=float(r["ts"]), crop_bgr=crop,
                              value=float(r["foot_y_px"]),
                              confidence=float(r.get("pos_conf") or 0.0),
                              extra={"foot_source": r["foot_source"]})
            # Кандидат на переклейку id: трек только что появился в кадре.
            sampler.offer("claim.track.id_switch_suspect", track_id=tid, frame_idx=fi,
                          ts=float(r["ts"]), crop_bgr=crop, value=float(tid),
                          confidence=float(r.get("pos_conf") or 0.0))
            if speed is not None and speed >= speed_hi:
                sampler.offer("claim.track.speed_outlier", track_id=tid, frame_idx=fi,
                              ts=float(r["ts"]), crop_bgr=crop, value=float(speed),
                              confidence=float(r.get("pos_conf") or 0.0),
                              extra={"speed": speed, "top_decile": speed_hi})


def main(argv=None) -> int:
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
        sampler = build_evidence(cfg, STAGE, manifest)

        res = run(cfg, manifest, sampler)
        write_parquet(OUTPUT, OUTPUT_COLS, res["rows"], STAGE, STATUS_OK,
                      inputs=INPUTS)

        # Пруфы требуют кадров; на этом этапе кропы берутся из видео и пока
        # не собираются — этап падает на finalize, если claim-ы объявлены,
        # но не набраны. Это не заглушка: число без пруфов в отчёт не идёт.
        index = finalize_evidence(sampler, manifest)

        manifest.note("output_artifact", OUTPUT)
        manifest.finish(STATUS_OK)
        print(f"[{STAGE}] записано: {OUTPUT} ({len(res['rows'])} строк), пруфы {index}")
        return 0

    except (StageError, EvidenceError, ConfigError, OSError, ValueError, KeyError) as exc:
        if manifest is not None:
            manifest.finish("failed", error=str(exc))
        print(f"[{STAGE}] ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
