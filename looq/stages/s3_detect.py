"""S3 detect — детекция людей.

Этап чисто ПИКСЕЛЬНЫЙ: метровых колонок здесь нет и calib/ не читается.
Проекция на план — работа S4. Иначе перекалибровка гомографии заставляла бы
перезапускать самый дорогой этап (правило 4).

Два артефакта:
  det/frames.parquet        одна строка = одна детекция
  det/frames_index.parquet  одна строка = один кадр записи, обработан он или нет

Второй обязателен: без него неразличимы "кадр обработан, людей нет"
(processed=true, n_detections=0) и "кадр не обработан" (processed=false).
Различие ломает знаменатель во всех процентах S8 — так и получаются нули в зонах.

    python -m looq.stages.s3_detect --config configs/s3_detect.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from looq import STATUS_OK
from looq.evidence import EvidenceError
from looq.io import ConfigError, RunManifest, load_config, require
from looq.pilot import PilotError, infer_params
from looq.stages._base import Col, StageError, build_evidence, finalize_evidence, write_parquet

STAGE = "s3_detect"

OUTPUT = "det/frames.parquet"
OUTPUT_KIND = "parquet"

OUTPUT_COLS: list[Col] = [
    Col("frame_idx", "int64",   False, "кадр",  "-",        "номер кадра от начала записи, 0-based"),
    Col("ts",        "float64", False, "с",     "-",        "секунды от начала записи; ts = frame_idx / fps"),
    Col("det_id",    "int32",   False, "-",     "-",        "номер детекции внутри кадра, 0-based"),
    Col("x1_px",     "float32", False, "px",    "frame_px", "левая граница рамки"),
    Col("y1_px",     "float32", False, "px",    "frame_px", "верхняя граница рамки"),
    Col("x2_px",     "float32", False, "px",    "frame_px", "правая граница рамки"),
    Col("y2_px",     "float32", False, "px",    "frame_px", "нижняя граница рамки"),
    Col("conf",      "float32", False, "[0,1]", "-",        "уверенность детектора"),
    Col("cls",       "int16",   False, "-",     "-",        "класс COCO; ожидается 0 (person)"),
]

FRAMES_INDEX = "det/frames_index.parquet"
FRAMES_INDEX_COLS: list[Col] = [
    Col("frame_idx",    "int64",   False, "кадр", "-", "номер кадра, 0-based; непрерывный, без дыр"),
    Col("ts",           "float64", False, "с",    "-", "секунды от начала записи"),
    Col("processed",    "bool",    False, "-",    "-", "детектор реально отработал по этому кадру"),
    Col("n_detections", "int32",   True,  "шт",   "-", "детекций записано; 0 валидно; null при processed=false"),
    Col("skip_reason",  "string",  True,  "-",    "-", "decode_error | out_of_roi_window | sampled_out; заполнена ровно при processed=false"),
]

SKIP_SAMPLED_OUT = "sampled_out"
SKIP_DECODE_ERROR = "decode_error"
SKIP_OUT_OF_WINDOW = "out_of_roi_window"


def _video_and_geometry(cfg: dict[str, Any]) -> tuple[Path, float, int]:
    """Путь к видео, fps и число кадров.

    Пока S0 не реализован, источник берётся из конфига. Когда raw/manifest.json
    станет настоящим, список сегментов должен приходить оттуда — здесь стоит
    явная проверка статуса, чтобы подмена не прошла молча.
    """
    import cv2
    from looq.stages._base import read_artifact_status
    from looq import STATUS_SKELETON

    manifest_path = cfg.get("input", {}).get("manifest")
    if manifest_path and read_artifact_status(manifest_path) not in (None, STATUS_SKELETON):
        raise StageError(
            f"{manifest_path} содержит настоящие данные S0, но чтение списка сегментов "
            f"из манифеста ещё не реализовано. Не хочу молча взять один файл из конфига "
            f"вместо часа записи — реализуйте выбор источника явно"
        )

    video = Path(require(cfg, "input", "video"))
    if not video.is_file():
        raise StageError(f"нет видео {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise StageError(f"cv2 не открыл {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not (1.0 < fps < 240.0) or n_frames < 2:
        raise StageError(f"неправдоподобные параметры {video}: fps={fps}, кадров={n_frames}")
    return video, fps, n_frames


def run(cfg: dict[str, Any], manifest: RunManifest, sampler) -> dict[str, Any]:
    import cv2
    from ultralytics import YOLO

    video, fps, n_frames = _video_and_geometry(cfg)
    params = infer_params(cfg)
    detect_cfg = require(cfg, "detect")
    stride = int(detect_cfg.get("frame_stride", 1))
    if stride < 1:
        raise ConfigError(f"detect.frame_stride должен быть >= 1, получено {stride}")
    max_det = int(detect_cfg.get("max_det", 300))
    # Отсечка по глубине. Применяется ЗДЕСЬ, до записи детекций, чтобы дальние
    # ненадёжные детекции вообще не порождали треков в S4.
    min_box_h = float(detect_cfg.get("min_box_height_px", 0.0))
    if min_box_h > 0:
        print(f"[{STAGE}] отсечка по глубине: рамки ниже {min_box_h:.0f} px "
              f"не записываются")
    max_frames = detect_cfg.get("max_frames")
    if max_frames is not None:
        # Обрезанные кадры в frames_index НЕ попадают: цикл до них не доходит.
        # Прежний комментарий обещал обратное и был неверен — константа
        # SKIP_OUT_OF_WINDOW не присваивалась никогда, и в frames_index таких
        # строк ноль. Знаменатель S8 берёт n_frames_total из scope, поэтому
        # обрезка видна там, а не здесь.
        n_frames = min(n_frames, int(max_frames))
        print(f"[{STAGE}] ОБРЕЗКА до {n_frames} кадров (detect.max_frames)")

    model = YOLO(params["weights"])
    print(f"[{STAGE}] {video}: {n_frames} кадров, {fps:.2f} fps, шаг {stride}")

    det_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    n_dropped_small = 0
    cap = cv2.VideoCapture(str(video))
    t0 = time.time()
    frame_idx = 0
    n_processed = 0
    try:
        while frame_idx < n_frames:
            ok, frame = cap.read()
            ts = frame_idx / fps
            if not ok:
                # Кадр не декодировался. Это НЕ "людей нет": знаменатель S8
                # должен уметь их различать.
                index_rows.append({"frame_idx": frame_idx, "ts": ts, "processed": False,
                                   "n_detections": None, "skip_reason": SKIP_DECODE_ERROR})
                frame_idx += 1
                continue
            if frame_idx % stride != 0:
                index_rows.append({"frame_idx": frame_idx, "ts": ts, "processed": False,
                                   "n_detections": None, "skip_reason": SKIP_SAMPLED_OUT})
                frame_idx += 1
                continue

            res = model.predict(frame, imgsz=params["imgsz"], classes=params["classes"],
                                conf=params["conf"], iou=params["iou"], max_det=max_det,
                                device=params["device"], half=params["half"], verbose=False)
            boxes = res[0].boxes
            n_det = 0
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                if min_box_h > 0:
                    keep = (xyxy[:, 3] - xyxy[:, 1]) >= min_box_h
                    n_dropped_small += int((~keep).sum())
                    xyxy, confs, clss = xyxy[keep], confs[keep], clss[keep]
                order = np.argsort(-confs)     # det_id детерминирован: по убыванию conf
                for det_id, k in enumerate(order):
                    x1, y1, x2, y2 = xyxy[k]
                    det_rows.append({
                        "frame_idx": frame_idx, "ts": ts, "det_id": det_id,
                        "x1_px": float(x1), "y1_px": float(y1),
                        "x2_px": float(x2), "y2_px": float(y2),
                        "conf": float(confs[k]), "cls": int(clss[k]),
                    })
                n_det = len(order)
                _offer_evidence(sampler, frame, frame_idx, ts, xyxy, confs, order,
                                frame.shape[0], float(params["conf"]))
            index_rows.append({"frame_idx": frame_idx, "ts": ts, "processed": True,
                               "n_detections": int(n_det), "skip_reason": None})
            n_processed += 1
            if n_processed % 200 == 0:
                el = time.time() - t0
                print(f"[{STAGE}]   кадр {frame_idx}/{n_frames}, обработано {n_processed}, "
                      f"{el:.0f} с, {n_processed / el:.1f} кадр/с")
            frame_idx += 1
    finally:
        cap.release()

    elapsed = time.time() - t0
    if not det_rows:
        raise StageError("детектор не нашёл ни одного человека за весь прогон "
                         "(правило 8: пустой результат не выдаётся за успех)")

    # Правило 4: throughput меряется и записывается на каждом прогоне.
    manifest.note("min_box_height_px", min_box_h)
    manifest.note("n_detections_dropped_small", n_dropped_small)
    manifest.note("elapsed_s", round(elapsed, 1))
    manifest.note("frames_processed", n_processed)
    manifest.note("throughput_fps", round(n_processed / elapsed, 2) if elapsed else None)
    manifest.note("n_detections", len(det_rows))
    manifest.note("processed_frac", round(n_processed / max(1, len(index_rows)), 4))
    print(f"[{STAGE}] детекций {len(det_rows)}, обработано кадров {n_processed} из "
          f"{len(index_rows)}, {elapsed:.0f} с, {n_processed / max(elapsed, 1e-9):.1f} кадр/с")
    return {"det_rows": det_rows, "index_rows": index_rows}


def _offer_evidence(sampler, frame, frame_idx, ts, xyxy, confs, order,
                    frame_h: int, conf_thr: float) -> None:
    """Пруфы отдельно по ближней и дальней половине кадра.

    Дальняя половина — там, где детектор врёт; средний AP её маскирует,
    поэтому и гейт, и пруфы делят кадр пополам.
    """
    mid_y = frame_h / 2.0
    for k in order:
        x1, y1, x2, y2 = xyxy[k]
        crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 4:
            continue
        conf = float(confs[k])
        half = "near" if y2 >= mid_y else "far"
        sampler.offer(f"claim.detect.{half}_half", track_id=0, frame_idx=int(frame_idx),
                      ts=float(ts), crop_bgr=crop, value=conf, confidence=conf,
                      extra={"bbox_h_px": round(float(y2 - y1), 1), "half": half})
        # Пограничные по уверенности: именно они определяют выбор порога conf.
        if conf < conf_thr * 1.5:
            sampler.offer("claim.detect.borderline", track_id=0, frame_idx=int(frame_idx),
                          ts=float(ts), crop_bgr=crop, value=conf, confidence=conf,
                          extra={"conf_thr": conf_thr})


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

        write_parquet(OUTPUT, OUTPUT_COLS, res["det_rows"], STAGE, STATUS_OK)
        write_parquet(FRAMES_INDEX, FRAMES_INDEX_COLS, res["index_rows"], STAGE, STATUS_OK)
        index = finalize_evidence(sampler, manifest)

        manifest.note("output_artifacts", [OUTPUT, FRAMES_INDEX])
        manifest.finish(STATUS_OK)
        print(f"[{STAGE}] записано: {OUTPUT} ({len(res['det_rows'])} строк), "
              f"{FRAMES_INDEX} ({len(res['index_rows'])} строк), пруфы {index}")
        return 0

    except (StageError, EvidenceError, PilotError, ConfigError, OSError, ValueError) as exc:
        if manifest is not None:
            manifest.finish("failed", error=str(exc))
        print(f"[{STAGE}] ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
