"""Обезличивание людей на картинках README — детектором, а не на глаз.

ЗАЧЕМ. README утверждает, что в репозитории нет изображений лиц. Три
фигуры — опорный кадр с горизонтом, кадр оверлея и кадр с зонами — это
уличные сцены с прохожими, и утверждение было верно «по духу», но не
буквально: кадры перепубликуют уже публичный поток, однако лица на них
различимы.

КАК. Детектор находит людей, и к верхней части каждой рамки применяется та
же функция, что и к пруф-кропам, — `looq.evidence.blur_face_region`:
пикселизация, затем гаусс. Параметры берутся из `configs/evidence.yaml`,
одни и те же для пруфов и для фигур. Никакого «посмотрел и замазал».

Скрипт идемпотентен: повторный прогон размоет уже размытое, результат от
этого не станет хуже, но и пересобирать фигуры каждый раз не нужно —
`make readme-figures` вызывает его сам.

    python scripts/anonymise_figures.py docs/img/overlay_frame.webp
    python scripts/anonymise_figures.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.evidence import EvidenceError, blur_face_region  # noqa: E402
from looq.io import load_config  # noqa: E402
from looq.pilot import infer_params  # noqa: E402

#: Фигуры-фотографии. Графики (гистограммы, скаттеры, план) людей не содержат,
#: и гонять по ним детектор незачем.
PHOTO_FIGURES = [
    "docs/img/calib_vanishing.webp",
    "docs/img/overlay_frame.webp",
    "docs/img/zones_reference.webp",
]

EVIDENCE_CONFIG = "configs/evidence.yaml"
DETECT_CONFIG = "configs/s3_detect.yaml"


def _detector(cfg: dict):
    from ultralytics import YOLO

    weights = ((cfg.get("model") or {}).get("weights"))
    if not weights or not Path(weights).is_file():
        raise SystemExit(
            f"нет весов детектора {weights!r} из {DETECT_CONFIG}. Обезличивание "
            f"фигур требует детектора: замазывать вручную нельзя, это была бы "
            f"работа на глаз")
    return YOLO(weights)


#: Ниже этой высоты рамки полоса головы вырождается: при top_frac 0.3 у рамки
#: в 20 px это 6 px, меньше min_kernel_px из конфига. Такие рамки считаются
#: отдельно и печатаются — молча отбрасывать их нельзя (правило 7).
MIN_BOX_H_PX = 20


def anonymise(path: Path, model, params: dict, priv: dict) -> tuple[int, int, int]:
    """Размывает голову каждому найденному человеку.

    Возвращает (размыто, пропущено по размеру, пропущено как неизменяемые).
    Вторые два числа печатаются: это не «ничего не нашлось», это отброшенные
    кандидаты, и читатель должен видеть, сколько их.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"не прочитан {path}")

    res = model.predict(img, imgsz=params.get("imgsz", 1280),
                        conf=float(params["_conf"]),
                        device=params.get("device", "cpu"),
                        half=bool(params.get("half", False)),
                        classes=[0], verbose=False)[0]

    n = tiny = flat = 0
    for box in np.asarray(res.boxes.xyxy.cpu(), dtype=np.float64):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 - x1 < 4 or y2 - y1 < MIN_BOX_H_PX:
            tiny += 1
            continue
        crop = img[y1:y2, x1:x2].copy()
        try:
            blurred, _ = blur_face_region(
                crop,
                top_frac=float(priv["face_blur_top_frac"]),
                kernel_frac=float(priv["blur_kernel_frac"]),
                sigma_frac=float(priv["blur_sigma_frac"]),
                pixelate_factor=int(priv["pixelate_factor"]))
        except EvidenceError:
            # Постусловие blur_face_region: область не изменилась. Значит она
            # уже однородна — плоская стена, асфальт, ранее размытый участок.
            # Лица там нет, и это не повод падать, но и не повод молчать.
            flat += 1
            continue
        img[y1:y2, x1:x2] = blurred
        n += 1

    # webp пишется тем же качеством, что и в make_readme_figures
    cv2.imwrite(str(path), img, [cv2.IMWRITE_WEBP_QUALITY, 80])
    return n, tiny, flat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="все фотографические фигуры")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="порог детектора. НАМЕРЕННО НИЖЕ боевого: цена лишнего "
                         "размытия — размытый столб, цена пропуска — опубликованное "
                         "лицо. Для обезличивания важна полнота, не точность")
    args = ap.parse_args(argv)

    targets = [Path(q) for q in PHOTO_FIGURES] if args.all else list(args.figures)
    if not targets:
        raise SystemExit("нечего обезличивать: укажите файлы или --all")

    priv = (load_config(EVIDENCE_CONFIG).get("privacy") or {})
    need = ("face_blur_top_frac", "blur_kernel_frac", "blur_sigma_frac",
            "pixelate_factor")
    missing = [k for k in need if k not in priv]
    if missing:
        raise SystemExit(f"в {EVIDENCE_CONFIG} нет ключей приватности {missing}")

    dcfg = load_config(DETECT_CONFIG)
    model = _detector(dcfg)
    params = infer_params(dcfg)
    params["_conf"] = float(args.conf)

    print(f"обезличиваю: top_frac={priv['face_blur_top_frac']}, "
          f"порог детектора {args.conf} (ниже боевого намеренно)")
    for t in targets:
        if not t.is_file():
            print(f"  пропуск {t}: нет файла")
            continue
        n, tiny, flat = anonymise(t, model, params, priv)
        print(f"  {t}: размыто {n}, отброшено мелких (<{MIN_BOX_H_PX} px) {tiny}, "
              f"однородных {flat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
