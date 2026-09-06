"""Сбор кадров-доказательств (пруфов).

Зачем модуль существует
-----------------------
Правило 1 CLAUDE.md: любое число в отчёте посчитано кодом из артефакта на диске.
Пруфы не порождают числа — они позволяют человеку проверить УЖЕ ПОСЧИТАННОЕ число
глазами. Из отчёта по claim_id можно перейти к кадрам, на которых это число стоит.

Приватность (правило 9)
-----------------------
Кропы лиц на диск не сохраняются. Технически это обеспечено структурно, а не
дисциплиной:

  * единственная функция модуля, пишущая изображение на диск, — ``_write_jpeg()``;
  * она вызывается ровно из одного места — ``EvidenceWriter.add()``, сразу после
    ``blur_face_region()``, и получает на вход только её результат;
  * других обращений к cv2.imwrite / open(..., "wb") для изображений в модуле нет;
  * обезличивание — пикселизация И гаусс, а не только гаусс: свёртка обратима,
    пикселизация уничтожает информацию мельче блока безвозвратно;
  * если обезличивание не изменило область лица, запись отменяется с ошибкой.

Отбор кадров (принципиально)
----------------------------
``EvidenceSampler`` отбирает кадры СТРАТИФИЦИРОВАННО по уверенности, а не top-N.
Top-N систематически показывает лучшие случаи и создаёт ложное впечатление
качества. Стратификация показывает и уверенные, и пограничные, и слабые примеры —
то есть то, что нужно для проверки числа, а не для его защиты.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from looq import SCHEMA_VERSION
from looq.io import atomic_write_bytes

# --------------------------------------------------------------------------- #
# Дефолты. Рабочие значения берутся из configs/evidence.yaml, где у каждого
# есть комментарий "почему такое значение". Здесь — только фолбэк для тестов.
# --------------------------------------------------------------------------- #

#: Доля высоты кропа сверху, которая обезличивается.
#: Поднята с 0.22 до 0.30 (решение владельца 2026-09-03): на дальнем плане человек
#: занимает меньше пикселей, голова оказывается выше в кропе, и 22 % её не накрывали.
DEFAULT_FACE_BLUR_TOP_FRAC = 0.30

#: Во сколько раз область головы уменьшается перед обратным увеличением.
#: Пикселизация уничтожает информацию ниже размера блока безвозвратно, тогда как
#: чистая гауссиана — обратимая свёртка, и деблюр по ней восстанавливает черты лица.
DEFAULT_PIXELATE_FACTOR = 16

#: Ядро гауссова размытия как доля меньшей стороны области головы.
#: Гаусс идёт ПОСЛЕ пикселизации: он убирает резкие границы блоков, по которым
#: иначе можно оценить исходные значения.
DEFAULT_BLUR_KERNEL_FRAC = 0.35

#: sigma как доля от размера ядра.
DEFAULT_BLUR_SIGMA_FRAC = 1.0 / 3.0

#: Нижняя граница ядра в пикселях: на маленьких кропах доля даёт 1-3 px,
#: чего для обезличивания недостаточно.
MIN_BLUR_KERNEL_PX = 5

DEFAULT_JPEG_QUALITY = 90

#: Сколько кадров оставить на claim_id.
DEFAULT_N_PER_CLAIM = 12

#: Число страт по уверенности: верхняя / средняя / нижняя.
DEFAULT_N_STRATA = 3

#: Потолок кандидатов на claim_id в памяти. Кроп 128x256x3 ~ 96 КБ,
#: 2000 кандидатов ~ 190 МБ на claim. Сверх потолка работает reservoir sampling.
DEFAULT_MAX_CANDIDATES_PER_CLAIM = 2000

DEFAULT_SEED = 20260903

#: Порядок колонок evidence/index.parquet. Первые девять зафиксированы заданием,
#: последние три добавлены по правилам 7 и 8: без них extra теряется молча,
#: а стратификацию невозможно проверить постфактум.
INDEX_COLUMNS: tuple[str, ...] = (
    "claim_id",
    "track_id",
    "frame_idx",
    "ts",
    "value",
    "confidence",
    "path",
    "stage",
    "model_name",
    "extra_json",
    "stratum",
    "schema_version",
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class EvidenceError(RuntimeError):
    """Пруфы собраны некорректно. Всегда громко (правило 8)."""


# --------------------------------------------------------------------------- #
# Обезличивание
# --------------------------------------------------------------------------- #

def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def blur_face_region(
    crop_bgr: np.ndarray,
    top_frac: float = DEFAULT_FACE_BLUR_TOP_FRAC,
    kernel_frac: float = DEFAULT_BLUR_KERNEL_FRAC,
    sigma_frac: float = DEFAULT_BLUR_SIGMA_FRAC,
    pixelate_factor: int = DEFAULT_PIXELATE_FACTOR,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Обезличивает верхние top_frac высоты кропа: пикселизация, затем гаусс.

    Порядок важен и обоснован:

      1. downsample в pixelate_factor раз (INTER_AREA — усреднение блока)
         и обратный upsample (INTER_NEAREST — честные блоки без интерполяции).
         Информация мельче блока уничтожается безвозвратно.
      2. гауссово размытие поверх, ядро пропорционально размеру кропа.
         Убирает резкие границы блоков, по которым иначе можно оценить исходные
         значения на краях.

    Чистая гауссиана этого не даёт: свёртка обратима, и по ней восстанавливают
    черты лица. Отсюда пикселизация первым шагом.

    Возвращает НОВЫЙ массив; входной не мутируется. Второй элемент кортежа —
    параметры, которые реально применились: они уходят в extra_json, чтобы
    обезличивание можно было проверить, а не принять на веру.
    """
    if not isinstance(crop_bgr, np.ndarray):
        raise EvidenceError(f"кроп должен быть np.ndarray, получено {type(crop_bgr).__name__}")
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        raise EvidenceError(f"кроп должен быть HxWx3 BGR, получено {crop_bgr.shape}")
    h, w = int(crop_bgr.shape[0]), int(crop_bgr.shape[1])
    if h < 1 or w < 1:
        raise EvidenceError(f"пустой кроп {crop_bgr.shape}")
    if not (0.0 < top_frac <= 1.0):
        raise EvidenceError(f"top_frac должен быть в (0, 1], получено {top_frac}")
    if pixelate_factor < 2:
        raise EvidenceError(
            f"pixelate_factor должен быть >= 2, получено {pixelate_factor}: "
            f"без пикселизации остаётся только обратимая свёртка (правило 9)"
        )

    head_h = max(1, int(round(top_frac * h)))
    k = _odd(max(MIN_BLUR_KERNEL_PX, int(round(kernel_frac * min(w, head_h)))))
    sigma = float(sigma_frac * k)
    px_w = max(1, w // pixelate_factor)
    px_h = max(1, head_h // pixelate_factor)

    out = np.ascontiguousarray(crop_bgr.copy())
    head_before = out[:head_h, :, :].copy()

    # 1. пикселизация: вниз с усреднением, вверх без интерполяции
    small = cv2.resize(head_before, (px_w, px_h), interpolation=cv2.INTER_AREA)
    head_after = cv2.resize(small, (w, head_h), interpolation=cv2.INTER_NEAREST)
    # 2. гаусс поверх блоков
    head_after = cv2.GaussianBlur(head_after, (k, k), sigma,
                                  borderType=cv2.BORDER_REPLICATE)
    out[:head_h, :, :] = head_after

    # Постусловие приватности: если в области лица была хоть какая-то структура,
    # обезличивание обязано её изменить. Иначе на диск не пишем.
    if float(np.var(head_before)) > 0.0 and np.array_equal(head_before, head_after):
        raise EvidenceError(
            "обезличивание не изменило область лица — запись на диск отменена (правило 9)"
        )

    return out, {
        "blur_top_frac": float(top_frac),
        "blur_head_h_px": int(head_h),
        "blur_pixelate_factor": int(pixelate_factor),
        "blur_pixel_block_px": [int(px_w), int(px_h)],
        "blur_kernel_px": int(k),
        "blur_sigma_px": round(sigma, 3),
    }


def _write_jpeg(path: Path, img_bgr: np.ndarray, quality: int) -> int:
    """ЕДИНСТВЕННАЯ точка записи изображения на диск во всём модуле.

    Вызывается ровно из одного места — EvidenceWriter.add(), сразу после
    blur_face_region(). Не вызывать откуда-либо ещё: это обходит обезличивание.

    Кроп приходит уже размытым по ВЕРХНЕЙ ДОЛЕ, и этого мало. Замер
    2026-09-06 по всем 668 кропам на диске: на 86 из них лицевые кейпоинты
    остались в резких областях — голова не попала в долю, потому что человек
    наклонён, перекрыт или срезан краем кропа. Поэтому запись идёт через
    ``looq.anonymise``, который ищет голову детектором и позой, а долю
    оставляет запасным путём. Импорт ленивый: anonymise берёт отсюда
    blur_face_region, и на уровне модуля вышел бы цикл.
    """
    from looq.anonymise import encode_image           # см. докстроку: цикл
    data = encode_image(img_bgr, ".jpg", quality=int(quality))
    atomic_write_bytes(path, data)
    return len(data)


# --------------------------------------------------------------------------- #
# EvidenceWriter
# --------------------------------------------------------------------------- #

class EvidenceWriter:
    """Пишет обезличенные кропы и индекс evidence/index.parquet."""

    def __init__(
        self,
        root: str | Path = "evidence",
        *,
        stage: str,
        model_name: str,
        face_blur_top_frac: float = DEFAULT_FACE_BLUR_TOP_FRAC,
        blur_kernel_frac: float = DEFAULT_BLUR_KERNEL_FRAC,
        blur_sigma_frac: float = DEFAULT_BLUR_SIGMA_FRAC,
        pixelate_factor: int = DEFAULT_PIXELATE_FACTOR,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ) -> None:
        if not stage:
            raise EvidenceError("EvidenceWriter требует stage: пруф без этапа непроверяем")
        if not model_name:
            raise EvidenceError(
                "EvidenceWriter требует model_name. Если модели на этапе нет, "
                'передайте явное "none" — молчаливого пропуска быть не должно (правило 6)'
            )
        self.root = Path(root)
        self.stage = str(stage)
        self.model_name = str(model_name)
        self.face_blur_top_frac = float(face_blur_top_frac)
        self.blur_kernel_frac = float(blur_kernel_frac)
        self.blur_sigma_frac = float(blur_sigma_frac)
        self.pixelate_factor = int(pixelate_factor)
        self.jpeg_quality = int(jpeg_quality)
        self._rows: list[dict[str, Any]] = []
        self._seen: set[tuple[str, int, int]] = set()

    # -- запись ------------------------------------------------------------- #

    def add(
        self,
        claim_id: str,
        track_id: int,
        frame_idx: int,
        ts: float,
        crop_bgr: np.ndarray,
        value: float,
        confidence: float,
        extra: Mapping[str, Any] | None = None,
        *,
        stratum: int | None = None,
    ) -> str:
        """Обезличивает кроп, кладёт его на диск, добавляет строку в индекс.

        Возвращает путь к файлу относительно корня проекта.
        """
        claim_id = self._check_id(claim_id, "claim_id")
        track_id = int(track_id)
        frame_idx = int(frame_idx)
        ts = float(ts)
        value = float(value)
        confidence = float(confidence)
        if not math.isfinite(confidence):
            raise EvidenceError(f"confidence должна быть конечным числом, получено {confidence}")
        if not math.isfinite(ts) or ts < 0:
            raise EvidenceError(f"ts должно быть неотрицательным конечным числом, получено {ts}")

        key = (claim_id, track_id, frame_idx)
        if key in self._seen:
            # Молча перезаписать файл значило бы потерять пруф. Правило 8.
            raise EvidenceError(
                f"повторный пруф для claim_id={claim_id} track_id={track_id} "
                f"frame_idx={frame_idx}: ключ индекса должен быть уникален"
            )

        # --- обезличивание строго перед записью -------------------------- #
        blurred, blur_meta = blur_face_region(
            crop_bgr,
            top_frac=self.face_blur_top_frac,
            kernel_frac=self.blur_kernel_frac,
            sigma_frac=self.blur_sigma_frac,
            pixelate_factor=self.pixelate_factor,
        )
        rel = Path(self.root) / claim_id / f"{track_id}_{frame_idx}.jpg"
        _write_jpeg(rel, blurred, self.jpeg_quality)
        # ----------------------------------------------------------------- #

        payload: dict[str, Any] = dict(extra or {})
        payload.update(blur_meta)

        self._seen.add(key)
        self._rows.append({
            "claim_id": claim_id,
            "track_id": track_id,
            "frame_idx": frame_idx,
            "ts": ts,
            "value": value,
            "confidence": confidence,
            "path": rel.as_posix(),
            "stage": self.stage,
            "model_name": self.model_name,
            "extra_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "stratum": -1 if stratum is None else int(stratum),
            "schema_version": SCHEMA_VERSION,
        })
        return rel.as_posix()

    # -- индекс ------------------------------------------------------------- #

    def finalize(self, *, allow_empty: bool = False,
                 metadata: Mapping[str, Any] | None = None) -> Path:
        """Пишет evidence/index.parquet. Возвращает путь к нему."""
        if not self._rows and not allow_empty:
            raise EvidenceError(
                "пруфы не собраны ни для одного claim_id. Пустой индекс — это не успех "
                "(правило 8). Если пусто по делу, вызовите finalize(allow_empty=True)"
            )
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise EvidenceError(
                f"для записи evidence/index.parquet нужны pandas и pyarrow: {exc}"
            ) from exc

        df = pd.DataFrame(self._rows, columns=list(INDEX_COLUMNS))

        # Индекс пруфов — СКВОЗНОЙ каталог по всем этапам, а не файл одного
        # прогона. Этап владеет только своими claim_id и заменяет только их.
        # Раньше finalize перезаписывал файл целиком, и прогон S7 стирал из
        # индекса пруфы S4 и S6: jpg на диске оставались, но witness-ссылок
        # на них уже не было, и дашборд молча оставался без доказательств.
        n_kept = 0
        index_path = self.root / "index.parquet"
        if index_path.is_file():
            try:
                prev = pd.read_parquet(index_path)
            except (OSError, ValueError) as exc:
                raise EvidenceError(
                    f"{index_path} есть, но не читается: {exc}. "
                    f"Молча затирать чужие пруфы нельзя") from exc
            # Этап владеет ВСЕМ, что записал под своим именем, а не только
            # совпавшими claim_id. Если фильтровать по claim_id, то claim,
            # который этап перестал выпускать, остаётся в индексе навсегда:
            # 2026-09-04 дашборд одновременно печатал у витрины M1 «никого не
            # засчитано» и показывал её пруфы из прошлого прогона.
            keep = prev[prev["stage"] != self.stage]
            # Ссылка без файла — это обещание пруфа, которого нет.
            alive = keep["path"].map(lambda x: Path(str(x)).is_file())
            n_dropped = int((~alive).sum())
            keep = keep[alive]
            n_kept = len(keep)
            if n_dropped:
                print(f"[evidence] выброшено {n_dropped} записей: файлов нет на диске")
            if n_kept:
                df = pd.concat([keep[list(INDEX_COLUMNS)], df], ignore_index=True)
        self._n_kept_from_previous = n_kept
        schema = pa.schema([
            ("claim_id", pa.string()),
            ("track_id", pa.int64()),
            ("frame_idx", pa.int64()),
            ("ts", pa.float64()),
            ("value", pa.float64()),
            ("confidence", pa.float64()),
            ("path", pa.string()),
            ("stage", pa.string()),
            ("model_name", pa.string()),
            ("extra_json", pa.string()),
            ("stratum", pa.int8()),
            ("schema_version", pa.string()),
        ])
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

        meta = {
            b"schema_version": SCHEMA_VERSION.encode(),
            b"stage": self.stage.encode(),
            b"model_name": self.model_name.encode(),
            b"blur": json.dumps({
                "top_frac": self.face_blur_top_frac,
                "pixelate_factor": self.pixelate_factor,
                "kernel_frac": self.blur_kernel_frac,
                "sigma_frac": self.blur_sigma_frac,
            }).encode(),
        }
        if metadata:
            for k, v in metadata.items():
                meta[str(k).encode()] = json.dumps(v, ensure_ascii=False).encode()
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})

        out = self.root / "index.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(out)
        return out

    # -- служебное ---------------------------------------------------------- #

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    @staticmethod
    def _check_id(value: str, field_name: str) -> str:
        s = str(value)
        if not s or not _SAFE_ID.match(s):
            raise EvidenceError(
                f"{field_name}={value!r}: допустимы только буквы, цифры, точка, "
                f"дефис и подчёркивание — значение попадает в путь на диске"
            )
        return s


# --------------------------------------------------------------------------- #
# EvidenceSampler
# --------------------------------------------------------------------------- #

@dataclass
class _Candidate:
    track_id: int
    frame_idx: int
    ts: float
    crop_bgr: np.ndarray
    value: float
    confidence: float
    extra: dict[str, Any] = field(default_factory=dict)


class EvidenceSampler:
    """Копит кандидатов по claim_id и отбирает N штук СТРАТИФИЦИРОВАННО.

    Не top-N. Кандидаты сортируются по confidence и делятся ПО РАНГУ на три страты
    (нижняя / средняя / верхняя). Деление по рангу, а не по значению, устойчиво
    к перекошенному распределению и к одинаковым confidence.

    Квоты: n // 3 на страту, остаток идёт в НИЖНИЕ страты — слабые примеры
    информативнее всего для аудита и не должны недобираться.
    """

    def __init__(
        self,
        writer: EvidenceWriter,
        n_per_claim: int = DEFAULT_N_PER_CLAIM,
        n_strata: int = DEFAULT_N_STRATA,
        seed: int = DEFAULT_SEED,
        max_candidates_per_claim: int = DEFAULT_MAX_CANDIDATES_PER_CLAIM,
    ) -> None:
        if n_per_claim < 1:
            raise EvidenceError(f"n_per_claim должен быть >= 1, получено {n_per_claim}")
        if n_strata < 1:
            raise EvidenceError(f"n_strata должен быть >= 1, получено {n_strata}")
        if max_candidates_per_claim < n_per_claim:
            raise EvidenceError(
                f"max_candidates_per_claim ({max_candidates_per_claim}) меньше "
                f"n_per_claim ({n_per_claim})"
            )
        self.writer = writer
        self.n_per_claim = int(n_per_claim)
        self.n_strata = int(n_strata)
        self.seed = int(seed)
        self.max_candidates_per_claim = int(max_candidates_per_claim)
        self._pool: dict[str, list[_Candidate]] = {}
        self._offered: dict[str, int] = {}
        self._declared: set[str] = set()
        self._stats: dict[str, dict[str, Any]] = {}

    # -- объявление claim-ов ------------------------------------------------ #

    def declare(self, claim_ids: Iterable[str]) -> None:
        """Объявить claim-ы, которые этап ОБЯЗАН набрать (из configs/*.yaml).

        finalize() падает, если по объявленному claim_id не пришло ни одного
        кандидата: значит либо метрика не считалась, либо пруфы забыли собрать.
        """
        for cid in claim_ids:
            self._declared.add(EvidenceWriter._check_id(cid, "claim_id"))

    @property
    def declared_claims(self) -> list[str]:
        return sorted(self._declared)

    # -- набор -------------------------------------------------------------- #

    def offer(
        self,
        claim_id: str,
        track_id: int,
        frame_idx: int,
        ts: float,
        crop_bgr: np.ndarray,
        value: float,
        confidence: float,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Предложить кандидата. Кроп копируется — вызывающий волен переиспользовать буфер."""
        claim_id = EvidenceWriter._check_id(claim_id, "claim_id")
        cand = _Candidate(
            track_id=int(track_id),
            frame_idx=int(frame_idx),
            ts=float(ts),
            crop_bgr=np.ascontiguousarray(np.asarray(crop_bgr).copy()),
            value=float(value),
            confidence=float(confidence),
            extra=dict(extra or {}),
        )
        pool = self._pool.setdefault(claim_id, [])
        seen = self._offered.get(claim_id, 0)
        self._offered[claim_id] = seen + 1

        if len(pool) < self.max_candidates_per_claim:
            pool.append(cand)
            return
        # Reservoir sampling: равномерная выборка сохраняет распределение confidence,
        # поэтому стратификация после неё остаётся корректной. Недобор фиксируется
        # в статистике, а не проглатывается (правило 7).
        rng = self._rng(claim_id, salt=f"reservoir:{seen}")
        j = rng.randrange(seen + 1)
        if j < self.max_candidates_per_claim:
            pool[j] = cand

    # -- отбор -------------------------------------------------------------- #

    def finalize(self, n: int | None = None, *, allow_empty: bool = False) -> Path:
        """Отбирает кадры по каждому claim_id, пишет их и индекс."""
        n = self.n_per_claim if n is None else int(n)
        if n < 1:
            raise EvidenceError(f"n должен быть >= 1, получено {n}")

        missing = sorted(c for c in self._declared if not self._pool.get(c))
        if missing:
            raise EvidenceError(
                f"по объявленным claim_id не набрано ни одного пруфа: {missing}. "
                f"Число без пруфов в отчёт не идёт (правило 1)"
            )

        for claim_id in sorted(self._pool):
            self._select_and_write(claim_id, n)

        return self.writer.finalize(
            allow_empty=allow_empty,
            metadata={"sampling_stats": self._stats,
                      "declared_claims": self.declared_claims},
        )

    def stats(self) -> dict[str, dict[str, Any]]:
        """Статистика отбора по claim_id. Идёт в run_manifest.json и в отчёт."""
        return {k: dict(v) for k, v in self._stats.items()}

    # -- внутреннее --------------------------------------------------------- #

    def _rng(self, claim_id: str, salt: str = "") -> random.Random:
        """Детерминированный RNG, независимый от порядка обхода claim-ов."""
        digest = hashlib.sha256(f"{self.seed}:{claim_id}:{salt}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _select_and_write(self, claim_id: str, n: int) -> None:
        pool = self._pool[claim_id]
        # Детерминированный порядок: по confidence, тай-брейк по ключу строки.
        order = sorted(range(len(pool)),
                       key=lambda i: (pool[i].confidence, pool[i].track_id,
                                      pool[i].frame_idx))
        m = len(order)
        n_strata = min(self.n_strata, m)

        # Границы страт по РАНГУ, а не по значению confidence.
        bounds = [round(i * m / n_strata) for i in range(n_strata + 1)]
        strata_idx = [order[bounds[i]:bounds[i + 1]] for i in range(n_strata)]

        # Квоты: остаток от деления идёт в нижние страты.
        quotas = [n // n_strata] * n_strata
        for i in range(n % n_strata):
            quotas[i] += 1

        rng = self._rng(claim_id, salt="select")
        chosen: list[list[int]] = []
        for i in range(n_strata):
            pool_i = strata_idx[i]
            take = min(quotas[i], len(pool_i))
            chosen.append(sorted(rng.sample(pool_i, take)))

        # Дефицит одной страты перераспределяем по остальным, круговым обходом.
        deficit = n - sum(len(c) for c in chosen)
        while deficit > 0:
            progressed = False
            for i in range(n_strata):
                if deficit == 0:
                    break
                taken = set(chosen[i])
                left = [j for j in strata_idx[i] if j not in taken]
                if left:
                    chosen[i].append(rng.choice(left))
                    chosen[i].sort()
                    deficit -= 1
                    progressed = True
            if not progressed:
                break  # кандидатов физически меньше, чем n — фиксируется в stats

        per_stratum: list[dict[str, Any]] = []
        n_written = 0
        for i in range(n_strata):
            confs = [pool[j].confidence for j in chosen[i]]
            per_stratum.append({
                "stratum": i,
                "quota": quotas[i],
                "selected": len(chosen[i]),
                "candidates": len(strata_idx[i]),
                "conf_min": min(confs) if confs else None,
                "conf_max": max(confs) if confs else None,
            })
            for j in chosen[i]:
                c = pool[j]
                self.writer.add(
                    claim_id=claim_id,
                    track_id=c.track_id,
                    frame_idx=c.frame_idx,
                    ts=c.ts,
                    crop_bgr=c.crop_bgr,
                    value=c.value,
                    confidence=c.confidence,
                    extra=c.extra,
                    stratum=i,
                )
                n_written += 1

        offered = self._offered.get(claim_id, m)
        self._stats[claim_id] = {
            "n_offered": offered,
            "n_retained": m,                      # после reservoir sampling
            "n_dropped_by_reservoir": max(0, offered - m),
            "n_requested": n,
            "n_selected": n_written,
            "shortfall": max(0, n - n_written),   # правило 7: недобор виден
            "n_strata": n_strata,
            "per_stratum": per_stratum,
            "seed": self.seed,
        }


def stratum_of(rank: int, m: int, n_strata: int = DEFAULT_N_STRATA) -> int:
    """Номер страты для элемента ранга rank из m по возрастанию confidence.

    Вынесено наружу, чтобы verify мог пересчитать стратификацию независимо
    от EvidenceSampler и сравнить с колонкой stratum в индексе.
    """
    if m < 1 or not (0 <= rank < m):
        raise ValueError(f"rank={rank} вне диапазона для m={m}")
    n_strata = min(n_strata, m)
    bounds = [round(i * m / n_strata) for i in range(n_strata + 1)]
    for i in range(n_strata):
        if bounds[i] <= rank < bounds[i + 1]:
            return i
    return n_strata - 1


__all__ = [
    "EvidenceWriter",
    "EvidenceSampler",
    "EvidenceError",
    "blur_face_region",
    "DEFAULT_PIXELATE_FACTOR",
    "stratum_of",
    "INDEX_COLUMNS",
]


def _unused(*_: Sequence[Any]) -> None:  # pragma: no cover
    """Заглушка, чтобы линтер не ругался на импорт Sequence в аннотациях."""
