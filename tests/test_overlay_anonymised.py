"""Обезличивание в оверлее — часть рендера, а не опция.

Два уровня проверки, и оба нужны.

Синтетический работает всегда, в том числе на чистом клоне без весов: он
проверяет, что `anonymise_frame` действительно меняет верхнюю полосу рамки и
что резкость там падает. Этого достаточно, чтобы поймать регресс вида
«кто-то вернул флаг и выключил размытие».

Измерительный требует весов и видео и пропускается без них. Он делает ровно
то, что просили: считает уверенные лицевые кейпоинты на кадре ДО и ПОСЛЕ
обезличивания и требует нуля после. Синтетика этого не докажет — на шуме
кейпоинтов нет по построению.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _render_overlay():
    spec = importlib.util.spec_from_file_location(
        "render_overlay", ROOT / "scripts" / "render_overlay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeBoxes:
    def __init__(self, xyxy):
        self._x = np.asarray(xyxy, dtype=np.float64)

    def __len__(self):
        return len(self._x)

    @property
    def xyxy(self):
        class _T:
            def __init__(self, v):
                self._v = v

            def cpu(self):
                return self._v
        return _T(self._x)


class _FakeKeypoints:
    """xy и conf в форме, которую отдаёт ultralytics."""

    def __init__(self, xy, conf):
        self._xy = np.asarray(xy, dtype=np.float64)
        self._cf = np.asarray(conf, dtype=np.float64)

    @staticmethod
    def _wrap(v):
        class _T:
            def __init__(self, x):
                self._x = x

            def cpu(self):
                return self._x
        return _T(v)

    @property
    def xy(self):
        return self._wrap(self._xy)

    @property
    def conf(self):
        return self._wrap(self._cf)


class _FakeResult:
    def __init__(self, xyxy, kp=None):
        self.boxes = _FakeBoxes(xyxy)
        #: Детектор без позы кейпоинтов не отдаёт — как настоящий yolo11m.
        self.keypoints = kp


class _FakeModel:
    """Детектор, возвращающий заранее известные рамки и, если задано, позу."""

    def __init__(self, xyxy, kp=None):
        self._xyxy = xyxy
        self._kp = kp

    def predict(self, *a, **kw):
        return [_FakeResult(self._xyxy, self._kp)]


PRIV = {"face_blur_top_frac": 0.30, "blur_kernel_frac": 0.35,
        "blur_sigma_frac": 0.333, "pixelate_factor": 16}


def _sharp_frame(h=400, w=600):
    """Кадр с высокочастотной шахматкой: любое размытие видно по резкости."""
    yy, xx = np.mgrid[0:h, 0:w]
    board = (((yy // 3) + (xx // 3)) % 2 * 255).astype(np.uint8)
    return np.dstack([board, board, board])


def _sharpness(img) -> float:
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.abs(cv2.Laplacian(g, cv2.CV_64F)).mean())


def test_anonymise_frame_blurs_the_head_band():
    """Верхняя полоса рамки теряет резкость, остальное остаётся как было."""
    ro = _render_overlay()
    frame = _sharp_frame()
    box = (100, 50, 200, 350)                 # 100x300, полоса головы 90 px
    before_head = _sharpness(frame[50:140, 100:200])
    before_body = _sharpness(frame[250:350, 100:200])

    done, _ = ro.anonymise_frame(frame, _FakeModel([box]), {"imgsz": 640}, PRIV)

    assert done == 1
    after_head = _sharpness(frame[50:140, 100:200])
    after_body = _sharpness(frame[250:350, 100:200])
    assert after_head < before_head * 0.5, (
        f"полоса головы не размылась: {before_head:.1f} -> {after_head:.1f}")
    assert after_body == pytest.approx(before_body, rel=1e-6), (
        "размытие задело тело, а должно было только голову")


def test_degenerate_boxes_are_counted_not_dropped_silently():
    """Слишком мелкая рамка идёт в счётчик отброшенных, а не исчезает."""
    ro = _render_overlay()
    frame = _sharp_frame()
    done, skipped = ro.anonymise_frame(
        frame, _FakeModel([(10, 10, 20, 25)]), {"imgsz": 640}, PRIV)
    assert done == 0 and skipped == 1


def test_anonymisation_has_no_off_switch():
    """Флага для отключения нет и быть не должно.

    Опция, которую можно забыть выставить, рано или поздно окажется
    невыставленной. Тест сторожит именно это: появление флага в разборе
    аргументов — регресс, а не улучшение.
    """
    src = (ROOT / "scripts" / "render_overlay.py").read_text(encoding="utf-8")
    for bad in ("--no-blur", "--no-anon", "--skip-blur", "--raw",
                "no_anonymise", "skip_anonymise"):
        assert bad not in src, f"появился выключатель обезличивания: {bad}"
    # и сам вызов стоит в цикле безусловно, без if
    assert "n_anon, n_skip = anonymise_frame(" in src
    call = src[src.index("n_anon, n_skip = anonymise_frame("):]
    head = src[:src.index("n_anon, n_skip = anonymise_frame(")]
    assert not head.rstrip().endswith(":"), "вызов обезличивания стоит под условием"


def test_person_without_head_keypoints_still_gets_the_band():
    """ЗАПАСНОЙ ПУТЬ. Позы нет — размывается верхняя доля рамки, а не ничего.

    Случай не выдуманный: перекрытый человек, снятый со спины, срезанный
    краем кадра. Модель позы на них кейпоинтов не даёт. Если бы размытие
    зависело от кейпоинтов, такой человек уходил бы в кадр незакрытым — и
    заметить это было бы нечем.
    """
    ro = _render_overlay()
    frame = _sharp_frame()
    box = (100, 50, 200, 350)                    # человек 100x300
    before_head = _sharpness(frame[50:140, 100:200])
    before_body = _sharpness(frame[250:350, 100:200])

    # keypoints=None — ровно то, что отдаёт детектор без позы
    done, _ = ro.anonymise_frame(frame, _FakeModel([box]), {"imgsz": 640}, PRIV)

    assert done == 1, "рамка человека без позы обязана быть обработана"
    assert _sharpness(frame[50:140, 100:200]) < before_head * 0.5, (
        "запасной путь не сработал: верхняя доля рамки осталась резкой")
    assert _sharpness(frame[250:350, 100:200]) == pytest.approx(
        before_body, rel=1e-6), "размылось тело, а должна была голова"


def test_head_box_is_blurred_whole_not_by_fraction():
    """У рамки головы размывается вся площадь, а не её верхняя доля."""
    ro = _render_overlay()
    frame = _sharp_frame()
    # нос, глаза, уши сгруппированы — рамка головы строится вокруг них
    kp = _FakeKeypoints(
        xy=[[[300, 200], [295, 195], [305, 195], [288, 198], [312, 198]]],
        conf=[[0.9, 0.9, 0.9, 0.9, 0.9]])
    person = (270, 170, 330, 360)

    ro.anonymise_frame(frame, _FakeModel([person], kp), {"imgsz": 640}, PRIV)

    # низ рамки головы: при размытии только верхней доли остался бы резким
    heads = ro._head_boxes_from_pose(_FakeResult([person], kp))
    assert heads, "рамка головы не построена"
    hx0, hy0, hx1, hy1 = [int(v) for v in heads[0]]
    low = frame[max(0, hy1 - 12):hy1, max(0, hx0):hx1]
    assert low.size and _sharpness(low) < 12.0, (
        "низ рамки головы остался резким — размылась только доля")


def test_head_and_person_are_marked_not_guessed():
    """Вид рамки задаётся явно и не выводится из её пропорции.

    Прежняя версия считала головой всё, что вытянуто вертикально слабее 1.7.
    Работало только потому, что HEAD_PAD = 0.9 держал рамку головы на 1.556.
    Уменьшение отступа тихо превратило бы голову в «человека» и оставило бы
    лицо под размытой лишь верхней третью.
    """
    ro = _render_overlay()
    src = (ROOT / "scripts" / "render_overlay.py").read_text(encoding="utf-8")
    assert "1.7 * w" not in src, "вернулась догадка о виде рамки по пропорции"
    assert ro.KIND_HEAD != ro.KIND_PERSON

    kp = _FakeKeypoints(xy=[[[300, 200], [295, 195], [305, 195],
                             [288, 198], [312, 198]]],
                        conf=[[0.9, 0.9, 0.9, 0.9, 0.9]])
    kinds = [k for _, k in ro._boxes_union(
        [_FakeModel([(270, 170, 330, 360)], kp)], _sharp_frame(), {"imgsz": 640})]
    assert ro.KIND_PERSON in kinds and ro.KIND_HEAD in kinds, (
        f"ожидались обе метки, получено {kinds}")


# --------------------------------------------------------------------------- #
# Измерение на реальном кадре: требует весов позы и видео
# --------------------------------------------------------------------------- #

_POSE = ROOT / "models" / "yolo11m-pose.pt"
_DET = ROOT / "models" / "yolo11m.pt"
_CLIP = ROOT / "raw" / "peak_hour.ts"

_REASON = "нужны веса обеих моделей и исходная запись"

#: Окно вокруг кейпоинта для замера резкости.
_WIN = 15

#: Порог резкости, выше которого черты лица считаются сохранившимися.
#: Выведен из НЕОБЕЗЛИЧЕННЫХ кадров: p5 резкости вокруг лицевых кейпоинтов на
#: исходном видео равен 13.4, то есть даже самое размытое настоящее лицо
#: резче этого. Всё ниже — не лицо, а пятно.
_SHARP_THR = 13.4


def _sharp_at(gray, x, y, win=_WIN):
    import cv2
    h, w = gray.shape
    x0, x1 = max(0, int(x) - win), min(w, int(x) + win)
    y0, y1 = max(0, int(y) - win), min(h, int(y) + win)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    return float(np.abs(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F)).mean())


def _facial_keypoints(pose, img):
    """(число уверенных лицевых кейпоинтов, их резкости)."""
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = pose.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]
    if r.keypoints is None or r.keypoints.conf is None:
        return 0, []
    xy = np.asarray(r.keypoints.xy.cpu())
    cf = np.asarray(r.keypoints.conf.cpu())
    sharp = []
    for pi in range(cf.shape[0]):
        for j in (0, 1, 2, 3, 4):          # нос, глаза, уши
            if cf[pi, j] >= 0.30:
                sharp.append(_sharp_at(gray, xy[pi, j, 0], xy[pi, j, 1]))
    return len(sharp), sharp


@pytest.mark.skipif(not (_POSE.is_file() and _DET.is_file() and _CLIP.is_file()),
                    reason=_REASON)
def test_anonymisation_leaves_no_sharp_facial_detail():
    """После обезличивания ни один лицевой кейпоинт не лежит в резкой области.

    Меряется на кадре СРАЗУ ПОСЛЕ обезличивания и ДО отрисовки боксов,
    стрелок и подписей. Это принципиально: на готовом кадре модель позы
    ставит кейпоинты на саму разметку — замер поймал «ухо» с уверенностью
    0.32 на зелёной надписи «1916 blue 2.5s», где лица нет вовсе. Тест на
    готовом кадре мерил бы качество нашей же графики.

    Сырое число кейпоинтов тоже не годится: модель находит ГОЛОВУ по плечам
    и корпусу и уверенно ставит «нос» на пикселизованное пятно. Опасно не
    то, что голову видно, а то, что видно лицо, — поэтому мерится резкость.
    """
    import cv2
    from ultralytics import YOLO

    ro = _render_overlay()
    cfg = ro.load_config("configs/s3_detect.yaml")
    models, priv = ro.load_anonymiser(cfg)
    params = ro.infer_params(cfg)
    pose = YOLO(str(_POSE))

    cap = cv2.VideoCapture(str(_CLIP))
    frames, idx = [], 0
    while len(frames) < 6 and idx < 3000:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % 500 == 0:
            frames.append(fr)
        idx += 1
    cap.release()
    assert frames, "не прочитан ни один кадр исходной записи"

    n_before = n_after = 0
    sharp_before, sharp_after = [], []
    for fr in frames:
        nb, sb = _facial_keypoints(pose, fr)
        n_before += nb
        sharp_before += sb
        ro.anonymise_frame(fr, models, params, priv)      # правит на месте
        na, sa = _facial_keypoints(pose, fr)
        n_after += na
        sharp_after += sa

    exposed = [v for v in sharp_after if v >= _SHARP_THR]
    assert n_before > 0, "на исходных кадрах не нашлось ни одного лица — нечего проверять"
    assert not exposed, (
        f"после обезличивания осталось {len(exposed)} лицевых кейпоинтов в "
        f"резких областях (резкость {[round(v, 1) for v in exposed]}, "
        f"порог {_SHARP_THR}). Было кейпоинтов {n_before}, стало {n_after}")
