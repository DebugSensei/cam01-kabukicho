"""Ручная разметка: ориентация (S5) и цвет верха (S7).

ЗАЧЕМ. Это единственные НЕКОЛЬЦЕВЫЕ проверки в проекте. Всё остальное —
согласие модели с самой собой. Гейты S5 и S7 без разметки не считаются
и обязаны валиться (правило 3).

ЧТО ЗАПИСЫВАЕТСЯ. Строка на кроп: видео, кадр, рамка, предсказание модели
и твой ответ. Предсказание кладётся В ФАЙЛ, поэтому метрику можно посчитать
даже после того, как артефакты перезапишет следующий прогон.

ПРОТОКОЛ ОРИЕНТАЦИИ. Ты кликаешь по земле в точке, КУДА развёрнут корпус.
Опорная точка человека и точка клика проходят через ТУ ЖЕ гомографию, и
угол считается на плане. Никаких «на глаз»: человек указывает точку,
геометрию считает код.

ПРОТОКОЛ ЦВЕТА. Цифра из палитры. Предсказание модели СКРЫТО — показать
его значило бы заякорить разметчика и завысить accuracy.

ПРИВАТНОСТЬ. Кропы на диск не пишутся вообще. Файл разметки хранит только
координаты, а картинка декодируется из видео заново. Лица размываются тем
же кодом, что и пруфы (looq.evidence.blur_face_region).

    python scripts/make_labels.py --mode=orient --n 50
    python scripts/make_labels.py --mode=color  --n 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.calib import apply_h  # noqa: E402
from looq.evidence import blur_face_region  # noqa: E402
from looq.io import atomic_write_text, load_config, read_json, sha256_file  # noqa: E402
from looq.pilot import iter_frames  # noqa: E402

LABEL_DIR = Path("labels")
#: Допуск сопоставления трека с детекцией — доля высоты рамки. Замерено:
#: медиана 0.006, p95 0.024, p99 0.044 высоты рамки.
TOL_FRAC, TOL_MIN_PX = 0.08, 4.0
PANEL_W, CROP_H = 420, 620

COLOR_KEYS = {"1": "black", "2": "white", "3": "grey", "4": "red", "5": "orange",
              "6": "yellow", "7": "green", "8": "blue", "9": "purple", "0": "pink",
              "o": "other"}


def _match_box(dets, fx, fy):
    """Ближайшая детекция в относительном допуске. Трекер сглаживает состояние,
    точного совпадения не бывает."""
    cx = ((dets["x1_px"] + dets["x2_px"]) / 2.0).to_numpy()
    y2 = dets["y2_px"].to_numpy()
    bh = (dets["y2_px"] - dets["y1_px"]).to_numpy()
    d = np.hypot(cx - fx, y2 - fy)
    j = int(np.argmin(d))
    if d[j] > max(TOL_MIN_PX, TOL_FRAC * bh[j]):
        return None
    r = dets.iloc[j]
    return [float(r.x1_px), float(r.y1_px), float(r.x2_px), float(r.y2_px)]


def _stratified(df, key: str, n: int, seed: int, max_per_track: int = 1):
    """Отбор по рангу уверенности, три страты, остаток в НИЖНЮЮ.

    Не top-N: сетка из лучших показывала бы, как хорошо всё работает, а не
    как оно работает. Один кроп на трек: два кропа одного человека — это не
    два наблюдения.
    """
    df = df.dropna(subset=[key]).copy()
    if df.empty:
        return df
    df = (df.sort_values(key, ascending=False)
            .groupby("track_id", as_index=False, group_keys=False).head(max_per_track))
    df = df.sort_values([key, "track_id", "frame_idx"]).reset_index(drop=True)
    m = len(df)
    if m <= n:
        return df
    b1, b2 = m // 3, 2 * m // 3
    strata = [df.iloc[:b1], df.iloc[b1:b2], df.iloc[b2:]]
    base, rem = n // 3, n % 3
    quota = [base + (1 if i < rem else 0) for i in range(3)]   # остаток вниз
    rng = random.Random(seed)
    out = []
    for s, q in zip(strata, quota):
        idx = list(range(len(s)))
        rng.shuffle(idx)
        out.append(s.iloc[sorted(idx[:min(q, len(s))])])
    got = __import__("pandas").concat(out, ignore_index=True)
    if len(got) < n:                       # дефицит в страте — добираем, но честно
        rest = df[~df.index.isin(got.index)]
        got = __import__("pandas").concat([got, rest.head(n - len(got))],
                                          ignore_index=True)
    return got.head(n)


def _panel(mode: str, i: int, total: int, extra: list[str]) -> np.ndarray:
    img = np.full((CROP_H, PANEL_W, 3), 28, np.uint8)
    lines = [f"{mode.upper()}  {i + 1} / {total}", ""]
    if mode == "color":
        lines += ["1 black   2 white   3 grey",
                  "4 red     5 orange  6 yellow",
                  "7 green   8 blue    9 purple",
                  "0 pink    o other (motley)", "",
                  "u unsure    x not a person", ""]
    else:
        # Про Enter обязательно писать: клик только ставит точку, запись
        # происходит по Enter/пробелу, и без подсказки это выглядит так,
        # будто окно не реагирует на клик.
        # Про корпус — тоже: модель считает угол по плечам, и разметка по
        # направлению движения меряла бы другую величину.
        lines += ["Label the TORSO, where the",
                  "CHEST points. NOT where the",
                  "person is walking.", "",
                  "CLICK the GROUND in front", "of the chest, then",
                  "ENTER or SPACE to confirm.", "",
                  "u unsure    x not a person", ""]
    lines += ["BACKSPACE  go back",
              "s          save and quit",
              "ESC        quit WITHOUT saving", ""] + extra
    for k, t in enumerate(lines):
        cv2.putText(img, t, (14, 34 + k * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (235, 235, 235) if k == 0 else (185, 190, 200), 1, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["orient", "color"],
                    help="что размечаем. Дефолта нет намеренно (правило 5)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--max-per-track", type=int, default=1,
                    help="кропов на трек. Больше одного — это НЕ больше "
                         "независимых наблюдений, и интервал так не сузится")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--video", type=Path, default=None,
                    help="видео, из которого брать кадры. По умолчанию берётся "
                         "из конфига — но конфиг мог быть переведён на другой "
                         "прогон, а артефакты остались от прежнего")
    ap.add_argument("--artifacts", type=Path, default=Path("."),
                    help="каталог с артефактами. Снимок нужен, если параллельно "
                         "идёт прогон и перезаписывает их")
    args = ap.parse_args(argv)

    import pandas as pd

    cfg_path = "configs/s5_orient.yaml" if args.mode == "orient" else "configs/s7_attrs.yaml"
    cfg = load_config(cfg_path)
    video = args.video or Path(cfg["input"].get("clip") or cfg["input"].get("video"))
    if not video.is_file():
        raise SystemExit(f"нет видео {video}")
    # Номера кадров в артефакте относятся к ТОМУ видео, на котором он посчитан.
    # Если конфиг уже переведён на другой прогон, а артефакты остались от
    # прежнего, кропы будут вырезаны не из тех кадров — молча и правдоподобно.
    print(f"[labels] видео: {video}")
    print(f"[labels] артефакты: {args.artifacts.resolve()}")
    hom = read_json("calib/homography.json")
    h = np.asarray(hom["H"], dtype=np.float64)

    A = args.artifacts
    tracks = pd.read_parquet(A / "track/tracks.parquet" if (A / "track").is_dir()
                             else A / "tracks.parquet")
    det = pd.read_parquet(A / "det/frames.parquet" if (A / "det").is_dir()
                          else A / "frames.parquet")
    ev_cfg = load_config("configs/evidence.yaml").get("privacy", {})

    if args.mode == "orient":
        src = pd.read_parquet(A / "pose/orient.parquet" if (A / "pose").is_dir()
                              else A / "orient.parquet")
        src = src[np.isfinite(src["body_yaw_deg"])]
        conf_key = "yaw_conf"
        picks = _stratified(src, conf_key, args.n, args.seed, args.max_per_track)
        pred_col, pred_name = "body_yaw_deg", "predicted_yaw_deg"
    else:
        attr = pd.read_parquet(A / "attr/tracks_attr.parquet" if (A / "attr").is_dir()
                               else A / "tracks_attr.parquet")
        attr = attr[attr["top_color_status"].isin(["ok", "low_agreement"])]
        # Один кадр на трек — РЕАЛЬНЫЙ средний кадр трека, а не медиана его
        # номеров: медиана попадает в дырку, если трек рвался, и кроп из такого
        # кадра вырезать неоткуда. На утреннем клипе так терялось 12 из 28.
        mid = (tracks.sort_values("frame_idx").groupby("track_id")["frame_idx"]
               .apply(lambda s: int(s.to_numpy()[len(s) // 2])).rename("frame_idx"))
        src = attr.merge(mid, on="track_id", how="inner")
        conf_key = "top_color_conf"
        picks = _stratified(src, conf_key, args.n, args.seed, args.max_per_track)
        pred_col, pred_name = "top_color_name", "predicted_color"

    if picks.empty:
        raise SystemExit("нечего размечать: в артефакте нет подходящих строк")
    print(f"отобрано {len(picks)} кропов из {len(src)} доступных "
          f"(стратификация по {conf_key}, один кроп на трек)")

    # Декодируем ВСЕ нужные кадры одним последовательным проходом: у .ts
    # перемотка врёт, а второй проход стоит столько же, сколько первый.
    want = np.asarray(sorted(picks["frame_idx"].unique()), dtype=np.int64)
    foot = tracks.set_index(["frame_idx", "track_id"])[["foot_x_px", "foot_y_px"]]
    det_by = {int(f): g for f, g in det.groupby("frame_idx")}
    items = []
    last = int(max(want)) if len(want) else 0
    print(f"читаю {len(want)} кадров из {video} ...")
    print(f"нужные кадры разбросаны до {last}, идём подряд без перемотки "
          f"(у .ts из HLS она врёт) — это займёт время")

    def _progress(i, left):
        pct = 100.0 * i / last if last else 100.0
        print(f"\r  {i}/{last} кадров ({pct:.0f}%), осталось найти {left}   ",
              end="", flush=True)

    for fi, frame in iter_frames(video, want, progress=_progress):
        rows = picks[picks["frame_idx"] == int(fi)]
        d = det_by.get(int(fi))
        if d is None:
            continue
        for r in rows.itertuples():
            try:
                fx, fy = foot.loc[(int(fi), int(r.track_id))]
            except KeyError:
                continue
            box = _match_box(d, float(fx), float(fy))
            if box is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
            if crop.size == 0 or crop.shape[0] < 24:
                continue
            crop, _blur_meta = blur_face_region(
                crop,
                top_frac=float(ev_cfg.get("face_blur_top_frac", 0.30)),
                kernel_frac=float(ev_cfg.get("blur_kernel_frac", 0.35)),
                sigma_frac=float(ev_cfg.get("blur_sigma_frac", 0.5)),
                pixelate_factor=int(ev_cfg.get("pixelate_factor", 16)))
            ctx = frame[max(0, y1 - 60):min(frame.shape[0], y2 + 220),
                        max(0, x1 - 260):min(frame.shape[1], x2 + 260)].copy()
            items.append({"track_id": int(r.track_id), "frame_idx": int(fi),
                          "bbox_px": box, "bbox_h_px": float(y2 - y1),
                          "crop": crop, "ctx": ctx,
                          "ctx_origin": [max(0, x1 - 260), max(0, y1 - 60)],
                          "foot_px": [float(fx), float(fy)],
                          "pred": getattr(r, pred_col),
                          "conf": float(getattr(r, conf_key))})
    if not items:
        raise SystemExit("ни один кроп не собран")

    rng = random.Random(args.seed)
    rng.shuffle(items)      # порядок перемешан: усталость не должна коррелировать
    print(f"готово {len(items)} кропов. Открываю окно.")

    labels: list[dict] = []
    click = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (x, y)

    win = f"labels [{args.mode}]"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    i, t_item = 0, time.time()
    while 0 <= i < len(items):
        it = items[i]
        crop = it["crop"]
        s = min(CROP_H / crop.shape[0], 300 / max(1, crop.shape[1]))
        shown = cv2.resize(crop, (max(1, int(crop.shape[1] * s)),
                                  max(1, int(crop.shape[0] * s))),
                           interpolation=cv2.INTER_CUBIC)
        left = np.full((CROP_H, max(320, shown.shape[1] + 20), 3), 18, np.uint8)
        left[:shown.shape[0], 10:10 + shown.shape[1]] = shown

        if args.mode == "orient":
            ctx = it["ctx"]
            cs = min(CROP_H / ctx.shape[0], 640 / max(1, ctx.shape[1]))
            ctx_s = cv2.resize(ctx, (int(ctx.shape[1] * cs), int(ctx.shape[0] * cs)))
            mid_img = np.full((CROP_H, ctx_s.shape[1], 3), 18, np.uint8)
            mid_img[:ctx_s.shape[0], :] = ctx_s
            ox, oy = it["ctx_origin"]
            fpx = (int((it["foot_px"][0] - ox) * cs), int((it["foot_px"][1] - oy) * cs))
            # Рамка цели: в плотной сцене одной точки у ног мало, чтобы понять,
            # чью ориентацию размечаем, а ошибка здесь портит эталон молча.
            bx1, by1, bx2, by2 = it["bbox_px"]
            cv2.rectangle(mid_img,
                          (int((bx1 - ox) * cs), int((by1 - oy) * cs)),
                          (int((bx2 - ox) * cs), int((by2 - oy) * cs)),
                          (60, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(mid_img, "THIS ONE",
                        (int((bx1 - ox) * cs), max(14, int((by1 - oy) * cs) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(mid_img, fpx, 6, (60, 255, 255), -1, cv2.LINE_AA)
            if click["pt"] is not None:
                cv2.arrowedLine(mid_img, fpx, click["pt"], (60, 255, 255), 2,
                                cv2.LINE_AA, tipLength=0.25)
            canvas = np.hstack([left, mid_img,
                                _panel(args.mode, i, len(items),
                                       [f"track #{it['track_id']}  frame {it['frame_idx']}",
                                        f"conf {it['conf']:.2f}",
                                        "click set" if click["pt"] else "no click yet"])])
        else:
            canvas = np.hstack([left, _panel(args.mode, i, len(items),
                                             [f"track #{it['track_id']}",
                                              f"box h {it['bbox_h_px']:.0f} px"])])
        cv2.imshow(win, canvas)
        k = cv2.waitKey(20) & 0xFF

        rec = None
        if k == 27:
            print("выход БЕЗ сохранения"); cv2.destroyAllWindows(); return 1
        if k == ord("s"):
            break
        if k == 8:                                    # BACKSPACE
            if labels:
                labels.pop(); i = max(0, i - 1); click["pt"] = None
            continue
        if k == ord("x"):
            rec = {"label": None, "flag": "not_a_person"}
        elif k == ord("u"):
            rec = {"label": None, "flag": "unsure"}
        elif (args.mode == "color" and 0 < k < 256
              and chr(k) in COLOR_KEYS):
            rec = {"label": COLOR_KEYS[chr(k)], "flag": None}
        elif args.mode == "orient" and click["pt"] is not None and k in (13, 32):
            ox, oy = it["ctx_origin"]
            ctx = it["ctx"]
            cs = min(CROP_H / ctx.shape[0], 640 / max(1, ctx.shape[1]))
            gx, gy = click["pt"][0] / cs + ox, click["pt"][1] / cs + oy
            a = apply_h(h, np.array([it["foot_px"]], dtype=np.float64))[0]
            b = apply_h(h, np.array([[gx, gy]], dtype=np.float64))[0]
            v = b - a
            if float(np.hypot(*v)) < 1e-6:
                click["pt"] = None
                continue
            yaw = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
            rec = {"label": round(yaw, 1), "flag": None,
                   "click_px": [round(gx, 1), round(gy, 1)],
                   "target_m": [round(float(b[0]), 3), round(float(b[1]), 3)]}
        if rec is None:
            continue

        pred = it["pred"]
        labels.append({
            "track_id": it["track_id"], "frame_idx": it["frame_idx"],
            "bbox_px": [round(v, 1) for v in it["bbox_px"]],
            "bbox_h_px": round(it["bbox_h_px"], 1),
            pred_name: (None if pred is None or (isinstance(pred, float)
                                                 and not np.isfinite(pred))
                        else (round(float(pred), 1) if args.mode == "orient" else str(pred))),
            "model_conf": round(it["conf"], 4),
            "predicted_hidden_during_labelling": True,
            "seconds_spent": round(time.time() - t_item, 1),
            **rec,
        })
        click["pt"] = None
        t_item = time.time()
        i += 1

    cv2.destroyAllWindows()
    if not labels:
        print("ничего не размечено, файл не пишется"); return 1

    out = args.out or LABEL_DIR / (f"s5_orient_{len(labels)}.jsonl" if args.mode == "orient"
                                   else f"s7_color_{len(labels)}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "schema_version": "1", "mode": args.mode, "n": len(labels),
        "video": str(video).replace("\\", "/"), "video_sha256": sha256_file(video),
        "config": cfg_path, "config_sha256": sha256_file(cfg_path),
        "homography_sha256": sha256_file("calib/homography.json"),
        "seed": args.seed, "selection": "stratified_by_rank_3_strata_remainder_low",
        "max_per_track": 1, "order": "shuffled",
        "prediction_hidden": True,
        "protocol_ru": ("orient: разметчик кликает по земле в точке, куда развёрнут "
                        "корпус; угол считается кодом через ту же гомографию. "
                        "color: цифра из палитры, предсказание скрыто."),
        "privacy_ru": "Кропы на диск не пишутся. Лица размыты перед показом.",
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    lines += [json.dumps(r, ensure_ascii=False) for r in labels]
    atomic_write_text(out, "\n".join(lines) + "\n")

    n_ok = sum(1 for r in labels if r["label"] is not None)
    print(f"\nзаписано: {out}")
    print(f"  размечено {n_ok}, unsure/не-человек {len(labels) - n_ok}")
    print(f"  медиана времени на кроп: "
          f"{np.median([r['seconds_spent'] for r in labels]):.1f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
