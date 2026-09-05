"""out/dashboard.html — дашборд под глаза.

Один самодостаточный файл: картинки и кропы вшиты в base64, внешних запросов
нет. Светлая тема по умолчанию, тёмная переключателем, интерфейс английский.

Дашборд НИЧЕГО НЕ СЧИТАЕТ. Все числа берутся из out/metrics.json (этап S8),
вместе с доверительными интервалами и покрытием. Единственное, что здесь
производится, — кропы-пруфы: их набирает EvidenceSampler, стратифицированно,
с обезличиванием лиц. Отбор именно стратифицированный, а не top-N: сетка из
двенадцати лучших кадров показывала бы, как хорошо всё работает, а не как оно
работает.

Чего здесь нет и не будет: пол, возраст, этничность. Приватность (правило 9)
и отсутствие ground truth для гейта (правило 7).

    python scripts/make_dashboard.py
    make dashboard
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looq.io import atomic_write_text, load_config, read_json, require  # noqa: E402
from looq.evidence import (EvidenceError, EvidenceSampler,  # noqa: E402
                           EvidenceWriter, blur_face_region)
from looq.pilot import iter_frames  # noqa: E402

OUT = Path("out/dashboard.html")
N_PROOFS = 12
#: Кадров одного трека в сетке пруфов. Двенадцать кадров одного человека
#: читаются как двенадцать человек — ровно та ошибка, из-за которой сетку у M2
#: приняли за 12 посетителей, хотя людей там было двое.
MAX_FRAMES_PER_TRACK = 2
#: Допуск сопоставления трека с его детекцией — ДОЛЯ ВЫСОТЫ РАМКИ, не пиксели.
#: Трекер сглаживает состояние, поэтому опорная точка трека не совпадает с
#: детекцией бит в бит. Абсолютный порог здесь неверен как критерий: 15 px для
#: рамки 40 px и для рамки 204 px — это разные вопросы.
#: ЗАМЕРЕНО на этом клипе: относительное расстояние до ближайшей детекции имеет
#: медиану 0.006, p95 = 0.024, p99 = 0.044 высоты рамки. Порог 0.08 лежит почти
#: вдвое выше p99 и покрывает 99.81% строк.
BOX_MATCH_TOL_FRAC = 0.08
BOX_MATCH_TOL_MIN_PX = 4.0     # для совсем мелких рамок доля вырождается

ZONE_HEX = ["#22c55e", "#f59e0b", "#3b82f6", "#ec4899"]

#: Названия витрин тремя языками. Японское — это то, что реально написано на
#: вывеске; английское и русское — транслитерация с пояснением рода заведения,
#: а не перевод «по смыслу»: заведение называется так, как называется.
ZONE_NAMES = {
    "facade_M1": ("Kakuni / Genkatsu — braised pork, tonkatsu",
                  "Какуни / Гэнкацу — тушёная свинина, тонкацу",
                  "角煮／げんかつ"),
    "facade_M2": ("Entrance / Ramen", "Вход / рамэн", "入口／らーめん"),
    "facade_M3": ("Shibaura Horumon — grilled offal",
                  "Сибаура Хорумон — жареные потроха", "芝浦ホルモン"),
    "facade_M4": ("Okonomiyaki", "Окономияки", "お好み焼き"),
}
#: Экранные образцы КЛАССОВ СВЕТЛОТЫ. Это не измеренный цвет одежды: измеренный
#: цвет лежит в hsv_* артефакта, и в этой записи он ахроматичен у всех треков.
CLASS_HEX = {"black": "#1f2430", "grey": "#9aa3b2", "white": "#e8ecf3",
             "red": "#ef4444", "orange": "#f97316", "yellow": "#eab308",
             "green": "#22c55e", "blue": "#3b82f6", "purple": "#8b5cf6",
             "pink": "#ec4899", "other": "#64748b"}

#: Английские формулировки для пунктов, которые S8 пишет по-русски. Ключ — поле
#: item из metrics.json. Неизвестный пункт НЕ выбрасывается: он выводится с
#: исходным текстом и пометкой untranslated, иначе ограничение могло бы тихо
#: исчезнуть со страницы.
LIMIT_EN = {
    "источник масштаба": (
        ("Source of scale", "Источник масштаба", "スケールの根拠"),
        ("Scale comes from the MEDIAN HEIGHT of the sampled pedestrians. Street "
         "width was rejected as the source: the 6.06 m satellite reference implies "
         "a median height of 1.93 m, outside the plausible 1.55–1.75 m.",
         "Масштаб взят из МЕДИАННОГО РОСТА выборки пешеходов. Ширина улицы как "
         "источник ОТВЕРГНУТА: спутниковый эталон 6.06 м даёт медианный рост "
         "1.93 м, что вне правдоподобных 1.55–1.75 м.",
         "スケールは歩行者の身長中央値に基づきます。通り幅は根拠として棄却： "
         "衛星基準6.06mでは中央身長1.93mとなり、妥当な1.55〜1.75mを外れます。"),
        ("Height is therefore NOT an independent check — it defines the scale. One "
         "independent check remains: the implied L1–L3 distance of 5.28 m falls "
         "inside the plausible range 4.6–5.6 m.",
         "Поэтому рост НЕ является независимой проверкой — он задаёт масштаб. "
         "Осталась одна независимая проверка: подразумеваемое расстояние L1–L3 "
         "в 5.28 м попадает в правдоподобный диапазон 4.6–5.6 м.",
         "したがって身長は独立検証ではなく、スケールを定義します。独立検証は1つ： "
         "含意されるL1–L3距離5.28mは妥当範囲4.6〜5.6mに収まります。")),
    "уклон улицы": (
        ("Street grade", "Уклон улицы", "通りの勾配"),
        ("The ground is modelled as FLAT, yet reconstructed height drifts "
         "systematically with depth. A 4.9% grade (2.81 deg) would explain the "
         "drift; the ground falls AWAY from the camera.",
         "Модель земли ПЛОСКАЯ, но рост систематически плывёт с глубиной. "
         "Уклон 4.9% (2.81 град) объяснил бы дрейф; земля уходит ВНИЗ от камеры.",
         "地面は平面としてモデル化していますが、身長が奥行きに応じて系統的に "
         "変動します。4.9%（2.81度）の勾配で説明可能。地面はカメラから下る方向。"),
        ("Lengths and speeds are distorted more in the far half than the near "
         "half. Sensitivity checked: shifting the vertical vanishing point by "
         "+-10% changes the drift by only 8% and never zeroes it — so the cause "
         "is the scene, not the calibration.",
         "Длины и скорости на дальнем плане искажены сильнее, чем на ближнем. "
         "Проверено чувствительностью: сдвиг вертикальной точки схода на +-10% "
         "меняет дрейф лишь на 8% и нигде не обнуляет его, значит дело в самой "
         "сцене, а не в калибровке.",
         "遠方ほど長さと速度の歪みが大きくなります。感度検証：鉛直消失点を±10% "
         "動かしても変動は8%しか変わらず消えないため、原因は較正ではなく現場です。")),
    "точка схода улицы против горизонта": (
        ("Street vanishing point vs horizon", "Точка схода улицы против горизонта",
         "通りの消失点と地平線"),
        ("The two estimates disagree by 104 px against a 87 px tolerance.",
         "Расхождение 104 px при допуске 87 px.",
         "2つの推定が87pxの許容に対し104pxずれています。"),
        ("Two independent estimates of the same quantity did not converge: the "
         "focal length, and therefore the scale, are less well determined than "
         "we would like.",
         "Две независимые оценки одной величины не сошлись; фокус и с ним "
         "масштаб определены хуже, чем хотелось бы.",
         "同一量の独立2推定が一致せず、焦点距離とスケールの確度は期待より低い。")),
}

#: Ключ — поле item из metrics.json. Значение — три языка для названия и
#: три для причины. Неизвестный пункт НЕ выбрасывается: выводится с исходным
#: текстом и пометкой untranslated, иначе ограничение тихо исчезло бы.
UNMEASURED_EN = {
    "AP@0.5 (S3)": (
        ("Detection AP@0.5", "AP@0.5 детекции", "検出AP@0.5"),
        ("no ground truth for 300 frames", "нет разметки 300 кадров",
         "300フレームの正解データなし")),
    "IDF1 (S4)": (
        ("Tracking IDF1 / ID switches", "IDF1 и склейки треков",
         "追跡IDF1・ID切替"),
        ("no ground truth", "нет разметки", "正解データなし")),
    # Ключи словаря обязаны совпадать с item из out/metrics.json, иначе
    # строка уедет в ветку "(untranslated)".
    "MAE угла (S5)": (
        ("Orientation angle MAE", "MAE угла ориентации", "方位角のMAE"),
        ("measured on {n} people, not the 200 the gate asks for",
         "измерен на {n} людях вместо 200, которых требует гейт",
         "ゲート要件200人に対し{n}人で計測")),
    "precision событий (S6)": (
        ("Attention-event precision", "Precision событий внимания",
         "注目イベントの適合率"),
        ("no ground truth for 100 events", "нет разметки 100 событий",
         "100イベントの正解データなし")),
    "точность цвета одежды (S7)": (
        ("Clothing colour accuracy", "Точность цвета одежды", "服装色の精度"),
        ("no labelled crops: white balance applied but not validated",
         "нет размеченных кропов: баланс белого применён, но не валидирован",
         "ラベル付きクロップなし。WB適用済みだが未検証")),
}


#: Три языка. Английский — значение по умолчанию, он же то, что стоит в HTML;
#: RU и JA подставляются переключателем на клиенте. Ключ короткий и латиницей,
#: чтобы его нельзя было спутать с самим текстом.
I18N = {
    "nav.camera":      ("Camera", "Камера", "カメラ"),
    "nav.location":    ("Location", "Локация", "所在地"),
    "nav.window":      ("Window", "Окно", "対象時間"),
    "nav.menu":        ("Menu", "Меню", "メニュー"),
    "nav.dash":        ("Dashboard", "Дашборд", "ダッシュボード"),
    "nav.colors":      ("Clothing", "Одежда", "服装"),
    "nav.replay":      ("Replay", "Реплей", "リプレイ"),
    "nav.overlay":     ("Overlay video", "Оверлей", "オーバーレイ"),
    "nav.frames":      ("Screenshots", "Скриншоты", "スクリーンショット"),
    "nav.bench":       ("Benchmark", "Бенчмарк", "ベンチマーク"),
    # out/report.html — артефакт этапа S9 по контракту, и до сих пор на него
    # не вела ни одна ссылка: единственная страница, где у каждого числа
    # напечатан compute_ref, была недостижима из интерфейса.
    "nav.report":      ("Provenance", "Провенанс", "来歴"),
    "nav.theme":       ("Light / Dark", "Светлая / Тёмная", "ライト / ダーク"),

    "hero.eyebrow":  ("Storefront attention", "Внимание к витринам", "店舗への注目"),
    "hero.title":    ("What the camera actually measured",
                      "Что камера действительно измерила",
                      "カメラが実際に計測したもの"),
    "hero.sub":      ("Every number on this page is read from out/metrics.json, "
                      "written by stage S8, together with its confidence interval "
                      "and coverage. Nothing here is estimated by eye. Attention "
                      "means a turn of the body or head toward a storefront — it "
                      "is not gaze, and it is not interest.",
                      "Каждое число на этой странице прочитано из out/metrics.json, "
                      "записанного этапом S8, вместе с доверительным интервалом и "
                      "покрытием. Ничего здесь не оценено на глаз. Внимание — это "
                      "поворот корпуса или головы в сторону витрины, а не взгляд и "
                      "не интерес.",
                      "本ページの数値はすべて、S8が出力した out/metrics.json から "
                      "信頼区間とカバレッジとともに読み込まれています。目視による "
                      "推定は一切ありません。「注目」とは店舗方向への体または頭の "
                      "向きであり、視線でも関心でもありません。"),

    "kpi.tracked":   ("People tracked", "Треков людей", "追跡した人数"),
    "kpi.turned":    ("Turned toward a storefront", "Повернулись к витрине",
                      "店舗の方を向いた"),
    "kpi.stopped":   ("Stopped near a storefront", "Остановились у витрины",
                      "店舗前で立ち止まった"),
    "kpi.garment":   ("Upper-garment class resolved", "Класс верха определён",
                      "上衣クラス判定済み"),

    "warn.head":     ("Read these three caveats before the percentages.",
                      "Три оговорки, которые надо прочесть до процентов.",
                      "割合を読む前に、以下の3点をお読みください。"),
    "warn.scale":    ("Scale comes from pedestrian height, not from the street.",
                      "Масштаб взят из роста пешеходов, а не из ширины улицы.",
                      "スケールは歩行者の身長由来であり、street幅ではありません。"),
    "warn.scale2":   ("The satellite reference (6.06 m) was rejected because it "
                      "implies a 1.93 m median height. Height is therefore no "
                      "longer an independent check; one independent check remains.",
                      "Спутниковый эталон 6.06 м отвергнут: он даёт медианный рост "
                      "1.93 м. Поэтому рост больше не независимая проверка; "
                      "осталась одна независимая проверка.",
                      "衛星基準6.06mは中央身長1.93mを意味するため棄却しました。"
                      "よって身長は独立検証ではなくなり、独立検証は1つ残ります。"),
    "warn.low":      ("Attention rates are low by construction.",
                      "Доли внимания низкие по построению.",
                      "注目率は定義上低くなります。"),
    "warn.low2":     ("A track counts only when the orientation ray actually "
                      "crosses the facade segment, grazing angles excluded. This "
                      "is a geometric test, not a guess about interest.",
                      "Трек засчитывается, только если луч ориентации реально "
                      "пересёк отрезок фасада; скользящие углы исключены. Это "
                      "геометрический тест, а не догадка об интересе.",
                      "方位レイが実際にファサード線分と交差した場合のみ計上し、"
                      "斜入射は除外します。幾何的判定であり、関心の推測ではありません。"),
    "warn.demo":     ("No demographics.", "Никакой демографии.", "属性推定なし。"),
    "warn.demo2":    ("Gender, age and ethnicity are not estimated and will not "
                      "be: privacy, and no ground truth to gate them against.",
                      "Пол, возраст и этничность не определяются и определяться не "
                      "будут: приватность и отсутствие разметки для гейта.",
                      "性別・年齢・民族は推定せず、今後も行いません。"
                      "プライバシーと、検証用の正解データがないためです。"),

    "kpi.n.tracked": ("frames processed", "кадров обработано", "処理フレーム"),
    "kpi.n.turned":  ("of tracks · body/head turn, NOT gaze",
                      "от треков · поворот корпуса или головы, НЕ взгляд",
                      "の割合 · 体/頭の向きであり視線ではない"),
    "kpi.n.stopped": ("speed below threshold inside an apron polygon",
                      "скорость ниже порога внутри прифасадной полосы",
                      "店前帯内で速度が閾値未満"),
    "kpi.n.garment": ("of tracks · white balance applied, accuracy NOT validated",
                      "от треков · баланс белого применён, точность НЕ валидирована",
                      "の割合 · ホワイトバランス適用、精度は未検証"),
    "cap.zone":  ("stratified by distance to the facade · faces anonymised before "
                  "anything is written to disk",
                  "стратифицировано по дистанции до фасада · лица обезличены до "
                  "записи на диск",
                  "ファサードまでの距離で層化 · 保存前に顔を匿名化"),
    "cap.color": ("stratified by agreement; class chosen by vote across up to 8 crops",
                  "стратифицировано по согласию; класс выбран голосованием по 8 кропам",
                  "一致度で層化。最大8クロップの多数決でクラスを決定"),
    "cap.arch":  ("click any frame to open it full size",
                  "клик по кадру открывает его крупно",
                  "画像をクリックすると拡大表示"),
    "cap.frames": ("frames of", "кадров от", "枚 /"),
    "cap.people": ("people", "человек", "人"),
    "cap.person": ("person", "человека", "人"),
    "sec.zones.eyebrow": ("Per storefront", "По витринам", "店舗別"),
    "sec.zones.title":   ("Attention by storefront", "Внимание по витринам",
                          "店舗別の注目度"),
    "sec.zones.sub":     ("The grid under each card shows the actual people the "
                          "number counts. Click any frame to open it full size.",
                          "Сетка под карточкой показывает тех самых людей, которых "
                          "считает число. Клик по кадру открывает его крупно.",
                          "各カードの下のグリッドは、その数値が数えた本人たちです。"
                          "画像をクリックすると拡大표示されます。"),
    "z.entered":   ("tracks entered the zone", "треков вошло в зону",
                    "ゾーンに入った追跡数"),
    "z.attn":      ("median attention time", "медианное время внимания",
                    "注目時間の中央値"),
    "z.dwell":     ("median time in zone", "медианное время в зоне",
                    "ゾーン滞在時間の中央値"),
    "z.counted":   ("tracks counted as turned", "треков засчитано повёрнутыми",
                    "「向いた」と判定された追跡数"),
    "z.turned":    ("turned toward it", "повернулись к ней", "この店舗を向いた"),
    "z.empty":     ("Nobody was counted as turned toward this storefront, so there "
                    "is nothing to show. An empty grid here is the correct picture, "
                    "not a missing feature.",
                    "К этой витрине никто не засчитан повёрнутым, показывать нечего. "
                    "Пустая сетка здесь — верная картина, а не недоделка.",
                    "この店舗を向いたと判定された人はいないため、表示するものが "
                    "ありません。空のグリッドは正しい結果であり、欠陥ではありません。"),

    "sec.fn.eyebrow": ("Funnel", "Воронка", "ファネル"),
    "sec.fn.title":   ("From passer-by to attention", "От прохожего до внимания",
                       "通行人から注目まで"),
    "sec.fn.sub":     ("Each step is a SUBSET of the one above it. The share is of "
                       "the top step, not of the previous one.",
                       "Каждая ступень — ПОДМНОЖЕСТВО предыдущей. Доля считается от "
                       "верхней ступени, а не от предыдущей.",
                       "各段は上段の部分集合です。割合は最上段に対する値です。"),
    "fn.tracked":  ("People tracked", "Треков людей", "追跡した人数"),
    "fn.entered":  ("Entered a storefront zone", "Вошли в зону витрины",
                    "店舗ゾーンに入った"),
    "fn.turned":   ("Turned toward a storefront", "Повернулись к витрине",
                    "店舗の方を向いた"),
    "fn.stopped":  ("Stopped there", "Остановились там", "そこで立ち止まった"),
    "sec.dist.eyebrow": ("Distribution", "Распределение", "分布"),
    "sec.dist.title":   ("Where people went and how long they stayed",
                         "Куда люди шли и сколько оставались",
                         "人の流れと滞在時間"),
    "dist.byzone":  ("Visitors by zone", "Посетители по зонам", "ゾーン別の来訪者"),
    "unit.visits":  ("VISITS", "ВХОДЫ", "入場回数"),
    "unit.tracks":  ("TRACKS", "ТРЕКИ", "追跡数"),
    "dist.sum.a":   ("people made", "человек совершили", "人が"),
    "dist.sum.b":   ("zone entries — on average", "входов в зоны — в среднем",
                     "回ゾーンに入場 — 平均"),
    "dist.sum.c":   ("zones per person: the street is a through route, people pass "
                     "several storefronts.",
                     "зоны на человека: улица сквозная, люди проходят мимо "
                     "нескольких витрин.",
                     "ゾーン/人。この通りは通り抜けであり、人は複数の店舗の前を "
                     "通過します。"),
    "dist.dwell":   ("Time in zone", "Время в зоне", "ゾーン滞在時間"),
    "dist.dwell.u": ("number of ZONE VISITS in each bucket, not people: one person "
                     "entering three zones counts three times",
                     "число ПОСЕЩЕНИЙ ЗОН в каждой корзине, а не людей: один "
                     "человек, зашедший в три зоны, считается трижды",
                     "各区間の「ゾーン訪問回数」であり人数ではありません。"
                     "3ゾーンに入った1人は3回として数えます"),
    "dist.attn.u":  ("number of TRACKS in each bucket, one track counted once",
                     "число ТРЕКОВ в каждой корзине, трек считается один раз",
                     "各区間の追跡数。1追跡は1回のみ計上"),
    "dist.byzone.u": ("tracks that entered each zone; one track can enter several",
                      "треков, вошедших в каждую зону; один трек может войти "
                      "в несколько",
                      "各ゾーンに入った追跡数。1追跡が複数ゾーンに入ることがあります"),
    "dist.attn":    ("Attention time", "Время внимания", "注目時間"),
    "sec.foot.eyebrow": ("Footfall", "Поток", "人流"),
    "sec.foot.title":   ("Presence over time", "Присутствие во времени",
                         "時間帯別の在圏"),
    "sec.foot.sub":     ("Mean simultaneous detections per 10-second bin, from the "
                         "presence curve computed by S8.",
                         "Среднее число одновременных детекций в 10-секундном "
                         "интервале, из кривой присутствия этапа S8.",
                         "10秒ビンごとの同時検出数の平均。S8の在圏カーブより。"),

    "sec.cloth.eyebrow": ("Clothing", "Одежда", "服装"),
    "sec.cloth.title":   ("Upper-garment lightness", "Светлота верхней одежды",
                          "上衣の明度"),

    "cl.head": ("Colour is corrected for the camera's cast, and NOT validated.",
                "Цвет скорректирован под сдвиг камеры и НЕ валидирован.",
                "色はカメラの色被りを補正済み、ただし未検証。"),
    "cl.why":  ("The first run called 21 of 25 tracks «blue» — every one with hue "
                "107–124, which is exactly the median hue of the whole frame (110). "
                "It was classifying the camera's colour cast, not clothing.",
                "Первый прогон назвал синими 21 трек из 25 — и все с тоном 107–124, "
                "ровно медианный тон всего кадра (110). Классифицировался цветовой "
                "сдвиг камеры, а не одежда.",
                "初回は25件中21件を「青」と判定し、いずれも色相107〜124 — "
                "これは全画面の中央色相110そのものです。衣服ではなくカメラの "
                "色被りを分類していました。"),
    "cl.gains": ("Fixed by white balance on the road surface inside the ROI, not by "
                 "raising a threshold. Channel gains B/G/R:",
                 "Исправлено балансом белого по мостовой внутри ROI, а не поднятием "
                 "порога. Коэффициенты каналов B/G/R:",
                 "閾値を上げるのではなく、ROI内の路面でホワイトバランスを取って "
                 "補正。チャネル係数 B/G/R:"),
    "cl.sat":  ("Road saturation before and after:", "Насыщенность мостовой до и после:",
                "路面の彩度、補正前と補正後:"),
    "cl.sat2": ("The drop proves the correction was APPLIED, not that it is CORRECT — "
                "the gains were computed from that same road. The only non-circular "
                "check is manual labelling.",
                "Падение доказывает, что коррекция ПРИМЕНИЛАСЬ, а не что она ВЕРНА: "
                "коэффициенты считались из этой же мостовой. Единственная "
                "некольцевая проверка — ручная разметка.",
                "この低下は補正が「適用された」証拠であり、「正しい」証拠では "
                "ありません。係数は同じ路面から算出されています。非循環的な検証は "
                "手作業のラベル付けのみです。"),
    "cl.chroma": ("Tracks with real chroma, above the achromatic floor:",
                  "Треков с реальной хромой, выше порога ахроматичности:",
                  "無彩色しきい値を超える、実際に彩度のある追跡数:"),
    "cl.notvalid": ("Accuracy is NOT MEASURED.", "Точность НЕ ИЗМЕРЕНА.",
                    "精度は未計測です。"),
    "cl.notvalid2": ("There are no labelled crops. top_color_conf is the share of "
                     "crops that voted for the class times the share of agreeing "
                     "pixels — not a probability of being right.",
                     "Размеченных кропов нет. top_color_conf — это доля кадров за "
                     "класс, умноженная на долю согласных пикселей, а не "
                     "вероятность правильности.",
                     "ラベル付きクロップがありません。top_color_conf はクラスに "
                     "投票したクロップの割合×一致画素の割合であり、正解確率では "
                     "ありません。"),
    "sec.geo.eyebrow": ("Geometry", "Геометрия", "幾何"),
    "sec.geo.title":   ("How this works, and how to check it",
                        "Как это работает и как это проверить",
                        "仕組みと検証方法"),
    "sec.geo.sub":     ("The calibration check you can make with your own eyes: the "
                        "street is straight, so trajectories on the ground plane "
                        "must run straight and parallel along it.",
                        "Проверка калибровки, которую можно сделать глазами: улица "
                        "прямая, значит траектории на плане обязаны идти прямо и "
                        "параллельно вдоль неё.",
                        "目視でできる較正チェック：通りは直線なので、平面図上の "
                        "軌跡も直線かつ平行に並ぶはずです。"),

    "sec.ev.eyebrow": ("Evidence", "Доказательства", "根拠"),
    "sec.ev.title":   ("Proof archive", "Архив пруфов", "証跡アーカイブ"),
    "sec.ev.sub":     ("Every claim that carries frames, with the frames themselves. "
                       "Selection is stratified, never top-N: a top-N grid would "
                       "systematically flatter the number it illustrates. Faces are "
                       "pixelated and blurred before anything reaches the disk.",
                       "Каждое утверждение, за которым стоят кадры, вместе с самими "
                       "кадрами. Отбор стратифицированный, а не top-N: сетка из "
                       "лучших систематически льстила бы числу. Лица пикселизуются "
                       "и размываются до записи на диск.",
                       "画像を伴うすべての主張と、その画像そのもの。選択は層化 "
                       "抽出であり上位N件ではありません。上位N件では数値を系統的に "
                       "良く見せてしまいます。顔は保存前にモザイクとぼかしを適用。"),

    "arch.stage":  ("stage", "этап", "ステージ"),
    "arch.frames": ("frames in the index", "кадров в индексе", "枚（索引内）"),
    "sec.hon.eyebrow": ("Honesty", "Честность", "透明性"),
    "sec.hon.title":   ("What is not measured", "Что не измерено", "未計測の項目"),
    "sec.hon.title2":  ("Limitations", "Ограничения", "制約"),
    "th.quantity":  ("Quantity", "Величина", "項目"),
    "th.whynone":   ("Why it has no number", "Почему числа нет", "数値がない理由"),
    "th.item":      ("Item", "Пункт", "項目"),
    "th.what":      ("What is going on", "Что происходит", "内容"),
    "th.conseq":    ("Consequence", "Следствие", "影響"),
}


#: Ключи, которые добавляют отдельные страницы. Живут рядом с базовым
#: словарём, чтобы t() и i18n_payload() видели ОДНО И ТО ЖЕ: иначе строка
#: попадёт в разметку, но не попадёт в словарь, и не переведётся.
RUNTIME_I18N: dict[str, tuple] = {}


def register_i18n(extra: dict) -> None:
    """Страница объявляет свои переводимые строки перед сборкой."""
    RUNTIME_I18N.update(extra)


def t(key: str) -> str:
    """Переводимая строка. В HTML кладётся английский, остальное — на клиенте."""
    row = I18N.get(key) or RUNTIME_I18N.get(key)
    if row is None:
        raise KeyError(f"нет перевода для {key!r}: зарегистрируйте его через "
                       f"register_i18n, иначе строка не переведётся")
    return f'<span data-i18n="{key}">{esc(row[0])}</span>'


CSS = """
:root{
  --bg:#f5f6f8; --panel:#fff; --ink:#0f1622; --muted:#6b7482; --line:#e6e9ef;
  --accent:#2f6bff; --accent-soft:#eaf0ff; --warn-bg:#fff8e6; --warn-line:#f2d492;
  --warn-ink:#6b4f10; --shadow:0 1px 2px rgba(16,24,40,.05),0 8px 24px rgba(16,24,40,.05);
}
:root[data-theme="dark"]{
  --bg:#0c1017; --panel:#151b26; --ink:#e8ecf3; --muted:#93a0b4; --line:#232c3b;
  --accent:#5b8bff; --accent-soft:#1a2440; --warn-bg:#2a2311; --warn-line:#5c4a1c;
  --warn-ink:#f0d79a; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
a{color:var(--accent)}
.wrap{max-width:1280px;margin:0 auto;padding:0 24px 80px}
header{position:sticky;top:0;z-index:40;background:var(--panel);
  border-bottom:1px solid var(--line)}
/* Шапка держится в ОДНУ полоску. Прежний flex-wrap ронял языки и тему на
   вторую строку, как только подписи удлинялись: RU и JA длиннее EN. Чипы
   ужимаются и прячутся на узких экранах, навигация не ужимается никогда. */
.hd{max-width:1280px;margin:0 auto;padding:12px 20px;display:flex;
  align-items:center;gap:10px;flex-wrap:nowrap}
.hd .chip{flex:0 1 auto;min-width:0}
.hd .chip .v{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.navgrp{display:flex;align-items:center;gap:9px;flex:none;margin-left:auto}
@media(max-width:1180px){.hd .chip:not(.keep){display:none}}
@media(max-width:900px){.hd .chip{display:none}}
/* На телефоне шапка ОБЯЗАНА переноситься: держать её в одну строку там
   означало бы вытолкнуть кнопки за экран. Одна полоска — требование
   настольной ширины, а не всех подряд. */
@media(max-width:720px){
  .hd{flex-wrap:wrap;row-gap:8px;padding:10px 14px}
  .navgrp{margin-left:0;width:100%;justify-content:flex-start;
    /* min-width:0 обязателен: flex-элемент по умолчанию не сжимается ниже
       ширины содержимого, и overflow-x на нём не включается — вместо
       прокрутки внутри группы разъезжалось всё тело страницы. */
    min-width:0;overflow-x:auto;-webkit-overflow-scrolling:touch;
    padding-bottom:2px}
  .navgrp::-webkit-scrollbar{height:0}
  .navgrp .btn{flex:none}
  .logo{font-size:16px}
}
.logo{font-weight:800;letter-spacing:.22em;font-size:19px}
.chip{border:1px solid var(--line);border-radius:12px;padding:7px 14px;line-height:1.25}
.chip .k{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.chip .v{font-weight:650;font-size:14px}
.sp{flex:1}
.langs{display:inline-flex;border:1px solid var(--line);border-radius:999px;
  overflow:hidden}
.lang{border:0;background:var(--panel);color:var(--muted);font:inherit;
  font-weight:700;font-size:12px;padding:8px 12px;cursor:pointer}
.lang.on{background:var(--accent);color:#fff}
/* Переключатель темы — иконка, а не надпись: он крайний справа и не должен
   переезжать на вторую строку вместе с длинными подписями языков. */
.icobtn{width:38px;height:38px;padding:0;border-radius:50%;display:inline-flex;
  align-items:center;justify-content:center;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);cursor:pointer;flex:none}
.icobtn:hover{border-color:var(--accent);color:var(--accent)}
.icobtn svg{width:19px;height:19px;display:block}
/* Солнце видно в светлой теме, луна в тёмной: кнопка показывает ТЕКУЩЕЕ
   состояние, а не то, куда переключит. */
.icobtn .moon{display:none}
:root[data-theme="dark"] .icobtn .sun{display:none}
:root[data-theme="dark"] .icobtn .moon{display:block}
.btn{border:1px solid var(--line);background:var(--panel);color:var(--ink);
  border-radius:999px;padding:8px 16px;font:inherit;font-weight:600;font-size:13px;
  cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;gap:7px}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn.pri{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.pri:hover{opacity:.9;color:#fff}
h1{font-size:36px;line-height:1.15;margin:34px 0 8px;letter-spacing:-.02em}
h2{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.eyebrow{font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--accent);font-weight:700;margin-bottom:6px}
.sub{color:var(--muted);margin:0 0 26px;max-width:78ch}
section{margin:34px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;
  padding:20px;box-shadow:var(--shadow)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(232px,1fr));gap:16px}
.kpi .lab{font-size:12px;color:var(--muted);margin-bottom:8px}
.kpi .big{font-size:42px;font-weight:750;letter-spacing:-.03em;line-height:1}
.kpi .note{font-size:12px;color:var(--muted);margin-top:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(292px,1fr));gap:16px}
.zname{display:flex;align-items:flex-start;gap:9px;font-weight:700;font-size:16px;
  line-height:1.3}
.zname .dot{margin-top:5px}
/* Длинное название витрины переносилось и сдвигало процент вниз, из-за чего
   карточки выходили разной высоты и сетка выглядела сломанной. Фиксируем
   высоту блока имени: две строки помещаются, третья обрезается многоточием. */
.zhead{min-height:44px;display:flex;align-items:flex-start;gap:9px}
.zhead .nm{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
  overflow:hidden}
.card{display:flex;flex-direction:column}
.dot{width:11px;height:11px;border-radius:3px;flex:none}
.pct{font-size:46px;font-weight:750;letter-spacing:-.03em;line-height:1;margin:12px 0 2px}
.ci{font-size:12px;color:var(--muted)}
.mini{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:14px 0 4px;
  padding-top:14px;border-top:1px solid var(--line)}
.mini .n{font-size:19px;font-weight:700}
.mini .l{font-size:11px;color:var(--muted)}
.sheet{display:grid;grid-template-columns:repeat(6,1fr);gap:5px;margin-top:12px}
.sheet img{width:100%;height:88px;object-fit:cover;border-radius:5px;
  background:var(--bg);cursor:zoom-in;border:1px solid var(--line)}
.cap{font-size:11px;color:var(--muted);margin-top:8px}
.none{font-size:12px;color:var(--muted);border:1px dashed var(--line);
  border-radius:9px;padding:14px;text-align:center;margin-top:12px}
.bar{display:flex;height:44px;border-radius:10px;overflow:hidden;border:1px solid var(--line)}
.bar div{display:flex;align-items:center;justify-content:center;font-size:12px;
  font-weight:700;color:#0f1622;min-width:2px}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;font-size:13px}
.legend span{display:inline-flex;align-items:center;gap:7px}
.warn{background:var(--warn-bg);border:1px solid var(--warn-line);color:var(--warn-ink);
  border-radius:14px;padding:16px 18px;font-size:13.5px}
.warn b{font-size:14.5px}
.warn div{margin:6px 0}
img.fig{width:100%;border-radius:11px;border:1px solid var(--line);cursor:zoom-in;
  background:var(--panel);display:block}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);
  vertical-align:top}
th{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);
  font-weight:600}
code{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;
  background:var(--accent-soft);padding:1px 5px;border-radius:4px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}
@media(max-width:820px){.two{grid-template-columns:1fr}
  .sheet{grid-template-columns:repeat(4,1fr)}}


.fn{display:grid;grid-template-columns:210px 1fr 120px;gap:10px;align-items:center;
  padding:9px 0;border-bottom:1px solid var(--line)}
.fn:last-child{border-bottom:0}
.fn-l{font-weight:650;font-size:13.5px}
.fn-bar{background:var(--accent-soft);border-radius:6px;height:16px;overflow:hidden}
.fn-bar i{display:block;height:100%;background:var(--accent)}
.fn-v{text-align:right;font-weight:750;font-variant-numeric:tabular-nums}
.fn-v small{color:var(--muted);font-weight:600;font-size:11px}
.fn-n{grid-column:1/-1;font-size:11px;color:var(--muted);margin-top:-4px}
.hb{display:grid;grid-template-columns:150px 1fr 54px;gap:10px;align-items:center;
  padding:7px 0}
.hb-l{font-size:13px}
.hb-bar{background:var(--accent-soft);border-radius:5px;height:12px;overflow:hidden}
.hb-bar i{display:block;height:100%;background:var(--accent)}
.hb-v{text-align:right;font-weight:700;font-variant-numeric:tabular-nums;font-size:13px}
.three{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
/* Единица измерения обязана быть на самой карточке: рядом стоят числа,
   считающие РАЗНОЕ, и без метки читатель их сложит. */
.card{position:relative}
.unit{position:absolute;top:14px;right:14px;font-size:10px;font-weight:800;
  letter-spacing:.09em;padding:3px 8px;border-radius:999px;
  background:var(--accent-soft);color:var(--accent)}
.dleg{display:flex;flex-direction:column;gap:7px;font-size:12.5px}
.dleg span{display:inline-flex;align-items:center;gap:8px}
#lb{position:fixed;inset:0;background:rgba(8,11,17,.93);display:none;z-index:99;
  align-items:center;justify-content:center;flex-direction:column;gap:14px;
  padding:28px;cursor:zoom-out}
#lb.on{display:flex}
/* Кропы крошечные — 30x98 у дальних людей. Заданные только максимумы
   означали, что картинка открывается 1:1, то есть маркой на весь экран,
   хотя подпись обещает «открыть крупно». Тянем по высоте.
   image-rendering:pixelated намеренно: увеличение в 6-8 раз сглаживанием
   ДОРИСОВАЛО БЫ детали, которых в данных нет. Честнее показать пиксели. */
#lb img{max-width:94vw;max-height:86vh;width:auto;height:auto;object-fit:contain;
  border-radius:9px;background:#000}
/* Мелкие кропы тянем по высоте и показываем пикселями: увеличение в семь раз
   сглаживанием дорисовало бы детали, которых в данных нет. Широкие кадры
   оверлея (16:9) наоборот тянем по ширине и сглаживаем. */
#lb img.small{height:min(78vh,900px);image-rendering:pixelated}
#lb .meta{color:#c9d3e2;font-size:13px;text-align:center;max-width:80ch}

/* ---- бургер ------------------------------------------------------------ */
/* На телефоне восемь кнопок в строку не помещаются никак: горизонтальная
   прокрутка внутри шапки прятала половину из них за краем, и найти язык или
   тему было нельзя. Всё уезжает в меню. */
.burger{display:none;width:38px;height:38px;padding:0;border-radius:11px;
  align-items:center;justify-content:center;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);cursor:pointer;flex:none;
  margin-left:auto}
.burger svg{width:20px;height:20px;display:block}
.burger .x{display:none}
.burger[aria-expanded="true"] .x{display:block}
.burger[aria-expanded="true"] .bars{display:none}
.burger[aria-expanded="true"]{border-color:var(--accent);color:var(--accent)}
@media(max-width:720px){
  .hd{position:relative}
  .burger{display:inline-flex}
  .navgrp{display:none;position:absolute;top:100%;left:0;right:0;
    flex-direction:column;align-items:stretch;gap:8px;
    background:var(--panel);border-bottom:1px solid var(--line);
    padding:12px 14px 16px;box-shadow:var(--shadow);z-index:50;
    margin-left:0;width:auto;overflow:visible}
  .navgrp.open{display:flex}
  .navgrp .btn{justify-content:center;padding:11px 16px;font-size:14px}
  .navgrp .lbl{display:inline}
  .langs{align-self:stretch}
  .lang{flex:1;padding:11px 0;font-size:13px}
  .icobtn{width:auto;height:auto;border-radius:999px;padding:11px 16px;
    gap:9px;font:inherit;font-weight:600;font-size:14px}
  .icobtn::after{content:attr(data-label)}
}

/* ---- адаптив ---------------------------------------------------------- */
/* Широкое содержимое прокручивается ВНУТРИ своего контейнера. Тело страницы
   не должно ехать по горизонтали ни на одной ширине: у бенчмарка таблица на
   четыре колонки выносила body на 613 px при экране 390. */
.tbl{overflow-x:auto;-webkit-overflow-scrolling:touch;
  /* .card это flex-контейнер, а flex-элемент не сжимается ниже ширины
     содержимого без min-width:0 — та же причина, по которой не включалась
     прокрутка в шапке. Без него таблица 540 px разъезжала всю страницу. */
  min-width:0;max-width:100%}
.tbl table{min-width:540px}
@media(max-width:900px){
  .wrap{padding:0 16px 60px}
  h1{font-size:30px;margin-top:26px}
  h2{font-size:20px}
  .card{padding:16px}
  .fn{grid-template-columns:1fr;gap:4px}
  .fn-bar{order:3}
  .fn-v{text-align:left;order:2}
  .hb{grid-template-columns:110px 1fr 46px}
}
@media(max-width:560px){
  .wrap{padding:0 12px 48px}
  h1{font-size:25px;line-height:1.2}
  h2{font-size:18px}
  .kpi .big{font-size:34px}
  .pct{font-size:38px}
  .card{padding:14px;border-radius:13px}
  .sheet{grid-template-columns:repeat(3,1fr);gap:4px}
  .sheet img{height:74px}
  .mini{grid-template-columns:1fr 1fr;gap:8px}
  .bar{height:38px}
  .legend{gap:10px;font-size:12px}
  .gal{grid-template-columns:1fr}
  #lb{padding:14px}
  #lb img{max-height:74vh}
  .unit{top:10px;right:10px}
}
@media(max-width:380px){
  .sheet{grid-template-columns:repeat(2,1fr)}
  .hb{grid-template-columns:1fr 1fr}
  .hb-bar{grid-column:1/-1}
}
"""

JS = """
(function(){
  var root=document.documentElement, KEY='looq-theme';
  try{var s=localStorage.getItem(KEY); if(s) root.setAttribute('data-theme',s);}catch(e){}
  document.getElementById('theme').onclick=function(){
    var d=root.getAttribute('data-theme')==='dark';
    root.setAttribute('data-theme', d?'light':'dark');
    try{localStorage.setItem(KEY, d?'light':'dark');}catch(e){}
  };
  var I18N = window.__I18N__ || {};
  function setLang(code){
    document.querySelectorAll('[data-i18n]').forEach(function(el){
      var row = I18N[el.getAttribute('data-i18n')];
      if (row && row[code]) el.textContent = row[code];
    });
    document.querySelectorAll('.lang').forEach(function(b){
      b.classList.toggle('on', b.dataset.lang===code);
    });
    document.documentElement.lang = code;
    var th = document.getElementById('theme');
    var row = I18N['nav.theme'];
    if (th && row && row[code]) { th.title = row[code];
      th.setAttribute('aria-label', row[code]); th.dataset.label = row[code]; }
    var bg=document.getElementById('burger'), mrow=I18N['nav.menu'];
    if (bg && mrow && mrow[code]) { bg.title = mrow[code];
      bg.setAttribute('aria-label', mrow[code]); }
    try{localStorage.setItem('looq-lang', code);}catch(e){}
  }
  document.querySelectorAll('.lang').forEach(function(b){
    b.onclick = function(){ setLang(b.dataset.lang); };
  });
  try{var L=localStorage.getItem('looq-lang'); if(L) setLang(L);}catch(e){}


  var burger=document.getElementById('burger'), nav=document.querySelector('.navgrp');
  if (burger && nav){
    burger.onclick=function(e){
      e.stopPropagation();
      var open=nav.classList.toggle('open');
      burger.setAttribute('aria-expanded', open?'true':'false');
    };
    // Клик по пункту закрывает меню: иначе после выбора языка панель
    // остаётся раскрытой и закрывает собой саму страницу.
    nav.addEventListener('click',function(e){
      if (e.target.closest('a')) close();
    });
    document.addEventListener('click',function(e){
      if (nav.classList.contains('open') && !nav.contains(e.target)
          && e.target!==burger) close();
    });
    document.addEventListener('keydown',function(e){
      if (e.key==='Escape') close();
    });
  }
  function close(){
    if (!nav) return;
    nav.classList.remove('open');
    if (burger) burger.setAttribute('aria-expanded','false');
  }

  var lb=document.getElementById('lb'), im=document.getElementById('lbimg'),
      mt=document.getElementById('lbmeta');
  document.addEventListener('click',function(e){
    var t=e.target;
    if(t.tagName==='IMG'&&(t.classList.contains('fig')||t.closest('.sheet'))){
      im.classList.remove('small');
      im.src=t.currentSrc||t.src;
      // Решаем ПОСЛЕ загрузки самой модалки: у ленивой миниатюры на момент
      // клика naturalWidth ещё нулевой, и кадр 1920x1080 попадал в режим
      // для крошечных кропов.
      var decide=function(){ im.classList.toggle('small', im.naturalWidth < 400); };
      if (im.complete && im.naturalWidth) decide(); else im.onload = decide;
      // Настоящий размер кропа обязан быть виден: увеличенная в семь раз
      // марка не должна выглядеть как снимок высокого разрешения.
      var px = (t.naturalWidth&&t.naturalHeight)
             ? t.naturalWidth+'×'+t.naturalHeight+' px · ' : '';
      mt.textContent = px + (t.dataset.meta||t.alt||'');
      lb.classList.add('on');
    } else if(lb.classList.contains('on')){ lb.classList.remove('on'); }
  });
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape') lb.classList.remove('on');
  });
})();
"""


#: Навигация между страницами. ОДНА таблица на все страницы: шапка была
#: скопирована в пять мест по-разному, и на трёх из них не было ни языков,
#: ни переключателя темы.
NAV = [("dashboard.html", "nav.dash"), ("replay.html", "nav.replay"),
       ("overlay.mp4", "nav.overlay"), ("frames.html", "nav.frames"),
       ("benchmark.html", "nav.bench"), ("report.html", "nav.report")]

THEME_SVG = (
    '<svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4.2"/>'
    '<path d="M12 2v2.6M12 19.4V22M2 12h2.6M19.4 12H22M4.9 4.9l1.9 1.9'
    'M17.2 17.2l1.9 1.9M19.1 4.9l-1.9 1.9M6.8 17.2l-1.9 1.9"/></svg>'
    '<svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20.5 14.3A8.5 8.5 0 1 1 9.7 3.5a6.8 6.8 0 0 0 10.8 10.8z"/></svg>')


#: Ссылка на видео в публичной сборке. mp4 в репозиторий не кладётся, и
#: относительная ссылка на него на Pages вела бы в 404.
PUBLIC_VIDEO_URL = "https://youtu.be/3ZXDQcmrOUI"

#: Что публикуется на Pages. Остальные страницы существуют только локально,
#: и вести на них из публичной шапки значило бы обещать несуществующее.
PUBLIC_PAGES = {"dashboard.html", "benchmark.html", "report.html"}

#: Выставляется один раз из main() при --public. Модульный, а не параметр:
#: header_html зовут четыре скрипта, и протаскивать флаг через все — шум.
PUBLIC_BUILD = False


def set_public_build(on: bool) -> None:
    global PUBLIC_BUILD
    PUBLIC_BUILD = bool(on)


def header_html(current: str, chips) -> str:
    """Шапка страницы. current — имя текущего файла, чтобы не вести на себя."""
    ch = "".join(
        f'<div class="chip{" keep" if i == 0 else ""}">'
        f'<div class="k">{t(k)}</div><div class="v">{esc(v)}</div></div>'
        for i, (k, v) in enumerate(chips))
    links = []
    for href, key in NAV:
        if href == current:
            continue
        if PUBLIC_BUILD:
            # Видео уезжает на YouTube, остальное неопубликованное выпадает.
            if key == "nav.overlay":
                links.append(f'<a class="btn pri" href="{PUBLIC_VIDEO_URL}" '
                             f'target="_blank" rel="noopener">&#9654; '
                             f'<span class="lbl">{t(key)}</span></a>')
                continue
            if href not in PUBLIC_PAGES:
                continue
        pri = " pri" if key == "nav.replay" else ""
        icon = "&#9654; " if key == "nav.replay" else ""
        links.append(f'<a class="btn{pri}" href="{href}">{icon}'
                     f'<span class="lbl">{t(key)}</span></a>')
    return f"""<header><div class="hd">
  <div class="logo">LOOQ</div>{ch}
  <button class="burger" id="burger" aria-expanded="false"
          aria-label="{I18N["nav.menu"][0]}"
          title="{I18N["nav.menu"][0]}"><svg class="bars" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg><svg class="x" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
  <span class="navgrp">{''.join(links)}
    <span class="langs">
      <button class="lang on" data-lang="en">EN</button>
      <button class="lang" data-lang="ru">RU</button>
      <button class="lang" data-lang="ja">JA</button>
    </span>
    <button class="icobtn" id="theme" title="{I18N["nav.theme"][0]}"
            aria-label="{I18N["nav.theme"][0]}"
            data-label="{I18N["nav.theme"][0]}">{THEME_SVG}</button>
  </span>
</div></header>"""


def i18n_payload(extra=None) -> str:
    """Словарь переводов в страницу. Общий для всех страниц."""
    d = {**I18N, **RUNTIME_I18N, **(extra or {})}
    return json.dumps({k: {"en": v[0], "ru": v[1], "ja": v[2]}
                       for k, v in d.items()}, ensure_ascii=False)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _priv_top_frac() -> float:
    """Порог обезличивания, который применяется ПРИ ВСТРАИВАНИИ."""
    return float((load_config("configs/evidence.yaml").get("privacy") or {})
                 .get("face_blur_top_frac", 0.0))


def b64_img(path: Path, max_w: int | None = None, quality: int = 92,
            anonymise: bool = False) -> str | None:
    """Картинка в data-URI. Пруфы вшиваются ПО ОДНОМУ, а не монтажом: монтаж
    ужимает кропы, и мелкий человек превращается в кашу.

    anonymise=True — прогнать кроп через обезличивание ЕЩЁ РАЗ, по текущему
    порогу из configs/evidence.yaml. Кропы на диске писались разными прогонами
    и несут разные пороги: часть сделана при 0.22, который владелец потом
    поднял до 0.30. Страница обязана показывать текущий порог независимо от
    того, когда кроп записан, а повторное размытие уже размытого безвредно.
    """
    img = cv2.imread(str(path))
    if img is None:
        return None
    if anonymise:
        priv = load_config("configs/evidence.yaml").get("privacy") or {}
        need = ("face_blur_top_frac", "blur_kernel_frac", "blur_sigma_frac",
                "pixelate_factor")
        missing = [k for k in need if k not in priv]
        if missing:
            # Правило 8: молча отдать кроп с диска, не зная порога, значило бы
            # опубликовать его с тем размытием, какое случайно оказалось.
            raise SystemExit(
                f"в configs/evidence.yaml нет ключей приватности {missing}: "
                f"вшивать пруфы без порога обезличивания нельзя")
        try:
            img, _ = blur_face_region(
                img,
                top_frac=float(priv["face_blur_top_frac"]),
                kernel_frac=float(priv["blur_kernel_frac"]),
                sigma_frac=float(priv["blur_sigma_frac"]),
                pixelate_factor=int(priv["pixelate_factor"]))
        except EvidenceError:
            pass          # область уже однородна: кроп обезличен своей стадией
    if max_w and img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            if ok else None)


def _n_labels() -> int:
    """Сколько людей реально размечено — по самому большому файлу в labels/.

    Тот же выбор, что делают verify_s5.py и make_benchmark.py: наибольший файл.
    Число нельзя вписывать в текст руками — оно меняется с каждой разметкой.
    """
    d = Path("labels")
    files = sorted(d.glob("s5_orient_*.jsonl")) if d.is_dir() else []
    if not files:
        return 0
    rows = max(files, key=lambda q: q.stat().st_size).read_text(
        encoding="utf-8").splitlines()
    return sum(1 for x in rows[1:] if x and json.loads(x).get("label") is not None)


def sheet_html(rows, note: str) -> str:
    """Сетка пруфов. В подписи ОБЯЗАТЕЛЬНО число разных людей: двенадцать
    кадров одного человека читаются как двенадцать человек, и это ровно та
    ошибка, из-за которой сетку у M2 приняли за 12 посетителей."""
    cells = []
    for r in rows:
        uri = b64_img(Path(r["path"]), anonymise=True)
        if uri is None:
            continue
        try:
            ex = json.loads(r.get("extra_json") or "{}")
        except (ValueError, TypeError):
            ex = {}
        # blur_top_frac из индекса — это порог, с которым кроп ЗАПИСАН, а
        # показывается он переразмытым по текущему. Печатать записанный
        # означало бы подписать под пикселями чужое число, на странице,
        # которая продаёт прослеживаемость. Показываем оба.
        written = ex.pop("blur_top_frac", None)
        meta = (f"track #{int(r['track_id'])} · frame {int(r['frame_idx'])} · "
                f"t={float(r['ts']):.1f}s · stratum {int(r['stratum'])}"
                + (f" · {json.dumps(ex, ensure_ascii=False)}" if ex else "")
                + (f" · blur {_priv_top_frac():.2f} applied here"
                   f" (written at {float(written):.2f})" if written is not None
                   else ""))
        cells.append(f'<img src="{uri}" alt="proof frame" data-meta="{esc(meta)}">')
    if not cells:
        return '<div class="none">no proof frames for this claim</div>'
    n_people = len({int(r["track_id"]) for r in rows})
    word = t("cap.person") if n_people == 1 else t("cap.people")
    return (f'<div class="sheet">{"".join(cells)}</div>'
            f'<div class="cap"><b>{len(cells)} {t("cap.frames")} {n_people} {word}</b>'
            f' · {note}</div>')


def collect_zone_proofs(video: Path, n: int, nonzero: set[str]) -> dict:
    """Кропы треков, повёрнутых к каждой витрине. Через EvidenceWriter."""
    import pandas as pd

    zf = pd.read_parquet("attn/track_zone_frames.parquet")
    det = pd.read_parquet("det/frames.parquet")
    tracks = pd.read_parquet("track/tracks.parquet")
    events = pd.read_parquet("attn/events.parquet")

    # ТО ЖЕ множество, что стоит за долей повёрнутых в S8, и притом ПОПАРНО:
    # ключ (трек, витрина), а не просто трек. Общий по всем витринам список
    # треков пропускал у M1 человека, засчитанного у M2, и витрина с честным
    # нулём получала сетку пруфов, противоречащую собственному числу.
    ok = events[events["event_type"].isin(["gaze", "stop_and_gaze"])
                & (~events["low_confidence"])]
    ok_pairs = set(zip(ok["track_id"], ok["zone_id"]))
    hits = zf[zf["gaze_hit"] & (~zf["grazing"])]
    hits = hits[[p in ok_pairs for p in zip(hits["track_id"], hits["zone_id"])]]
    if hits.empty:
        return {}

    writer = EvidenceWriter(root="evidence", stage="dashboard", model_name="none")
    sampler = EvidenceSampler(writer, n_per_claim=n, n_strata=3)
    claims = {z: f"claim.zone.{z.replace('facade_', '')}.oriented"
              for z in hits["zone_id"].unique()}
    # Объявляем ТОЛЬКО витрины с ненулевой долей: доказывать нужно утверждение,
    # а не его отсутствие. Если доля не ноль, а кадров нет — finalize упадёт.
    sampler.declare([c for z, c in claims.items() if z in nonzero])

    foot = tracks.set_index(["frame_idx", "track_id"])[["foot_x_px", "foot_y_px"]]
    by_frame = {int(f): g for f, g in hits.groupby("frame_idx")}
    det_by_frame = {}
    for f, g in det.groupby("frame_idx"):
        b = g[["x1_px", "y1_px", "x2_px", "y2_px"]].to_numpy(np.float64)
        det_by_frame[int(f)] = (b, (b[:, 0] + b[:, 2]) / 2.0, b[:, 3])

    # Кадров на трек не больше, чем нужно, чтобы сетка охватила ВСЕХ засчитанных
    # людей. Иначе пул забивается одним долгим треком: у M2 из 61 попадания
    # 12 кадров сетки приходились на двух человек, и сетка читалась как 12.
    import math
    per_zone_tracks = {z: g["track_id"].nunique()
                       for z, g in hits.groupby("zone_id")}
    cap = {z: max(1, min(MAX_FRAMES_PER_TRACK, math.ceil(n / max(1, k))))
           for z, k in per_zone_tracks.items()}
    allow: dict[tuple, list] = {}
    for (z, t), g in hits.groupby(["zone_id", "track_id"]):
        f = np.sort(g["frame_idx"].unique())
        k = min(cap[z], len(f))
        idx = np.unique(np.linspace(0, len(f) - 1, k).round().astype(int))
        allow[(z, int(t))] = set(int(f[i]) for i in idx)

    for fi, frame in iter_frames(video, np.asarray(sorted(by_frame), dtype=np.int64)):
        g = by_frame.get(int(fi))
        dv = det_by_frame.get(int(fi))
        if g is None or dv is None:
            continue
        boxes, cxs, y2s = dv
        for r in g.itertuples():
            if int(fi) not in allow.get((r.zone_id, int(r.track_id)), ()):
                continue
            try:
                fx, fy = foot.loc[(int(fi), int(r.track_id))]
            except KeyError:
                continue
            j = int(np.argmin(np.hypot(cxs - fx, y2s - fy)))
            tol = max(BOX_MATCH_TOL_MIN_PX,
                      BOX_MATCH_TOL_FRAC * (boxes[j][3] - boxes[j][1]))
            if np.hypot(cxs[j] - fx, y2s[j] - fy) > tol:
                continue
            x1, y1, x2, y2 = boxes[j]
            crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 6:
                continue
            dist = 0.0 if r.gaze_dist_m is None else float(r.gaze_dist_m)
            # Уверенность для стратификации — близость к фасаду: в сетку попадут
            # и ближние, и дальние случаи, а не только удобные.
            sampler.offer(claims[r.zone_id], track_id=int(r.track_id),
                          frame_idx=int(fi), ts=float(r.ts), crop_bgr=crop,
                          value=dist, confidence=float(1.0 / (1.0 + dist)),
                          extra={"zone": r.zone_id, "gaze_dist_m": round(dist, 2)})
    sampler.finalize()
    return {"claims": claims, "stats": sampler.stats()}


def funnel_html(steps) -> str:
    """Воронка. Каждая ступень — ПОДМНОЖЕСТВО предыдущей, иначе это не воронка,
    а просто четыре числа рядом. Доля считается от первой ступени."""
    top = max(1, steps[0][1])
    rows = []
    for label, val, note in steps:
        frac = val / top
        rows.append(
            f'<div class="fn"><div class="fn-l">{label}</div>'
            f'<div class="fn-bar"><i style="width:{max(frac * 100, 0.6):.2f}%"></i></div>'
            f'<div class="fn-v">{val}<small> {frac * 100:.1f}%</small></div>'
            f'<div class="fn-n">{esc(note)}</div></div>')
    return "".join(rows)


def donut_svg(parts, size: int = 190) -> str:
    """Кольцо долей. Подписи рядом, а не на секторах: на тонких секторах
    подпись не читается и её приходится «примерно» располагать."""
    total = sum(v for _, v, _ in parts) or 1
    r, cx = size / 2 - 16, size / 2
    circ = 2 * np.pi * r
    off, segs = 0.0, []
    for _, v, col in parts:
        ln = circ * v / total
        segs.append(f'<circle cx="{cx}" cy="{cx}" r="{r:.1f}" fill="none" '
                    f'stroke="{col}" stroke-width="22" '
                    f'stroke-dasharray="{ln:.2f} {circ - ln:.2f}" '
                    f'stroke-dashoffset="{-off:.2f}" transform="rotate(-90 {cx} {cx})"/>')
        off += ln
    return (f'<svg viewBox="0 0 {size} {size}" style="width:{size}px;height:{size}px">'
            f'{"".join(segs)}<text x="{cx}" y="{cx - 2}" text-anchor="middle" '
            f'font-size="26" font-weight="700" fill="var(--ink)">{len(parts)}</text>'
            f'<text x="{cx}" y="{cx + 16}" text-anchor="middle" font-size="10" '
            f'fill="var(--muted)">ZONES</text></svg>')


def bars_html(rows) -> str:
    """Горизонтальные полосы: название, полоса, значение."""
    top = max([v for _, v in rows] or [1]) or 1
    return "".join(
        f'<div class="hb"><div class="hb-l">{esc(k)}</div>'
        f'<div class="hb-bar"><i style="width:{v / top * 100:.1f}%"></i></div>'
        f'<div class="hb-v">{v}</div></div>' for k, v in rows)


def presence_svg(points, w: int = 1200, h: int = 240) -> str:
    if not points:
        return '<div class="none">presence curve unavailable</div>'
    ys = [float(p["mean_detections"]) for p in points]
    xs = [float(p["t_start_s"]) for p in points]
    ymax = (max(ys) * 1.18) or 1.0
    pad_l, pad_b, pad_t = 44, 26, 12

    def px(i):
        return pad_l + (w - pad_l - 12) * (i / max(1, len(xs) - 1))

    def py(v):
        return pad_t + (h - pad_t - pad_b) * (1 - v / ymax)

    line = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ys))
    area = f"{pad_l},{py(0):.1f} {line} {px(len(xs) - 1):.1f},{py(0):.1f}"
    ticks = "".join(
        f'<line x1="{pad_l}" y1="{py(v):.1f}" x2="{w - 12}" y2="{py(v):.1f}" '
        f'stroke="var(--line)"/><text x="6" y="{py(v) + 4:.1f}" font-size="11" '
        f'fill="var(--muted)">{v:.0f}</text>' for v in [0, ymax / 2, ymax * .95])
    dots = "".join(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3.2" '
                   f'fill="var(--accent)"/>' for i, v in enumerate(ys))
    labs = "".join(
        f'<text x="{px(i):.1f}" y="{h - 6}" font-size="11" fill="var(--muted)" '
        f'text-anchor="middle">{xs[i]:.0f}s</text>'
        for i in range(0, len(xs), max(1, len(xs) // 8)))
    return (f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto">{ticks}'
            f'<polygon points="{area}" fill="var(--accent)" opacity=".12"/>'
            f'<polyline points="{line}" fill="none" stroke="var(--accent)" '
            f'stroke-width="2.4"/>{dots}{labs}</svg>')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/s3_detect.yaml")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n", type=int, default=N_PROOFS)
    ap.add_argument("--public", action="store_true",
                    help="сборка для публичного Pages: только сетки витрин. "
                         "Архив пруфов и сетки цвета НЕ СОБИРАЮТСЯ — не "
                         "прячутся стилями, а не попадают в разметку. "
                         "Обоснование в docs/DECISIONS.md")
    args = ap.parse_args(argv)
    set_public_build(args.public)

    import pandas as pd

    cfg = load_config(args.config)
    video = Path(require(cfg, "input", "video"))
    m = read_json("out/metrics.json")
    met, scope = m["metrics"], m["scope"]
    zones = read_json("zones/zones.geojson")
    hom = read_json("calib/homography.json")
    events = pd.read_parquet("attn/events.parquet")

    facs = [f["properties"] for f in zones["features"]
            if f["properties"]["zone_type"] == "facade"]
    nonzero = {p["zone_id"] for p in facs
               if (met.get(f"orientation_rate_{p['zone_id']}", {}).get("value") or 0) > 0}
    print(f"витрин {len(facs)}, с ненулевой долей повёрнутых {len(nonzero)}")
    collect_zone_proofs(video, args.n, nonzero)

    ev = pd.read_parquet("evidence/index.parquet")
    attr = pd.read_parquet("attr/tracks_attr.parquet")
    ok_attr = attr[attr["top_color_status"] == "ok"]

    def val(key, default=None):
        return (met.get(key) or {}).get("value", default)

    n_tracks = int(val("unique_tracks_total") or 0)
    gaze_ev = events[events["event_type"].isin(["gaze", "stop_and_gaze"])
                     & (~events["low_confidence"])]
    lookers = int(gaze_ev["track_id"].nunique())
    # low_confidence здесь НЕ фильтруется, и это не небрежность. Флаг ставится
    # по скользящему углу, неизмеренной ориентации и вырожденному окну — всё
    # это качество ОРИЕНТАЦИИ, а остановка считается по скорости и от угла не
    # зависит. Отбрасывать такие события значило бы терять настоящие остановки
    # из-за неизвестного угла. Правило совпадает с compute_zone_stoppers в S8,
    # поэтому карточка и out/metrics.json показывают одно число.
    stoppers = int(events[events["event_type"].isin(["stop", "stop_and_gaze"])]
                   ["track_id"].nunique())
    look_rate = lookers / n_tracks if n_tracks else 0.0

    kpis = [
        (t("kpi.tracked"), f"{n_tracks}",
         f"{scope['duration_s'] / 60:.0f} min · {scope['n_frames_processed']} / "
         f"{scope['n_frames_total']} {t('kpi.n.tracked')}"),
        (t("kpi.turned"), f"{lookers}",
         f"{look_rate * 100:.1f}% {t('kpi.n.turned')}"),
        (t("kpi.stopped"), f"{stoppers}", t("kpi.n.stopped")),
        (t("kpi.garment"), f"{len(ok_attr)}",
         f"{len(ok_attr) / max(1, len(attr)) * 100:.0f}% {t('kpi.n.garment')}"),
    ]
    kpi_html = "".join(
        f'<div class="card kpi"><div class="lab">{a}</div>'
        f'<div class="big">{esc(b)}</div><div class="note">{c}</div></div>'
        for a, b, c in kpis)

    cards = []
    for i, p in enumerate(facs):
        zid = p["zone_id"]
        short = zid.replace("facade_", "")
        rate = met.get(f"orientation_rate_{zid}", {})
        v = rate.get("value") or 0.0
        lo, hi = rate.get("ci95_low"), rate.get("ci95_high")
        ci = (f"95% CI {lo * 100:.1f}–{hi * 100:.1f}% · Wilson, resampled by track"
              if lo is not None else "no confidence interval")
        gs, dw = val(f"gaze_seconds_median_{zid}"), val(f"dwell_median_{zid}")
        rows = ev[ev["claim_id"] == f"claim.zone.{short}.oriented"] \
            .sort_values("stratum").head(args.n).to_dict("records")
        proofs = (sheet_html(rows, t("cap.zone"))
                  if rows else
                  f'<div class="none">{t("z.empty")}</div>')
        cards.append(f"""<div class="card">
<div class="zname zhead"><span class="dot" style="background:{ZONE_HEX[i % 4]}"></span>
<span class="nm">{esc(short)} · <span data-i18n="zone.{short}">{esc(ZONE_NAMES.get(zid, (p['name_ru'],))[0])}</span></span></div>
<div class="pct">{v * 100:.1f}%</div>
<div class="ci">{t("z.turned")} · {esc(ci)}</div>
<div class="mini">
  <div><div class="n">{esc(val(f'visitors_{zid}', 0))}</div>
       <div class="l">{t("z.entered")}</div></div>
  <div><div class="n">{'n/a' if gs is None else f'{gs:.1f} s'}</div>
       <div class="l">{t("z.attn")}</div></div>
  <div><div class="n">{'n/a' if dw is None else f'{dw:.1f} s'}</div>
       <div class="l">{t("z.dwell")}</div></div>
  <div><div class="n">{int(rate.get('n') or 0)}</div>
       <div class="l">{t("z.counted")}</div></div>
</div>{proofs}</div>""")

    counts = ok_attr["top_color_name"].value_counts()
    bar, legend, colour_cards = [], [], []
    for name, n in counts.items():
        share = n / max(1, len(ok_attr))
        hexc = CLASS_HEX.get(name, "#64748b")
        bar.append(f'<div style="width:{share * 100:.2f}%;background:{hexc}" '
                   f'title="{esc(name)} {share * 100:.0f}%">'
                   f'{f"{share * 100:.0f}%" if share > .08 else ""}</div>')
        legend.append(f'<span><i class="dot" style="background:{hexc};'
                      f'display:inline-block"></i>{esc(name)} — {n} '
                      f'({share * 100:.0f}%)</span>')
        sub = ok_attr[ok_attr["top_color_name"] == name]
        rows = ev[ev["claim_id"] == f"claim.attrs.color.{name}"] \
            .sort_values("stratum").head(args.n).to_dict("records")
        colour_cards.append(f"""<div class="card">
<div class="zname"><span class="dot" style="background:{hexc}"></span>{esc(name)}</div>
<div class="pct">{n}<span style="font-size:20px;color:var(--muted)">
 / {len(ok_attr)}</span></div>
<div class="ci">{share * 100:.0f}% of tracks with a resolved class</div>
<div class="mini">
  <div><div class="n">{sub['top_color_conf'].median() * 100:.0f}%</div>
       <div class="l">median agreement (votes x pixels)</div></div>
  <div><div class="n">{int(sub['n_crops_used'].median())}</div>
       <div class="l">median crops per track</div></div>
</div>{"" if args.public else sheet_html(rows, t("cap.color"))}</div>""")

    s_floor = float(load_config("configs/s7_attrs.yaml")["attrs"]["s_achromatic_max"])
    chromatic = int((ok_attr["hsv_s"] >= s_floor).sum())
    # Коэффициенты и насыщенность мостовой берутся из манифеста прогона, а не
    # пишутся словами: иначе текст переживёт данные и начнёт им противоречить,
    # как случилось с прежним блоком «хроматики не обнаружено».
    wb_gains_txt, wb_sat_txt = "n/a", "n/a"
    try:
        wbn = read_json("run_manifest.json")["stages"]["s7_attrs"]["notes"]["white_balance"]
        wb_gains_txt = " / ".join(f"{g:.3f}" for g in wbn["gains_bgr"])
        wb_sat_txt = f"{wbn['sat_road_before']:.0f} → {wbn['sat_road_after']:.0f}"
    except (OSError, KeyError, TypeError, ValueError):
        pass

    archive = []
    # В публичной сборке архива НЕТ. Не скрыт, а не построен: спрятанный
    # стилями архив всё равно уехал бы в HTML и читался бы в исходнике.
    for claim, g in ([] if args.public else ev.groupby("claim_id")):
        rows = g.sort_values("stratum").head(args.n).to_dict("records")
        archive.append(f"""<div class="card">
<div class="zname" style="font-size:14px"><code>{esc(claim)}</code></div>
<div class="cap">{t("arch.stage")} <b>{esc(g['stage'].iloc[0])}</b> · {len(g)} {t("arch.frames")}</div>
{sheet_html(rows, t("cap.arch"))}</div>""")

    EXTRA_I18N: dict[str, tuple] = {}
    lim_rows = []
    for lim in m.get("limitations", []):
        item = lim.get("item", "")
        got = LIMIT_EN.get(item)
        if got:
            k = f"lim.{abs(hash(item)) % 100000}"
            EXTRA_I18N[k + ".t"] = got[0]
            EXTRA_I18N[k + ".x"] = got[1]
            EXTRA_I18N[k + ".c"] = got[2]
            lim_rows.append(
                f'<tr><td><b><span data-i18n="{k}.t">{esc(got[0][0])}</span></b></td>'
                f'<td><span data-i18n="{k}.x">{esc(got[1][0])}</span></td>'
                f'<td><span data-i18n="{k}.c">{esc(got[2][0])}</span></td></tr>')
        else:
            lim_rows.append(
                f"<tr><td><b>{esc(item)} (untranslated)</b></td>"
                f"<td>{esc(lim.get('text_ru', ''))}</td>"
                f"<td>{esc(lim.get('consequence_ru', ''))}</td></tr>")
    unm_rows = []
    for u in m.get("unmeasured", []):
        item = u.get("item", "")
        got = UNMEASURED_EN.get(item)
        if got:
            # {n} подставляется из файла разметки: раньше здесь стояло
            # замороженное «24 людях», пока измерение шло уже по 50.
            got = tuple(tuple(x.replace("{n}", str(_n_labels())) for x in tri)
                        for tri in got)
            k = f"unm.{abs(hash(item)) % 100000}"
            EXTRA_I18N[k + ".t"] = got[0]
            EXTRA_I18N[k + ".w"] = got[1]
            unm_rows.append(
                f'<tr><td><b><span data-i18n="{k}.t">{esc(got[0][0])}</span></b></td>'
                f'<td><span data-i18n="{k}.w">{esc(got[1][0])}</span></td></tr>')
        else:
            unm_rows.append(
                f"<tr><td><b>{esc(item)} (untranslated)</b></td>"
                f"<td>{esc(u.get('reason_ru', ''))}</td></tr>")

    figs = {k: b64_img(Path(f"out/img/{v}"), max_w=1600)
            for k, v in {"zones": "zones_ref.jpg", "plan": "plan_all.png",
                         "gaze": "gaze_example.jpg"}.items()}
    missing = [k for k, v in figs.items() if v is None]
    if missing:
        raise SystemExit(f"нет картинок {missing}: сначала python scripts/make_figures.py")

    # ---- воронка: каждая ступень подмножество предыдущей ------------------ #
    zf_all = pd.read_parquet("attn/track_zone_frames.parquet")
    # S6 пишет строку, если человек в apron ЛИБО в окне фасада, поэтому
    # «попал в зону» — это и есть все треки в этом файле. Прежнее выражение
    # объединяло подмножество in_apron со всем множеством, то есть фильтр
    # in_apron не делал ничего и только притворялся, что делает.
    in_zone = set(zf_all["track_id"])
    turned = set(gaze_ev["track_id"])
    stopped = set(events[events["event_type"].isin(["stop", "stop_and_gaze"])]
                  ["track_id"])
    funnel = funnel_html([
        (t("fn.tracked"), n_tracks, "S4 · track/tracks.parquet"),
        (t("fn.entered"), len(in_zone), "S6 · attn/track_zone_frames.parquet"),
        (t("fn.turned"), len(turned & in_zone),
         "S6 · gaze/stop_and_gaze, low_confidence excluded"),
        # Каждая ступень — подмножество предыдущей, как и написано над
        # воронкой. Без пересечения с turned последняя ступень оказывалась
        # шире своего родителя и подпись врала.
        # Ступень воронки, а не общее число остановившихся: «из повернувшихся
        # ещё и остановились». Карточка выше показывает всех остановившихся,
        # и это разные величины — здесь пересечение обязательно, иначе
        # ступень окажется шире родительской и подпись «подмножество» соврёт.
        (t("fn.stopped"), len(stopped & turned & in_zone),
         "S6 · stop/stop_and_gaze among turned"),
    ])

    # ---- доли по зонам ---------------------------------------------------- #
    zparts, zleg = [], []
    for i, pz in enumerate(facs):
        zid = pz["zone_id"]
        v = int(val(f"visitors_{zid}", 0) or 0)
        col = ZONE_HEX[i % 4]
        zparts.append((zid, v, col))
        tot = sum(int(val(f"visitors_{q['zone_id']}", 0) or 0) for q in facs) or 1
        zleg.append(f'<span><i style="width:11px;height:11px;border-radius:3px;'
                    f'background:{col};display:inline-block"></i>'
                    f'{esc(zid.replace("facade_", ""))} — {v} ({v / tot * 100:.0f}%)</span>')
    # Входы в зоны, а не люди: один трек, прошедший вдоль улицы, входит
    # в несколько зон. Сумма по зонам обязана совпадать с числом событий S6,
    # иначе кольцо и гистограмма считают разное.
    n_entries = int(sum(v for _, v, _ in zparts))
    entries_per_person = n_entries / max(1, n_tracks)
    donut = donut_svg(zparts)
    donut_leg = "".join(zleg)

    # ---- распределения времени: гистограммы по корзинам -------------------- #
    def _buckets(series, edges):
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            n = int(((series >= a) & (series < b)).sum())
            lab = f"{a:g}-{b:g} s" if np.isfinite(b) else f"{a:g}+ s"
            out.append((lab, n))
        return out

    dwell_s = (events["t_end"] - events["t_start"])
    dwell_bars = bars_html(_buckets(dwell_s, [0, 5, 10, 20, 40, 80, np.inf]))
    per_tr = (zf_all[zf_all["gaze_hit"] & (~zf_all["grazing"])
                     & zf_all["track_id"].isin(turned)]
              .groupby("track_id").size() * float(
                  np.median(np.diff(np.sort(zf_all["ts"].unique())))
                  if zf_all["ts"].nunique() > 1 else 0.1))
    attn_bars = (bars_html(_buckets(per_tr, [0, 1, 2, 4, 8, np.inf]))
                 if len(per_tr) else
                 '<div class="none">nobody was counted as turned</div>')

    # Подпись окна берётся из выбора часа, а не пишется словами: «daytime clip»
    # было верно для трёхминутного отладочного куска и стало ложью для часа пик.
    window_label = f"{scope['duration_s'] / 60:.0f} min"
    try:
        hc = read_json("out/hour_choice.json")["best_window"]
        window_label = (f"{hc['start_jst'][11:16]}–{hc['end_jst'][11:16]} JST"
                        f" · {scope['duration_s'] / 60:.0f} min")
    except (OSError, KeyError, ValueError):
        pass

    unit = "m" if hom.get("scale_known") else "unit"
    _i18n = {k: {"en": v[0], "ru": v[1], "ja": v[2]}
             for k, v in {**I18N, **EXTRA_I18N}.items()}
    for _zid, _nm in ZONE_NAMES.items():
        _i18n[f"zone.{_zid.replace('facade_', '')}"] = {
            "en": _nm[0], "ru": _nm[1], "ja": _nm[2]}
    i18n_json = json.dumps(_i18n, ensure_ascii=False)   # включает зоны
    page = f"""<!doctype html><html lang="en" data-theme="light"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAM-01 Kabukicho — storefront attention</title><style>{CSS}</style>
{header_html("dashboard.html", [("nav.camera", "CAM-01 Kabukicho"), ("nav.location", "Ichiban-gai, Shinjuku"), ("nav.window", window_label)])}

<div class="wrap">
<div class="eyebrow">{t("hero.eyebrow")}</div>
<h1>{t("hero.title")}</h1>
<p class="sub">{t("hero.sub")}</p>

<section class="kpis">{kpi_html}</section>

<section class="warn"><b>{t("warn.head")}</b>
<div>&bull; <b>{t("warn.scale")}</b> {t("warn.scale2")}</div>
<div>&bull; <b>{t("warn.low")}</b> {t("warn.low2")}</div>
<div>&bull; <b>{t("warn.demo")}</b> {t("warn.demo2")}</div>
</section>

<section>
  <div class="eyebrow">{t("sec.zones.eyebrow")}</div>
  <h2>{t("sec.zones.title")}</h2>
  <p class="sub">{t("sec.zones.sub")}</p>
  <div class="grid">{''.join(cards)}</div>
</section>

<section>
  <div class="eyebrow">{t("sec.fn.eyebrow")}</div>
  <h2>{t("sec.fn.title")}</h2>
  <p class="sub">{t("sec.fn.sub")}</p>
  <div class="card">{funnel}</div>
</section>

<section>
  <div class="eyebrow">{t("sec.dist.eyebrow")}</div>
  <h2>{t("sec.dist.title")}</h2>
  <p class="sub" style="font-size:16px;color:var(--ink);max-width:none;
     margin:10px 0 20px"><b>{n_tracks}</b> {t("dist.sum.a")}
     <b>{n_entries}</b> {t("dist.sum.b")} <b>{entries_per_person:.1f}</b>
     {t("dist.sum.c")}</p>
  <div class="three">
    <div class="card"><span class="unit">{t("unit.visits")}</span>
      <div class="zname">{t("dist.byzone")}</div><div class="cap">{t("dist.byzone.u")}</div>
      <div style="display:flex;gap:16px;align-items:center;margin-top:12px">
        {donut}<div class="dleg">{donut_leg}</div></div></div>
    <div class="card"><span class="unit">{t("unit.visits")}</span>
      <div class="zname">{t("dist.dwell")}</div><div class="cap">{t("dist.dwell.u")}</div>
      <div style="margin-top:10px">{dwell_bars}</div></div>
    <div class="card"><span class="unit">{t("unit.tracks")}</span>
      <div class="zname">{t("dist.attn")}</div><div class="cap">{t("dist.attn.u")}</div>
      <div style="margin-top:10px">{attn_bars}</div></div>
  </div>
</section>

<section>
  <div class="eyebrow">{t("sec.foot.eyebrow")}</div>
  <h2>{t("sec.foot.title")}</h2>
  <p class="sub">{t("sec.foot.sub")}</p>
  <div class="card">{presence_svg(m['presence_curve']['points'])}</div>
</section>

<section>
  <div class="eyebrow">{t("sec.cloth.eyebrow")}</div>
  <h2>{t("sec.cloth.title")}</h2>
  <div class="warn" style="margin-bottom:16px"><b>{t("cl.head")}</b>
  <div>&bull; {t("cl.why")}</div>
  <div>&bull; {t("cl.gains")} <b>{wb_gains_txt}</b>. {t("cl.sat")}
  <b>{wb_sat_txt}</b>. {t("cl.sat2")}</div>
  <div>&bull; {t("cl.chroma")} <b>{chromatic}</b> / {len(ok_attr)}.</div>
  <div>&bull; <b>{t("cl.notvalid")}</b> {t("cl.notvalid2")}</div></div>
  <div class="card"><div class="bar">{''.join(bar)}</div>
    <div class="legend">{''.join(legend)}</div></div>
  <div class="grid" style="margin-top:16px">{''.join(colour_cards)}</div>
</section>

<section>
  <div class="eyebrow">{t("sec.geo.eyebrow")}</div>
  <h2>{t("sec.geo.title")}</h2>
  <p class="sub">{t("sec.geo.sub")} 1 {unit}.</p>
  <div class="two">
    <img class="fig" src="{figs['plan']}" alt="ground plane with all trajectories"
      data-meta="Ground plane, all trajectories of the clip, 1 {unit} grid">
    <div>
      <img class="fig" src="{figs['zones']}" alt="reference frame with traced zones"
        data-meta="Reference frame with the four traced storefronts">
      <img class="fig" src="{figs['gaze']}" style="margin-top:16px"
        alt="storefront lit by an orientation ray"
        data-meta="A storefront lights up when the orientation ray crosses its
ground edge; the id of the person is printed next to it">
    </div>
  </div>
</section>

<section>
  <div class="eyebrow">{t("sec.ev.eyebrow")}</div>
  <h2>{t("sec.ev.title")}</h2>
  <p class="sub">{t("sec.ev.sub")}</p>
  <div class="grid">{''.join(archive)}</div>
</section>

<section>
  <div class="eyebrow">{t("sec.hon.eyebrow")}</div>
  <h2>{t("sec.hon.title")}</h2>
  <div class="card"><div class="tbl"><table><tr><th>{t("th.quantity")}</th><th>{t("th.whynone")}</th></tr>
  {''.join(unm_rows)}</table></div></div>
  <h2 style="margin-top:22px">{t("sec.hon.title2")}</h2>
  <div class="card"><div class="tbl"><table>
  <tr><th>{t("th.item")}</th><th>{t("th.what")}</th><th>{t("th.conseq")}</th></tr>
  {''.join(lim_rows)}</table></div></div>
</section>
</div>

<div id="lb"><img id="lbimg" alt=""><div class="meta" id="lbmeta"></div></div>
<script>window.__I18N__ = {i18n_json};</script>
<script>{JS}</script></html>"""

    if args.public:
        n_emb = page.count("data:image/jpeg;base64")
        bad = [h for h, _ in NAV
               if h not in PUBLIC_PAGES and not h.startswith("http")
               and f'href="{h}"' in page]
        if bad:
            raise SystemExit(
                f"публичная страница ссылается на неопубликованное: {bad}")
        print(f"[dashboard] ПУБЛИЧНАЯ СБОРКА: кропов вшито {n_emb}, "
              f"архив и сетки цвета не собраны")
    atomic_write_text(args.out, page)
    print(f"готово: {args.out} ({args.out.stat().st_size / 1e6:.2f} МБ)")
    print(f"  треков {n_tracks}, повёрнутых {lookers}, остановившихся {stoppers}")
    print(f"  карточек витрин {len(cards)}, классов светлоты {len(colour_cards)}, "
          f"claim-ов в архиве {len(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
