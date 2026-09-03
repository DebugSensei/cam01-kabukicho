# CAM-01 Kabukicho.
#
# Всё запускается отсюда. Ничего не запускается из ноутбуков и ничего не
# запускается вручную мимо Makefile (CLAUDE.md, раздел "Команды").
#
# Каждая цель этапа состоит из двух строк: сам этап, затем его гейт.
# make останавливается на первом ненулевом коде возврата, поэтому этап,
# не прошедший гейт, не считается сделанным (правило 3).
#
# Рецепты намеренно написаны без bash-измов: под Windows make может подобрать
# и sh.exe, и cmd.exe. Только $(PY) -m ... и простые echo.

PY ?= python

.PHONY: all ingest calib zones detect track orient attn attrs aggregate report \
        verify run-all chain test hints ground pick-hour overlay overlay-plan figures hue-floor replay dashboard benchmark frames visuals serve serve-stop serve-nodocker depth-cutoff labels-orient labels-color rebuild-visuals rebuild-video zones-stage blur-check density-curve clean-artifacts help

all: report

# --- этапы ------------------------------------------------------------------ #

ingest:
	$(PY) -m looq.stages.s0_ingest --config configs/s0_ingest.yaml
	$(PY) verify/verify_s0.py

# Клики-затравки для S1, путь auto_vp_height. Интерактивно, окно cv2.
hints:
	$(PY) scripts/pick_hints.py --config configs/s1_calib.yaml

# Запасной путь S1: четыре угла прямоугольного участка мостовой, лупа x4.
# Даёт плоскость земли с точностью до масштаба, метров не даёт.
ground:
	$(PY) scripts/pick_ground.py --config configs/s1_calib.yaml

calib:
	$(PY) -m looq.stages.s1_calib --config configs/s1_calib.yaml
	$(PY) verify/verify_s1.py

# Ручная обводка зон: 20 кликов, интерактивно, окно cv2 с лупой.
# Пишет zones/zones.json В ПИКСЕЛЯХ. Повторный запуск перезаписывает файл,
# предыдущую обводку кладёт в zones/zones_prev.json.
zones:
	$(PY) scripts/pick_zones.py --config configs/s2_zones.yaml

# Сам этап S2: проецирует обводку в МЕТРЫ ПЛАНА через гомографию S1
# и пишет zones/zones.geojson.
zones-stage:
	$(PY) -m looq.stages.s2_zones --config configs/s2_zones.yaml
	$(PY) verify/verify_s2.py

detect:
	$(PY) -m looq.stages.s3_detect --config configs/s3_detect.yaml
	$(PY) verify/verify_s3.py

track:
	$(PY) -m looq.stages.s4_track --config configs/s4_track.yaml
	$(PY) verify/verify_s4.py

orient:
	$(PY) -m looq.stages.s5_orient --config configs/s5_orient.yaml
	$(PY) verify/verify_s5.py

attn:
	$(PY) -m looq.stages.s6_attn --config configs/s6_attn.yaml
	$(PY) verify/verify_s6.py

# S7 опционален и режется первым при отставании от графика.
attrs:
	$(PY) -m looq.stages.s7_attrs --config configs/s7_attrs.yaml
	$(PY) verify/verify_s7.py

aggregate:
	$(PY) -m looq.stages.s8_aggregate --config configs/s8_aggregate.yaml
	$(PY) verify/verify_s8.py

report:
	$(PY) -m looq.stages.s9_report --config configs/s9_report.yaml
	$(PY) verify/verify_s9.py

# --- гейты ------------------------------------------------------------------ #

verify:
	$(PY) verify/run_all.py

# Сквозной прогон трубы с понижением НЕИЗМЕРЕННОГО до предупреждения.
# Провал реального порога флаг НЕ прощает — S5 на нём и валится.
run-all:
	$(PY) -m looq.stages.s1_calib   --config configs/s1_calib.yaml
	$(PY) -m looq.stages.s2_zones   --config configs/s2_zones.yaml
	$(PY) -m looq.stages.s3_detect  --config configs/s3_detect.yaml
	$(PY) -m looq.stages.s4_track   --config configs/s4_track.yaml
	$(PY) -m looq.stages.s5_orient  --config configs/s5_orient.yaml
	$(PY) -m looq.stages.s6_attn    --config configs/s6_attn.yaml
	$(PY) -m looq.stages.s8_aggregate --config configs/s8_aggregate.yaml
	$(PY) -m looq.stages.s9_report  --config configs/s9_report.yaml
	-$(PY) verify/run_all.py --allow-unmeasured

# Прогон всей лестницы S0-S9 с игнорированием кодов возврата этапов: нужен,
# чтобы увидеть, где именно каждый гейт говорит "нет", не останавливаясь на
# первом. Завершается запуском всех гейтов, поэтому сама цель падает.
chain:
	-$(PY) -m looq.stages.s0_ingest --config configs/s0_ingest.yaml
	-$(PY) -m looq.stages.s1_calib --config configs/s1_calib.yaml
	-$(PY) -m looq.stages.s2_zones --config configs/s2_zones.yaml
	-$(PY) -m looq.stages.s3_detect --config configs/s3_detect.yaml
	-$(PY) -m looq.stages.s4_track --config configs/s4_track.yaml
	-$(PY) -m looq.stages.s5_orient --config configs/s5_orient.yaml
	-$(PY) -m looq.stages.s6_attn --config configs/s6_attn.yaml
	-$(PY) -m looq.stages.s7_attrs --config configs/s7_attrs.yaml
	-$(PY) -m looq.stages.s8_aggregate --config configs/s8_aggregate.yaml
	-$(PY) -m looq.stages.s9_report --config configs/s9_report.yaml
	$(PY) verify/run_all.py

# --- служебное -------------------------------------------------------------- #

test:
	$(PY) -m pytest tests/ -v

# Видео с наложением посчитанного: рамки, id, стрелки поворота корпуса, хвосты
# траекторий, точки ног по источнику, зоны, загорающиеся от попадания луча.
# Ничего не пересчитывает — рисует то, что лежит в артефактах.
overlay:
	$(PY) scripts/render_overlay.py --config configs/s3_detect.yaml

# То же плюс второе окно: план земли, синхронно с кадром. Прямые параллельные
# траектории на плане — это и есть проверка калибровки глазом.
overlay-plan:
	$(PY) scripts/render_overlay.py --config configs/s3_detect.yaml --plan 		--out out/overlay_plan.mp4 --frame-dir out/overlay_plan_frames

# Статические картинки для дашборда: план со всеми траекториями, опорный кадр
# с зонами, пример попадания луча.
figures:
	$(PY) scripts/make_figures.py

# Замер: с какой насыщенности тон перестаёт повторять тон сцены. Отсюда берётся
# s_achromatic_max в configs/s7_attrs.yaml.
hue-floor:
	$(PY) scripts/hue_floor.py

# Бенчмарк: что измерено против того, что только покрыто.
benchmark:
	$(PY) scripts/make_benchmark.py --hide-unmeasured

# Галерея кадров оверлея.
frames:
	$(PY) scripts/make_frames_page.py

# Весь визуал одной командой, в правильном порядке.
visuals: overlay figures replay dashboard benchmark frames

# Пересборка визуала ПОСЛЕ прогона: оверлей режется по самому плотному окну,
# час целиком это 36000 кадров и файл, который не открывается.
rebuild-visuals:
	$(PY) scripts/depth_cutoff.py
	$(PY) scripts/make_figures.py
	$(PY) scripts/make_replay.py
	$(PY) -m looq.stages.s7_attrs --config configs/s7_attrs.yaml
	$(PY) scripts/make_dashboard.py
	$(PY) scripts/make_benchmark.py --hide-unmeasured
	$(PY) scripts/make_frames_page.py

# ОДНО видео: самый плотный кусок часа. План рисуется в реплее рядом и
# синхронно, поэтому второе окно внутри видео — лишняя проходка по часу.
# Кадры каждые 30 обработанных уходят в out/overlay_frames как скриншоты.
rebuild-video:
	$(PY) scripts/render_overlay.py --densest-sec 180

# Замер глубины отсечки: медианная высота рамки по бинам глубины.
depth-cutoff:
	$(PY) scripts/depth_cutoff.py

# Ручная разметка. Единственные НЕКОЛЬЦЕВЫЕ проверки в проекте: всё остальное
# это согласие модели с самой собой. Гейты S5 и S7 без них валятся по делу.
labels-orient:
	$(PY) scripts/make_labels.py --mode=orient --n 50

labels-color:
	$(PY) scripts/make_labels.py --mode=color --n 50

# Просмотр результатов на localhost:8080. Внутри только nginx со статикой out/.
# Не python -m http.server: реплей перематывает видео, для чего нужен HTTP
# Range, которого http.server не умеет.
serve:
	docker compose up -d --build
	@echo "открой http://localhost:8080"

serve-stop:
	docker compose down

# Запасной сервер без Docker: та же статика, тот же порт, тот же Range.
# Docker Desktop на этой машине падал дважды посреди работы, и демонстрация
# не должна от него зависеть.
serve-nodocker:
	$(PY) scripts/serve.py

# Страница реплея: видео, синхронный вид сверху, шкала событий. Без сервера.
replay:
	$(PY) scripts/make_replay.py

# Дашборд под глаза: карточки витрин с крупным процентом и сеткой пруф-кадров.
dashboard:
	$(PY) scripts/make_dashboard.py

# Контроль обезличивания глазами. Требует уже собранных пруфов (S3 и далее).
# Блокер в docs/JOURNAL.md снимается только после подтверждения владельцем.
blur-check:
	$(PY) scripts/check_blur.py --n 20

# Выбор самого людного часа по всем записям raw/live_*.ts. Кликов не требует,
# склеивает выбранное окно в raw/peak_hour.ts. График идёт в отчёт.
pick-hour:
	$(PY) scripts/pick_hour.py --step-sec 60 --window-min 60

# Кривая присутствия по ОДНОМУ файлу: обоснование выбора часа, график идёт в отчёт.
# Требует аргументов, поэтому передаются переменными:
#   make density-curve SRC=raw/peak_1900-2200JST.ts START="2026-09-03 19:00"
SRC   ?= raw/peak_1900-2200JST.ts
START ?= 2026-09-03 19:00
STEP  ?= 60
density-curve:
	$(PY) scripts/density_curve.py $(SRC) --start-jst "$(START)" --step-sec $(STEP)

clean-artifacts:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p,ignore_errors=True) for p in ['det','track','pose','attn','attr','out','evidence']]"
	$(PY) -c "import pathlib; [pathlib.Path(d).mkdir(exist_ok=True) or pathlib.Path(d,'.gitkeep').touch() for d in ['det','track','pose','attn','attr','out','evidence']]"

help:
	@echo "targets: ingest calib zones detect track orient attn attrs aggregate report"
	@echo "         verify  - run all gates"
	@echo "         chain   - run S0..S9 ignoring stage exit codes, then all gates"
	@echo "         hints   - click S1 vanishing-point seeds (interactive)"
	@echo "         ground  - click a rectangular pavement patch, fallback S1 (interactive)"
	@echo "         zones   - trace S2 zones by hand, 20 clicks (interactive)"
	@echo "         zones-stage  - project traced zones to plane metres"
	@echo "         pick-hour    - scan raw/live_*.ts, pick busiest hour, concat"
	@echo "         overlay      - render out/overlay.mp4 from artifacts (H.264)"
	@echo "         replay       - build out/replay.html (video + plan + timeline)"
	@echo "         dashboard    - build out/dashboard.html (cards + proof grids)"
	@echo "         test    - pytest"
	@echo "         blur-check   - contact sheet of anonymised crops, eyeball it"
	@echo "         density-curve SRC=.. START=.. - presence curve, picks the busiest hour"
	@echo "raw/ is NOT cleaned by clean-artifacts: re-downloading an hour costs too much"
