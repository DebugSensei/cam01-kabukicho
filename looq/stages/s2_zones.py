"""S2 zones — проекция обводки витрин на план земли.

Читает ручную обводку `zones/zones.json` (ПИКСЕЛИ) и гомографию S1, пишет
`zones/zones.geojson` в координатах ПЛАНА.

ЧТО ПРОЕЦИРУЕТСЯ, А ЧТО НЕТ
---------------------------
Обведённый четырёхугольник витрины — это ВЕРТИКАЛЬНАЯ поверхность, фасад.
Прогонять его через гомографию земли нельзя: получится четырёхугольник,
которого нет ни на земле, ни где-либо ещё. На земле лежит ровно одно —
нижнее ребро BL->BR, след витрины. Оно и становится facade-отрезком.

Поэтому в geojson идут:
  * facade   — LineString из двух точек, спроецированный след витрины;
  * apron    — прифасадная полоса, построенная от facade наружу, в сторону
               улицы, на apron_depth;
  * roi      — Polygon, обводка пола: она лежит на земле и проецируется честно.
Сам четырёхугольник витрины сохраняется как polygon_px в свойствах — только
для отрисовки в отчёте, downstream его не читает.

Сторону улицы определяет не знак в коде, а центр ROI: apron строится в ту
полуплоскость, где лежит ROI. Ошибиться полуплоскостью так невозможно.

    python -m looq.stages.s2_zones --config configs/s2_zones.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from looq import SCHEMA_VERSION, STATUS_OK, STATUS_SKELETON
from looq.calib import apply_h
import cv2

from looq.geometry import point_in_polygon_m, segment_normal_m
from looq.evidence import EvidenceError
from looq.io import (ConfigError, RunManifest, load_config, read_json, require,
                     sha256_file, write_json)
from looq.stages._base import StageError, read_artifact_status

STAGE = "s2_zones"
OUTPUT = "zones/zones.geojson"


def run(cfg: dict[str, Any], manifest: RunManifest) -> dict[str, Any]:
    hom_path = require(cfg, "input", "homography")
    if read_artifact_status(hom_path) == STATUS_SKELETON:
        raise StageError(f"{hom_path} помечен status=skeleton: S1 не отработал")
    hom = read_json(hom_path)
    # ТА ЖЕ матрица, что берёт S4. Раньше здесь стоял H_px_to_unit, который
    # переводит в единицы ВЫСОТЫ КАМЕРЫ, а не в метры: зоны оказывались
    # в 4.4 раза мельче треков, фасады выходили по полметра, и в сектор
    # взгляда попадали все подряд. Ключ H — единственный, чьи единицы
    # совпадают с track/tracks.parquet.
    h = np.asarray(hom["H"], dtype=np.float64)
    scale_known = bool(hom.get("scale_known", True))
    calib_status = hom.get("calib_status", "calibrated")
    unit = "m" if scale_known else "conventional_unit"

    px_path = Path(require(cfg, "input", "zones_px"))
    if not px_path.is_file():
        raise StageError(
            f"нет обводки {px_path}. Сначала обведите зоны:\n"
            f"    python scripts/pick_zones.py --config {cfg['_config_path']}")
    traced = read_json(px_path)
    if traced.get("coordinate_frame") != "frame_px":
        raise StageError(f"{px_path}: ожидались координаты frame_px")

    apron_depth = float(require(cfg, "zones", "apron_depth_m"))
    if not scale_known:
        # Метров нет, значит глубина apron в метрах бессмысленна. Берём её как
        # долю характерного размера ROI — иначе полоса окажется либо в точку,
        # либо во весь кадр.
        apron_depth_units = float((cfg.get("zones") or {}).get(
            "apron_depth_units_when_unscaled", 0.15))
    else:
        apron_depth_units = apron_depth

    roi_m = apply_h(h, np.asarray(traced["roi_px"], dtype=np.float64))
    roi_centre = roi_m.mean(axis=0)
    roi_traced_m = roi_m.copy()

    features: list[dict[str, Any]] = [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [_ring(roi_m)]},
        "properties": {"zone_id": "roi_main", "zone_type": "roi",
                       "name_ru": "область достоверности", "is_measured": bool(scale_known)},
    }]

    for z in traced["zones"]:
        seg_m = apply_h(h, np.asarray(z["ground_segment_px"], dtype=np.float64))
        a, b = seg_m[0], seg_m[1]
        length = float(np.hypot(*(b - a)))
        normal = segment_normal_m(a, b, roi_centre)
        facade_deg = float(np.degrees(np.arctan2(normal[1], normal[0])) % 360.0)

        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [_pt(a), _pt(b)]},
            "properties": {
                "zone_id": f"facade_{z['id']}", "zone_type": "facade",
                "storefront_id": z["id"], "name_ru": z["name"],
                "facade_normal_deg": facade_deg,
                "facade_len_units": length,
                "source": "manual_trace_bottom_edge",
                "is_measured": bool(scale_known),
                # Только для отрисовки: это вертикальная плоскость, downstream
                # её не читает.
                "polygon_px": z["polygon_px"],
            },
        })
        apron = np.array([a, b, b + normal * apron_depth_units,
                          a + normal * apron_depth_units])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [_ring(apron)]},
            "properties": {
                "zone_id": f"apron_{z['id']}", "zone_type": "apron",
                "storefront_id": z["id"], "parent_zone_id": f"facade_{z['id']}",
                "name_ru": z["name"],
                "apron_depth_units": apron_depth_units,
                "source": "derived_from_facade",
                "is_measured": bool(scale_known),
            },
        })

    # ROI обязан накрывать полосу витрин. Обводка велась по мостовой, и все
    # четыре фасада оказались СНАРУЖИ неё (замер: фасады y 7.0..9.2 м при ROI
    # до y 7.1 м). Метрика витрины, посчитанная вне собственной области
    # достоверности, — абсурд, поэтому ROI расширяется выпуклой оболочкой
    # обводки вместе с концами фасадов, отодвинутыми на roi_facade_margin_m
    # наружу вдоль нормали. Обводка сохраняется отдельно, чтобы было видно,
    # что расширено, а что обведено рукой.
    margin = float(require(cfg, "zones", "roi_facade_margin_m"))
    extra = []
    for f in features:
        if f["properties"]["zone_type"] != "facade":
            continue
        seg = np.asarray(f["geometry"]["coordinates"], dtype=np.float64)
        nrm = segment_normal_m(seg[0], seg[1], roi_centre)
        extra.extend([seg[0] - nrm * margin, seg[1] - nrm * margin])
    if extra:
        pts = np.vstack([roi_traced_m, np.asarray(extra)])
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(np.float64)
        n_out = sum(1 for p in extra if not point_in_polygon_m(tuple(p), roi_traced_m))
        print(f"[{STAGE}] ROI расширен до витрин: концов фасадов вне обводки "
              f"{n_out} из {len(extra)}, вершин {len(roi_traced_m)} -> {len(hull)}")
        features[0] = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [_ring(hull)]},
            "properties": {"zone_id": "roi_main", "zone_type": "roi",
                           "name_ru": "область достоверности",
                           "is_measured": bool(scale_known),
                           "source": "convex_hull(traced_roi + facade_ends)",
                           "facade_margin_m": margin,
                           "polygon_traced_m": [_pt(p) for p in roi_traced_m],
                           "polygon_px": traced["roi_px"]},
        }

    doc = {
        "type": "FeatureCollection",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": STATUS_OK,
        "coordinate_frame": "plane_m" if scale_known else "plane_unit",
        "coordinate_order": "[x, y]",
        "unit": unit,
        "is_geographic": False,
        "calib_status": calib_status,
        "scale_known": scale_known,
        "warning_ru": ("Координаты — план земли, НЕ широта/долгота."
                       + ("" if scale_known else
                          " Масштаб НЕ ОПРЕДЕЛЁН: длины в условных единицах.")),
        "source_px": str(px_path).replace("\\", "/"),
        # Провенанс проекции: зоны в метрах имеют смысл только вместе с той
        # гомографией, которой их спроецировали. Без этой привязки устаревший
        # geojson молча используется с новой калибровкой — так и случилось
        # 2026-09-04, когда S2 не перегнали после смены масштаба.
        "homography_sha256": sha256_file(hom_path),
        "homography_calib_status": calib_status,
        "homography_key_used": "H",   # тот же ключ, что у S4: единицы обязаны совпадать
        "features": features,
    }
    manifest.note("n_features", len(features))
    manifest.note("scale_known", scale_known)
    print(f"[{STAGE}] зон: {len(traced['zones'])} витрин -> {len(features)} фич "
          f"(roi + facade и apron на каждую)")
    print(f"[{STAGE}] единицы: {unit}, calib_status={calib_status}")
    for f in features:
        if f["properties"]["zone_type"] == "facade":
            print(f"[{STAGE}]   {f['properties']['zone_id']}: длина "
                  f"{f['properties']['facade_len_units']:.3f} {unit}, "
                  f"нормаль {f['properties']['facade_normal_deg']:.1f} град")
    return doc


def _pt(p) -> list[float]:
    return [round(float(p[0]), 6), round(float(p[1]), 6)]


def _ring(pts) -> list[list[float]]:
    ring = [_pt(p) for p in np.asarray(pts)]
    ring.append(ring[0])          # GeoJSON требует замкнутое кольцо
    return ring


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
        doc = run(cfg, manifest)
        write_json(OUTPUT, doc)
        manifest.note("output_artifact", OUTPUT)
        manifest.finish(STATUS_OK)
        print(f"[{STAGE}] записано: {OUTPUT}")
        return 0
    except (StageError, EvidenceError, ConfigError, OSError, ValueError, KeyError) as exc:
        if manifest is not None:
            manifest.finish("failed", error=str(exc))
        print(f"[{STAGE}] ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
