# CAM-01 artifact contracts

`SCHEMA_VERSION = 1`

Stages are isolated and talk **only through files** (rule 2). A file's schema is the API
between stages. The schema changes only by an explicit decision, recorded in
`docs/DECISIONS.md`, with a bump of `SCHEMA_VERSION` in `looq/__init__.py`.

| Stage | Artifact | Format | One row = | Version |
|---|---|---|---|---|
| S0 ingest | `raw/*.ts` + `raw/manifest.json` | json | — | 1 |
| S1 calib | `calib/homography.json` | json | — | 1 |
| S2 zones | `zones/zones.geojson` | geojson | one zone | 1 |
| S3 detect | `det/frames.parquet` | parquet | one detection | 1 |
| S3 detect | `det/frames_index.parquet` | parquet | one recorded frame | 1 |
| S4 track | `track/tracks.parquet` | parquet | one track-frame | 1 |
| S5 pose+orient | `pose/orient.parquet` | parquet | one track-frame | 1 |
| S6 attention | `attn/events.parquet` | parquet | one episode | 1 |
| S6 attention | `attn/track_zone_frames.parquet` | parquet | frame x zone | 1 |
| S7 attrs | `attr/tracks_attr.parquet` | parquet | one track | 1 |
| S8 aggregate | `out/metrics.json` | json | — | 1 |
| S9 report | `out/report.html` | html | — | 1 |
| — | `run_manifest.json` | json | — | 1 |
| — | `evidence/index.parquet` | parquet | one evidence crop | 1 |

Every artifact carries its own version inside itself: json/geojson under the key
`schema_version`, parquet in file-level metadata `schema_version`.

Every artifact also carries `status`:

* `ok` — written by a full run of the stage;
* `skeleton` — written by the scaffold, no data inside.

**Every `verify_s<N>.py` must reject an artifact with `status = "skeleton"`** regardless
of thresholds (rule 8). A scaffold stage always returns exit code 1, so an empty artifact
is never passed off as a success.

---

## 1. Coordinate frames — the most error-prone section

The rule: **the column name tells you the coordinate frame. There are no columns without
a suffix.**

| Suffix | Coord frame | Unit | Origin and axes |
|---|---|---|---|
| `_px` | `frame_px` — the image frame | pixel | top-left corner of the frame; x right, y **down** |
| `_m` | `plane_m` — the ground plane | metre | set by S1 in `calib/homography.json`; +x along the street, +y across it, z = 0 |
| `_mps` | `plane_m` | m/s | — |
| `_deg` | `plane_m` | degree | see below |

The transform is implemented in one place only, `looq/geometry.py::apply_homography`.
The matrix in `homography.json` **always maps pixels to metres**. There is no reverse
direction in the contract, so the two sides cannot be confused.

### 1.1. Angles

```
0 degrees = the direction of the plan's +x axis (along the street),
measured anticlockwise seen from above,
range [0, 360).
```

Angles exist only in the `plane_m` frame. There is no angle in pixels anywhere in the
contracts: an angle in the image depends on perspective and means nothing on its own.

> **Orientation is a turn of the body or head, not the direction of gaze.**

This wording goes into the UI and the report **verbatim**. It lives in the code as
`looq.geometry.ORIENTATION_DISCLAIMER`, and the report imports it rather than restating it.
The words «взгляд», «смотрит», «разглядывает» — gaze, is looking, is eyeing up — are
banned from the Russian text of the report
(`configs/s9_report.yaml: forbidden_words_ru`), and `verify_s9` greps the built HTML.

### 1.2. Time

| Column | Type | Meaning |
|---|---|---|
| `frame_idx` | int64 | frame number from the start of the recording, 0-based. **The only join key on time.** |
| `ts` | float64 | seconds from the start of the recording, `ts = frame_idx / fps`. For plots and duration thresholds. |

Joining tables on `ts` is forbidden: a float as a join key silently drops rows.
Wall-clock time lives only in `raw/manifest.json`.

### 1.3. A missing value

A missing value is `null`, and only `null`. The sentinels `0`, `-1`, `999` and `NaN` in
float columns are forbidden: they are indistinguishable from a computed value and turn
"not measured" into "measured and equal to zero" (rules 7 and 8).

### 1.4. Rule 7 structurally: an indirect value carries a flag column

Any value **obtained indirectly rather than measured must carry a flag column** that
separates it out. Not a note in the documentation — a column in the data.

The first such case is `foot_source` in S4. In a crowd the legs are occluded and the foot
point is reconstructed from the hips or from the bottom of the box. That is not a
measurement, and it has to be visible in the data rather than lost.

| `foot_source` | Meaning | Direct? |
|---|---|---|
| `ankle` | the ankle is visible in the pose, the point is on the ground | **yes** |
| `bbox_bottom` | bottom of the box; under occlusion or clipping at the frame edge this is not a foot | no |
| `hip_est` | extrapolated down from the hips using the height estimate | no |

The mapping lives in the code: `looq.geometry.FOOT_SOURCE_IS_DIRECT`, and the indirect
share is computed by `looq.geometry.indirect_share()`.

**S8 prints the indirect share as its own number**, rather than dissolving it into an
average (`configs/s8_aggregate.yaml: report_indirect_shares`).

---

## 2. S0 ingest — `raw/*.ts` + `raw/manifest.json`

MPEG-TS segments exactly as HLS delivered them, no re-encoding, plus a manifest.

The manifest holds: `schema_version`, `status`, a `source` block (url, `yt-dlp`/`ffmpeg`
versions, recording start and end times), a `target` block (`fps_nominal`, `frame_w_px`,
`frame_h_px`, codec), a `segments` array (one entry = one `.ts`: `seg_idx`, `path`,
`sha256`, `bytes`, `pts_start_s`, `pts_end_s`, `n_frames`, `fps_measured`, `frame_w_px`,
`frame_h_px`, `frame_idx_start`, `frame_idx_end`, `decode_status`), a derived `gaps` array
and a derived `checks` block — exactly the numbers the gate prints.

`gaps` and `checks` are derived, but they sit in the artifact deliberately: otherwise the
gap logic lives in the gate rather than in the data, and the two drift apart.

**S0 gate:** no gaps in the segments, fps and resolution stable.
Thresholds in `configs/s0_ingest.yaml`: `max_total_gap_s`, `max_single_gap_s`,
`fps_tolerance`, `require_stable_resolution`. A change of resolution invalidates the S1
homography — fail at once.

---

## 3. S1 calib — `calib/homography.json`

The method is **`self_calib_pedestrians`** (`configs/s1_calib.yaml`, key
`method`), and that is what the published artifact records. There are no hand
measurements of the geometry: the horizon and the focal length come from pedestrian
pairs, and the scale from the median height of the sample. `auto_vp_height` /
`vanishing_points` is a legacy path kept in `looq/stages/s1_calib.py` as a synonym; it
was **not** used for this run, and §3.1.1 below describes its clicked seeds and does not
apply to the current artifact. The scheme is in `docs/S1_TASK.md`, the implementation in
`looq/calib.py`.

**Units before the scale is fixed.** Everything is computed in CAMERA-HEIGHT UNITS: the
camera stands at exactly 1.0 unit above the origin. Plan coordinates and a person's height
are therefore in the same units, and a single multiplier `scale_m_per_unit` converts all
of it to metres at once. Separate multipliers for the plan and for height would drift
apart, and the street-width check would stop checking anything.

Artifact keys, as they are in the published `calib/homography.json` — all 52 of
them:

| Group | Keys |
|---|---|
| geometry | `H`, `H_px_to_unit`, `K`, `R`, `focal_px`, `focal_over_diagonal`, `vp_horizontal`, `vp_vertical`, `horizon_line`, `camera_height_m`, `axis_convention`, `direction`, `frame_w_px`, `frame_h_px`, `reference_frame_idx`, `clip` |
| fit quality | `horizon_inlier_frac`, `horizon_n_pairs`, `horizon_residual_px`, `vertical_inlier_frac`, `vertical_n_segments`, `vp_horizon_tol_used_px`, `vp_street_dist_from_centre_px`, `vp_street_to_horizon_px` |
| height sample | `n_people_used`, `height_median_m`, `height_iqr_m`, `height_p10`, `height_p90`, `height_depth_slope`, `height_depth_slope_ci95`, `height_depth_slope_covers_zero` |
| scale | `scale_m_per_unit`, `scale_known`, `scale_rescale_factor`, `calib_status` |
| street width (implied, a check not a source) | `street_width_L1L3_implied_m`, `street_width_implied_plausible`, `street_width_implied_range_m`, `street_width_reference_m`, `street_width_units`, `street_grade` |
| pilot samples | `pilot_heights_m`, `pilot_depths_m` |
| what the stage says about itself | `method`, `status`, `stage`, `schema_version`, `banner_ru`, `assumptions_ru`, `scale_source_ru`, `independent_checks_ru` |

**Thirteen keys the S1 gate asks for and this artifact does not carry**, which is why it
fails 7 of its 9 checks: `origin_px`, `hint_line_residual_px`, `vp_inliers_used`, `vp_holdout_residual_px`, `vp_holdout_n`, `pilot_filter_stats`, `speed_median_mps`, `speed_n_tracks`, `street_width_measured_m`, `street_width_delta_m`, `facade_baselines`, `pilot_speeds_mps`, `vp_seed_pairwise_spread_px`. Earlier
revisions of this document listed them as if they were present. They are not, and their
absence is not a documentation slip — it is the state of the calibration, recorded in
`docs/DECISIONS.md` and in the Limitations table of the README. The gate prints each
missing key by name rather than reporting a single failure.

**The pilot samples sit in the artifact deliberately.** The gate recomputes the medians,
the spread and the regression **itself** instead of reading `height_median_m` and
`height_depth_slope`. Reading the finished numbers would check the stage's arithmetic, not
the homography.

**Assumptions, not measurements** (`assumptions_ru`, they go into the report): the
principal point at the frame centre, square pixels and zero skew, the street direction
orthogonal to the vertical, and the sampled median height equal to
`scale.target_height_m` = 1.65 m.

### 3.1.1. Seeds: `configs/calib_hints.yaml`, schema_version = 2

Produced by `scripts/pick_hints.py`. **14 clicks, five named lines:**

| id | role | side | what is clicked | points |
|---|---|---|---|---|
| `L1` | `street_ground` | left | foot of the left wall (where it meets the roadway) | 3 |
| `L2` | `street` | left | cornice / lower edge of the row of signs | 3 |
| `L3` | `street_ground` | right | the right kerb | 3 |
| `L4` | `street` | right | top line of the right-hand facades | 3 |
| `V1` | `vertical` | — | two points on one vertical | 2 |

`ground_pair: [L1, L3]` — **the only lines on the ground plane**. The street width is
measured from those alone: `L2` and `L4` run high up the walls, the distance between them
is not the street width, and the satellite reference measures the ground.

All four lines with role `street`/`street_ground` are parallel to the street in the world
and therefore meet in **one** point. The horizontal VP seed is their least-squares
intersection (`looq.calib.lines_common_vp`), not the intersection of two lines: the wall
lines run high above the ground and give a wide baseline. Measured on synthetic data with
2 px of click noise: median VP error **1.5 px from four lines against 3.1 px from the two
ground lines**. The spread of the pairwise intersections goes into the artifact as
`vp_seed_pairwise_spread_px` — a single least-squares point would hide any disagreement
between the lines.

`V1` defines **one** line, and one line does not determine a vanishing point. But the true
vertical VP must lie on it, so candidates are searched on that line only
(`vp_candidates_on_line`): RANSAC with a hard constraint, not a free search.

The clicks are **a RANSAC seed, not control points**. They are not part of the held-out
sample: otherwise the gate would be checking the quality of the clicks rather than of the
calibration. Three points per line instead of two give redundancy, and the spread of each
goes into `hint_line_residual_px` (a dict by line plus `max`). The threshold
`vp.hint_line_max_residual_px` = 6.0 px is a **warning, not an error**: whether to
re-click is for a human to decide.

`vp.seed_max_deviation_deg` = 25 bounds how far the RANSAC-refined vanishing point may
move from the seed. The deviation is measured **as an angle from the frame centre, not in
pixels**: the VP of a nearly horizontal street runs off thousands of pixels away, where a
hundred pixels is a fraction of a degree, while for a near VP the same hundred changes the
direction completely. Exceeding it means RANSAC latched onto a different family of
parallel lines.

Debug images: `calib/debug_topdown.png` (top-down view — the street must come out as a
straight band of constant width), `calib/debug_vp.png`, `calib/debug_heights.png`.

Application:

```
[X, Y, W] = H_px_to_m @ [x_px, y_px, 1]
x_m = X / W,  y_m = Y / W,  requires |W| > 1e-12
```

Held-out points are **not a separate file** but a `split` column in one table: `holdout`
points are physically never fed to the solver, and the gate reads the error on them only.
Two files of points are easy to desynchronise.

### 3.1. Independent check on the scale — street width

Additional artifact keys:

| Key | Type | Description |
|---|---|---|
| `facade_baselines` | list of 2 | the base lines of the opposite facades, set by the user's clicks. Each: `{"name": ..., "points_m": [[x_m, y_m], [x_m, y_m]]}` — **plane_m coordinates** |
| `street_width_measured_m` | float | the width computed by code from `facade_baselines` |
| `street_width_reference_m` | float | `6.06` — Google/Airbus satellite 2026, section 下村ビル ↔ 和田久ビл |
| `street_width_delta_m` | float | `measured - reference` |

The reference **takes no part in fitting the homography** — otherwise the check would be
checking itself. It lives in `configs/s1_calib.yaml`, section `control`, marked "NOT a
calibration input".

The measurement is `looq.geometry.facade_lines_separation_m`: the distance is measured
symmetrically (each end of each line to the line through the other), so the artifact also
receives the spread across the four measurements and the angle between the facade lines. A
large spread is diagnostic in itself: the facades are not parallel and "street width" is
poorly defined.

`verify_s1.py` **recomputes the width itself** and compares it with
`street_width_measured_m`. A discrepancy over 1 mm means a bug in the stage, not in the
homography.

### 3.2. What in the S1 gate is independent and what is not

The wording CLAUDE.md asks the gate to satisfy:

> The scale is estimated from the sampled median height; independently confirmed two
> ways — by the median pedestrian speed and by the street width from satellite.

**The published artifact does not satisfy it, and says so in its own fields.**
`scale_source_ru` reads: the median height 1.68 m; street width as a source of scale
REJECTED, because the 6.06 m reference gives a median height of 1.93 m, outside
1.55–1.75. `independent_checks_ru` reads: height is no longer an independent check — it
sets the scale; one check remains, the implied L1–L3 width of 5.28 m falling inside the
plausible range. Neither `pilot_speeds_mps` nor `facade_baselines` is in the artifact, so
the speed and street-width checks cannot be computed at all, and `verify_s1.py` fails both
by name. The table below is therefore what the gate WOULD check, with the current state of
each in the last column.

| Check | Threshold | Role | Independent? |
|---|---|---|---|
| reprojection error on held-out points | < 0.5 m | fit quality | — |
| **median height** | 1.55–1.75 m | **source of scale** | **no** |
| **height spread** (IQR, p90−p10) | ≤ 0.18 / ≤ 0.35 m | geometry | **yes** |
| median speed | 1.0–1.6 m/s | check | yes |
| street width | 6.06 ± 0.8 m | check | yes |

**The median height does not confirm the scale.** The scale was fitted from it, so its
landing inside the range is close to tautological. It detects a gross failure of the fit,
and nothing more. `verify_s1.py` prints it on a separate line marked "not independent", so
that three green lines in a row do not read as three confirmations.

**The height spread is independent and therefore valuable.** IQR and p90−p10 are not set
by the scale, they are set by the geometry: with a bad homography the height estimate
drifts with depth, and the spread widens even while the median stays put. Checked on
synthetic data: with the scale drifting 0.8×…1.25×, the median stays at 1.72 m (passes)
while the IQR moves from 0.09 to 0.36 m (fails). The function is
`looq.geometry.height_spread_stats`.

The pilot samples sit in the artifact itself so that the gate can compute the medians and
the spread **itself** instead of reading finished numbers — for the same reason as with the
street width. `pilot_heights_m` and `pilot_depths_m` are there. `pilot_speeds_mps` is not,
which is why the median-speed check has nothing to run on.

The height and speed ranges were widened from the originals in CLAUDE.md by an owner
decision on 2026-09-03. The reason is in `docs/DECISIONS.md`.

---

## 4. S2 zones — `zones/zones.geojson`

### 4.1. The seed: `zones/zones.json` — the tracing IN PIXELS

Produced by `scripts/pick_zones.py` (`make zones`), **exactly 20 clicks**: four storefronts
at 4 corners each plus the 4 corners of the ROI. This is NOT the stage artifact: the S2
artifact is `zones.geojson` in plan metres, and the stage does the projection itself
through the S1 homography. A polygon in pixels and a polygon in metres are different
things, and confusing the two is precisely the class of error the `_px` and `_m` suffixes
exist against.

The corners of each polygon go strictly clockwise from the top left:
**TL, TR, BR, BL**. The order is fixed for a reason: the bottom edge `BL→BR` is the
**storefront's footprint on the ground**, and it alone enters the gaze computation. Put the
corners in another order and the footprint is wrong, and the error surfaces only in S6.

`ground_segment_px` is **computed** from `polygon_px` as the bottom edge rather than asked
for separately: something you ask for can be confused with what was traced, something you
compute cannot.

Schema: `schema_version`, `clip`, `clip_sha256`, `frame_idx`,
`coordinate_frame: "frame_px"`, `corner_order`, `picking_thresholds`, `roi_px` (4 points),
`zones` — a list of `{id, name, polygon_px (4), ground_segment_px (BL, BR)}`.
`id` is the ASCII token of the name (`M1`), because `cv2.putText` does not draw kanji; the
full name lives in `name`.

Checks on save, all at once and fatal (re-clicking one problem at a time is slow): the
polygon is non-convex or self-intersecting; the corner order is inverted (BR/BL not below
TL/TR); the bottom edge is shorter than `picking.min_ground_edge_px`; the zone lies
entirely outside the ROI horizontally; two zones overlap by more than
`picking.max_zone_overlap_frac` of the area of the smaller one. Touching edge to edge is
allowed — adjacent storefronts do stand that way.

A re-run overwrites the file and puts the previous tracing in `zones/zones_prev.json`.

---

**The coordinates in `zones.geojson` are GROUND-PLANE METRES, not latitude/longitude.**
GeoJSON is read as lon/lat by default, so the `FeatureCollection` must carry the fields
`coordinate_frame: "plane_m"`, `coordinate_order: "[x_m, y_m]"`, `is_geographic: false`,
`warning_ru`. The reader fails if `coordinate_frame != "plane_m"`.

Zone types (`properties.zone_type`): `roi` (the reliability boundary, exactly one),
`facade` (a LineString of 2 points — the facade segment for `gaze_score`),
`apron` (the apron polygon for `stop_score`),
`entrance` (the entry polygon for `entry`), `exclusion`.

`properties`: `zone_id`, `zone_type`, `name_ru`, `storefront_id`, `parent_zone_id`,
`facade_normal_deg` (for `facade` only, angle convention from §1.1), `apron_depth_m`,
`source`, `is_measured`, `confidence_note_ru`.

`is_measured = false` for zones drawn by eye with no control points behind them — they go
into the report carrying that label (rule 7).

**S2 gate:** ≥30% of detections fall inside at least one zone on 100 random frames; the ROI
boundary is justified by a recall-versus-depth curve.

---

## 5. S3 detect — `det/frames.parquet`

One row = **one detection**. Primary key: `(frame_idx, det_id)`.

| Column | dtype | null | Unit | Coord frame | Description |
|---|---|---|---|---|---|
| `frame_idx` | int64 | no | frame | — | frame number from the start of the recording, 0-based |
| `ts` | float64 | no | s | — | `frame_idx / fps` |
| `det_id` | int32 | no | — | — | detection index within the frame, 0-based |
| `x1_px` | float32 | no | px | `frame_px` | left edge of the box |
| `y1_px` | float32 | no | px | `frame_px` | top edge of the box |
| `x2_px` | float32 | no | px | `frame_px` | right edge of the box |
| `y2_px` | float32 | no | px | `frame_px` | bottom edge of the box |
| `conf` | float32 | no | [0,1] | — | detector confidence |
| `cls` | int16 | no | — | — | COCO class; 0 (`person`) expected |

Box coordinates are in the **native decode resolution**, not the model's `imgsz`. S3 undoes
the detector's letterbox and resize itself, before writing.

There are no metre columns in S3: the stage is purely pixel-based and does not read
`calib/`. Projection onto the plan is S4's job. Otherwise re-calibrating the homography
would force a re-run of the most expensive stage.

### 5.1. The second S3 artifact — `det/frames_index.parquet`

One row = **one recorded frame**, processed or not. Primary key: `frame_idx`, contiguous,
no holes, covering the whole recording.

| Column | dtype | null | Unit | Description |
|---|---|---|---|---|
| `frame_idx` | int64 | no | frame | frame number, 0-based |
| `ts` | float64 | no | s | seconds from the start of the recording |
| `processed` | bool | no | — | the detector actually ran on this frame |
| `n_detections` | int32 | yes | count | detections written; `0` is a valid value; `null` when `processed = false` |
| `skip_reason` | string | yes | — | `decode_error` \| `out_of_roi_window` \| `sampled_out`; filled exactly when `processed = false` |

Why a separate table: without it **"frame processed, nobody there"**
(`processed = true, n_detections = 0`) and **"frame not processed"**
(`processed = false`) are indistinguishable. The difference breaks the denominator in every
S8 percentage — that is exactly how zones end up at zero.

Invariants: `frame_idx` is contiguous and unique; `n_detections` is null ⟺
`processed = false` ⟺ `skip_reason` is not null; the sum of `n_detections` over
`processed = true` equals the number of rows in `det/frames.parquet`.

**Every S8 percentage is computed against `processed = true`**, and the share of
unprocessed frames goes into the report as its own number
(`configs/s8_aggregate.yaml: denominator`, `report_unprocessed_share`).

Invariants: `x2_px > x1_px`, `y2_px > y1_px`; the box lies inside the frame; `det_id` is
unique within the frame; `conf >= configs/s3_detect.yaml: conf_thr`; the table is not empty.

**S3 gate:** AP@0.5 on 300 labelled frames, **the near and far halves separately** — an
average over both masks the collapse at distance.

---

## 6. S4 track — `track/tracks.parquet`

One row = **one track-frame**. Primary key: `(frame_idx, track_id)`.
The link to S3 is `(frame_idx, det_id)`.

| Column | dtype | null | Unit | Coord frame | Description |
|---|---|---|---|---|---|
| `track_id` | int32 | no | — | — | track identifier, unique within a run |
| `frame_idx` | int64 | no | frame | — | frame number |
| `ts` | float64 | no | s | — | seconds from the start of the recording |
| `foot_x_px` | float32 | no | px | `frame_px` | foot point in the image, x |
| `foot_y_px` | float32 | no | px | `frame_px` | foot point in the image, y |
| `foot_x_m` | float32 | **yes** | m | `plane_m` | foot point on the plan, x; `null` beyond the horizon |
| `foot_y_m` | float32 | **yes** | m | `plane_m` | foot point on the plan, y; `null` beyond the horizon |
| `foot_source` | string | no | — | — | `ankle` \| `bbox_bottom` \| `hip_est` — **the indirectness flag, §1.4** |
| `pos_conf` | float32 | yes | [0,1] | — | confidence of the foot point |
| `speed_mps` | float32 | yes | m/s | `plane_m` | from the smoothed trajectory; `null` at the window edges |

Here pixels and metres sit in one table — which is exactly why the suffixes are mandatory.

Invariants: `foot_x_m` is null ⟺ `foot_y_m` is null; `speed_mps >= 0` and
`<= configs/s4_track.yaml: max_plausible_mps` (anything faster is a projection error, not a
runner); `foot_source` only from the enumeration; ReID embeddings are **never written to
disk** (rule 9).

**S4 gate:** IDF1 and the number of ID switches on 3 minutes, separately on a sparse and a
dense stretch.

---

## 7. S5 pose+orient — `pose/orient.parquet`

One row = **one track-frame** for which a pose was attempted.
Primary key: `(frame_idx, track_id)`.

| Column | dtype | null | Unit | Coord frame | Description |
|---|---|---|---|---|---|
| `track_id` | int32 | no | — | — | track identifier |
| `frame_idx` | int64 | no | frame | — | frame number |
| `ts` | float64 | no | s | — | seconds from the start of the recording |
| `body_yaw_deg` | float32 | **yes** | degree | `plane_m` | **body turn**; 0 = +x of the plan, anticlockwise, [0,360); `null` = not measured |
| `head_yaw_deg` | float32 | **yes** | degree | `plane_m` | **head turn**; same convention; `null` = not measured |
| `yaw_conf` | float32 | yes | [0,1] | — | confidence of the angle |
| `n_kpts_valid` | int8 | no | count | — | visible keypoints, 0..17; `0` is a valid value |

`body_yaw_deg` and `head_yaw_deg` are **a turn of the body and of the head, not the
direction of gaze**. The wording of §1.1 goes into the UI and the report verbatim.

### 7.1. How the angle is computed

Shoulders and ears are **not on the ground**, so the ordinary homography will not do for
them: shoulders projected onto the ground give a point where the person is not standing,
and the error grows with distance from the camera. The points are back-projected onto the
**horizontal plane at their own height** (`looq.calib.backproject_to_height`), and that
height is taken as a fraction of **this** person's height
(`orient.shoulder_height_ratio`, `ear_height_ratio`), not the sample average.

`body_yaw_deg` is computed from the shoulder pair (COCO 5, 6), `head_yaw_deg` from the ear
pair (COCO 3, 4).

**The 180° ambiguity resolves itself.** The model labels shoulders anatomically — left and
right, not first and second. A person facing direction `f` carries the left shoulder to the
left of `f`, so rotating the vector "left → right" by +90° anticlockwise gives the
direction of the body. Resolving the ambiguity **from the direction of travel is not
allowed**: orientation would then stop being independent of the trajectory, and the whole
point of S6 is to compare the two against each other.

**S5 reads the video again.** By contract `track/tracks.parquet` holds no boxes, only the
foot point, so there is nowhere to take a crop from. Pose is computed per frame and matched
to tracks by nearest foot point, no further than `orient.match_max_dist_px`. One pose goes
to one track: otherwise two people standing side by side would get the same angle.

`null` in the angles is the share of non-coverage. It goes into the report as its own
number rather than being replaced by "was not looking" (rule 7). A row is written even when
the pose could not be matched: "tried and failed" and "never tried" are different things,
and the non-coverage share can only be computed if both are visible.

**S5 gate:** angle MAE on 200 hand-labelled people. Angle differences are computed with
wrapping (`looq.geometry.yaw_diff_deg`): naive subtraction gives 359° instead of 1°.

---

## 8. S6 attention — `attn/events.parquet`

One row = **one episode** `(track_id, zone_id, event_type)`. Primary key: `event_id`.

| Column | dtype | null | Unit | Coord frame | Description |
|---|---|---|---|---|---|
| `event_id` | string | no | — | — | deterministic: `track:zone:type:frame_start` |
| `track_id` | int32 | no | — | — | track identifier |
| `zone_id` | string | no | — | — | zone from `zones/zones.geojson` |
| `t_start` | float64 | no | s | — | start of the episode, seconds from the start of the recording |
| `t_end` | float64 | no | s | — | end of the episode, inclusive |
| `stop_score` | float32 | yes | [0,1] | — | see the formula below |
| `gaze_score` | float32 | yes | [0,1] | — | see the formula below |
| `event_type` | string | no | — | — | `stop` \| `gaze` \| `stop_and_gaze` \| `approach` \| `entry` |

The formulas are fixed by CLAUDE.md and do not change without a record in
`docs/DECISIONS.md`:

```
stop_score : share of time in the apron polygon at a speed < 0.3 m/s lasting > 1.5 s
gaze_score : share of the track's time in the window during which the ORIENTATION SECTOR CROSSES the facade segment
approach   : the distance to the facade decreasing
entry      : the track disappeared inside the entry polygon
```

### 8.1. `gaze_score`: a geometric test, not an angular one

The definition was rewritten on 2026-09-03. The primary test is the **intersection of the
orientation sector with the facade segment**:

1. a ray is cast from the person's position on the plan `(foot_x_m, foot_y_m)` in the
   direction `yaw` (S5: `head_yaw_deg`, or `body_yaw_deg` where that is missing);
2. the ray length is capped at `gaze.max_dist_m` = 8.0 m;
3. the ray is widened into a sector of half-width `gaze.yaw_uncertainty_deg`;
4. **the sector crossed the `facade` segment — the frame is counted.**

`gaze_score` = counted frames as a share of the track's frames in the window.

`yaw_uncertainty_deg` is **the angular uncertainty of the yaw estimate, not a decision
threshold**. Its value is taken equal to the measured MAE of the orientation model from the
S5 gate, not set by hand. Measured 2026-09-05 on `labels/s5_orient_50.jsonl` (n=50,
labelled on the same hour as the metrics): **21.2 deg**, and that is the value standing in
`configs/s6_attn.yaml`. The previous `15.0` was a placeholder marked NOT CALIBRATED.

**Grazing angle.** If the ray meets the facade at more than `gaze.grazing_max_deg` (70°
from the normal), the facade is seen edge-on and the intersection is unreliable. The event
is marked `low_confidence`: **it does not enter the main aggregate, but it is kept in the
data** (rule 7).

**Struck out 2026-09-03.** The previous wording was "the orientation ray crosses the facade
segment at an angle < 30° at a distance < 8 m". It read two ways: the angle to the facade
**normal** (approached head-on) or to the facade **line** (walked along it), and the
difference gave numbers several times apart. The geometric test removes the ambiguity by
itself: there is no longer anything to decide about what the angle is measured against. The
reason is in `docs/DECISIONS.md`.

`gaze_score` is computed from **the orientation of the body or head**, not from gaze. The
column name is technical; in the report it is "the share of time oriented toward the
storefront".

Invariants: `t_start <= t_end`; all `*_score ∈ [0,1]`; episodes of the same
`(track_id, zone_id, event_type)` do not overlap in time; `stop*` ⟹ `stop_score` is not
null; `gaze*` ⟹ `gaze_score` is not null; the table is not empty.

**S6 gate:** precision of "stopped + turned toward the storefront" on 100 hand-labelled
events.

---

### 8.2. The second S6 artifact — `attn/track_zone_frames.parquet`

The per-frame trace of a track-zone pair: one row for every frame where the person was in
the apron, hit the facade with the sector, or was inside the facade window.
`attn/events.parquet` is the aggregate over episodes; this file holds what that was built
from, and it is the file read by S8 (`gaze_seconds_median_*`), by the dashboard (the
"entered the zone" funnel step), by the overlay and by the replay.

| Column | Type | null | Unit | Coord frame | Meaning |
|---|---|---|---|---|---|
| `frame_idx` | int64 | no | frame | — | frame number |
| `track_id` | int32 | no | — | — | track identifier |
| `zone_id` | string | no | — | — | zone from `zones.geojson` |
| `ts` | float64 | no | s | — | seconds from the start of the recording |
| `x_m` | float32 | yes | m | `plane_m` | position on the plan, x |
| `y_m` | float32 | yes | m | `plane_m` | position on the plan, y |
| `in_apron` | bool | no | — | — | the point is inside the apron |
| `is_slow` | bool | no | — | — | speed below the stop threshold |
| `gaze_hit` | bool | no | — | — | the ORIENTATION sector crossed the facade |
| `gaze_dist_m` | float32 | yes | m | `plane_m` | distance to the intersection point |
| `grazing` | bool | no | — | — | grazing angle: the intersection is unreliable |
| `yaw_deg` | float32 | yes | degree | `plane_m` | the ORIENTATION angle used |

Both S6 artifacts are swapped on disk together (`commit_parquet`): done separately, a crash
between the two writes would leave S8 with a mix of a fresh file and an old one.

This section arrived later than the rest: the file was written from the beginning and read
by four consumers, but had no contract — which is to say rule 2 was being broken exactly
where it is hardest to notice.

## 9. S7 attrs — `attr/tracks_attr.parquet`

One row = one track. Primary key: `track_id`.

**Schema on implementation.** S7 is optional and is cut first if the schedule slips.
Skipping the stage is a legal state: S8 reports the corresponding metrics as
`measured: false`, not as zeros.

Fixed already: gender, age and ethnicity are forbidden, for two reasons — privacy (rule 9)
and no ground truth to gate against (rule 7). This is an explicit refusal, not a backlog
item (`configs/s7_attrs.yaml: forbidden`).

**S7 gate:** accuracy of the upper-garment colour on 150 crops + the coverage share.

---

## 10. S8 aggregate — `out/metrics.json`

**Schema on implementation.** Fixed now: every metric comes in an envelope with the value,
a confidence interval, `n`, the source stage, that stage's quality metric, a `measured`
flag and a reference to the code that computed the number (rule 1).

`measured` and `calibration_status` are two different fields: a computed number with an
uncalibrated threshold is not "unmeasured", but it is not "reliable" either.

Confidence intervals are resampled **by tracks, not by frames**: frames within a track are
strongly correlated, and a per-frame interval would come out falsely narrow
(`configs/s8_aggregate.yaml: resample_unit`).

The share of indirect values is printed as its own number (§1.4).

**S8 gate:** the sums add up, no NaN, confidence intervals present.

---

## 11. S9 report — `out/report.html`

**Schema on implementation.** Fixed now: every number in the report carries a reference to
its stage and to that stage's quality metric; the report is self-contained (no CDN); there
is a block of what is not measured and a block of the thresholds used, with "NOT
CALIBRATED" labels.

**S9 gate:** every number references its stage and its quality metric.

---

## 12. `run_manifest.json`

One file at the root, a section per stage; the stage's latest run overwrites the previous
one. Written by `looq/io.py::RunManifest`.

Keys of a stage section: `stage`, `schema_version`, `started_at`/`finished_at`
(UTC ISO-8601), `status`, `config_path` + `config_sha256`, a `model` block (`weights`,
`weights_sha256`, `weights_present`, `imgsz`, `device`, `precision`, `tracker`,
`tracker_version`), `git_sha`, `git_dirty`, `libraries`, `gpu` (device name and VRAM size),
`notes`.

"We use YOLO" is not an acceptable answer: the manifest holds the weights file name and its
sha256 (rule 6). If git is unavailable, `git_sha: "no-git"`, and the stage does not fail.

---

## 13. `evidence/index.parquet`

One row = one evidence crop. Written by `looq/evidence.py::EvidenceWriter`.

`claim_id` is **a stable identifier of a claim made in the report**, not of a metric and
not of a track. It is what takes you from a number in the report to the frames that number
stands on. Format: `claim.<family>.<area>`, for example `claim.attn.stop_and_orient`.
`claim_id` carries no run identifier, so that the same claim can be compared across runs.
The list of required claims per stage is in `configs/*.yaml`, key `evidence_claims`; a
stage with no declared claims will not start.

| Column | dtype | null | Description |
|---|---|---|---|
| `claim_id` | string | no | the claim this crop supports |
| `track_id` | int64 | no | track identifier |
| `frame_idx` | int64 | no | frame number |
| `ts` | float64 | no | seconds from the start of the recording |
| `value` | float64 | no | the value of the claim at this point, not an aggregate |
| `confidence` | float64 | no | confidence of the source, [0,1] |
| `path` | string | no | path to the jpeg: `evidence/<claim_id>/<track_id>_<frame_idx>.jpg` |
| `stage` | string | no | the stage that produced the evidence crop |
| `model_name` | string | no | weights name from `run_manifest`; `"none"` if there is no model |
| `extra_json` | string | no | arbitrary fields from the caller + the blur parameters applied |
| `stratum` | int8 | no | confidence stratum index: 0 lower, 1 middle, 2 upper |
| `schema_version` | string | no | contract version |

The first nine columns are fixed by the brief. The last three were added under rules 7 and
8: without `extra_json` the `extra` from `add()` is lost, without `stratum` there is no way
to check that the selection was stratified, and without `schema_version` a change of
contract cannot be traced.

### 13.1. Privacy (rule 9)

Face crops are never saved to disk. Before writing, the top **30 %** of the crop's height
is anonymised in two steps:

1. **pixelation** — downsample by `pixelate_factor` = 16 (`INTER_AREA`) and upsample back
   (`INTER_NEAREST`). Information finer than a block is destroyed irrecoverably;
2. **a Gaussian** over the blocks, kernel proportional to the crop size. It removes the
   sharp edges from which the original values could otherwise be estimated.

The order matters: a plain Gaussian is an **invertible convolution**, and deblurring it
recovers facial features. So the pixelation goes first.

The fraction was raised from 0.22 to 0.30 (owner decision, 2026-09-03): at distance a
person occupies fewer pixels, the head sits higher in the crop, and 22 % did not cover it.

The guarantee is structural: the module's only image-writing function is called from
exactly one place — immediately after anonymisation. If anonymisation did not change the
face region, the write is aborted with an error. The parameters applied (`blur_top_frac`,
`blur_head_h_px`, `blur_pixelate_factor`, `blur_pixel_block_px`, `blur_kernel_px`,
`blur_sigma_px`) are written to `extra_json` — anonymisation can be checked rather than
taken on trust. `evidence/` is in `.gitignore`.

The parameters are checked on synthetic data in `tests/test_evidence.py`, **but not on real
crops**. The check by eye is `scripts/check_blur.py` (a contact sheet of the first 20 crops
after S3). Until the owner confirms, a blocker stands in `docs/DECISIONS.md`.

### 13.2. Frame selection: stratified, not top-N

`EvidenceSampler` sorts candidates by `confidence` and splits them **by rank** into three
strata: lower, middle, upper. Each takes its own quota, and the remainder of the division
goes to the **lower** strata — weak examples are the most informative for an audit.

Top-N is forbidden on principle: it systematically shows the best cases and creates a false
impression of quality.

A shortfall is recorded, not swallowed: `n_offered`, `n_retained`, `n_selected`,
`shortfall` and the confidence bounds per stratum are written into the metadata of
`index.parquet` and into `run_manifest.json` (rule 7).
