# S1 — Automatic camera calibration

No geometry was measured by hand. Scale comes from the height of the people in frame.
Timebox: 50 minutes.

Debug clip: `raw/clip_debug_2030JST.ts` (3 min, 1080p30, Kabukicho, 20:30 JST).

To create: `looq/calib.py`, `looq/pilot.py`, `looq/stages/s1_calib.py`,
`scripts/pick_hints.py`, `verify/verify_s1.py` (extend).

---

## Answers to the two questions asked

**Where pilot_heights_m and pilot_speeds_mps come from.**
S1 is self-contained. It does NOT wait for the S3/S4 artifacts and cannot wait for them:
S3 writes `foot_x_m`, which needs a finished homography. The order in the Makefile is
strictly S1 → S3 → S4.

The shared detection and tracking code moves into `looq/pilot.py` and is called by two
consumers: S1's internal pilot and the full S3/S4 stages. Inference parameters are read
from `configs/s3_detect.yaml`, so that the calibration and the pipeline count the same
people.

S1's pilot results are never written to `det/` or `track/`. They live inside
`calib/pilot/` and in the `pilot_*` fields of the S1 artifact. This prevents the pilot
from standing in for a real run.

**How the control points for the reprojection check are produced.**
Not from points but from lines. Vanishing points are estimated by RANSAC over segments
found by LSD/Hough across the whole frame. The inlier set of each VP is split 70/30 with
a fixed seed: 70% take part in the final VP estimate, 30% are held out. The residual is
the median angular distance from the held-out segments to the estimated VP, converted
into pixels at a characteristic segment length.

The user's clicks only seed RANSAC and are not part of the held-out sample.

---

## Step 1. User hints

`scripts/pick_hints.py` — a cv2 window on the clip's reference frame (frame 0, or the one
given by `--frame`). The user clicks 8 points in a strict order, each one labelled on
screen:

1–3. three points on the base of the LEFT facade (the line where wall meets pavement)
4–6. three points on the base of the RIGHT facade
7–8. two points on one vertical (a pole, a building corner, a sign post)

Three points per line instead of two — to have redundancy and to estimate how crookedly
the user clicked. The spread of the points about the fitted line is written into the
artifact as `hint_line_residual_px`.

Right click undoes the last point, Enter saves, Esc cancels.
Result: `configs/calib_hints.yaml` with the points in pixels, plus the path and index of
the frame that was clicked on.

## Step 2. Vanishing points

The horizontal VP is the intersection of the two base lines (left and right), then
refined by RANSAC: all LSD/Hough segments of the frame are taken, and a segment counts as
an inlier if its line passes within `vp_inlier_tol_px` of the candidate VP. Optimisation
over the inliers, 30% holdout.

The vertical VP likewise, seeded from clicks 7–8, with inliers among the segments whose
slope is closer to vertical.

Degenerate cases must fail rather than return infinity: near-parallel base lines, a VP
inside the frame, fewer than `vp_min_inliers` inliers.

## Step 3. Ground plane up to scale

The horizon line and the ground-plane homography are built from the two VPs.

Axes: `+x` along the street, in the direction of the horizontal VP (into the frame).
`+y` across the street. The origin is the projection of the centre of the frame's bottom
edge onto the plane. The convention is recorded in the artifact's `axis_convention`.

## Step 4. Pilot detection and tracking

`looq/pilot.py`: yolo11m, `imgsz=1280, classes=[0], conf=0.25, half=True, device=0`.

Frame sampling is CONTINUOUS BURSTS spread evenly over the length of the clip
*(changed 2026-09-03: was 300 single frames)*. Single frames gave a step of 0.6 s between
observations, at which ByteTrack links unreliably in a crowd and the median speed comes
out biased. Bursts give the same coverage in time, but within a burst the frames are
consecutive and the linking is honest.

The height sample takes detections where:
- `conf > 0.5`
- the box does not touch the frame edges (a cropped person gives a rubbish height)
- maximum IoU with neighbouring boxes `< 0.2` (in a crowd the bottom of the box sits on
  someone else's back)
- box height `> 60 px`

ByteTrack over the same detections; speeds come from tracks longer than 2 seconds.

## Step 5. Scale from height

Single-view metrology (Criminisi): for each detection the bottom of the box is the foot
point, the top is the crown of the head. Height in arbitrary scale units follows from the
vertical VP and the horizon line.

`scale_m_per_unit` is chosen so that the MEDIAN height of the sample equals
`target_height_m: 1.65` (# mean adult height of the Japanese population, mixed-gender
sample).

Robustness: median, with 10% tails dropped from each side before the fit.

## Step 6. Independent checks

None of them takes part in fitting the scale.

**Speed.** Median speed of the pilot tracks → expected 1.0–1.6 m/s.
*(Corrected 2026-09-03: this said 1.1–1.5, which disagreed with the list of gate failures
below. The correct value is 1.0–1.6, as in the failure list and in
configs/s1_calib.yaml.)*

**Street width.** Distance between the facade base lines projected onto the plan →
expected 6.06 ± 0.8 m. The gate recomputes it from the homography itself and does not
read the artifact's ready-made field.

**Scale drift with depth** (accepted on your suggestion). Linear regression
`est_height_m ~ foot_y_m`. The slope must be indistinguishable from zero: the check is
that the 95% confidence interval of the slope covers zero. A non-zero slope means the
homography lies systematically with depth — and that is exactly the defect a pooled IQR
hides. Record in JOURNAL that the check was added on the implementer's suggestion.

**Height spread.** IQR and p90−p10, thresholds 0.18 / 0.35, marked NOT CALIBRATED.

**Median height** is NOT independent; it is printed with that note, as a detector of a
failed fit.

---

## Artifact `calib/homography.json`

```
H (3x3), vp_horizontal, vp_vertical, horizon_line,
scale_m_per_unit, origin_px, axis_convention,
hint_line_residual_px, vp_inliers_used, vp_holdout_residual_px,
n_people_used, height_median_m, height_iqr_m, height_p10, height_p90,
height_depth_slope, height_depth_slope_ci95,
speed_median_mps, speed_n_tracks,
street_width_measured_m, street_width_reference_m, street_width_delta_m,
method: "auto_vp_height", schema_version
```

## Gate `verify/verify_s1.py`

Fails if:
- median height outside 1.55–1.75 m *(with the note "not independent")*
- height IQR > 0.18 or p90−p10 > 0.35
- the confidence interval of the regression slope does not cover zero
- median speed outside 1.0–1.6 m/s
- street width outside 6.06 ± 0.8 m
- fewer than 200 people in the sample
- residual of the held-out lines > `vp_holdout_max_px`

Prints all seven quantities either way, whether the gate failed or not.

## Debug artifacts

`calib/debug_topdown.png` — top-down view, foot points over 300 frames, one-metre grid.
The street must look like a straight band of constant width. If it is wedge-shaped or
curved, the calibration is wrong, and that is visible by eye in a second.

`calib/debug_vp.png` — the reference frame with the inliers of both VPs drawn, the horizon
line, and the user's points labelled.

`calib/debug_heights.png` — a histogram of heights and a scatter of `est_height_m`
against `foot_y_m` with the regression line.
