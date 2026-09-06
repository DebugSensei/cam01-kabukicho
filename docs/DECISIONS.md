# CAM-01 Kabukicho decisions

This document records **the decisions taken and why**, not the course of the work.
Chronology, dead ends and daily "what got done today" reports have been stripped out of
the log. Every entry answers four questions: what was decided, why, what number backs it
and what it costs.

Every number here was computed by code from an artifact on disk: `calib/homography.json`,
`out/metrics.json`, `run_manifest.json`, `out/depth_cutoff.json`, `out/hour_choice.json`,
`labels/s5_orient_50.jsonl`. Numbers that were never measured are labelled not measured
and collected in the last section.

Final run: one recorded hour, 16:25–17:25 JST 2026-09-04, `raw/peak_hour.ts`.

---

## 1. Conventions the rest of the decisions rest on

**Angle zero.** `0° = the +x axis of the plan (along the street), counter-clockwise,
[0, 360)`. The reason: that is exactly `degrees(atan2(dy_m, dx_m)) % 360` — not one sign
conversion. A compass bearing was rejected: a local frame has no north, and CW/CCW
confusion is precisely the class of error the convention exists to prevent. The cost:
these angles are not comparable with outside sources without an explicit conversion.

**Coordinate-frame suffixes are mandatory.** `_px` is frame pixels, `_m` is plan metres.
In `track/tracks.parquet` both frames sit in one table (`foot_x_px` and `foot_x_m`), so a
label at table level would not have helped. A column with spatial meaning and no suffix is
a review error.

**An indirect value must carry a flag.** The first case is `foot_source` in S4 (`ankle` is
direct; `bbox_bottom` and `hip_est` are indirect), and the mapping is fixed in
`looq.geometry.FOOT_SOURCE_IS_DIRECT`. S8 prints the indirect share as a number of its
own: on the final run it is **1.0 at n = 481 751 rows**.

**`det/frames_index.parquet` — one row per frame of the recording.** Without it, "frame
processed, nobody in it" and "frame not processed" are indistinguishable, and that breaks
the denominator of every S8 percentage — which is how zones come out zero. Every
percentage is taken over `processed = true` and the unprocessed share is printed
separately: **36 000 of 108 000 frames (33.33 %)** processed, stride 3 frames, so no
metric has a time resolution better than 0.1 s.

**An empty artifact is never passed off as success.** A stage stub writes a file with
`status="skeleton"` and returns **exit 1**; `verify_s<N>` must reject `skeleton` whatever
the thresholds say. `write_parquet` checks rows against the contract schema: an extra key,
a missing key and `None` in a not-null column are errors, not something to fill in.

**Provenance is not allowed to lie.** A killed run used to leave `status: running` in the
manifest; `RunManifest.start()` now marks such an entry `aborted` and keeps an
`aborted_runs` list. Provenance that lies about state is worse than no provenance.

**Orientation is a turn of the body or head, not the direction of gaze.** The wording
lives in the code as `looq.geometry.ORIENTATION_DISCLAIMER` and is imported by the report
rather than retyped by hand in every place.

---

## 2. Calibration

### 2.1 Geometry from a hand seed, scale from pedestrians

**Decision.** Vanishing points, focal length and horizon come from a one-time hand tracing
of lines; pedestrians take part only in the scale estimate. Canonical wording for the report:

> Geometry is set by a one-time hand seed, scale is estimated automatically from
> pedestrian height.

**Why.** These are fundamentally different roles and the report must not conflate them: a
click is external information about the scene, height is a statistic over the run's data.

**Cost.** A run cannot be reproduced from scratch without the seed file `configs/calib_hints.yaml`;
the quality of the geometry depends on the accuracy of the clicks, which is not a measurable quantity.

### 2.2 Seed: 14 clicks over five named lines instead of 8 over three groups

**Decision.** `L1`/`L3` on the ground, `L2`/`L4` high on the walls, `V1` vertical,
`ground_pair: [L1, L3]`.

**Why.** All four lines `L1..L4` are parallel to the street in the world and converge on one
point, so the seed is taken as a least-squares point over four lines rather than as the
intersection of two. The wall lines give a wide baseline and condition the problem better.
`ground_pair` closes a separate hole: street width can be measured **only from the ground
lines**, otherwise the satellite reference would be compared against the wrong distance.

**Numbers** (synthetic, click noise 2 px per point, 120 repeats):

| Seed | Median vanishing-point error |
|---|---|
| from 4 lines (L1–L4) | **1.5 px** |
| from 2 ground lines (L1, L3) | 3.1 px |

The test `test_four_lines_beat_two_under_click_noise` requires a gain of at least 25 %.

**Cost.** Tracing became more expensive for a human, and the old `schema_version: 1` format
is rejected with an explicit error: 8 clicks must not be silently read as 14.

### 2.3 The tolerance on the refined vanishing point's deviation is in degrees, not pixels

**Decision.** `seed_max_deviation_deg: 25`, deviation measured as an angle from the frame centre.

**Why.** The vanishing point of a nearly horizontal street runs off thousands of pixels away,
and out there a hundred pixels is a fraction of a degree; for a near vanishing point the same
hundred changes the direction completely. A pixel threshold would be either useless or
prohibitive, depending on where the camera happens to sit.

For the same reason the RANSAC inlier tolerance was raised from 4 to 25 px: 4 px demanded
better than 0.25° accuracy from the segment detector. Measured on a real frame: **3 inliers
at 4 px, 21 at 20 px**. This is not a threshold nudged to fit the result, it is the removal
of a physically unattainable requirement.

### 2.4 The vanishing-point path on a real frame was rejected

**Decision.** The `method: vanishing_points` result was rejected in full, the run marked
`calib_status: stub_affine`, metres shown nowhere.

**Why.** Five independent quantities disagreed at once, not one.

| Quantity | Obtained | Expected |
|---|---|---|
| height IQR | 5.39 m | ≤ 0.18 |
| height p90−p10 | 21.19 m | ≤ 0.35 |
| height drift with depth | +0.0195 m/m, zero not covered | zero covered |
| median speed | 113 m/s | 1.0–1.6 |
| street width | 12.80 m | 6.06 ± 0.8 |
| camera height | 2.13 m | 4–8 |
| focal length | 541 px (0.25 of the diagonal) | — |

A height spreading over 21 metres and a speed of 113 m/s mean a wrongly reconstructed
vertical vanishing point, and with it a wrong focal length.

**Cost.** One full end-to-end S1→S9 run was made on the `stub_affine` stub (frame coordinates
divided by frame height); all its numbers are unusable and do not go into the report.

### 2.5 Self-calibration from pedestrians: first rejected, then accepted in a limited form

**Rejected** as a full replacement for the hand seed for two reasons: a hard deadline with the
manual path already written and covered by tests, and instability on this scene — in a dense
crowd the bottom of the box lands on someone else's back, the foot-to-crown segment is shorter
than the true one, and the error is systematic and does not average out over the number of people.

**Accepted** in a form where the clicks are needed only for the street direction (L1/L3), while
the horizon and the focal length come from the already computed `det/frames.parquet`. Result:

| Quantity | vanishing_points | self_calib_pedestrians |
|---|---|---|
| focal length | 541 px (0.25 diag.) | **1029.8 px (0.47 diag.)** |
| camera height | 2.13 m | 5.04 m (before the scale recomputation) |
| people in the sample | 400 | **10 150** |
| horizon inliers | — | 38.7 % over 16 810 pairs, residual 12.2 px |
| vertical inliers | — | 40 % over 425 segments |

**Cost.** The method is sensitive to how far away the vanishing point sits (see 2.7) and rests
on a population height distribution that is itself not measured.

### 2.6 Three flaws in the method, found before the result was accepted

**The vertical vanishing point cannot be obtained from boxes.** For a bbox the bottom and the
top are taken at the centre of the box, that is with the same `x`: the segment is vertical by
construction, all such segments are parallel and never intersect. Real keypoints are needed:
ankle and nose are offset in `x` by **15 px at the median**, and that offset carries the
perspective. An explicit degeneracy check was added to the code.

**RANSAC for the vertical vanishing point built consensus on noise.** Foot-to-crown segments
are almost parallel (median tilt from vertical **5.4°**, spread 8.9°), pairwise intersections
are wild outliers, and RANSAC placed the vanishing point inside the frame, which is
geometrically impossible for a downward-tilted camera. Replaced by robust least squares with
Tukey reweighting: **(942, 4604)**, far below the frame. The inlier threshold was moved from
pixels to degrees for the reason in 2.3.

**The focal-length formula was badly conditioned.** The per-component equations
`f² = c'·v_x/a` and `f² = c'·v_y/b` are formally correct, but on a camera without roll the
horizon is nearly horizontal, `a` is close to zero, and the first one blows up: on synthetic
data the focal error came out at **30 % with an exact vertical vanishing point**. Replaced by
`f² = d(principal point → horizon) · d(principal point → vertical VP)` — the same pole-polar
relation expressed through norms. The error became **0.4 %**.

Synthetic data, camera with known parameters, people of height N(1.68, σ):

| height σ | horizon inliers | focal error |
|---|---|---|
| 0.00 m | 100 % | exact |
| **0.07 m (realistic)** | 37 % | **−0.4 %** |
| 0.12 m | 23 % | method fails |

The inlier fraction turned out to be a poor quality indicator: at a realistic height spread
the focal length is already accurate at 37 %.

### 2.7 The tolerance on the vanishing point disagreeing with the horizon scales with distance

**Decision.** Instead of a fixed 40 px — `tol = 20 + 0.072 · distance from the frame centre`.

**Why.** The horizon error from pedestrians grows linearly as the vanishing point moves away;
a fixed tolerance is too lax for a near point and prohibitive for a far one.

| \|VP − centre\| | horizon error (synthetic, height σ 0.07) |
|---|---|
| 566 px | 4.7 px |
| 943 px | 22.4 px |
| 1612 px | 40.1 px |
| 4565 px | 110.9 px |

Slope ≈ `0.024 × distance`; the tolerance carries a threefold margin.

**The number on real data.** The street vanishing point sits **934.4 px** from the centre, the
tolerance works out at **87.3 px**, the disagreement is **103.7 px**. The check fires, but now
under a defensible rule: the disagreement exceeds the method's own noise by roughly a factor of
four, not by an order of magnitude as the first wording made it sound.

**Cost.** A weaker conclusion: the vanishing point disagreeing with the horizon is an argument
against the quality of the focal length, not standalone proof of a scale error. The main
evidence remains height (section 3).

### 2.8 Three calibration statuses instead of two

**Decision.** `calibrated` / `angles_ok_scale_unverified` / `stub_affine`. Only a focal length
out of range and low inlier fractions block the geometry; the disagreement with the horizon,
height and the depth drift hit the scale, not the angles.

**Why.** Previously any problem with the scale threw away working angles. The orientation of the
plane is set directly by two vanishing points, the horizon enters only through the magnitude of
the focal length, and the focal length is scale and depth dependence.

**Cost.** The report has to distinguish three modes in the text and in the banner; in status
`angles_ok_scale_unverified` the height, speed and street-width fields are absent from the
artifact **deliberately** — empty keys would read as "we computed it and it came out zero".

### 2.9 One scale for the plan and for height

**Decision.** Until the scale is fitted, everything is computed in camera-height units: the
camera sits at exactly 1.0 unit. Plan coordinates and a person's height are expressed in the
same units, and `scale_m_per_unit` converts all of it to metres at once.

**Why.** The alternative — computing height through the horizon and the vertical vanishing
point and the plan separately — was rejected: the two factors drift apart, and then the
street-width check stops checking anything, because plan and height live on different scales.
`test_scale_recovers_camera_height_and_heights` requires the recovered factor to equal the true
camera height.

### 2.10 The fallback `manual_ground_plane` path and the canonical plan sign

**Decision.** A homography from the four corners of a rectangular patch of pavement with a known
aspect ratio; it gives the ground plane up to scale. `CANONICAL_PLANE_SIGN = −1` was introduced,
conforming to it is mandatory, and a flip is recorded in the artifact as `plane_y_flipped`.

**Why.** The rectangle can be traced so that the plan comes out mirrored relative to the camera
path (checked with the Jacobian of the frame → plan mapping: the camera path has sign −1, the
rectangle in the test layout +1). The consequence is silent: on a mirrored plan a +90° turn from
left shoulder to right points backwards, and **every orientation would have been reversed by
180°**. The number would have looked plausible and no gate would have fired.

**Side benefit.** The plan-based angle method needs neither shoulder height nor K and R:
`shoulder_height_ratio` and `ear_height_ratio` were removed from `configs/s5_orient.yaml` —
both were NOT CALIBRATED.

**Cost.** In this mode there are no metres: the stop threshold is switched to relative mode
(see 6.7), and the height and speed fields are absent.

### 2.11 `--allow-unmeasured` downgrades "NOT MEASURED", not "FAILED"

**Decision.** The flag turns an unmeasured quantity into a warning and does not touch a
genuinely failed threshold.

**Why.** These are different things, and mixing them would mean a green gate on bad numbers. In
`verify_s1` on the fallback path: without the flag, 4 unmeasured quantities and exit code 1;
with the flag, exit code 0, but the line "NOT APPLICABLE without metres" is printed in both cases.

---

## 3. Scale

### 3.1 The source of scale is a median height of 1.68 m. Street width rejected

**Decision.** `calib_status: scale_from_height`, `scale_m_per_unit = 4.3937 m`,
rescale factor 0.8714.

**Why the width was rejected.** The reference **6.06 m ± 0.4** was measured from
satellite, building to building (下村ビル ↔ 和田久ビル), while the clicked points are
`L1` = the foot of the left wall and `L3` = **the right kerb**. The kerb is set back
inward from the opposite building. Scaling from the reference puts the median height at
**1.928 m** — outside the 1.55–1.75 band, too high by a factor of **1.17**.

**The number that supports this reading.** The width at which the median height becomes
1.68 m:

```
L1↔L3 = 6.06 × 1.68 / 1.928 = 5.281 m
```

The plausible band **4.6–5.6 m** was set by the owner — it falls inside. The kerb
setback from the opposite building comes out at **0.78 m**, plausible for the edge of a
pavement.

Fitting the width to the height would break rule 3 outright, so the 6.06 m reference is
left untouched in the config: it stays a control value, and the discrepancy is carried
into the report's limitations.

**Final calibration numbers:**

| Quantity | Value |
|---|---|
| focal length | 1029.8 px (0.4675 of the diagonal) |
| camera height | 4.394 m |
| median height | 1.68 m (by construction) |
| height IQR | 0.299 m |
| height p10 / p90 | 1.490 / 1.967 m |
| people in the sample | 10 150 |
| implied L1↔L3 width | 5.281 m against the 6.06 m reference |

**The cost, stated plainly.** Height **is no longer an independent check** — it defines
the scale, and its landing inside 1.55–1.75 has become a tautology. One check remains
and it is weaker: the implied width falls inside the plausible band. That tests "the
number is not absurd", not "the number is right". The wording is carried into the
artifact banner and into the report's limitations block.

**What would restore an independent check.** Any tape measurement in frame, or a
satellite distance between the clicked lines themselves rather than between the
buildings.

### 3.2 Height spread is the part that really is independent of scale

**Decision.** The height IQR and p90−p10 were added to the S1 gate, with thresholds
`height_iqr_max_m: 0.18` and `height_p90_p10_max_m: 0.35`.

**Why.** The median height is tautological under any scheme that fits the scale to it;
it detects a gross failure of that fit and nothing more. The IQR and p90−p10 are not set
by the scale — geometry sets them: with a bad homography the height estimate drifts with
depth, and the spread opens up even while the median stays put.

**Numbers** (synthetic, 200 pilot tracks):

| Case | Median | IQR | p90−p10 | Verdict |
|---|---|---|---|---|
| normal population N(1.68, 0.07) | 1.683 | 0.092 | 0.182 | ok |
| scale drifting 0.8×…1.25× with depth | 1.719 — **passes** | **0.362** | **0.610** | FAIL |

**The cost.** The thresholds come from synthetic data with a factor-of-two margin and
are marked NOT CALIBRATED. On real data the IQR is 0.299 against a threshold of 0.18:
the spread here is physical (height is measured from the box — hair, shadows,
occlusions) rather than a scale artefact, and the threshold does not describe it.

The median keeps its role in the gate but is relabelled: it prints on its own line
marked `[НЕ НЕЗАВИСИМА: масштаб подобран по ней, это детектор сбоя подгонки]`. Without
the label, three green lines in a row would read as three confirmations.

### 3.3 Height regressed on depth, a sharper version of the same check

**Decision.** A linear regression `est_height_m ~ foot_y_m`; the slope must be
indistinguishable from zero (the 95 % interval covers zero).

**Why.** A pooled IQR mixes near and far and partly hides the drift. The regression
looks at the dependence on depth itself. Checked by
`test_depth_slope_catches_wrong_focal`: with the focal length corrupted by 25 % the
slope interval does not cover zero, and with the correct calibration it does.

**The number on real data.** `height_depth_slope = −0.0163 m/m`, zero **not covered**.
The cause is worked through in section 4.

### 3.4 Speed as a check: OLS over a burst instead of differencing neighbouring frames

**Decision.** Pilot speed is the OLS slope over every point of a burst; bursts are
contiguous (10 frames, spaced 6.2 s apart) and the tracker is reset on the first frame
of each burst.

**Why.** At 30 fps a pedestrian covers about 4 cm per frame, while the jitter of the
foot point on the plane is 10–20 cm. Speed is non-negative, so the noise does not cancel
under averaging — it pushes the median up: on synthetic data with a true 1.30 m/s,
differencing neighbouring frames returned **6.5 m/s**, five times too high.

**Bias numbers** (400 tracks, true speed 1.30 m/s, position noise 0.15 m):

| Burst | Duration | Median | Bias |
|---|---|---|---|
| **10 frames** | 0.33 s | 1.415 | **+8.9 %** |
| 20 frames | 0.67 s | 1.308 | +0.6 % |
| 30 frames | 1.00 s | 1.293 | −0.6 % |
| 45 frames | 1.50 s | 1.302 | +0.2 % |

**The cost.** 10 frames was kept: a bias of +8.9 % does not fail the 1.0–1.6 band, but
it is a systematic error in a number that goes into the report. The recommendation
`burst_frames: 30` is written into the config comment. A trap that comes with it: at a
burst of 0.33 s the threshold `min_track_s: 2.0` yields zero tracks **by construction**,
so the stage fails loudly when `min_track_s > burst_frames / fps`, and
`pilot_burst_duration_s` sits next to the share of long tracks.

### 3.5 Widening the bands before the first measurement is not nudging a threshold

The height and speed bands are wider than the brief set them: height 1.60–1.70 →
1.55–1.75, speed 1.1–1.5 → 1.0–1.6. Rule 3 formally forbids touching a threshold to make
a gate pass, but these were widened **before** the first measurement rather than after a
failure, and a third quantity was added at the same time. The strength of the check moved
out of the narrowness of each band and into the conjunction: three wide conditions
together are stricter than two narrow ones. The gate recomputes the width and the medians
**itself** rather than reading finished numbers out of the artifact — otherwise it would
be checking the stage's arithmetic and not the homography (a disagreement between the
recorded and the recomputed value of more than 1 mm is a failure).

---

## 4. Ground plane and depth

### 4.1 Height drift with depth is not a calibration error

**Decision.** The vertical vanishing point is ruled out as the cause of the drift.

**Why.** Sensitivity was tested: the vanishing point was shifted along `y` relative to the
principal point, everything else unchanged.

| VP shift | VP.y | focal | camera height | median height | slope | zero covered |
|---|---|---|---|---|---|---|
| −30 % | 3384 | 862 | 5.48 m | 2.123 | −0.02576 | no |
| −10 % | 4197 | 977 | 5.16 m | 2.001 | −0.02125 | no |
| **0 %** | **4604** | **1030** | **5.04 m** | **1.952** | **−0.01972** | no |
| +10 % | 5010 | 1080 | 4.94 m | 1.910 | −0.01848 | no |
| +30 % | 5823 | 1174 | 4.77 m | 1.838 | −0.01659 | no |
| +100 % | 8667 | 1456 | 4.39 m | 1.680 | −0.01291 | no |

A ±10 % shift changes the slope by only 8 %. Zero is covered nowhere, not even with the
vanishing point doubled. A side finding: at +100 % the median height lands exactly on
1.680 m while the slope remains — so tuning the vertical alone can produce the right
median on a plane that is still bent. The median and the slope measure different things.

### 4.2 The 4.9 % grade is an upper bound on the curvature of the flat model, NOT a street gradient

**Decision.** The figure of 4.9 % (2.81°, a rise of about 1.6 m over the 32 m observed) is
published **only together with the caveat** and is not called a measured street gradient.

**Model.** If the ground slopes, a person at distance `d` stands at a height of
`dz = −d·tanθ` relative to the assumed plane, the camera sits above their feet not by
`H` but by `H − dz`, and height is measured as `h·(1 − d·tanθ/H)`, from which
`tanθ = −slope·H/h`.

**Why this is not a measurement of the gradient.** The same drift could come from a
systematic box error that grows with depth — if, for instance, the detector clips the legs
of distant people more often. These data cannot tell the two apart. So the report says
"a grade that explains the drift", not "the street has a 4.9 % grade".

**The cost.** Lengths and speeds are distorted more far from the camera than near it; this
is a separate row in the "Limitations" section of `out/metrics.json`.

**An open discrepancy that needs checking.** The 4.9 % estimate is computed from a slope of
**−0.0188 m/m** (the value before the scale rescale, together with the H = 5.04 and
h = 1.93 in force at the time), while the artifact records a slope of **−0.0163 m/m** —
the result of multiplying by the factor 0.8714 in `_rescale_from_height`. A slope is a
ratio of two lengths, both of which are rescaled by the same factor, so it must be
invariant to scale. The two recorded numbers therefore differ by exactly that factor, and
one of them is wrong. Marked as an open question rather than fixed after the fact.

### 4.3 Depth cutoff: 70 px, derived from the focal length

**Decision.** `min_box_height_px: 70` in S3.

**Why 70.** The threshold is derived from the measured focal length, not assigned:
`h_px = f·H/Z` with f = 1030 px and H = 1.68 m gives **69.2 px** at Z = 25 m.
The 45 px first proposed cut nothing: the median box height falls below 52 px nowhere in
the frame, and 45 px corresponds to Z ≈ 38 m — past the end of the street.

**The numbers** (`out/depth_cutoff.json`, the recorded hour, median box height by depth bin):

| depth, m | rows | median box height, px |
|---|---|---|
| 0–3 | 32 084 | 240.0 |
| 3–6 | 64 348 | 249.0 |
| 6–9 | 78 708 | 209.2 |
| 9–12 | 74 221 | 169.5 |
| 12–15 | 61 477 | 138.0 |
| 15–18 | 53 326 | 114.8 |
| 18–21 | 46 292 | 99.0 |
| 21–24 | 36 619 | 87.0 |
| 24–27 | 23 257 | 78.0 |
| 27–30 | 6 688 | 73.5 |
| 30–33 | 216 | 72.0 |

The 70 px threshold falls at roughly **30 m**. On the three-minute debug clip, where the
cutoff was applied after the fact, **80.5 % of rows and 35 tracks out of 109** remained:
many tracks are dropped and few rows — distant tracks are short and broken. On the
hour-long run the cutoff is applied inside S3, so `rows_kept_frac = 1.0` and the number of
discarded detections is recorded separately: **129 918**.

**The cost, stated plainly.** This is **not recall**: a proper recall measurement needs
hand labelling, and that was not done. The threshold is also set in pixels, and converting
it to metres rests on the height-derived scale, which is not independently confirmed — the
cutoff depth in metres inherits that uncertainty. Both caveats sit in the artifact.

### 4.4 Shoulders are not on the ground

**Decision.** Shoulder points are not put through the ground homography;
`backproject_to_height` intersects the camera ray with a horizontal plane at the required
height, and that height is taken as a fraction of **this** person's height, not of the
sample mean.

**Why.** The homography maps the z = 0 plane only. Shoulders sit about 1.4 m above it, and
projecting them to the ground lands on a point where the person is not standing; the
further away the person, the larger the error. The claim is held by the test
`test_projecting_shoulders_to_ground_is_wrong`, which requires the naive projection to be
off by more than a metre — if someone "simplifies" the code back, the test fails.

### 4.5 The person's foot point: 100 % indirect

**Decision.** S4 writes `bbox_bottom`, S5 adds `foot_x_m_refined` from the ankles, and S6
takes the refined value where it exists.

**Why this way and not another.** The required chain `ankle → bbox_bottom → hip_est` runs
into the direction of the dependencies: pose is computed in S5, that is after S4, and S4
cannot read S5's artifact. Three options: (1) a second pose pass inside S4 — two full pose
passes over an hour of video, which does not fit the budget; (2) moving keypoints into a
shared step — a contract change; (3) an honest 100 % indirect. The third was chosen, and
with `pose_enabled: true` the stage **fails with an explanation** rather than quietly
making a second pass: the cost of the decision has to be chosen deliberately. S5 meanwhile
gives the ankles for free — zero extra model passes.

**Numbers from the hour-long run.** `indirect_foot_share = 1.0` (n = 481 751);
coverage of the ankle-refined point **37.6 %**.

---

## 5. Zones and ROI

### 5.1 Two artifacts instead of one

**Decision.** `zones/zones.json` — the tracing in `frame_px` (a seed, the analogue of
`calib_hints.yaml`); `zones/zones.geojson` — the S2 artifact in `plane_m`, projected
through the S1 homography. The geojson records `coordinate_frame: plane_m`,
`is_geographic: false` and the warning "coordinates are the ground plane, NOT
latitude/longitude", plus the sha256 of the homography used.

**Why.** Clicks land in frame pixels, the contract asks for plane metres; putting both in
one file would break the coordinate-suffix rule.

**Cost.** The Makefile targets diverged: `make zones` is the tracing, `make zones-stage`
is the stage.

### 5.2 Corner order is checked, not assumed

**Decision.** Besides convexity, points 3–4 are checked to be genuinely below points 1–2
on the `y` axis.

**Why.** The bottom edge `BL→BR` is the storefront's footprint on the ground, and the only
thing attention is computed from. With the wrong order the "bottom" edge comes out as the
top one, and the error surfaces only in S6, where it can no longer be recognised. The
convexity check also catches self-intersection: in a simple convex quadrilateral the cross
products of adjacent edges all share one sign, while a bow tie gives mixed ones — no
separate test needed.

A polygon is rejected on ENTER, immediately, not at save time: redrawing one
quadrilateral is cheaper than learning on the twentieth click that the second one was
malformed. The thresholds `min_ground_edge_px` and `max_zone_overlap_frac` live in the
config and go into the artifact as `picking_thresholds`: which threshold accepted a
tracing must be visible from the file.

### 5.3 The ROI extended to the storefronts

**Decision.** ROI = the convex hull of the hand tracing together with the facade
endpoints, pushed **0.5 m** outward along the normal. Eight vertices instead of four. The
hand tracing is kept in a separate field, `polygon_traced_m`.

**Why.** The tracing followed the roadway, and **all 8 facade endpoints lay outside it**:
facades at y = 7.0–9.2 m against an ROI reaching y = 7.1 m. A storefront metric outside
its own region of validity is absurd.

**Cost and open question.** After the change of scale the ROI boundary has **not been
re-justified by a recall curve**, which the S2 gate requires. The storefronts sit 2–6 m to
the right of the corridor people walk along; that is geometrically plausible (people walk
down the middle of the street), but no check confirms it. Nothing outside the ROI is drawn
on the plan or in the overlay — it is visible that nothing there is counted.

### 5.4 The mirrored plan: the drawing lied, not the geometry

**Decision.** One line fixed in `PlanView`: `p[:, ::-1]` (a transposition — a reflection
about the diagonal, determinant −1) replaced by the rotation
`np.stack([-p[:,1], p[:,0]], axis=1)`.

**Why.** The Jacobian of the stored `H` is −1, that is `CANONICAL_PLANE_SIGN`: the plane
is right-handed, the geometry is sound. Measured from `H`: a `+y_m` step moves **95.6 px
to the left** in the frame, so the homography agrees with the camera (storefronts on the
left) while the canvas put `+y` on the right.

**Why this is dangerous.** A global reflection is almost harmless — every intersection
predicate is invariant under it. Exactly one thing breaks: `facing = [-v[1], v[0]]` in
`looq/calib.py`, because a +90° rotation does not commute with a reflection, and every
orientation would have silently flipped by 180°. That is precisely why
`CANONICAL_PLANE_SIGN` exists. Four tests in `tests/test_planview.py`, the main one
checking the sign of the signed area.

---

## 6. Orientation and attention

### 6.1 `gaze_score` redefined: the test is geometric, not angular

**Before.** "The orientation ray crosses the facade segment at an angle < 30° at a distance < 8 m."

**Why it changed.** The wording read two ways: the angle to the facade **normal** (walked
straight at it) or to the facade **line** (walked along it). The difference gives numbers
several times apart. Picking one of the two readings with a config key was rejected: a
choice between two wrong readings does not make the formula right.

**Now.**

```
ray from the person's position on the plane in the yaw direction, length <= gaze.max_dist_m (8 m)
-> widened into a sector of +- gaze.yaw_uncertainty_deg
-> the sector crossed the facade segment — frame counted
gaze_score = share of counted frames out of the track's frames inside the window
```

The geometry settles what the angle is measured against; the question simply does not
arise. The old function name `gaze_ray_hits_facade_m` is gone — it named an angular test —
and is now `orientation_sector_hits_facade_m`.

### 6.2 `yaw_uncertainty_deg` is a measured uncertainty, not a decision threshold

**Decision.** The value equals the measured MAE of the orientation model: **21.2°**
instead of the 15.0 placeholder. `configs/s6_attn.yaml` carries that number and the date
it was calibrated.

**The number.** `labels/s5_orient_50.jsonl`, n = 50: MAE **21.2°**, 95 % bootstrap
**[15.5, 28.8]**, median error 15.6°, p90 36.2°. `python verify/verify_s5.py` prints it.

**It was measured twice, and the first measurement is why the second exists.** The first
round used `labels/s5_orient_24.jsonl`, n = 24: MAE **25.5°** [18.7, 32.7] — but that
labelling was taken on a three-minute morning clip while the metrics are computed over the
evening hour, and a sector half-width is not a number to carry across scenes on trust. The
second round was labelled on `raw/peak_hour.ts` itself, the recording the report describes,
and gave 21.2°. Both are on disk; the config uses the second.

**Why bootstrap.** The angle-error distribution is not normal and is bounded at zero.

**The cost.** The old 15.0 made the sector half as wide as justified and systematically
undercounted facade hits, which understated attention. Going from 25.5° to 21.2° narrows
the sector again, and that is not cosmetic: the half-width decides who counts as turned
toward a storefront.

### 6.3 The S5 gate decides on the UPPER bound of the interval

**Decision.** A point estimate below the threshold with an interval that covers the
threshold is not a pass, it is "not confirmed".

**Result.** MAE **21.2°** is under the 25.0 threshold, but the interval
**[15.5, 28.8]** covers it, so the gate reads **not confirmed**, not passed. The threshold
was not touched (rule 3). The shortfall in sample size — **50 hand-labelled people against
the 200 in CLAUDE.md** — is printed on its own line rather than hidden. The first round,
n = 24 on a morning clip, gave 25.5° [18.7, 32.7], which was over the threshold outright.

**The cost.** One red gate in the final report. That is more honest than a moved threshold.

### 6.4 The hand-labelling protocol

**Decision.** The labeller clicks the point on the ground the body is turned towards;
**the code computes the angle** through the same homography. The model's prediction is
hidden and written into the label file. Selection is stratified by confidence rank, the
remainder going into the bottom stratum, **one crop per track**. Crops are not written to
disk, and faces are blurred by the same code as the evidence crops.

**Why.** Estimating an angle by eye is exactly the "look and say" that is forbidden; a
click on the ground turns a human judgement into a measurable quantity. The prediction is
hidden so that it cannot anchor the labeller and inflate accuracy. One crop per track,
because two crops of the same person are not two observations. Keeping the prediction in
the file makes the metric recomputable after the next run overwrites the artifacts.

Along the way: the middle frame of a track is taken as **a frame that actually exists**
rather than the median of the frame numbers — the median landed in a hole in a broken
track, which lost 12 crops of 28.

**The number.** `body_yaw_deg` is computed for only **24 tracks of 93** on the morning
clip; no more independent orientation observations exist there.

### 6.5 The 180° ambiguity is resolved by anatomy, not by motion

**Decision.** Body direction = the vector "left shoulder → right" rotated +90°
anticlockwise; left and right shoulder are labelled by the pose model.

**Why not from the trajectory.** An orientation derived from motion stops being
independent of it, and the whole point of S6 is to compare where a person is going against
where they are turned. Agreement would become an artifact of the method.

**The second trap found.** The first implementation took the direction from the
intersection of the shoulder line with the horizon. A vanishing point in homogeneous
coordinates has an **arbitrary sign**, and at some angles the method returned a 180°
reversal: the disagreement with the camera-space method was +90° at times and −90° at
others. Replaced by the difference of the images `h(L)` and `h(R)` — not the shoulders'
positions on the plane (shoulders are not on the ground) but two points on the line that
matters; their difference gives the direction with its sign already attached. After the
replacement the two independent methods agree: **0.0°** apart on the same plane, and
across planes a pure rotation with a spread of **2·10⁻⁵°**.

**Checked on synthetic data before the run** (not a single look at frames):

| Check | Result |
|---|---|
| back-projection to the ground against the homography | agreement to 1e−9 |
| a point at a height of 1.40 m returns to its own place | to within 1 cm |
| naive projection of the shoulders onto the ground (control) | error over a metre |
| body angle, 6 values from 0 to 330° | error under 0.5° |
| a 180° reversal | exactly 180° |
| degenerate pair (shoulders coincide) | nan, not a random number |

### 6.6 Three fixes that dropped the "turned toward a storefront" share from 50–94 % to 0–23 %

**The plane unit changed meaning.** Under the `stub_affine` placeholder the unit was the
**frame** height; under self-calibration it is the **camera** height. A sector distance of
"0.5 units" meant half a frame, that is, a sector spanning the whole scene. Now
**1.6 units = 8 m** at a camera height of 5.04 m.

**A track was counted on a single frame.** It was enough for the sector to touch the
facade once; over 20 seconds past a storefront that happens to nearly everyone. A
share-of-time threshold was introduced: `gaze.min_score_for_event = 0.20`. A momentary
head turn from a passer-by is not attention to a storefront. NOT CALIBRATED.

**The denominator was wrong.** Only a track that entered the apron — a narrow strip —
counted as a visitor. Now a track counts as having passed the storefront if it came within
the sector distance of the facade, regardless of orientation.

**The cost.** The 0.20 threshold was chosen without calibration and directly moves the
headline number of the report.

### 6.7 The stop threshold: a relative mode when there are no metres

**Decision.** Two branches. With `calib_status = angles_ok_scale_unverified`,
`mode: relative` applies — stopped = slower than **25 % of the median speed of the
flow**; the `absolute` mode (0.3 m/s from the brief) stays for a confirmed scale.

**Why.** A ratio of speeds does not depend on the scale, so the threshold is defensible
without metres and tunes itself to the scene as well. The 1.5 s duration always works: fps
does not depend on the scale. For the same reason the speed-outlier threshold was made
relative (the top decile): 4 m/s in arbitrary units is meaningless.

**The cost.** The threshold is marked NOT CALIBRATED, and the median of the flow is
computed from speeds that themselves inherit the calibration's uncertainty. Stop shares
over the hour come out at **0–0.15 %** — that is either the strictness of the threshold or
the real picture; there is nothing to tell them apart with.

### 6.8 The grazing-angle guard

**Decision.** `grazing_max_deg: 70.0`: if the ray meets the facade at more than 70° from
the normal, the facade is seen edge-on and the crossing is unreliable. The event is marked
`low_confidence`, stays out of the main aggregate, but **is kept in the data**.

**Why.** What is unreliable must not be dropped silently. NOT CALIBRATED.

### 6.9 Speed in S4 is not clipped, it is marked null

A speed above `max_plausible_mps` is neither zeroed nor clipped, it becomes `null`: it is a
red flag on the projection, not a runner, and substituting a number for it is not allowed.
Speed is computed by a sliding least-squares fit over a window, `null` at the window edges.

### 6.10 The "attention seconds" metric was reconciled to the same set of tracks

**Decision.** `gaze_seconds_median_facade_*` is computed only over tracks with counted
`gaze`/`stop_and_gaze` events that are not marked `low_confidence`.

**Why.** The first version counted every ray hit indiscriminately and reported "0.4 s of
attention" at storefront M1 with zero turned tracks: two numbers about the same thing
contradicting each other.

### 6.11 The numbers of the final hour

This table is generated from `out/metrics.json` by
[`scripts/decisions_table.py`](../scripts/decisions_table.py), not typed. The
previous version was carried over from an earlier run and disagreed with the
artifact in every cell, including which storefront leads: it said M2, while the
artifact, the dashboard and the README all say M3. Two documents naming a
different best storefront is a defect this project has already shipped twice,
and a hand-typed copy of a computed table will drift again on the next run.

| storefront | visitors | stopped | turned (95 % Wilson) | median attention | median time in zone |
|---|---|---|---|---|---|
| M1 角煮/げんかつ | 2 198 | 0.05 % (n=1) | **2.46 %** [1.89, 3.19] (n=54) | 1.3 s | 18.4 s |
| M2 入口/らーめん | 2 551 | 0 % (n=0, upper 0.15 %) | **7.53 %** [6.57, 8.62] (n=192) | 2.4 s | 17.0 s |
| M3 芝浦ホルモン | 2 588 | 0.08 % (n=2) | **13.33 %** [12.08, 14.69] (n=345) | 1.4 s | 16.8 s |
| M4 お好み焼き | 2 706 | 0.15 % (n=4) | **3.40 %** [2.78, 4.15] (n=92) | 0.6 s | 16.3 s |

3 359 tracks in total, 10 043 events; the sum reconciliation in S8 balances on all 9 checks.

The leader is **M3** at **13.33 %** [12.08, 14.69] over n=345 turned tracks, and its interval does not overlap the next storefront's.

**The caveats that travel with the table.** Orientation coverage is **45.6 %**
for the body and **22.4 %** for the head — the turned share is computed over a
biased subsample (large, unoccluded people). Tracks, not people: the tracker
breaks trajectories and merges different ones, and IDF1 is not measured. The
median time in zone is close to the typical track length and reflects the
duration of observation more than a pause at the storefront.

---

## 7. Colour

### 7.1 The first run classified the camera's colour cast, not the clothing

**Symptom.** 21 "blue" tracks out of 25, all with a hue in 107..124. The median hue of
**the whole frame** is 110.

**Measurement** `scripts/hue_floor.py` (median hue per saturation bin, 3 frames,
6.2 million pixels):

| s bin | share of pixels | median hue | gap from the scene hue |
|---|---|---|---|
| 0–30 | 0.512 | 110 | 0 |
| 30–45 | 0.168 | 109 | 1 |
| 45–60 | 0.074 | 113 | 3 |
| 60–80 | 0.079 | 110 | 0 |
| 80–100 | 0.054 | 109 | 1 |
| 100–130 | 0.044 | 107 | 3 |
| **130–180** | 0.039 | **23** | **87** |
| 180–256 | 0.030 | 99 | 11 |

Hue only breaks away from 110 at s ≥ 130, where the neon signs live.

### 7.2 The threshold of 130 was a masking error and has been returned to 45

**Decision.** `s_achromatic_max` returned from 130 to **45**.

**Why 130 was wrong.** The value rested on a measurement, but the measurement answered a
different question. A threshold of 130 did not measure chroma — it declared almost
everything achromatic and thereby **masked** the camera's colour cast: about **85 % of the
sample collapsed into "grey"**, which swallowed dark blue, beige and black. After it,
chromatic tracks numbered zero out of 28, and that looked like a solved problem when it
was a hidden one.

**What was done instead.** The cast is **compensated**, not hidden: white balance on the
road surface inside the ROI.

**The cost.** 45 is the original value, NOT CALIBRATED: never checked against labelled crops.

### 7.3 White balance on a reference surface

**Decision.** The grey point is the median BGR of the road surface inside the ROI: a mask
via `fillPoly` minus person boxes with an 8 px margin, minus blown-out pixels, minus
saturated ones. Normalisation is by the **mean of the three channels**.

**Why this way.** Not by green and not by the maximum: the `v_black_max` and
`v_white_min` thresholds are absolute, and any other normalisation would silently shift
what they mean; normalising by the maximum would additionally conjure white shirts out of
nowhere. Boxes are grown by 8 px because they hug the person tightly while a shadow and
the edge of a coat fall outside them. Blown-out pixels are excluded because they have
already lost their chroma and **understate** the visible cast.

**Numbers.**

| Quantity | Value |
|---|---|
| BGR coefficients | **0.8955 / 1.0160 / 1.1123** |
| coefficient drift across frames, p10–p90 | B 0.888–0.904, G 1.010–1.019, R 1.106–1.129 |
| road-surface saturation before / after | **62 → 23** |
| permitted range of the coefficients | 0.5–2.0, outside it the stage fails |

**A caveat that goes into the report.** "The road surface is grey" is an **assumption, not
a measurement** (`reference_is_measured: false`): if the paving is warm, grey-world on it
over-corrects towards blue — the same error from the other side. And the drop in
background saturation proves the correction **was applied**, not that it is **right**: the
coefficients were computed from that same background. The only non-circular check is hand
labelling.

### 7.4 The class is chosen by a vote over crops, not by a median HSV

**Decision.** 8 crops per track, the winner decided by vote; `hsv_*` in the artifact is the
median over only the frames that voted for the winning class.

**Why.** A median over frames blends different lighting conditions into one number that
occurred on no frame: a person under a red sign and the same person in shadow give an
"average" hue equal to no observation. The median over agreeing frames only — so that the
recorded hue does not contradict the recorded class.

**What `top_color_conf` means.** The share of frames for the class, multiplied by the share
of agreeing pixels. Both are shares of agreement, **neither is a probability of being right**.

**Numbers for the final hour.** Coverage **48.2 %**: ok 1620, box too small 772, low
agreement 967. Distribution: blue 606, grey 458, black 408, orange 61, red 39, white 31,
green 8, purple 4, pink 3, yellow 2. The chromatic classes are back — that is exactly the
difference between compensating the cast and masking it.

### 7.5 The S7 gate: 50 crops can reject 0.70 but cannot confirm it

**Decision.** The volume threshold was lowered from 150 to 50 crops deliberately, and the
reason is recorded.

**Why this changes what the gate means.** 78 crops across 28 tracks are available. Fifty
crops at `max_per_track = 1` give about **28 independent observations**, not 50. The Wilson
interval at such an n is wide: at an observed 0.70 it is roughly **[0.56, 0.81]** and
covers the threshold itself. So the gate can **reject** a bad classifier but **cannot
confirm** 0.70.

**Consequences, mandatory for the report.** Accuracy is printed only together with the
interval; the decision is taken on the lower bound; when the threshold is covered it prints
"not confirmed" rather than "ok"; a confusion matrix over 50 crops and 10 classes is sparse
(~5 examples per class) and goes in as an illustration, not as a metric. `CLAUDE.md` and
`configs/s7_attrs.yaml: gates.n_labelled_crops` stay at 150 as the recorded requirement;
the actual n and its cost are printed alongside.

### 7.6 Gender, age and ethnicity are forbidden

**Decision.** `forbidden: ["gender", "age", "ethnicity"]`, no blocks on the dashboard, and
the page says so explicitly.

**Why.** Privacy, and the absence of ground truth for a gate. The second matters no less
than the first: a number that cannot be checked is not published, whatever the subject.

---

## 8. Evidence crops and provability

### 8.1 Collecting evidence crops is a mandatory step of a stage

**Decision.** `_base.build_evidence()` is called in the scaffold of every stage and fails if
the config has no non-empty `evidence_claims`; `finalize_evidence()` fails if a declared
claim collected no frames at all.

**Why.** A number whose frames cannot be shown does not go into the report.

### 8.2 Selection is stratified, not top-N

**Decision.** Candidates are split by confidence rank into three strata, quota 4/4/4 at n = 12,
with the remainder of the division going to the **lower** strata. The memory ceiling is
reservoir sampling with a fixed seed (20260903).

**Why.** Weak examples are the most informative for an audit; uniform sampling preserves the
confidence distribution, so stratifying afterwards stays correct. Shortfalls and the number of
discarded candidates are written into the statistics rather than swallowed: for
`claim.detect.near_half`, 461 944 candidates were offered, 2 000 kept, 459 944 discarded.

Three columns beyond the required nine were added to the index: `extra_json` (otherwise `extra`
from `add()` would be lost silently), `stratum` (otherwise there is no way to check that the
selection is stratified), `schema_version`.

### 8.3 The evidence index is one catalogue keyed by stage

**Decision.** A stage replaces everything it wrote **under its own name** and does not touch
anything else. Rows whose file has disappeared from disk are dropped.

**Why.** `EvidenceWriter.finalize()` rewrote the index whole: an S7 run erased the evidence
crops of S4 and S6 — the jpgs stayed on disk, the witness links did not, and the dashboard was
quietly left with nothing to back it. The key is `stage` and not `claim_id`: with a filter on
`claim_id`, a claim a stage had stopped emitting would stay in the index forever — which is how
the dashboard printed "nobody counted" for storefront M1 and showed its evidence crops from an
earlier run at the same time. A link with no file is a promise of evidence that does not exist.

### 8.4 Storefront evidence crops are counted per (track, storefront) pair

**Decision.** The filter for "counted" tracks is pairwise, not global.

**Why.** A track counted as turned at M2 passed the global filter at M1 too, and a storefront
with an honest zero got a grid of evidence crops contradicting its own number.

### 8.5 Deduplication: no more than 2 frames per track

**Decision.** The evidence pool is balanced across tracks, and the caption names the number of
people: "12 frames of 2 people".

**Why.** Twelve frames of one person read as twelve visitors.

### 8.6 Anonymisation: pixelation plus a Gaussian, with a postcondition

**Decision.** The top **0.30** of the crop height (was 0.22): downsample ×16 `INTER_AREA` →
upsample `INTER_NEAREST`, then a Gaussian. The write to disk is cancelled with an error if the
face region did not change. The parameters are written into `extra_json`.

**Why.** A Gaussian is an invertible convolution; deblurring against it restores facial
features. Pixelation destroys information finer than the block irreversibly, and the Gaussian
on top removes the block edges. The fraction was raised from 0.22 to 0.30 because at distance
the head sits higher in the crop and 22 % did not cover it. The test
`test_pixelation_destroys_subblock_detail` checks exactly the irreversible part: the residual of
the image relative to the block grid falls by more than a factor of 10 — on a pure Gaussian the
test would not pass. The contact sheet enlarges crops with `INTER_NEAREST`: interpolation would
smooth the blocks and the anonymisation would look better than it is.

**Integrity tests** (`tests/test_evidence_files.py`): there is one path to disk and it is
guarded by anonymisation (checked against the source, not taken on trust); every index row
points at a file that exists; every row has `blur_*` metadata, which only `blur_face_region` can
set; the height of the anonymised region agrees with the declared fraction.

**A false alarm, recorded in a test.** Relative and absolute sharpness of the upper region
flagged 25 "suspicious" files. The cause was not a leak but the metric: on short crops the top
30 % catches clothing below the blur boundary, and the Laplacian spikes on a striped shirt.
Recorded in a test so that nobody reinvents this check.

### 8.7 Lightbox: a crop opened 1:1

**Decision.** The CSS set only `max-width` and `max-height`, so a small image had nothing to
grow into: a 30×98 crop opened as a 1:1 stamp on a 1440×900 screen, although the caption
promises "open large". The image now stretches to the height: the same crops open at 215×702,
an enlargement of 2.2×–7.2×.

`image-rendering: pixelated` is deliberate: a sevenfold enlargement with smoothing would draw in
detail that is not in the data. The real crop size was added to the caption so that a
full-screen stamp does not read as a high-resolution shot.

### 8.8 Replay counters reduced to one set

**Decision.** The page printed 279 ray hits while the dashboard, from the same data, gave
3 turned tracks. The chain was measured and published:
**279 raw → 243 after dropping grazing angles → 62 frames across 3 tracks** once tied to S6
events. The counters were renamed so that instantaneous stops being confused with cumulative:
`storefronts lit NOW`, `people turned so far`, `ray-hit frames so far`.

**Why.** Two numbers about the same thing have no right to differ by an order of magnitude in
two places of one report.

### 8.9 The report must carry limitations, not only numbers

**Decision.** `out/metrics.json` contains the sections `unmeasured`, `limitations` and
`reconciliation`; S9 draws "Limitations" as a separate table before the "Not measured" block.
Every item with a consequence, not only a statement. Every number carries `source_stage`,
`source_artifact` and `compute_ref` — the file, function and line that computes it.

**Three current limitations:** the source of scale (height is not an independent check); the
street grade (a flat ground model against a drift in height); the vanishing point disagreeing
with the horizon by 104 px against an 87 px tolerance.

**Separately.** S6 collects no evidence crops: it works from artifacts and reads no frames. The
contact sheets in the report come from the evidence crops of S3, S4 and S5.

### 8.10 Hand labelling is the only non-circular check in the project

Everything else is the model agreeing with itself. The S5 and S7 gates fail without labelling
and rightly so; that is correct behaviour, not a shortcoming.

---

## 9. Performance and budget

### 9.1 A heavy stage is timed on a clip first

**S1 timed on the debug clip** (5400 frames, 1080p30): warm-up (model load, CUDA,
3 frames) 23.2 s; 30 sampled frames 19.3 s. Those 19.3 s include **sequential decoding of
the whole clip**, not inference alone, and it does not grow with the number of samples:
300 frames come to about 30 s per pass, and S1's two passes to about a minute.

**Why the read is sequential.** On a `.ts` built from HLS segments, `CAP_PROP_POS_FRAMES`
lies at segment boundaries, and the sample would drift. Slower, but the timestamp is honest.

### 9.2 Numbers from the final hour-long run

| Stage | Time | Volume |
|---|---|---|
| S3 detect | 1978.5 s, **18.2 frames/s** | 36 000 of 108 000 frames, **556 987 detections**, 129 918 dropped by the 70 px cutoff |
| S4 track | 430.1 s | **3 359 tracks**, 481 751 rows, 100 % indirect foot points |
| S5 pose+orient | 1598.3 s | 238 895 pose persons, 221 788 matched; body coverage 45.6 %, head 22.4 %, ankles 37.6 % |
| S6 attn | — | 9 610 events |
| S7 attrs | — | coverage 48.2 % |

All in fp16 on an RTX 4070 Laptop, 8187 MiB VRAM. The stage **fails if CUDA is
unavailable**: `half=True` on CPU silently returns rubbish.

### 9.3 The peak hour is chosen by code, not by eye

**Decision.** `scripts/pick_hour.py` scans the recordings at a 60 s step and looks for the
busiest continuous 60-minute window.

**Numbers.** 53 files, 756 measurements; the chosen window is **16:25–17:25 JST, 19.02
people in frame on average** against 8.4 for the rejected morning window. 7 files
concatenated, and the sha256 of the result recorded.

**Two decisions about reproducibility.** Time comes from the **file names**, not from the
system clock — that was the main hole in the first version of the presence script: JST
labels were computed from the wall-clock time of the run, and the chart shifted on a
re-run. Concatenation is byte-for-byte: MPEG-TS has no global header, so there is nothing
to re-encode. Checked: two consecutive runs give a byte-identical csv.

**The script's inference parameters are the same as in `configs/s3_detect.yaml`** (imgsz
1280, conf 0.25, classes [0], fp16) — otherwise the presence curve and the pipeline itself
would be counting different people. The script is not a stage and has no gate, but its
number reaches the report, so `density_meta.json` sits beside it with the sha256 of the
input file and of the weights, the parameters and the library versions.

**The search for the densest window** requires the window to fit entirely inside the
recording: otherwise the last windows are averaged over fewer measurements and win undeservedly.

### 9.4 Matching a track to a detection: the threshold is relative

**Decision.** The tolerance on foot-point disagreement is **0.08 of the box height**.

**Why an absolute threshold is the wrong criterion here.** 15 px on a 40 px box and on a
204 px box are different questions. The first version demanded a match tighter than 1 px:
75 % of people got a box. A 6 px threshold raised coverage to 98.3 %, but failed storefront
M3, whose only frame had a disagreement of 15.2 px at a box height of 204 px.

**Measured relative distance:** median 0.006, p95 0.024, **p99 0.044**, maximum 0.36.
The 0.08 threshold is almost twice p99, and coverage is **99.81 %**. This is not fitted to
M3: 0.08 was chosen as an ordinary margin over p99 and will give a different number on another recording.

### 9.5 The orientation arrow's length is set in pixels, not in metres

A 1.6 m segment, projected in full, runs across the whole screen for a person facing the
camera: the far end ends up nearer than the camera. Only the direction is taken from the plan.

### 9.6 Small presentation decisions

Overlay: one video instead of two (the plan is drawn beside it in the replay and in sync,
and a second window inside the video is another pass over the hour-long recording, ten
minutes). The whole hour is 36 000 frames and a file that will not open, so the overlay is
cut to the densest window; the frame folder is cleared before a render, otherwise last
run's frames are indistinguishable by name.

Serving: nginx rather than `python -m http.server`, because the replay seeks within an
hour-long video, which needs HTTP Range (checked: the mp4 is served with code 206). Plus a
fallback `scripts/serve.py` without Docker with the same Range support — a demonstration
should not depend on whether Docker Desktop is alive.

The plan's frame in the replay is taken from the confidence region rather than from every
position in the hour: sparse distant points stretched the view, and the people in frame
ended up as a blob in the corner of an empty field. The trail is cumulative — without it
you see eight points in emptiness at any instant, and "the street is straight, people walk
along it" cannot be checked (canvas fill after 6 s: 8.2 % against ~2 %). One storefront
must be one colour everywhere: in the overlay, on the plan and in the dashboard.

The `colors.html` page was deleted: it duplicated the clothing section of the dashboard and
contradicted its own numbers ("no chromatic colour found on any track" above a table with
blue 606 and black 408; "threshold raised from 45 to 45"). Two pages about one thing, one
of them lying, is worse than one page that is right.

---

## 10. What is not measured and what remains open

### 10.1 Unmeasured quality metrics

| Metric | Stage | Reason |
|---|---|---|
| AP@0.5, near and far half separately | S3 | no labelling for 300 frames |
| IDF1 and ID switches | S4 | no labelling |
| Angle MAE on 200 people | S5 | 50 of 200 labelled on `raw/peak_hour.ts`; MAE 21.2° [15.5, 28.8] — the interval covers the 25.0 threshold, so the gate reads not confirmed |
| precision of "stopped + looking" | S6 | no labelling for 100 events |
| upper-garment colour accuracy | S7 | no labelled crops; the sample threshold was lowered to 50, which cannot confirm 0.70 |
| recall by depth | S3 | the 70 px cutoff is a proxy, not recall |
| recall curve for the ROI boundary | S2 | not re-justified after the scale changed |

### 10.2 Open questions

**The recorded drift contradicts itself.** The 4.9 % grade was computed from a slope of
−0.0188 m/m, while the artifact stores −0.0163 m/m (multiplied by the scale rescale
factor). A slope is a ratio of two lengths and is invariant to scale; one of the two
numbers is wrong. See 4.2.

**The street vanishing point disagrees with the horizon.** 103.7 px against a tolerance
of 87.3 px. Two independent estimates of one quantity did not converge; the focal length,
and with it the scale, are less well determined than we would like.

**The orientation sample is biased.** The "turned toward a storefront" shares are computed
over the 45.6 % of tracks that have an angle at all, and those are the large, unoccluded
people.

**The MAE was measured on data other than the data the metrics come from.** The labelling
is a three-minute morning clip, the metrics an evening hour; occlusion is heavier in a
dense scene.

**The stop threshold and `min_score_for_event` are not calibrated**, and they directly
determine the two headline numbers of the report.

**The S5 gate is not green.** MAE 21.2° is under the 25.0 threshold but its interval [15.5, 28.8] covers it, and 50 people are labelled against the 200 the rules ask for. The threshold was not touched.

**The anonymisation blocker was cleared by code, not by a person.** The parameters are
checked by tests and contact sheets; confirmation by the owner on real crops remains his
to give.

## 11. Prose audits: commit messages and the README

The code was checked by gates and tests from the beginning. The prose — the commit
messages and the README — was checked by nothing, and that turned out to be the most
productive hole in the process across the whole project.

### 11.1. Commit messages against the repository

After the history was squashed into 12 commits, the messages were read back against
the code and the artifacts. **Five claims did not hold**, and all five had been
written by me on the same day:

| Commit | What it claimed | What is actually there |
|---|---|---|
| S0 ingest | records the stream, checks for gaps in the segments, fps and resolution | `s0_ingest.py` is 28 lines of scaffold, `status=not_implemented`, zero recording calls |
| Scaffold | "faces are already anonymised, nothing identifiable reaches disk" | 173 of 257 crops at a threshold of 0.22, which the owner had rejected |
| S1 | scale cross-checked two independent ways | the artifact holds neither `pilot_speeds_mps` nor `facade_baselines`; it says itself that one check remains |
| S2 | the ROI drops a track before the attention stage | the word `roi` appears in neither `s4_track.py` nor `s5_orient.py`; the ROI only trims pictures |
| Scaffold | every artifact's schema is pinned in CONTRACTS.md | `attn/track_zone_frames.parquet` had no contract, with four consumers |

Four wordings were rewritten. The fifth was fixed on the substance: section 8.2 was
added to `docs/CONTRACTS.md`.

### 11.2. The README against the code

Then every claim the README makes about what the system can do was read against the
code that has to implement it. **Fifteen did not hold.** Two are worth naming:

- **"Every stage has a gate"** — **two gates of the ten** compute a metric. The other
  eight print honestly what they cannot do and return 1. In the terminal that was
  visible; in the README it was not.
- **"Zone entry: tracks whose ground position enters a storefront's apron
  polygon"** — no such metric exists. `visitors_*` counts tracks that came within
  8 m of the facade, and **the two quantities differ by a factor of thirty**.

The rest are of the same kind: a manual step described as automatic, a stale test
count, `make run-all` without S0 and S7, "no face imagery" with two street frames in
`docs/img`, and `zones/zones.json` absent from the repository when S2 will not start
without it.

All twenty were fixed before publication.

### 11.3. Caveat: one slice went through without an adversarial check

The README audit ran in four slices, and every finding was meant to face an
independent attempt to refute it. For one slice that attempt **did not run** — it hit
the session limit. Its findings were closed anyway, and the key facts (the number of
working gates, what `make run-all` covers, the absence of `zones.json` and of
`torch`) were re-checked by hand with commands.

The check was run separately later. The caveat stays here as it is: the decision to
act on unverified findings was taken before it, and recording that is more honest
than pretending after the fact that the order was right.

### 11.4. What follows from this

Prose about a system drifts from the system faster than the system drifts from
itself. Gates catch regressions in the code; nobody looks at the README, and it goes
quietly stale with every change that touches it. Twenty discrepancies against nine
code defects is a ratio, not an accident.

A mechanical re-read of the prose against the code is cheap and finds a lot. On the
next project it belongs in the same row as the tests, rather than being done once
before publication.

## 12. What is published on GitHub Pages, and why not all of it

### 12.1. Decision

Two pages are published: `dashboard.html` and `benchmark.html`. The evidence-crop
set is cut to **12 frames per storefront, 48 in all**, selected stratified by
confidence — the same selection as locally.

Not published: `report.html`, the full archive of evidence crops (394 crops),
`overlay.mp4`, the replay page. The video goes to YouTube; the header button points
there.

`report.html` was in the original decision and was taken out on a measurement. The
page embeds four full-size JPEG figures, and the same check the overlay is held to —
facial keypoints at `conf >= 0.5` whose 24x24 neighbourhood has a Laplacian variance
above 13.4 — finds **five sharp facial keypoints on each of two of them**. Those
figures are drawn before any anonymisation runs. Publishing the page would put
unblurred faces on an indexed URL, which is exactly what 12.2 argues against; the
decision cannot be exempt from its own reasoning. The page is still built by every
run and is the internal engineering report.

### 12.2. Reasoning

**Blurring a face does not make the data non-personal.** The top 30 % of the crop
covers the head, but the crop still carries clothing, build, gait, companions, an
exact time and an exact place. Together that is enough to recognise a person —
especially for someone who was there. Anonymisation here lowers the risk, it does
not remove it.

**A public indexed URL changes the scale of the risk.** While the pages sat
locally, one person saw a crop. On Pages it enters search results and lives in
caches indefinitely. That is a different quantity, not the same one with a
correction.

**APPI.** The recording is made in Japan, and crops of passers-by on a Kabukicho
street fall under the Japanese personal data protection act. Publishing 394 crops
of real people to demonstrate a methodology is a risk out of proportion to the
benefit: the methodology can be checked on 48 as well.

**What the reader loses.** The ability to see every frame behind every number.
That is a real loss for a page whose value is traceability, and it should be
named rather than hidden: the full archive is available from a local run, and the
README says so next to the link.

### 12.2.1. `run_manifest.json` ships, and why it did not before

Rule 6 exists so that "we use YOLO" is not an acceptable answer: the weights
name, their sha256, the input resolution, the device, the precision and the
library versions are written to `run_manifest.json` on every run. The file was
gitignored, which made the rule unverifiable by anyone who cloned the
repository — and this document names the manifest as a source of its own
numbers three paragraphs into section 0. It now ships, next to
`zones/zones.json` and `calib/homography.json` and for the same reason. It
holds no personal data: model and config hashes, timings, library versions and
the four storefront names.

Publishing it turned up something first. The manifest recorded `s2_zones` with
`status: failed` and an error about a missing `zones/zones_FRESHCLONE.json`,
against a config named `configs/_s2_freshclone_test.yaml`. That was not the
run: it was a throwaway invocation used to check that S2 fails loudly on a
clone with no traced zones, and it had overwritten the real entry. The manifest
keeps one record per stage, so a test run silently replaces a real one — worth
knowing, and the reason the file is worth reading before it is published rather
than after.

S2 was re-run with the real config to restore a truthful entry: 8.5 s, and
`zones/zones.geojson` came out byte-identical (sha256 f97fecb8466025f7 before
and after), so nothing downstream is affected. Every stage now reads `ok`
except `s0_ingest`, which reads `not_implemented`, which is true.

### 12.3. How it is done

`--public` on `make_dashboard.py` and `make_benchmark.py`. In that mode the
archive and the colour grids are **not built** rather than hidden with styles: a
block hidden by styles would still go into the markup and be readable in the page
source.

In public mode the header links only to published pages, and the video link goes
to YouTube. The build **fails** if a link to an unpublished file is left in the
finished page: the check costs one line, and a broken link on Pages is otherwise
found only by a reader.

Files are exempted in `.gitignore` **by name**. Not the `out/` folder and not a
`*.html` mask: the mask would have pulled in `replay.html` with its link to a local
mp4, and `report.html` with the figures above.

## 13. Overlay anonymisation

### 13.1. Why it was necessary

The page refused to publish 394 blurred crops on privacy grounds — and then
embedded three minutes of video of the same street, faces unblurred, at
1920x1080. More data, and better quality, than what we had refused to
publish.

Softening the wording in 12.2 was rejected by the owner: a principle is not
adjusted to fit what has already been done. Removing the player was rejected
too — the video is the main thing the demonstration shows. One option left:
render it again.

### 13.2. How it is done

Anonymisation is **part of the render, not an option**. There is no flag to
turn it off and there will not be: an option that can be forgotten will
eventually be forgotten, and the price of forgetting here is a published face.
The test `test_anonymisation_has_no_off_switch` guards against such a flag.

Order matters: anonymisation is the first thing that happens to a frame,
BEFORE the boxes and arrows are drawn. Otherwise the blur would wipe out the
tracing instead of the faces.

Three decisions, each forced by a measurement:

**A separate detector pass, not the stored detections.** In
`det/frames.parquet` the detections are filtered by the production confidence
threshold and by the 70 px box-height cutoff: anyone smaller or less certain
is simply not there. Right for drawing boxes, wrong for anonymisation.

**Detector threshold 0.05 instead of the production one.** The price of a blur
too many is a blurred lamppost; the price of a miss is a published face.
Recall matters more than precision, and this is not the trade-off to nudge for
a better-looking frame.

**Two models instead of one.** Measured over eight frames: the detector at
0.05 missed 11 people whose faces the pose model found with confidence. The
boxes are taken as a union; duplicates are not removed, since blurring one
head twice does no harm.

**The blur follows the head that was found, not a band.** A band across the top
30% of a person's box is an approximation, and it broke on frame f060369: the
person was looking down, the nose landed inside the band and the chin fell
below its hard edge. The facial keypoints say where the head ACTUALLY is; a box
is built around them with 90% of their size as padding, and all of it is blurred.

### 13.2.1. Fallback path: a person with no head keypoints

**The person box is ALWAYS blurred, whether or not a pose was found.** Head
boxes go ON TOP of it, not instead of it. The case is not hypothetical: an
occluded person, one filmed from behind, one cut off by the frame edge — the
pose model returns no keypoints for any of them. If the blur depended on
keypoints, such a person would go into the frame uncovered, and nothing would
catch it.

Checked by `test_person_without_head_keypoints_still_gets_the_band`. The test
was verified by mutation: make the person box depend on a pose being present
and it fails.

**The kind of box is marked explicitly.** The first version told a head from a
person by aspect ratio: anything less vertically elongated than 1.7 counted as
a head and was blurred whole. That worked only by luck — `HEAD_PAD = 0.9` held
the head box at a ratio of 1.556, 0.14 from the threshold. Reducing the padding
to 0.6 (a reasonable wish: less blur than necessary) would have quietly turned
a head into a "person", of which only the top third is blurred, and left the
face exposed. The kind is now set when the boxes are collected, and dropping
`HEAD_PAD` to 0.4 in the mutation check breaks nothing.

### 13.3. The measurement

Confident facial keypoints (nose, eyes, ears; conf >= 0.30) are counted by a
pose model — a **different** one from the detector that did the blurring:
identical models would prove only that a model agrees with itself.

What is measured is not the number of keypoints but the **sharpness around
them**. The model locates the HEAD from the shoulders and the torso and
confidently puts a "nose" on a pixelated blob; the danger is not that the head
is visible but that the face is. The threshold of 13.4 is the p5 sharpness on
un-anonymised frames: even the blurriest real face is sharper.

The measurement is taken on the frame immediately after anonymisation and
BEFORE anything is drawn. That was learned the expensive way: on a finished
frame the model placed an "ear" with confidence 0.32 on the green caption
`1916 blue 2.5s`, where there is no face at all. A test on the finished frame
would be measuring the quality of our own graphics.

**20 frames, evenly spaced across the hour:**

| | before | after |
|---|---|---|
| confident facial keypoints | 354 | 1 |
| sharpness around them, median | 26.0 | 1.2 |
| sharpness, maximum | 72.0 | 1.2 |
| **keypoints above the 13.4 threshold** | **341** | **0** |

Checked by `tests/test_overlay_anonymised.py`, which skips on a fresh clone
with no weights and no recording — the synthetic part runs always.

### 13.4. What the measurement does NOT prove

That no face is recognisable by a human. It proves that where the pose model
finds a face, no high-frequency detail is left. Clothing, build, companions,
time and place still identify a person — exactly the argument for not
publishing the full crop archive in 12.2, and covering the faces does not
retire it.

### 13.5. Two surfaces, two decisions

The repository and the YouTube link are governed differently, on purpose, and
the difference is the owner's decision rather than an unfinished task.

**Everything in the repository is anonymised.** The seven figures in
`docs/img/`, the images embedded in the published pages, the crops written by
the stages — all of it goes through the single write path of section 14, and
the gate measures zero sharp facial keypoints across the lot. This is the
surface that is permanent: a public git repository keeps its blobs reachable
by SHA after a delete, and GitHub Pages is indexed and cached.

**The recording on YouTube is the earlier render and is not anonymised.** The
owner chose to keep it for one demonstration and to remove it afterwards. That
choice is defensible where the repository one is not, for a reason worth
stating plainly: a YouTube video can actually be deleted, and a commit cannot.
The underlying stream is public either way; what differs is who is
redistributing it and for how long.

The anonymised render exists — `out/overlay.mp4`, 89 MB, measured in 13.3 — and
replaces the linked one after the demonstration. Until it does, the claim
"the renderer anonymises by construction" is a claim about the code and about
every file in this repository, not about that particular upload, and README
says so at the link.

---

## 14. Anonymisation as a path, not a step

### 14.1. What was found

Section 13 made the overlay video anonymised by construction. It closed one
file. On 2026-09-06 the same question was asked of every other path that
writes an image, and the answer was worse than expected.

Measured with the project's own metric — the one in
`tests/test_overlay_anonymised.py`: pose at `imgsz=1280 conf=0.25`, facial
keypoints COCO 0-4 at `conf >= 0.30`, sharpness = mean absolute Laplacian in a
30x30 window, exposed above 13.4, the fifth percentile of sharpness around
facial keypoints on **un-anonymised** frames.

| Where | Images with sharp facial keypoints |
|---|---|
| `raw/probe_*.jpg`, the control | 18 of 18, up to 21 keypoints |
| `docs/img/*.webp`, tracked in git | 0 of 7 |
| **`out/dashboard.html`, live on GitHub Pages** | **14 of 51; one carried 27 of 31** |
| `out/report.html`, not published | 2 of 4 |
| `out/img/zones_ref.jpg` | 26 |
| `calib/debug_{vp,selfcalib,stub_affine}.png` | 10-11 each |
| `out/check_all_compact.jpg` | 4 |
| `out/overlay_plan_frames/` | 47 of 60 |
| `evidence/` crops | 86 of 668 at the time of the audit; re-measured after the rebuild as 79 of 668 — 75 of the 620 old, 4 of the 48 new |

The published file was downloaded from the live site and compared with the
local one: byte for byte identical. This was not a risk, it was a publication.

### 14.2. The cause was one default

```python
def b64_img(path, max_w=None, quality=92, anonymise: bool = False):
```

Two call sites in the same file. `sheet_html` passed `anonymise=True` for the
evidence crops. The figure block, `make_dashboard.py:1394`, did not, and got
the default. One remembered, one forgot.

Nothing about that is unusual, and that is the point: a step a caller can skip
will eventually be skipped. The defect was not in the page. It was in
anonymisation being a step at all.

### 14.3. The fix: one write path

`looq/anonymise.py` is now the only place in the project that turns pixels
into bytes. `save_image`, `encode_image`, `data_uri` and `save_figure`
anonymise first, always. There is no parameter that disables it, because that
parameter was the defect.

Every previous writer now calls it: `make_figures`, `make_readme_figures`,
`make_dashboard`, `contact_sheet_all`, `check_blur`, `density_curve`,
`pick_hour`, `anonymise_figures`, `render_overlay`, `s1_calib` (six debug
images), `s9_report` and `looq.evidence._write_jpeg`.

`render_overlay` kept a second copy of the anonymiser. It does not any more —
it imports the one implementation. Two copies of the same guarantee drift:
you fix one and forget the other, which is the defect of 14.2 wearing a
different hat.

**No exemption for images without people.** A matplotlib figure with no faces
is tempting to exempt, and an exemption by image type is exactly the loophole
being closed — "it's only a plot" reads the same whether or not the plot has
an `imshow` of a frame in it. Measurement removes the argument: a detector
pass over a people-free figure costs 0.13 s and returns zero boxes.

**Detector, not a fixed fraction.** `blur_face_region` blurs the top share of
a crop. Where the head is not in that share — a bent, occluded or edge-cropped
person — the face survives; that is why 79 of 668 crops on disk are exposed,
and why 11 crops embedded through the *old* `anonymise=True` path were still
exposed on the published page. The detector plus pose finds the head where it
is. The fixed fraction stays as the mandatory fallback for a person whose pose
gave no head keypoints, and is not removed.

### 14.4. The gate found a defect in its own fix

The first version of `_imgsz_for` ran inference at a resolution matched to the
image: 320 for a small crop. The gate failed on four crops. Diagnosis: at 320
the pose model found **no head keypoints at all** on a 114x248 crop, so only
the fallback applied — the top 30 % of the person box — while the face lay at
82 % of the crop height. The same crop at 1280 gives a nose at confidence 0.95.

The rule is now explicit: **the anonymiser looks at no lower a resolution than
the gate does**. `CHECK_IMGSZ = 1280`. Otherwise it cannot see what the check
will see, and that is not a hypothetical — it cost four crops on the first
build.

### 14.5. Enforcement

Three tests in `tests/test_single_image_write.py`:

- `test_no_raw_image_write_outside_the_module` walks the **syntax tree** of
  every file in `looq/`, `scripts/` and `verify/` for `cv2.imwrite`,
  `cv2.imencode`, `savefig`, `imsave`, a PIL `.save`, and for
  `from cv2 import imwrite`. Not grep: grep does not see a renamed import, and
  it cannot tell a call from the docstring in `looq/evidence.py` that
  describes this very prohibition.
- `test_anonymise_has_no_off_switch` checks the signatures for a parameter
  that could disable anonymisation, and checks by AST that `encode_image`
  calls the anonymiser and that `save_image` goes through `encode_image`.
- `test_no_sharp_faces_anywhere` is the measurement: every image tracked in
  git plus every `data:` URI in `out/dashboard.html` and `out/benchmark.html`
  — 58 images — must carry zero sharp facial keypoints. It skips without model
  weights, because a gate that cannot run stops being run and stops
  protecting.

Both prohibitions were mutation-tested rather than assumed. Appending a
`cv2.imwrite` to `scripts/make_figures.py` fails the first with the file and
line; calling `_install_models_for_tests` from `scripts/make_dashboard.py`
fails the second the same way. A gate nobody has tried to break is a comment.

**`tests/conftest.py` is a compromise, not a design.** It should be read as
one, because it was not planned and exists only to repair a consequence.

Routing `looq.evidence` through the detector made crop writing depend on 40 MB
of model weights. In a real run that is correct and even desirable — S3, S5 and
S7 load those weights anyway, and failing without them is rule 8. On a clean
clone there are no weights, and the effect was measured on a clone of the
pushed commit: **12 failed, 97 passed, 4 skipped**. Twelve evidence tests write
synthetic noise crops to check the blur band and the index bookkeeping, and
they could no longer write anything at all.

`tests/conftest.py` installs a model that honestly finds nothing, and only when
the weights are genuinely absent; with weights present the real models run.
"Found nothing" is the true answer on a noise crop, not a skipped step. The
clone is back to 109 passed, 4 skipped.

A seam is a liability, so it is closed from the other side: calling
`_install_models_for_tests` or `_anonymise_with` from `looq/`, `scripts/` or
`verify/` fails `test_anonymise_has_no_off_switch`, and that prohibition was
mutation-tested. But the honest summary is that fifty lines of test scaffolding
exist because a production module acquired a heavy dependency, and a design
that needed no scaffolding would have been better.

The alternative was considered and rejected by the owner: leave `evidence.py`
on the fixed band, delete this machinery, and accept that 79 of 668 crops sit
on disk with a visible face while rule 9 says crops of faces are not written to
disk. Their reasoning, recorded as given: a rule the project breaks itself is
worse than fifty lines of scaffolding, and 70 s per run is an acceptable price.
The published surface was clean either way, because the pages re-anonymise
every crop through the detector as they embed it; what the detector pass at
write time buys is the files on disk.

### 14.6. What was not done, and why

The crops on disk under `evidence/` are not regenerated. Doing so means
re-running S3, S5 and S7, and the owner has ruled that out. They are
gitignored, they never leave the machine, and the pages now re-anonymise every
crop through the detector at embed time, so the published surface is clean
while the local files are not. `out/overlay_plan_frames/` is stale output from
before section 13 and is gitignored for the same reason.

`out/report.html` remains unpublished (section 12), now for a measured reason
rather than an editorial one.

### 14.7. What this still does not prove

The anonymiser is detector-limited. A person neither network finds at
confidence 0.05 is not blurred. Measured on the 60 saved overlay frames:
4 sharp facial keypoints across all of them fell inside a person box, meaning
the detector missed those people at render time and found them afterwards.
"By construction" describes the path, not a guarantee that no face survives.
The gate measures the published surface, and the published surface is at zero;
that is a smaller claim than "no face is ever recognisable", and it is the one
supported by the numbers.
