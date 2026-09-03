# CAM-01 Kabukicho — project rules

## What this is

An offline pipeline: 1 hour recorded from a single fixed camera (YouTube live, Kabukicho
Ichiban-gai, Shinjuku) → presence, stop, orientation and attention metrics per storefront
zone + a dashboard + a report on how far the numbers can be trusted.

The deadline is hard. Priority: a working honest system with a smaller scope, not a
complete system with unverified numbers.

## Hardware

MSI Katana A17, RTX 4070 Laptop, 8 GB VRAM. Everything in fp16, everything must fit in
8 GB. On OOM — cut the batch or the resolution and record it in DECISIONS.md. Do not
change the model silently.

## Hard rules

**1. The LLM is not in the inference loop.**
You write code and explain finished numbers. You never look at frames and never draw
conclusions by eye. Every number in the report must be computed by code from an artifact
on disk, and I must be able to point at the line that computes it. If you are asked to
"watch the video and tell me" — refuse, and propose a metric instead.

**2. Stages are isolated and talk only through files.**
No end-to-end "just do all of it". Each stage reads the previous stage's artifact and
writes its own. Schemas live in `docs/CONTRACTS.md`. A schema changes only by an explicit
decision, with a version bump.

**3. No stage counts as done until it has passed its gate.**
A gate is `verify/verify_s<N>.py`, returning 0/1 and printing metrics.
"I checked it visually" is not a gate. "The script printed ok" is a gate.
Do not tune a threshold to make a gate pass. Explain the reason for the failure in
DECISIONS.md first.

**4. The time budget is checked before the full run.**
Any heavy stage runs first on a 3-minute clip; measure throughput, multiply by 20 and
compare against the budget. If it does not fit — cut fps/resolution straight away, not
halfway through a large run.

**5. No silent defaults.**
Every threshold (conf, NMS IoU, the "stopped" speed, the "looking" angle, the gaze
distance) lives in `configs/*.yaml`, carries a comment saying why it has that value, and
reaches the report. If a value was picked at random, the comment says "not calibrated".

**6. Model versions are pinned.**
The weights name, sha256, input resolution, device, precision and tracker version are
written to `run_manifest.json` on every run. "We use YOLO" is not an acceptable answer.

**7. Anything not measured is labelled as not measured.**
Attribute coverage, the share of indirect positions, the ROI boundary — printed into the
report explicitly. 40% coverage with an honest figure beats 100% with rubbish.

**8. Fail loudly.**
If a stage cannot do its job correctly — crash with a clear error. Do not substitute
stubs, do not return empty arrays as success, do not draw a dashboard full of zeros.

**9. Privacy.**
Face crops are not written to disk. ReID embeddings live only inside a run.
Only aggregates leave the machine. `raw/` is in .gitignore.

## Stages and gates

| Stage | Artifact | Gate |
|---|---|---|
| S0 ingest | `raw/*.ts` + `manifest.json` | no gaps in the segments, fps and resolution stable |
| S1 calib | `calib/homography.json` | held-out reprojection error < 0.5 m; scale estimated from the sample's median height, independently confirmed two ways — median pedestrian speed (1.0–1.6 m/s) and satellite street width (6.06 ± 0.8 m); plus height spread as an independent check on the geometry |
| S2 zones | `zones/zones.geojson` (plane coordinates) | ≥30% of detections fall inside at least one zone on 100 random frames; the ROI boundary is justified by a recall-vs-depth curve |
| S3 detect | `det/frames.parquet` | AP@0.5 on 300 labelled frames, near and far half separately |
| S4 track | `track/tracks.parquet` | IDF1 and ID switches over 3 minutes, sparse and dense stretch separately |
| S5 pose+orient | `pose/orient.parquet` | angle MAE on 200 hand-labelled people |
| S6 attention | `attn/events.parquet` | precision of "stopped+looking" on 100 hand-labelled events |
| S7 attrs | `attr/tracks_attr.parquet` | upper-garment colour accuracy on 150 crops + coverage share |
| S8 aggregate | `out/metrics.json` | sums add up, no NaN, confidence intervals present |
| S9 report | `out/report.html` | every number references a stage and that stage's quality metric |

S7 is optional. It is the first thing cut if the schedule slips.

## Key definitions

The threshold for interest in a zone is computed as follows, and the formula does not
change without a record in DECISIONS.md:

```
stop_score  : share of time in the apron polygon at speed < 0.3 m/s lasting > 1.5 s
gaze_score  : share of the time a track spends in the window during which the
              ORIENTATION SECTOR CROSSES the facade segment. Sector = a ray from
              the person's position on the plane in the yaw direction, length
              <= gaze_max_dist_m (8 m), half-width +-yaw_uncertainty_deg.
              Crossed the segment — counted.
approach    : distance to the facade decreasing
entry       : track disappeared inside the entry polygon
```

`yaw_uncertainty_deg` is **the angular uncertainty of the yaw estimate, not a decision
threshold**. Its value is set equal to the measured MAE of the orientation model from the
S5 gate. Until S5 is measured — 15.0, marked NOT CALIBRATED.

Grazing-angle guard: if the ray meets the facade at more than `grazing_max_deg` (70° from
the normal), the facade is seen edge-on and the crossing is unreliable. The event is
marked `low_confidence`, stays out of the main aggregate but is kept in the data (rule 7).

~~Previous wording: "the orientation ray crosses the facade segment at an angle < 30°
at a distance < 8 m".~~ Struck out 2026-09-03: "at an angle < 30°" read two ways — the
angle to the facade normal, or to the facade line. The primary test was made geometric,
and the geometry removes the ambiguity by itself. The reason is in `docs/DECISIONS.md`.

Orientation is a body/head turn, **not gaze**. The wording in the UI and in the report
must reflect that.

## Commands

Everything through `python -m looq.stages.<name> --config configs/<name>.yaml`.
Nothing runs from notebooks. Nothing runs by hand around the Makefile.

## Journal

`docs/DECISIONS.md` — one entry per stage: what was decided, why, which numbers came out,
what is still unclear. This is the source for the final explanation to the CTO. Write as
you go, not at the end.
