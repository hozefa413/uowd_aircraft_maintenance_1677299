# AeroTrust — Aircraft Defect Intelligence

A React + FastAPI showcase app for an aircraft surface defect detection research
project, built to run as a Databricks App. It presents the full project —
dataset lineage, model comparisons, architecture, and a live inspection demo —
backed by real Databricks Model Serving endpoints and Unity Catalog models.

See [`research/`](research/README.md) for the MLflow pyfunc adapter source code
behind every served model, the dataset audit evidence, and per-experiment
training configs/metrics. See [`databricks/`](databricks/README.md) for the
Model Serving endpoint inventory and the training job.

## What's in the app

- Project overview, dataset lineage (v1, v2, v5) and model laboratory pages
- Experiment results with shared-evaluator comparisons across model families
- Architecture diagrams per model family (YOLO, two-stage, transformer + two-stage)
- Inspection demo: pick a model, upload or select a sample image, run live
  inference through a Model Serving endpoint, see boxes and class confidences
- A warm-up button that pings every serving endpoint in parallel, for getting
  scale-to-zero compute ready before a live demo
- Experiment Studio: configure training parameters and launch a real, capped
  training job on Databricks Serverless GPU for the YOLO family

## Models

All 12 tracked model configurations are registered in Unity Catalog and
served live:

| Model | Family | Compute | Accuracy evidence |
|---|---|---|---|
| `yolo11n_baseline` | YOLO11n | CPU Small | — |
| `e4_yolo11s` | YOLO11s | CPU Small | AP50 40.66% |
| `e13_yolo26s` | YOLO26s | CPU Small | AP50 50.33% |
| `e19_yolo26s` | YOLO26s | CPU Small | AP50 50.87% |
| `e15_frcnn_r101` | Faster R-CNN R101-FPN | GPU Small | AP50 59.81% (test) |
| `e17_cascade_r101_12ep` | Cascade R-CNN R101-FPN | GPU Small | AP50 61.07% (test) |
| `e17x_cascade_r101_25ep` | Cascade R-CNN R101-FPN | GPU Small | — |
| `e18_frcnn_r101_plus` | Faster R-CNN R101-FPN (variant) | GPU Small | AP50 58.62% (test) |
| `e20_dinov2_frcnn` | DINOv2 + SFP + Faster R-CNN | GPU Small | AP50 64.76% (test) |
| `e21a_frcnn_r50` | Faster R-CNN R50-FPN | GPU Small | **none recorded** |
| `e21b_frcnn_r101` | Faster R-CNN R101-FPN | GPU Small | **none recorded** |
| `e22_dinov3_frcnn` | DINOv3 + SFP + Faster R-CNN | GPU Small | AP50 63.91% (val only) |

`e21a_frcnn_r50` and `e21b_frcnn_r101` have no training metadata anywhere in
the workspace (confirmed empty experiment folders) — only the raw checkpoints
exist. They're served because each was independently verified (strict
`state_dict` load, real forward pass, full MLflow `predict()` smoke test
through the real inference contract) to be a correctly-loading, functioning
model, not because any accuracy number for them was found or assumed. No
accuracy claim is made for either in the app. See
[`databricks/README.md`](databricks/README.md) for exactly how each of the
four most recently added models (E18/E21A/E21B/E22) was verified and
registered, and the full endpoint list.

## Dataset

Dataset v5 is the current audited baseline: 9,350 images, 20,471 annotations,
six classes (Crack, Dent, Fastener Damage, Missing rivet, Paint-off, Rupture),
split 6,515 / 1,430 / 1,405. The test split's SHA-256 fingerprint is locked —
see `research/dataset_audit/` for the full audit.

v1 and v2 are earlier dataset inventories, not cleaned successors of v5 — they
have different source composition and split counts, so the lineage page shows
them as separate historical records rather than a before/after comparison.

## Results

E15, E17 and E20 are compared under the same evaluator:

| Model | AP50 | tiny AP50 | mAP50-95 |
|---|---|---|---|
| E15 — Faster R-CNN R101 | 59.81% | 35.97% | 28.50% |
| E17 — Cascade R-CNN R101 | 61.07% | 33.92% | 32.26% |
| E20 — DINOv2 + SFP + Faster R-CNN | 64.76% | 36.08% | 33.20% |

E20 has the best overall AP50, but its tiny-object AP50 is only +0.11pp over
E15 — a paired bootstrap on v5 shows the overall gain is significant while the
tiny-defect gain isn't. YOLO and two-stage metrics come from different
evaluation implementations, so they're not directly comparable; the UI labels
this where it matters.

## Run locally

```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Needs Databricks authentication (e.g. `databricks auth login`, or an existing
profile) for model loading and serving calls. Never put a PAT in source code.

## Build frontend

```bash
cd frontend
npm install
npm run build
```

This creates `frontend/dist`, which FastAPI serves at `/`. Build before
uploading to the Workspace — don't rely on the App container running the
Node build for you.

## Deploy to Databricks Apps

1. Build the frontend (above) and upload the whole project folder, including
   `frontend/dist`, to a Workspace folder.
2. Create a Databricks App pointing at that folder, using `app.yaml`.
3. Grant the app's service principal `USE CATALOG` on `aerotrust`,
   `USE SCHEMA` on `aerotrust.ml_models`, `CAN_QUERY` on each serving endpoint,
   and `EXECUTE` on each registered model — see
   [`databricks/README.md`](databricks/README.md) for the full list.
4. Set `AEROTRUST_SAMPLE_IMAGE_DIR` in `app.yaml` to a Unity Catalog Volume
   path if you want the sample-image dropdown, and grant `READ_VOLUME` on it.
5. Deploy, then run the warm-up button and test each model from the live app.

## Sample images

The inspection demo reads sample images from a Unity Catalog Volume via the
Files API (Databricks Apps don't mount `/Volumes` as a filesystem) — see
`databricks/README.md` for the manifest format. Direct upload also works
without any sample-image configuration.
