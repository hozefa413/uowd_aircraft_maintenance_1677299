# Deployment checklist

## Before deploying
- [ ] Build `frontend/dist` with Node 20+ (`npm install && npm run build`).
- [ ] Upload the entire project folder, including `frontend/dist`, to the Databricks Workspace.
- [ ] Confirm `requirements.txt` dependencies install in the App runtime.
- [ ] Configure `AEROTRUST_CATALOG=aerotrust` and `AEROTRUST_SCHEMA=ml_models` if using non-default names.
- [ ] Configure `AEROTRUST_SAMPLE_IMAGE_DIR` only if you have an App-readable directory / Unity Catalog Volume with approved sample images.

## App service principal
- [ ] Grant `USE CATALOG` on `aerotrust`.
- [ ] Grant `USE SCHEMA` on `aerotrust.ml_models`.
- [ ] Grant `EXECUTE` on the four registered models:
  - `aerotrust.ml_models.yolo11n_baseline`
  - `aerotrust.ml_models.e4_yolo11s`
  - `aerotrust.ml_models.e13_yolo26s`
  - `aerotrust.ml_models.e19_yolo26s`
- [ ] Grant read access to sample-image Volume if configured.
- [ ] Do not store a personal access token in the app.

## Acceptance tests
- [ ] App `/api/v1/health` returns `{"status":"ok", ...}`.
- [ ] Each of four models can be loaded from the App identity.
- [ ] Upload one representative image and verify detections and pixel-coordinate boxes.
- [ ] Switch model and run same image again.
- [ ] Confirm output class labels use the six-class canonical mapping.
- [ ] Confirm app handles invalid image, oversized image and model access failure cleanly.
- [ ] Keep the locked test split out of iterative UI validation.

## Explicitly not complete yet
- Training Job submission from Experiment Studio.
- Seven Torch adapters / UC registrations.
- Confidence calibration, OOD/abstention, inspector sign-off persistence and durable audit event storage.
