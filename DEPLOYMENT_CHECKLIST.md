# Deployment checklist

## Before deploying
- [ ] Build `frontend/dist` with Node 20+ (`npm install && npm run build`).
- [ ] Upload the entire project folder, including `frontend/dist`, to the Databricks Workspace.
- [ ] Confirm `requirements.txt` dependencies install in the App runtime.
- [ ] Set `AEROTRUST_SAMPLE_IMAGE_DIR` in `app.yaml` to the sample-image Volume path.

## App service principal
- [ ] Grant `USE CATALOG` on `aerotrust` and `USE SCHEMA` on `aerotrust.ml_models`.
- [ ] Grant `CAN_QUERY` on each serving endpoint and `EXECUTE` on each registered model:
  - `yolo11n_baseline`, `e4_yolo11s`, `e13_yolo26s`, `e19_yolo26s`
  - `e15_frcnn_r101`, `e17_cascade_r101_12ep`, `e17x_cascade_r101_25ep`, `e20_dinov2_sfp_frcnn`
- [ ] Grant `READ_VOLUME` on the sample-image Volume.
- [ ] Grant `CAN_MANAGE_RUN` on the Experiment Studio training job (see `databricks/README.md`).
- [ ] Do not store a personal access token in the app.

## Acceptance tests
- [ ] App `/api/v1/health` returns `{"status":"ok", ...}`.
- [ ] Run the warm-up button and confirm all 8 serving endpoints respond.
- [ ] Upload a representative image, run each of the 8 ready models, and verify
      boxes, class labels and confidences all render (not just raw detection counts).
- [ ] Confirm the six-class canonical mapping is used for labels.
- [ ] Confirm the app handles an invalid image, an oversized image, and a model
      access failure without crashing.
- [ ] Launch one capped Experiment Studio training run and confirm it reaches a
      terminal state in MLflow.
- [ ] Keep the locked test split out of iterative UI validation.

## Known gaps
- 4 of 12 tracked checkpoints (`e18_frcnn_r101_plus`, `e21a_frcnn_r50`,
  `e21b_frcnn_r101`, `e22_dinov3_frcnn`) have no verified inference adapter yet
  and are not selectable in the UI.
- Only the YOLO family has a training pipeline; two-stage/transformer
  architectures have inference adapters only.
- No confidence calibration, OOD/abstention handling, inspector sign-off
  persistence, or durable audit event storage.
