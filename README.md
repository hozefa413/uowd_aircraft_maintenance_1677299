# AeroTrust Showcase — React + FastAPI for Databricks Apps

See [`research/`](research/README.md) for the real MLflow pyfunc adapter source
code behind every served model, dataset-quality audit evidence, and
per-experiment training configs/metrics. See [`databricks/`](databricks/README.md)
for the Model Serving endpoint inventory and the Serverless GPU training job.

## What this delivers
A research-journey showcase with:
- Executive command center and project evidence/status
- Dataset lineage comparison (historical v1, E7 bundle v2, audited v5)
- Model laboratory and architecture comparison
- Experiment results with evidence labels and caveats
- Parameterized experiment design panel (configuration preview; no claim of training-job execution)
- Architecture and end-to-end pipeline view
- Live inspection demo: select one of the four registered YOLO models, upload an image, run MLflow PyFunc inference, view annotated boxes and detections
- Seven Torch checkpoints displayed as adapter-required, not falsely exposed as runnable models

## Important evidence discipline
Dataset v1 and v2 are historical lineage records, not verified successive versions of the exact same immutable records. Their counts/splits differ and should not be interpreted as a direct before/after quality comparison. Dataset v5 is the current audited research baseline:
- 9,350 images; 20,471 valid annotations
- Train/validation/test: 6,515 / 1,430 / 1,405
- 6 classes: Crack, Dent, Fastener Damage, Missing rivet, Paint-off, Rupture
- 306 cross-split near-duplicate candidates are unconfirmed; aircraft/session independence remains unverified.
- Locked test fingerprint: `2db49819b0ba156bedc8efcbd25152de26f83b0bc27b0b27b87babf7fea2ff2b`

## Results interpretation
E15, E17 and E20 values are from the shared evaluator:
- E15: AP50 59.81%, tiny AP50 35.97%, mAP50-95 28.50%
- E17: AP50 61.07%, tiny AP50 33.92%, mAP50-95 32.26%
- E20: AP50 64.76%, tiny AP50 36.08%, mAP50-95 33.20%
E20 has best reported overall AP50, but tiny AP50 is only +0.11 percentage points vs E15.
YOLO and two-stage metrics may use different evaluation implementations; UI labels this caveat. Some historical metrics are unavailable and shown as unavailable rather than invented.

## Registered models
The live inference selector includes only models confirmed REGISTERED, version 1:
- `aerotrust.ml_models.yolo11n_baseline`
- `aerotrust.ml_models.e4_yolo11s`
- `aerotrust.ml_models.e13_yolo26s`
- `aerotrust.ml_models.e19_yolo26s`

Seven PyTorch checkpoints have artifacts preserved but require architecture-specific adapters and registration. The UI marks them as not yet inference-ready.

## Build frontend
Use Node 20+ on a development machine or build environment:
```bash
cd frontend
npm install
npm run build
```
The build creates `frontend/dist`, which FastAPI serves at `/`. Upload the complete folder including `frontend/dist` to the Workspace before deploying the App. Do not rely on the Databricks App container to run the Node build unless your workspace runtime explicitly supports Node.

## Sample image gallery
The inspection page supports direct image upload. It can also show a sample-image dropdown when you configure `AEROTRUST_SAMPLE_IMAGE_DIR` to an app-readable directory containing approved JPG/PNG inspection images. Prefer a Unity Catalog Volume path mounted/accessible to the App service principal. Grant read access to that volume. The backend exposes `/api/v1/sample-images`; no synthetic or fabricated aircraft samples are included.

## Run locally (if workspace credentials are configured)
```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Databricks model loading requires Databricks authentication and permissions. Do not put a PAT in source code.

## Deploy to Databricks Apps
1. Build the frontend and upload the complete project folder (including `frontend/dist`) to a Workspace folder.
2. Create a Databricks App using this project folder and the `app.yaml` command.
3. Add app dependencies from `requirements.txt` if your app configuration flow requires it.
4. Give the app service principal `USE CATALOG` on `aerotrust`, `USE SCHEMA` on `aerotrust.ml_models`, and `EXECUTE` on the four registered models, or add them through the App resource configuration if available.
5. Deploy. Test all four models from the live app with validation images.
6. For the seven PyTorch models, add them to the live selector only after verified architecture adapters are implemented, registered, and smoke-tested.

## Parameterized training
The Experiment Studio page currently creates a reproducible parameter/configuration preview and captures the intended dimensions (architecture, dataset, image size, batch size, epochs, learning rate, optimizer, seed, augmentation, confidence threshold). It intentionally does not launch Databricks Jobs. Connect it to your actual training Job IDs and parameter contract before enabling job submission.

## Sample images
The UI accepts uploaded images in the demo panel. It does not fabricate annotated examples. Use actual validation images or approved aircraft inspection imagery for public demos.
