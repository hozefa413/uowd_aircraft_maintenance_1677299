# Databricks workspace infrastructure

This folder documents the Databricks-side resources the app depends on. They
live in the workspace/account, not in this repo's deploy path, so this is the
reference for recreating them.

## Model Serving endpoints

| Endpoint | Model | Compute |
|---|---|---|
| `aerotrust-yolo11n-baseline` | `aerotrust.ml_models.yolo11n_baseline` | CPU Small, scale-to-zero |
| `aerotrust-e4-yolo11s` | `aerotrust.ml_models.e4_yolo11s` | CPU Small, scale-to-zero |
| `aerotrust-e13-yolo26s` | `aerotrust.ml_models.e13_yolo26s` | CPU Small, scale-to-zero |
| `aerotrust-e19-yolo26s` | `aerotrust.ml_models.e19_yolo26s` | CPU Small, scale-to-zero |
| `aerotrust-e15-frcnn-r101` | `aerotrust.ml_models.e15_frcnn_r101` | GPU_SMALL, scale-to-zero |
| `aerotrust-e17-cascade-r101` | `aerotrust.ml_models.e17_cascade_r101_12ep` | GPU_SMALL, scale-to-zero |
| `aerotrust-e17x-cascade-r101` | `aerotrust.ml_models.e17x_cascade_r101_25ep` | GPU_SMALL, scale-to-zero |
| `aerotrust-e20-dinov2-frcnn` | `aerotrust.ml_models.e20_dinov2_sfp_frcnn` | GPU_SMALL, scale-to-zero |
| `aerotrust-e18-frcnn-r101-plus` | `aerotrust.ml_models.e18_frcnn_r101_plus` | GPU_SMALL, scale-to-zero |
| `aerotrust-e21a-frcnn-r50` | `aerotrust.ml_models.e21a_frcnn_r50` | GPU_SMALL, scale-to-zero |
| `aerotrust-e21b-frcnn-r101` | `aerotrust.ml_models.e21b_frcnn_r101` | GPU_SMALL, scale-to-zero |
| `aerotrust-e22-dinov3-frcnn` | `aerotrust.ml_models.e22_dinov3_frcnn` | GPU_SMALL, scale-to-zero |

The app's service principal needs `CAN_QUERY` on each, plus `EXECUTE` on the
underlying registered model (for the direct-load fallback path) and
`USE_CATALOG`/`USE_SCHEMA` on `aerotrust`/`aerotrust.ml_models`.

### E18 / E21A / E21B / E22 — registered without new adapter code

All four reuse the exact same `two_stage_pyfunc_adapter.py` already serving
E15/E17/E17x/E20 — only a new config JSON per model, no new Python. E18 and
E22 have real recorded training metrics (`summary.json` in their workspace
experiment folders); E21A/E21B have **no training metadata anywhere in the
workspace** (confirmed empty experiment folders), so no accuracy metric is
claimed for them in the app. Before registering any of the four, each
checkpoint was verified by: strict `state_dict` load against the proposed
config, a real forward pass, and (post-registration) a full `predict()` smoke
test through the actual MLflow pyfunc wrapper — not assumed from the
checkpoint's filename alone. This caught a real issue: E21A/E21B's RPN head
has 3 anchors/level (one scale per FPN level), not the project's usual 2
scales/level used by every other resnet-family model here; the config
reflects the verified value, not the usual default.

E22 (DINOv3 ViT-B/16) uses the adapter's `family: "vit"` branch with
`timm_name: "hf-hub:timm/vit_base_patch16_dinov3.lvd1689m"` — this pulls the
backbone architecture from Hugging Face Hub at model-load time, so the
serving container needs outbound network access to huggingface.co and a
`timm` version new enough to recognize DINOv3 (`timm==1.0.31`, already pinned
in `requirements.txt`, works).

## Experiment Studio training job

`notebooks/studio_train.py` is a parameterized YOLO training launcher
(architecture/image size/batch/epochs/lr/optimizer/seed/augmentation), run via
the Databricks Job in `jobs/studio_train_job.json` on Serverless GPU
(`GPU_1xA10`). It trains against the canonical dataset at
`aerotrust/aerotrust_v5_resplit_local/data.yaml` and logs to MLflow under
`aerotrust/studio_runs/`. Only the YOLO family (YOLO11n/11s/26s) has a real
training pipeline — the two-stage/transformer architectures have inference
adapters only, no training code.

To recreate the job: upload `notebooks/studio_train.py` to a workspace path,
update `notebook_path` in `jobs/studio_train_job.json` to match, then
`databricks jobs create --json @jobs/studio_train_job.json`. Grant the app's
service principal `CAN_MANAGE_RUN` on the resulting job, and set
`AEROTRUST_STUDIO_JOB_ID` in `app.yaml` if the new job ID differs from the
backend's default.

## Sample images

The Inspection demo's "pick a demo image" dropdown reads from a Unity Catalog
Volume (`AEROTRUST_SAMPLE_IMAGE_DIR` in `app.yaml`), via the Files API — Apps
do not mount `/Volumes` as a filesystem. The volume needs a `demo_images.csv`
manifest (columns: `demo_file,category,main_class,n_gt,talking_point,...`) for
the rich dropdown labels; without it, the app falls back to a plain file
listing. Grant the app's service principal `READ_VOLUME` on the volume.
