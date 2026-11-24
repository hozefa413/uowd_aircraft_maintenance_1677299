# Research evidence & model code

This folder holds the real research artifacts behind the AeroTrust app —
extracted from the Databricks workspace, code and small evidence files only.
Large binaries (PyTorch checkpoints, training images, prediction pickles) are
intentionally **not** included; they live in the Databricks workspace/Unity
Catalog, not in this repo.

## `model_code/` — the actual MLflow pyfunc adapters

These are the real Python source files extracted from each registered Unity
Catalog model's logged artifacts ("models-from-code" MLflow logging) — this is
the code that runs inside every Model Serving endpoint and direct-load
fallback in the app.

- `adapters/yolo_pyfunc_adapter.py` — shared by all 4 registered YOLO models
  (`yolo11n_baseline`, `e4_yolo11s`, `e13_yolo26s`, `e19_yolo26s`). Wraps
  Ultralytics YOLO inference behind the app's `image_base64` contract.
- `adapters/two_stage_pyfunc_adapter.py` — contract `aerotrust-detector-v1`,
  shared by `e15_frcnn_r101`, `e17_cascade_r101_12ep`, `e17x_cascade_r101_25ep`,
  `e18_frcnn_r101_plus`, `e21a_frcnn_r50`, `e21b_frcnn_r101`, and
  `e22_dinov3_frcnn`. Implements a ViT + Simple Feature Pyramid (ViTDet-style)
  backbone option (`family: "vit"`, used by E22's DINOv3 backbone) and a
  from-scratch Faster R-CNN / Cascade R-CNN head on top of torchvision
  (`family: "resnet"`, used by the other six), behind the app's `image_id` +
  `image_b64` contract. One file, config-driven — no new adapter code was
  written for any of the four most recently added models.
- `adapters/e20_dinov2_pyfunc_adapter.py` — E20's own earlier, narrower variant
  of the above (DINOv2 ViT-B/14-reg4 + SFP + Faster R-CNN specifically),
  logged separately before the shared adapter existed.
- `configs/<model_id>.json` — the per-model config logged alongside each
  adapter (class names, image size, score threshold, checkpoint SHA-256,
  cascade on/off, etc.) — this is what parameterizes the shared adapter code
  per model.
- `examples/sample_input.json` — the example payload MLflow logged for the
  `aerotrust-detector-v1` contract (identical across all eight non-YOLO
  models: E15/E17/E17x/E18/E20/E21A/E21B/E22).

## `dataset_audit/` — dataset v5 quality evidence

The audit behind the "locked test fingerprint" and quality claims in the main
README: class/size/resolution distributions, near-duplicate candidates,
annotation quality, and the full conclusion write-up. The one file
deliberately excluded is `image_level_inventory.csv` (3.9 MB, one row per
image) — the distribution summaries here already capture what it would show.

## `experiments/` — per-experiment training configs and metrics

Small evidence files only (args.yaml, results.csv, summary.json, bootstrap
CSVs) for each experiment that has them: E4, E13, E19 (YOLO family, with real
Ultralytics `args.yaml` training configs), E15/E17/E18/E20/E22 (metrics +
summaries only — these architectures have no training code in the workspace,
only inference adapters), and the YOLO11n baseline. E21A/E21B have no
metadata files in the workspace at all, only raw checkpoints, so nothing to
pull.

**Note on the `args.yaml` files:** these are historical records of what was
actually run — several reference paths that no longer resolve (a different
user's workspace folder, or `/local_disk0` scratch space that doesn't persist
across clusters). They document provenance, not a ready-to-rerun config. The
actual portable, working training path is `databricks/notebooks/studio_train.py`.
