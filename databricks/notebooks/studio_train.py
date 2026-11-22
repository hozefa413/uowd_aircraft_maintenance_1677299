# Databricks notebook source
# MAGIC %md
# MAGIC ## AeroTrust Experiment Studio — live YOLO training launcher
# MAGIC Parameterized training for the YOLO family (YOLO11n / YOLO11s / YOLO26s) against the
# MAGIC canonical, portable v5 dataset already stored under this workspace at
# MAGIC `aerotrust/aerotrust_v5_resplit_local/data.yaml`. Logs to MLflow; does not auto-register
# MAGIC to Unity Catalog (kept separate from the vetted production model catalog).
# COMMAND ----------
dbutils.widgets.text("architecture", "yolo11n", "Architecture (yolo11n/yolo11s/yolo26s)")
dbutils.widgets.text("image_size", "1024", "Image size (must be a multiple of 32)")
dbutils.widgets.text("batch_size", "8", "Batch size")
dbutils.widgets.text("epochs", "2", "Epochs")
dbutils.widgets.text("learning_rate", "0.001", "Initial learning rate")
dbutils.widgets.text("optimizer", "AdamW", "Optimizer (AdamW/SGD/Adam)")
dbutils.widgets.text("seed", "42", "Random seed")
dbutils.widgets.text("augmentation", "baseline", "Augmentation profile (baseline/realistic/randaugment/small_object_oversampling)")
dbutils.widgets.text("run_name", "studio_test_run", "Run name (used for the output folder and MLflow run name)")

# COMMAND ----------
import os, csv, shutil, random

ARCH = dbutils.widgets.get("architecture")
IMGSZ = int(dbutils.widgets.get("image_size"))
BATCH = int(dbutils.widgets.get("batch_size"))
EPOCHS = int(dbutils.widgets.get("epochs"))
LR0 = float(dbutils.widgets.get("learning_rate"))
OPTIMIZER = dbutils.widgets.get("optimizer")
SEED = int(dbutils.widgets.get("seed"))
AUG = dbutils.widgets.get("augmentation")
RUN_NAME = dbutils.widgets.get("run_name")

assert IMGSZ % 32 == 0, f"image_size must be a multiple of 32 (YOLO stride), got {IMGSZ}"
assert ARCH in ("yolo11n", "yolo11s", "yolo26s"), f"Unsupported architecture for the live launcher: {ARCH}"

WORKSPACE_ROOT = "/Workspace/Users/hozefamustan.kanchwala@databricks.com/aerotrust"
DATASET_ROOT = f"{WORKSPACE_ROOT}/aerotrust_v5_resplit_local"
OUTPUT_ROOT = f"{WORKSPACE_ROOT}/studio_runs/{RUN_NAME}"

assert os.path.isdir(DATASET_ROOT), f"Canonical dataset missing: {DATASET_ROOT}"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Ultralytics resolves a data.yaml's relative `path:` against its own configured
# datasets_dir / cwd, not against the yaml file's own location. Generate a fresh
# yaml with fully absolute paths to sidestep that entirely.
CANONICAL_DATA_YAML = f"{OUTPUT_ROOT}/resolved_data.yaml"
with open(CANONICAL_DATA_YAML, "w") as f:
    f.write(
        f"path: {DATASET_ROOT}\n"
        f"train: {DATASET_ROOT}/images/train\n"
        f"val: {DATASET_ROOT}/images/val\n"
        f"test: {DATASET_ROOT}/images/test\n"
        "names:\n  0: Crack\n  1: Dent\n  2: Fastener Damage\n  3: Missing rivet\n  4: Paint-off\n  5: Rupture\n"
    )
print("Using dataset config:", CANONICAL_DATA_YAML)
print("Output root:", OUTPUT_ROOT)

# COMMAND ----------
# Augmentation profile -> real Ultralytics hyperparameters.
# "baseline" and "randaugment" mirror values actually used in recorded E4/E13 runs.
# "small_object_oversampling" additionally duplicates images containing the smallest
# annotated boxes into a scratch train split, matching the documented E4 technique
# (see class_oversampling_audit.csv / image_oversampling_audit.csv in that run's folder).
AUG_PROFILES = {
    "baseline": dict(mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, fliplr=0.5, auto_augment=None),
    "realistic": dict(mosaic=0.5, mixup=0.0, hsv_h=0.015, hsv_s=0.4, hsv_v=0.2, degrees=3.0, translate=0.05, scale=0.3, fliplr=0.5, auto_augment=None),
    "randaugment": dict(mosaic=0.5, mixup=0.0, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5, auto_augment="randaugment"),
    "small_object_oversampling": dict(mosaic=0.5, mixup=0.0, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5, auto_augment="randaugment"),
}
assert AUG in AUG_PROFILES, f"Unknown augmentation profile: {AUG}"
aug_kwargs = {k: v for k, v in AUG_PROFILES[AUG].items() if v is not None}

# COMMAND ----------
def build_oversampled_split(small_area_frac=0.01, duplicate_factor=3):
    """Duplicate train images whose smallest annotated box is below `small_area_frac`
    of the image area, `duplicate_factor` extra times, into a scratch split that
    reuses the canonical validation/test splits unchanged."""
    src_root = f"{WORKSPACE_ROOT}/aerotrust_v5_resplit_local"
    dst_root = f"{OUTPUT_ROOT}/oversampled_dataset"
    for sub in ("images/train", "labels/train"):
        os.makedirs(f"{dst_root}/{sub}", exist_ok=True)
    label_dir = f"{src_root}/labels/train"
    image_dir = f"{src_root}/images/train"
    small_count = 0
    for name in os.listdir(label_dir):
        if not name.endswith(".txt"):
            continue
        stem = name[:-4]
        img_candidates = [f for f in os.listdir(image_dir) if f.startswith(stem + ".")]
        if not img_candidates:
            continue
        img_name = img_candidates[0]
        with open(f"{label_dir}/{name}") as f:
            rows = [l.split() for l in f if l.strip()]
        min_area = min((float(r[3]) * float(r[4]) for r in rows if len(r) >= 5), default=1.0)
        copies = duplicate_factor if min_area < small_area_frac else 1
        if min_area < small_area_frac:
            small_count += 1
        for i in range(copies):
            suffix = f"_dup{i}" if i else ""
            shutil.copyfile(f"{image_dir}/{img_name}", f"{dst_root}/images/train/{stem}{suffix}{os.path.splitext(img_name)[1]}")
            shutil.copyfile(f"{label_dir}/{name}", f"{dst_root}/labels/train/{stem}{suffix}.txt")
    print(f"Oversampled {small_count} small-object images x{duplicate_factor} into {dst_root}")
    data_yaml = f"{dst_root}/data.yaml"
    with open(data_yaml, "w") as f:
        f.write(
            f"path: {dst_root}\n"
            f"train: {dst_root}/images/train\n"
            f"val: {src_root}/images/val\n"
            f"test: {src_root}/images/test\n"
            "names:\n  0: Crack\n  1: Dent\n  2: Fastener Damage\n  3: Missing rivet\n  4: Paint-off\n  5: Rupture\n"
        )
    return data_yaml

data_yaml_path = build_oversampled_split() if AUG == "small_object_oversampling" else CANONICAL_DATA_YAML

# COMMAND ----------
import mlflow
from ultralytics import YOLO

os.makedirs(f"{WORKSPACE_ROOT}/studio_runs", exist_ok=True)
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(f"{WORKSPACE_ROOT}/studio_runs/mlflow_experiment")

model = YOLO(f"{ARCH}.pt")

with mlflow.start_run(run_name=RUN_NAME) as run:
    mlflow.log_params({
        "architecture": ARCH, "image_size": IMGSZ, "batch_size": BATCH, "epochs": EPOCHS,
        "learning_rate": LR0, "optimizer": OPTIMIZER, "seed": SEED, "augmentation": AUG,
        "dataset": "aerotrust_v5_resplit_local",
    })
    results = model.train(
        data=data_yaml_path, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, lr0=LR0,
        optimizer=OPTIMIZER, seed=SEED, deterministic=True, device="0", workers=2,
        project=OUTPUT_ROOT, name="training", exist_ok=True, pretrained=True, verbose=True,
        **aug_kwargs,
    )
    metrics = getattr(results, "results_dict", {}) or {}
    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    best_pt = f"{OUTPUT_ROOT}/training/weights/best.pt"
    if os.path.exists(best_pt):
        mlflow.log_artifact(best_pt, artifact_path="weights")
    print("MLflow run:", run.info.run_id)
    print("Output folder:", OUTPUT_ROOT)

print("STUDIO_TRAIN_DONE_OK")
