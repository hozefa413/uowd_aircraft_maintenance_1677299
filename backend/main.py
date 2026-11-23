import os, io, json, base64, time, logging, threading, csv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from PIL import Image
import pandas as pd
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError

log = logging.getLogger("aerotrust")

APP_TITLE = "AeroTrust | Aircraft Defect Intelligence"
CATALOG, SCHEMA = os.getenv("AEROTRUST_CATALOG", "aerotrust"), os.getenv("AEROTRUST_SCHEMA", "ml_models")
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

app = FastAPI(title=APP_TITLE, version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CLASSES = ["Crack", "Dent", "Fastener Damage", "Missing rivet", "Paint-off", "Rupture"]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


# Real project evidence exported from the Databricks Workspace at
# /Workspace/Users/hozefamustan.kanchwala@databricks.com/aerotrust — see README for provenance.
MODEL_CATALOG = _load("models.json")
MODEL_BY_ID = {m["id"]: m for m in MODEL_CATALOG}
DATASETS = _load("datasets.json")
EXPERIMENTS = _load("experiments.json")
PROJECT_JOURNEY = _load("project_journey.json")
PARAMETER_IMPACT = _load("parameter_impact.json")

MODEL_CACHE = {}
MODEL_CACHE_LOCK = threading.Lock()
_ws_client = None


def workspace_client():
    global _ws_client
    if _ws_client is None:
        _ws_client = WorkspaceClient()
    return _ws_client


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "aerotrust-showcase", "version": "1.0.0"}


@app.get("/api/v1/overview")
def overview():
    return {
        "dataset_current": "dataset-v5",
        "classes": CLASSES,
        "counts": {
            "datasets": len(DATASETS), "experiments": len(EXPERIMENTS), "models_total": len(MODEL_CATALOG),
            "models_registered": sum(m["ready"] for m in MODEL_CATALOG),
            "models_adapter_required": sum(not m["ready"] for m in MODEL_CATALOG),
        },
        "headline": {
            "best_reported_map50": 64.76, "best_model": "E20 — DINOv2 + SFP + Faster R-CNN",
            "dataset_images": 9350, "annotations": 20471,
        },
        "evidence_note": "Metrics are transcribed from recorded project results. Comparability is limited where evaluator protocols differ; E15/E17/E20 share the reported evaluator.",
    }


@app.get("/api/v1/datasets")
def datasets():
    return DATASETS


@app.get("/api/v1/models")
def models(registered_only: bool = False):
    return [m for m in MODEL_CATALOG if (m["ready"] or not registered_only)]


@app.get("/api/v1/experiments")
def experiments():
    return EXPERIMENTS


@app.get("/api/v1/project-journey")
def project_journey():
    return PROJECT_JOURNEY


@app.get("/api/v1/parameter-impact")
def parameter_impact():
    return PARAMETER_IMPACT


STUDIO_JOB_ID = int(os.getenv("AEROTRUST_STUDIO_JOB_ID", "470973301608665"))
STUDIO_SUPPORTED_ARCH = {"yolo11n", "yolo11s", "yolo26s"}
STUDIO_MAX_EPOCHS = 5  # hard safety cap on real Serverless GPU runs, independent of the UI slider


class StudioRunRequest(BaseModel):
    architecture: str
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    seed: int
    augmentation: str
    run_name: str = None


@app.post("/api/v1/studio/run")
def studio_run(req: StudioRunRequest):
    if req.architecture not in STUDIO_SUPPORTED_ARCH:
        raise HTTPException(
            409,
            f"Live training is only wired for the YOLO family ({', '.join(sorted(STUDIO_SUPPORTED_ARCH))}) in this workspace. "
            f"'{req.architecture}' has no training pipeline here — only inference adapters exist for it.",
        )
    if req.image_size % 32 != 0:
        raise HTTPException(400, "image_size must be a multiple of 32 (YOLO backbone stride).")
    capped_epochs = min(req.epochs, STUDIO_MAX_EPOCHS)
    run_name = (req.run_name or f"studio_{req.architecture}_{int(time.time())}")[:60]
    try:
        run = workspace_client().jobs.run_now(
            job_id=STUDIO_JOB_ID,
            notebook_params={
                "architecture": req.architecture,
                "image_size": str(req.image_size),
                "batch_size": str(req.batch_size),
                "epochs": str(capped_epochs),
                "learning_rate": str(req.learning_rate),
                "optimizer": req.optimizer,
                "seed": str(req.seed),
                "augmentation": req.augmentation,
                "run_name": run_name,
            },
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to launch training job: {type(e).__name__}: {e}")
    host = str(workspace_client().config.host).rstrip("/")
    return {
        "run_id": run.run_id,
        "job_id": STUDIO_JOB_ID,
        "run_name": run_name,
        "epochs_requested": req.epochs,
        "epochs_actually_run": capped_epochs,
        "epochs_capped": capped_epochs < req.epochs,
        "databricks_run_url": f"{host}/jobs/{STUDIO_JOB_ID}/runs/{run.run_id}",
        "compute": "Serverless GPU (GPU_1xA10)",
        "status": "PENDING",
    }


@app.get("/api/v1/studio/run/{run_id}")
def studio_run_status(run_id: int):
    try:
        run = workspace_client().jobs.get_run(run_id)
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch run status: {type(e).__name__}: {e}")
    state = run.state
    return {
        "run_id": run_id,
        "life_cycle_state": _enum_value(state.life_cycle_state) if state else None,
        "result_state": _enum_value(state.result_state) if state else None,
        "state_message": state.state_message if state else None,
    }


def _enum_value(x):
    return x.value if hasattr(x, "value") else x


@app.get("/api/v1/serving-endpoints")
def serving_endpoints():
    """Live status of the Databricks Model Serving endpoints backing the ready models."""
    out = []
    for m in MODEL_CATALOG:
        if not m.get("serving_endpoint"):
            continue
        entry = {"model_id": m["id"], "endpoint_name": m["serving_endpoint"]}
        try:
            ep = workspace_client().serving_endpoints.get(m["serving_endpoint"])
            state = ep.state
            entry["ready"] = _enum_value(state.ready) if state else "UNKNOWN"
            entry["config_update"] = _enum_value(state.config_update) if state else "UNKNOWN"
            cfg = ep.config if (ep.config and ep.config.served_entities) else ep.pending_config
            served = cfg.served_entities[0] if (cfg and cfg.served_entities) else None
            if served:
                entry["workload_size"] = _enum_value(served.workload_size)
                entry["workload_type"] = _enum_value(served.workload_type)
                entry["scale_to_zero_enabled"] = served.scale_to_zero_enabled
        except DatabricksError as e:
            entry["ready"] = "UNAVAILABLE"
            entry["error"] = str(e)
        out.append(entry)
    return out


def load_uc_model(model):
    mid = model["id"]
    if mid in MODEL_CACHE:
        return MODEL_CACHE[mid]
    with MODEL_CACHE_LOCK:
        if mid not in MODEL_CACHE:
            uri = f'models:/{model["registry_name"]}/{model["version"]}'
            MODEL_CACHE[mid] = mlflow.pyfunc.load_model(uri)
        return MODEL_CACHE[mid]


def _build_payload_record(model, encoded_image, filename):
    if model.get("payload_style") == "image_id_b64":
        return {"image_id": filename or "request", "image_b64": encoded_image}
    return {"image_base64": encoded_image}


def _normalize_detection(d):
    """Adapters disagree on field names: YOLO emits confidence/bbox, the
    two-stage and DINOv2 adapters emit score/box_xyxy. Normalize to one
    schema so the frontend can treat every model identically."""
    return {
        "class_id": d.get("class_id"),
        "class_name": d.get("class_name"),
        "confidence": float(d["confidence"] if "confidence" in d else d.get("score", 0)),
        "bbox": d["bbox"] if "bbox" in d else d.get("box_xyxy", []),
    }


def _parse_prediction(row):
    raw = row.get("detections_json", row.get("detections", "[]"))
    detections = json.loads(raw) if isinstance(raw, str) else (raw or [])
    detections = [_normalize_detection(d) for d in detections]
    reported_ms = row.get("inference_time_ms", row.get("inference_ms"))
    model_name = row.get("model_name", row.get("model_id"))
    return detections, reported_ms, model_name


def query_serving_endpoint(model, encoded_image, filename):
    resp = workspace_client().serving_endpoints.query(
        name=model["serving_endpoint"],
        dataframe_records=[_build_payload_record(model, encoded_image, filename)],
    )
    row = resp.predictions[0] if resp.predictions else {}
    detections, reported_ms, model_name = _parse_prediction(row)
    return detections, reported_ms, model_name


def query_direct_uc_load(model, encoded_image, filename):
    loaded = load_uc_model(model)
    payload = pd.DataFrame([_build_payload_record(model, encoded_image, filename)])
    pred = loaded.predict(payload)
    row = pred.iloc[0].to_dict() if isinstance(pred, pd.DataFrame) else pred
    detections, reported_ms, model_name = _parse_prediction(row)
    return detections, reported_ms, model_name


_WARMUP_IMAGE_B64 = None


def _warmup_image_b64():
    global _WARMUP_IMAGE_B64
    if _WARMUP_IMAGE_B64 is None:
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(buf, format="JPEG")
        _WARMUP_IMAGE_B64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return _WARMUP_IMAGE_B64


def _warmup_one(model):
    t0 = time.perf_counter()
    if not model.get("serving_endpoint"):
        return {"model_id": model["id"], "name": model["name"], "success": False, "error": "No serving endpoint configured for this model", "elapsed_ms": 0}
    try:
        query_serving_endpoint(model, _warmup_image_b64(), "warmup.jpg")
        return {"model_id": model["id"], "name": model["name"], "success": True, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
    except Exception as e:
        return {"model_id": model["id"], "name": model["name"], "success": False, "error": f"{type(e).__name__}: {e}", "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/api/v1/warmup/{model_id}")
async def warmup_one(model_id: str):
    """Send a trivial inference request to a single serving endpoint, forcing
    scale-to-zero compute to scale up ahead of a live demo. Warming one model
    per HTTP request (rather than fanning all of them into one request) means
    a slow cold start on one endpoint can't time out the whole batch at the
    Databricks Apps proxy and return an HTML error page in place of JSON."""
    model = MODEL_BY_ID.get(model_id)
    if not model:
        raise HTTPException(404, "Unknown model identifier")
    return await run_in_threadpool(_warmup_one, model)


@app.post("/api/v1/inference/{model_id}")
async def inference(model_id: str, image: UploadFile = File(...)):
    model = MODEL_BY_ID.get(model_id)
    if not model:
        raise HTTPException(404, "Unknown model identifier")
    if not model["ready"] or not model["version"]:
        raise HTTPException(409, "This checkpoint is preserved but does not yet have a verified registered inference adapter.")
    content = await image.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 20 MB limit")
    try:
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        log.warning("Image decode failed: content_len=%d filename=%r error=%s: %s", len(content), image.filename, type(e).__name__, e)
        raise HTTPException(400, "Upload must be a valid JPG or PNG image")
    encoded = base64.b64encode(content).decode("utf-8")

    t0 = time.perf_counter()
    serving_path, serving_error = "model_serving_endpoint", None
    try:
        if not model.get("serving_endpoint"):
            raise RuntimeError("No serving endpoint configured for this model")
        detections, reported_ms, model_name = await run_in_threadpool(query_serving_endpoint, model, encoded, image.filename)
    except Exception as e:
        serving_error = f"{type(e).__name__}: {e}"
        log.warning("Model Serving query failed for %s, falling back to direct UC load: %s", model_id, serving_error)
        serving_path = "direct_unity_catalog_load"
        try:
            detections, reported_ms, model_name = await run_in_threadpool(query_direct_uc_load, model, encoded, image.filename)
        except Exception as e2:
            raise HTTPException(500, f"Model inference failed on both Model Serving and direct load: {type(e2).__name__}: {e2}")
    wall_ms = (time.perf_counter() - t0) * 1000

    return {
        "model": model,
        "image": {"width": pil.width, "height": pil.height, "filename": image.filename},
        "detections": detections,
        "reported_inference_ms": float(reported_ms) if reported_ms is not None else round(wall_ms, 2),
        "request_wall_ms": round(wall_ms, 2),
        "model_name": model_name or model["name"],
        "serving_path": serving_path,
        "serving_endpoint": model.get("serving_endpoint") if serving_path == "model_serving_endpoint" else None,
        "serving_fallback_reason": serving_error if serving_path == "direct_unity_catalog_load" else None,
    }


SAMPLE_IMAGE_DIR = os.getenv("AEROTRUST_SAMPLE_IMAGE_DIR", "")


def _volume_path(name):
    return f"{SAMPLE_IMAGE_DIR.rstrip('/')}/{name}"


@app.get("/api/v1/sample-images")
def sample_images():
    """List approved sample images from a Unity Catalog Volume, via the Files API
    (Databricks Apps do not mount /Volumes as a local filesystem)."""
    if not SAMPLE_IMAGE_DIR:
        return {"configured": False, "items": []}
    try:
        entries = list(workspace_client().files.list_directory_contents(SAMPLE_IMAGE_DIR))
    except Exception as e:
        log.warning("Could not list sample image volume %s: %s", SAMPLE_IMAGE_DIR, e)
        return {"configured": False, "items": []}
    names = {os.path.basename(e.path) for e in entries if not e.is_directory}
    items = []
    if "demo_images.csv" in names:
        try:
            raw = workspace_client().files.download(_volume_path("demo_images.csv")).contents.read()
            for row in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
                name = row.get("demo_file", "")
                if name not in names:
                    continue
                items.append({
                    "name": name,
                    "url": f"/api/v1/sample-images/{name}",
                    "category": row.get("category"),
                    "main_class": row.get("main_class"),
                    "n_gt": row.get("n_gt"),
                    "talking_point": row.get("talking_point"),
                })
        except Exception as e:
            log.warning("Could not read demo_images.csv from %s: %s", SAMPLE_IMAGE_DIR, e)
    if not items:
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}:
                items.append({"name": name, "url": f"/api/v1/sample-images/{name}"})
    return {"configured": True, "items": items[:100]}


@app.get("/api/v1/sample-images/{filename}")
def get_sample_image(filename: str):
    if not SAMPLE_IMAGE_DIR:
        raise HTTPException(404, "Sample image directory is not configured")
    safe_name = os.path.basename(filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(415, "Unsupported sample image type")
    try:
        raw = workspace_client().files.download(_volume_path(safe_name)).contents.read()
    except Exception as e:
        raise HTTPException(404, f"Sample image not found: {type(e).__name__}: {e}")
    return Response(content=raw, media_type="image/png" if ext == ".png" else "image/jpeg")


@app.get("/api/v1/runtime")
def runtime():
    return {
        "tracking_uri": "databricks", "registry_uri": "databricks-uc",
        "catalog": CATALOG, "schema": SCHEMA,
        "app_identity_note": "The deployed Databricks App service principal needs Unity Catalog EXECUTE access to each model and CAN_QUERY access to each Model Serving endpoint.",
    }


# Serve the built React UI if the frontend build exists.
static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
