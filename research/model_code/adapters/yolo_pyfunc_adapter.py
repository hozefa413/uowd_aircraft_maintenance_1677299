
import os, json, base64, time, io
import numpy as np
import pandas as pd
import torch
import mlflow
from PIL import Image

def _decode_image(encoded):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))

class AerotrustModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.meta = json.load(open(context.artifacts["metadata"], "r"))
        self.name = self.meta["experiment"]
        self.family = self.meta["family"]
        self.classes = self.meta["class_names"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.family == "YOLO":
            from ultralytics import YOLO
            self.model = YOLO(context.artifacts["checkpoint"])
        else:
            factory = self.meta.get("factory")
            if not factory:
                raise RuntimeError("No verified Torch factory for " + self.name)
            import sys, importlib
            if factory["source_dir"] not in sys.path:
                sys.path.insert(0, factory["source_dir"])
            module = importlib.import_module(factory["module"])
            builder = getattr(module, factory["function"])
            self.predictor = builder(
                checkpoint_path=context.artifacts["checkpoint"],
                device=self.device,
                metadata=self.meta
            )

    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        if "image_base64" not in model_input.columns:
            raise ValueError("Input requires image_base64")
        outputs = []
        for _, row in model_input.iterrows():
            image = _decode_image(row["image_base64"])
            start = time.perf_counter()
            if self.family == "YOLO":
                result = self.model.predict(
                    source=image,
                    imgsz=int(self.meta["image_size"]),
                    conf=float(self.meta.get("confidence_threshold", 0.001)),
                    iou=float(self.meta.get("nms_iou", 0.7)),
                    device=self.device,
                    verbose=False
                )[0]
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    scores = result.boxes.conf.detach().cpu().numpy()
                    ids = result.boxes.cls.detach().cpu().numpy().astype(int)
                    for box, score, cid in zip(boxes, scores, ids):
                        names = result.names
                        cname = names.get(int(cid), str(cid)) if isinstance(names, dict) else str(cid)
                        detections.append({
                            "class_id": int(cid), "class_name": cname,
                            "confidence": float(score),
                            "bbox": [float(x) for x in box]
                        })
            else:
                detections = self.predictor(image)
            elapsed = (time.perf_counter() - start) * 1000.0
            normalized = []
            for d in detections or []:
                box = [float(x) for x in d["bbox"]]
                if len(box) != 4:
                    continue
                cid = int(d["class_id"])
                normalized.append({
                    "class_id": cid,
                    "class_name": d.get("class_name",
                        self.classes[cid] if 0 <= cid < len(self.classes) else str(cid)),
                    "confidence": float(d["confidence"]),
                    "bbox": box
                })
            outputs.append({
                "detections_json": json.dumps(normalized),
                "inference_time_ms": float(elapsed),
                "model_name": self.name
            })
        return pd.DataFrame(outputs)

mlflow.models.set_model(AerotrustModel())
