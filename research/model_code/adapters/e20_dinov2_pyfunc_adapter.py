
"""AeroTrust E20 detector: DINOv2 ViT-B/14-reg4 backbone + simple feature pyramid (ViTDet-style) + Faster R-CNN head.
MLflow "models from code" file: architecture, preprocessing and postprocessing live here; weights and config are artifacts.
Input : DataFrame with column image_b64 (base64 JPEG/PNG bytes) and optional image_id.
Params: score_threshold (float), max_detections (int).
Output: DataFrame with image_id, width, height, detections (JSON list of {class_id, class_name, score, box_xyxy, rel_area})."""
import base64, io, json
from collections import OrderedDict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fnn, mlflow
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops import MultiScaleRoIAlign


class ViTSFP(nn.Module):
    """Module layout identical to E20's training code (vit / up4 / up2 / h4 / h8 / h16 / h32)."""
    def __init__(self, timm_name, out_ch=256):
        super().__init__()
        import timm
        from timm.layers import LayerNorm2d
        self.vit = timm.create_model(timm_name, pretrained=False, num_classes=0, dynamic_img_size=True)
        self.patch = self.vit.patch_embed.patch_size[0]
        self.n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        C = self.vit.num_features
        def head(cin):
            return nn.Sequential(nn.Conv2d(cin, out_ch, 1, bias=False), LayerNorm2d(out_ch),
                                 nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), LayerNorm2d(out_ch))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(C, C // 2, 2, 2), LayerNorm2d(C // 2), nn.GELU(), nn.ConvTranspose2d(C // 2, C // 4, 2, 2))
        self.up2 = nn.ConvTranspose2d(C, C // 2, 2, 2)
        self.h4, self.h8, self.h16, self.h32 = head(C // 4), head(C // 2), head(C), head(C)
        self.out_channels = out_ch

    def forward(self, x):
        H, W = x.shape[-2:]
        h, w = max(self.patch, round(H / self.patch) * self.patch), max(self.patch, round(W / self.patch) * self.patch)
        xv = x if (h, w) == (H, W) else Fnn.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        tok = self.vit.forward_features(xv)[:, self.n_prefix:, :]
        f = tok.transpose(1, 2).reshape(x.shape[0], -1, h // self.patch, w // self.patch)
        rs = lambda t, s: Fnn.interpolate(t, size=(H // s, W // s), mode="bilinear", align_corners=False)
        p32 = self.h32(rs(f, 32))
        return OrderedDict([("0", self.h4(rs(self.up4(f), 4))), ("1", self.h8(rs(self.up2(f), 8))),
                            ("2", self.h16(rs(f, 16))), ("3", p32), ("pool", Fnn.max_pool2d(p32, 1, 2))])


def build_e20(cfg):
    sizes = tuple(tuple(s) for s in cfg["anchor_sizes"])
    m = FasterRCNN(ViTSFP(cfg["timm_name"]), num_classes=cfg["num_classes"],
                   rpn_anchor_generator=AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes)),
                   rpn_head=RPNHead(256, len(sizes[0]) * 3), box_roi_pool=MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2),
                   size_divisible=cfg["size_divisible"], image_mean=cfg["mean"], image_std=cfg["std"],
                   min_size=cfg["image_size"], max_size=cfg["image_size"],
                   box_score_thresh=0.001, box_nms_thresh=cfg["box_nms_thresh"], box_detections_per_img=cfg["max_det"])
    return m


def clean_state_dict(ck):
    sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    return {k.replace("module.", "", 1): (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}


class E20Detector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.cfg = json.load(open(context.artifacts["config"]))
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = build_e20(self.cfg)
        m.load_state_dict(clean_state_dict(torch.load(context.artifacts["weights"], map_location="cpu", weights_only=False)), strict=True)
        self.model = m.eval().to(self.dev)

    @torch.no_grad()
    def _run(self, imgs):
        tens = [TF.to_tensor(im).to(self.dev) for im in imgs]
        if self.dev.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                return self.model(tens)
        return self.model(tens)

    def predict(self, context, model_input, params=None):
        p = params or {}
        thr = float(p.get("score_threshold", self.cfg["default_score_threshold"]))
        mx = int(p.get("max_detections", 100))
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        ids = df["image_id"].astype(str).tolist() if "image_id" in df else [str(i) for i in range(len(df))]
        imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in df["image_b64"]]
        out = []
        for i in range(0, len(imgs), 2):
            for im, iid, r in zip(imgs[i:i + 2], ids[i:i + 2], self._run(imgs[i:i + 2])):
                W, H = im.size
                keep = (r["scores"] >= thr).nonzero().flatten()[:mx]
                dets = []
                for j in keep.tolist():
                    x1, y1, x2, y2 = [float(v) for v in r["boxes"][j]]
                    c = int(r["labels"][j]) - 1
                    dets.append(dict(class_id=c, class_name=self.cfg["class_names"][c], score=round(float(r["scores"][j]), 4),
                                     box_xyxy=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                     rel_area=round((x2 - x1) * (y2 - y1) / float(W * H), 6)))
                out.append(dict(image_id=iid, width=W, height=H, detections=json.dumps(dets)))
        return pd.DataFrame(out)


mlflow.models.set_model(E20Detector())
