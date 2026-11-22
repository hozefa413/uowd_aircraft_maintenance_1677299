
"""AeroTrust detector adapter (contract aerotrust-detector-v1).
Families: torchvision Faster R-CNN with ResNet-50/101 FPN (v1 or v2 head), the AeroTrust Cascade R-CNN head, and a
DINOv2 / DINOv3 ViT backbone with a simple feature pyramid (ViTDet-style). Weights and config are MLflow artifacts.
Input : DataFrame with image_b64 (base64 JPEG/PNG) and optional image_id.   Params: score_threshold, max_detections.
Output: DataFrame with image_id, width, height, detections = JSON list of {class_id, class_name, score, box_xyxy, rel_area}."""
import base64, io, json
from collections import OrderedDict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fnn, torchvision, mlflow
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, TwoMLPHead
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import MultiScaleRoIAlign, boxes as box_ops

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class ViTSFP(nn.Module):
    """Same module layout and forward as the E20 / E22 training code."""
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


CASCADE_IOUS = (0.5, 0.6, 0.7)
CASCADE_WEIGHTS = ((10., 10., 5., 5.), (20., 20., 10., 10.), (30., 30., 15., 15.))


class StagePredictor(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_ch, num_classes)
        self.bbox_pred = nn.Linear(in_ch, 4)

    def forward(self, x):
        x = x.flatten(1)
        return self.cls_score(x), self.bbox_pred(x)


class CascadeRoIHeads(nn.Module):
    """Inference part of the E17 / E18 Cascade R-CNN head (3 stages, class-agnostic boxes, mean of stage scores)."""
    def __init__(self, out_channels, num_classes, score_thresh, nms_thresh, detections_per_img):
        super().__init__()
        self.box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        self.heads = nn.ModuleList([TwoMLPHead(out_channels * 7 * 7, 1024) for _ in CASCADE_IOUS])
        self.preds = nn.ModuleList([StagePredictor(1024, num_classes) for _ in CASCADE_IOUS])
        self.coders = [det_utils.BoxCoder(w) for w in CASCADE_WEIGHTS]
        self.score_thresh, self.nms_thresh, self.detections_per_img, self.num_classes = score_thresh, nms_thresh, detections_per_img, num_classes

    def _stage(self, s, features, rois, image_shapes):
        return self.preds[s](self.heads[s](self.box_roi_pool(features, rois, image_shapes)))

    def _decode(self, s, reg, rois, image_shapes):
        n = [len(r) for r in rois]
        b = self.coders[s].decode(reg.float(), [r.float() for r in rois]).reshape(-1, 4)
        return [box_ops.clip_boxes_to_image(bb, sh) for bb, sh in zip(b.split(n), image_shapes)]

    def forward(self, features, proposals, image_shapes, targets=None):
        rois, probs = [p.float() for p in proposals], []
        for s in range(len(CASCADE_IOUS)):
            cls, reg = self._stage(s, features, rois, image_shapes)
            probs.append(Fnn.softmax(cls.float(), -1))
            final_reg, final_rois = reg, rois
            if s < len(CASCADE_IOUS) - 1:
                rois = self._decode(s, reg, rois, image_shapes)
        scores = torch.stack(probs).mean(0)
        boxes = self._decode(len(CASCADE_IOUS) - 1, final_reg, final_rois, image_shapes)
        out, C = [], self.num_classes
        for bx, sc in zip(boxes, scores.split([len(r) for r in final_rois])):
            bx = bx[:, None, :].expand(-1, C - 1, 4).reshape(-1, 4)
            lb = torch.arange(1, C, device=sc.device).view(1, -1).expand(sc.shape[0], -1).reshape(-1)
            sc = sc[:, 1:].reshape(-1)
            k = torch.where(sc > self.score_thresh)[0]; bx, sc, lb = bx[k], sc[k], lb[k]
            k = box_ops.remove_small_boxes(bx, 1e-2); bx, sc, lb = bx[k], sc[k], lb[k]
            k = box_ops.batched_nms(bx, sc, lb, self.nms_thresh)[: self.detections_per_img]
            out.append({"boxes": bx[k], "scores": sc[k], "labels": lb[k]})
        return out, {}


def build(cfg):
    sizes = tuple(tuple(s) for s in cfg["anchor_sizes"]); A = len(sizes[0]) * 3; nc = cfg["num_classes"]
    ag = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes))
    common = dict(min_size=cfg["image_size"], max_size=cfg["image_size"], size_divisible=32, image_mean=MEAN, image_std=STD,
                  rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=1000,
                  box_score_thresh=0.001, box_nms_thresh=cfg["box_nms_thresh"], box_detections_per_img=cfg["max_det"])
    if cfg["family"] == "vit":
        m = FasterRCNN(ViTSFP(cfg["timm_name"]), num_classes=nc, rpn_anchor_generator=ag, rpn_head=RPNHead(256, A),
                       box_roi_pool=MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2), **common)
    elif cfg["head_v2"]:
        res = getattr(torchvision.models, cfg["arch"])(weights=None, norm_layer=nn.BatchNorm2d)
        m = FasterRCNN(_resnet_fpn_extractor(res, 5, norm_layer=nn.BatchNorm2d), num_classes=nc, rpn_anchor_generator=ag,
                       rpn_head=RPNHead(256, A, conv_depth=2),
                       box_head=FastRCNNConvFCHead((256, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d), **common)
    else:
        m = FasterRCNN(resnet_fpn_backbone(backbone_name=cfg["arch"], weights=None, trainable_layers=5), num_classes=nc,
                       rpn_anchor_generator=ag, rpn_head=RPNHead(256, A), **common)
    if cfg.get("cascade"):
        m.roi_heads = CascadeRoIHeads(256, nc, 0.001, cfg["box_nms_thresh"], cfg["max_det"])
    return m


def clean_state_dict(ck):
    sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    return {k.replace("module.", "", 1): (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}


class AeroTrustDetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.cfg = json.load(open(context.artifacts["config"]))
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = build(self.cfg)
        m.load_state_dict(clean_state_dict(torch.load(context.artifacts["weights"], map_location="cpu", weights_only=False)), strict=True)
        self.model = m.eval().to(self.dev)

    @torch.no_grad()
    def _run(self, imgs):
        t = [TF.to_tensor(im).to(self.dev) for im in imgs]
        if self.dev.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                return self.model(t)
        return self.model(t)

    def predict(self, context, model_input, params=None):
        p = params or {}
        thr = float(p.get("score_threshold", self.cfg["default_score_threshold"])); mx = int(p.get("max_detections", 100))
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        ids = df["image_id"].astype(str).tolist() if "image_id" in df else [str(i) for i in range(len(df))]
        imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in df["image_b64"]]
        out = []
        for i in range(0, len(imgs), 2):
            for im, iid, r in zip(imgs[i:i + 2], ids[i:i + 2], self._run(imgs[i:i + 2])):
                W, H = im.size; dets = []
                for j in (r["scores"] >= thr).nonzero().flatten()[:mx].tolist():
                    x1, y1, x2, y2 = [float(v) for v in r["boxes"][j]]; c = int(r["labels"][j]) - 1
                    dets.append(dict(class_id=c, class_name=self.cfg["class_names"][c], score=round(float(r["scores"][j]), 4),
                                     box_xyxy=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                     rel_area=round((x2 - x1) * (y2 - y1) / float(W * H), 6)))
                out.append(dict(image_id=iid, width=W, height=H, detections=json.dumps(dets)))
        return pd.DataFrame(out)


mlflow.models.set_model(AeroTrustDetector())
