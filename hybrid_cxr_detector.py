"""
Accuracy-first hybrid chest X-ray detector.

Single-file PyTorch prototype that combines the requested research directions:
- Barlow Twins / BarlowTwins-CXR style SSL pretraining for lung anatomy features.
- YOLO-CXR-inspired small-lesion branch.
- NSEC-YOLO-inspired noise suppression and global context branch.
- RT-DETR-inspired NMS-free query branch.
- D-FINE-inspired distributional bounding-box refinement.

This file is designed to be trainable and easy to replace with exact paper
implementations later. It is not a claim of SOTA by itself; that requires the
same dataset split, label ontology, metric implementation, and ablation protocol
as the target papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

try:
    from torchvision.ops import roi_align
except Exception:  # pragma: no cover
    roi_align = None


@dataclass
class HybridCXRConfig:
    num_classes: int = 15
    image_channels: int = 1
    base_channels: int = 64
    hidden_dim: int = 256
    num_queries: int = 300
    num_decoder_layers: int = 6
    num_encoder_layers: int = 1
    bbox_bins: int = 48
    dfine_layers: int = 3
    topk_per_head: int = 150
    router_hidden_dim: int = 192
    projector_dim: int = 8192
    projector_hidden_dim: int = 2048
    barlow_lambda: float = 5e-3
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    cls_loss_weight: float = 1.0
    box_l1_weight: float = 5.0
    giou_weight: float = 2.0
    quality_weight: float = 0.5
    dfine_distill_weight: float = 0.35
    min_box_size: float = 1e-4


def box_xywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def sanitize_boxes_xyxy(boxes: Tensor, min_size: float = 1e-4) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    x1 = x1.clamp(0.0, 1.0)
    y1 = y1.clamp(0.0, 1.0)
    x2 = x2.clamp(0.0, 1.0)
    y2 = y2.clamp(0.0, 1.0)
    nx1 = torch.minimum(x1, x2 - min_size).clamp(0.0, 1.0)
    ny1 = torch.minimum(y1, y2 - min_size).clamp(0.0, 1.0)
    nx2 = torch.maximum(x2, nx1 + min_size).clamp(0.0, 1.0)
    ny2 = torch.maximum(y2, ny1 + min_size).clamp(0.0, 1.0)
    return torch.stack((nx1, ny1, nx2, ny2), dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp_min(0) * (boxes[..., 3] - boxes[..., 1]).clamp_min(0)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp_min(1e-7), union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area.clamp_min(1e-7)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class RepRefConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = ConvBNAct(channels, channels, 3)
        self.conv1 = ConvBNAct(channels, channels, 1)
        self.mix = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        return self.mix(torch.cat((self.conv3(x), self.conv1(x)), dim=1)) + x


class C2fBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int = 2):
        super().__init__()
        self.pre = ConvBNAct(in_ch, out_ch, 1)
        self.blocks = nn.ModuleList([RepRefConv(out_ch) for _ in range(depth)])
        self.post = ConvBNAct(out_ch * (depth + 1), out_ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pre(x)
        outs = [x]
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return self.post(torch.cat(outs, dim=1))


class CoordinateLocalAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.coord_h = nn.Conv2d(channels, channels, 1)
        self.coord_w = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h_pool = x.mean(dim=3, keepdim=True)
        w_pool = x.mean(dim=2, keepdim=True)
        coord = torch.sigmoid(self.coord_h(h_pool) + self.coord_w(w_pool))
        return x * self.channel(x) * self.local(x) * coord


class AdaptiveNoiseSuppression(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.smooth = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        smooth = self.smooth(x)
        residual = x - smooth
        local_var = F.avg_pool2d(residual.pow(2), 5, stride=1, padding=2)
        gate = self.gate(torch.cat((x, smooth, local_var), dim=1))
        return gate * x + (1.0 - gate) * smooth


class CXRBackbone(nn.Module):
    def __init__(self, cfg: HybridCXRConfig):
        super().__init__()
        c = cfg.base_channels
        self.stem = nn.Sequential(ConvBNAct(cfg.image_channels, c, 7, 2), ConvBNAct(c, c, 3, 1))
        self.stage2 = nn.Sequential(ConvBNAct(c, c * 2, 3, 2), C2fBlock(c * 2, c * 2, depth=2))
        self.stage3 = nn.Sequential(ConvBNAct(c * 2, c * 4, 3, 2), C2fBlock(c * 4, c * 4, depth=3))
        self.stage4 = nn.Sequential(ConvBNAct(c * 4, c * 8, 3, 2), C2fBlock(c * 8, c * 8, depth=3))
        self.stage5 = nn.Sequential(ConvBNAct(c * 8, c * 8, 3, 2), C2fBlock(c * 8, c * 8, depth=2))
        self.out_channels = {"p2": c * 2, "p3": c * 4, "p4": c * 8, "p5": c * 8}

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.stem(x)
        p2 = self.stage2(x)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


class MultiScaleBiFPNPAN(nn.Module):
    def __init__(self, in_channels: Dict[str, int], hidden_dim: int):
        super().__init__()
        keys = ["p2", "p3", "p4", "p5"]
        self.lateral = nn.ModuleDict({k: nn.Conv2d(in_channels[k], hidden_dim, 1) for k in keys})
        self.top = nn.ModuleDict({k: C2fBlock(hidden_dim, hidden_dim, depth=1) for k in keys})
        self.down = nn.ModuleDict({k: ConvBNAct(hidden_dim, hidden_dim, 3, 2) for k in ["p2", "p3", "p4"]})
        self.pan = nn.ModuleDict({k: C2fBlock(hidden_dim * 2, hidden_dim, depth=1) for k in ["p3", "p4", "p5"]})
        self.attn = nn.ModuleDict({k: CoordinateLocalAttention(hidden_dim) for k in keys})

    def forward(self, feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        p2 = self.lateral["p2"](feats["p2"])
        p3 = self.lateral["p3"](feats["p3"])
        p4 = self.lateral["p4"](feats["p4"])
        p5 = self.lateral["p5"](feats["p5"])
        t5 = self.top["p5"](p5)
        t4 = self.top["p4"](p4 + F.interpolate(t5, size=p4.shape[-2:], mode="nearest"))
        t3 = self.top["p3"](p3 + F.interpolate(t4, size=p3.shape[-2:], mode="nearest"))
        t2 = self.top["p2"](p2 + F.interpolate(t3, size=p2.shape[-2:], mode="nearest"))
        o2 = t2
        o3 = self.pan["p3"](torch.cat((t3, self.down["p2"](o2)), dim=1))
        o4 = self.pan["p4"](torch.cat((t4, self.down["p3"](o3)), dim=1))
        o5 = self.pan["p5"](torch.cat((t5, self.down["p4"](o4)), dim=1))
        out = {"p2": o2, "p3": o3, "p4": o4, "p5": o5}
        return {k: self.attn[k](v) for k, v in out.items()}


class BarlowProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_offdiag: float = 5e-3):
        super().__init__()
        self.lambda_offdiag = lambda_offdiag

    @staticmethod
    def off_diagonal(x: Tensor) -> Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        batch = z1.size(0)
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
        c = (z1.T @ z2) / batch
        on_diag = torch.diagonal(c).add(-1).pow(2).sum()
        off_diag = self.off_diagonal(c).pow(2).sum()
        return on_diag + self.lambda_offdiag * off_diag


class BarlowCXRPretrainer(nn.Module):
    def __init__(self, cfg: HybridCXRConfig):
        super().__init__()
        self.backbone = CXRBackbone(cfg)
        self.projector = BarlowProjector(self.backbone.out_channels["p5"], cfg.projector_hidden_dim, cfg.projector_dim)
        self.loss_fn = BarlowTwinsLoss(cfg.barlow_lambda)

    def encode(self, x: Tensor) -> Tensor:
        p5 = self.backbone(x)["p5"]
        return F.adaptive_avg_pool2d(p5, 1).flatten(1)

    def forward(self, view1: Tensor, view2: Tensor) -> Dict[str, Tensor]:
        z1 = self.projector(self.encode(view1))
        z2 = self.projector(self.encode(view2))
        return {"loss": self.loss_fn(z1, z2), "z1": z1, "z2": z2}


class DenseCXRHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, num_levels: int):
        super().__init__()
        self.stem = nn.ModuleList([C2fBlock(hidden_dim, hidden_dim, depth=1) for _ in range(num_levels)])
        self.cls = nn.ModuleList([nn.Conv2d(hidden_dim, num_classes, 1) for _ in range(num_levels)])
        self.obj = nn.ModuleList([nn.Conv2d(hidden_dim, 1, 1) for _ in range(num_levels)])
        self.box = nn.ModuleList([nn.Conv2d(hidden_dim, 4, 1) for _ in range(num_levels)])

    @staticmethod
    def decode_boxes(raw_box: Tensor) -> Tensor:
        b, _, h, w = raw_box.shape
        device = raw_box.device
        dtype = raw_box.dtype
        y, x = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing="ij")
        grid = torch.stack((x, y), dim=0).unsqueeze(0)
        scale = torch.tensor([w, h], device=device, dtype=dtype).view(1, 2, 1, 1)
        xy = (raw_box[:, 0:2].sigmoid() + grid) / scale
        wh = (raw_box[:, 2:4].sigmoid().pow(2) * 2.0) / scale
        boxes = box_xywh_to_xyxy(torch.cat((xy, wh), dim=1).permute(0, 2, 3, 1))
        return sanitize_boxes_xyxy(boxes.flatten(1, 2))

    def forward(self, feats: List[Tensor]) -> List[Dict[str, Tensor]]:
        outs = []
        for i, f in enumerate(feats):
            f = self.stem[i](f)
            outs.append({
                "logits": self.cls[i](f).flatten(2).transpose(1, 2),
                "objectness": self.obj[i](f).flatten(2).transpose(1, 2).squeeze(-1),
                "boxes": self.decode_boxes(self.box[i](f)),
            })
        return outs


class YOLOCXRSmallLesionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.sff = nn.Sequential(nn.Conv2d(hidden_dim * 3, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True))
        self.local = nn.ModuleList([CoordinateLocalAttention(hidden_dim) for _ in range(3)])
        self.refine = nn.ModuleList([RepRefConv(hidden_dim) for _ in range(3)])
        self.head = DenseCXRHead(hidden_dim, num_classes, num_levels=3)

    def forward(self, feats: Dict[str, Tensor]) -> List[Dict[str, Tensor]]:
        p2, p3, p4 = feats["p2"], feats["p3"], feats["p4"]
        p2_fused = self.sff(torch.cat((p2, F.interpolate(p3, size=p2.shape[-2:], mode="nearest"), F.interpolate(p4, size=p2.shape[-2:], mode="nearest")), dim=1)) + p2
        levels = [p2_fused, p3, p4]
        levels = [self.refine[i](self.local[i](levels[i])) for i in range(3)]
        return self.head(levels)


class NSECYOLOHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.denoise = nn.ModuleDict({k: AdaptiveNoiseSuppression(hidden_dim) for k in ["p3", "p4", "p5"]})
        self.global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_dim, hidden_dim, 1), nn.SiLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 1), nn.Sigmoid())
        self.head = DenseCXRHead(hidden_dim, num_classes, num_levels=3)

    def forward(self, feats: Dict[str, Tensor]) -> List[Dict[str, Tensor]]:
        levels = []
        for k in ["p3", "p4", "p5"]:
            f = self.denoise[k](feats[k])
            levels.append(f * (1.0 + self.global_context(f)))
        return self.head(levels)


class UNetFPNAnatomyHead(nn.Module):
    """U-Net decoder over FPN features for anatomy-preserving multi-scale detection."""
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.decode4 = C2fBlock(hidden_dim * 2, hidden_dim, depth=1)
        self.decode3 = C2fBlock(hidden_dim * 2, hidden_dim, depth=1)
        self.decode2 = C2fBlock(hidden_dim * 2, hidden_dim, depth=1)
        self.head = DenseCXRHead(hidden_dim, num_classes, num_levels=3)

    def forward(self, feats: Dict[str, Tensor]) -> List[Dict[str, Tensor]]:
        d4 = self.decode4(torch.cat((feats["p4"], F.interpolate(feats["p5"], size=feats["p4"].shape[-2:], mode="bilinear", align_corners=False)), dim=1))
        d3 = self.decode3(torch.cat((feats["p3"], F.interpolate(d4, size=feats["p3"].shape[-2:], mode="bilinear", align_corners=False)), dim=1))
        d2 = self.decode2(torch.cat((feats["p2"], F.interpolate(d3, size=feats["p2"].shape[-2:], mode="bilinear", align_corners=False)), dim=1))
        return self.head([d2, d3, d4])


class SinePositionEncoding(nn.Module):
    def __init__(self, dim: int, temperature: int = 10000):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        y, x = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing="ij")
        y = y / max(h - 1, 1)
        x = x / max(w - 1, 1)
        base = max(self.dim // 4, 1)
        dim_t = torch.arange(base, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / base)
        pos_x = x[..., None] / dim_t
        pos_y = y[..., None] / dim_t
        pos = torch.cat((pos_x.sin(), pos_x.cos(), pos_y.sin(), pos_y.cos()), dim=-1)[..., : self.dim]
        if pos.size(-1) < self.dim:
            pos = F.pad(pos, (0, self.dim - pos.size(-1)))
        return pos.flatten(0, 1)


class RTDETRHead(nn.Module):
    def __init__(self, cfg: HybridCXRConfig):
        super().__init__()
        self.cfg = cfg
        self.pos = SinePositionEncoding(cfg.hidden_dim)
        self.level_embed = nn.Parameter(torch.randn(3, cfg.hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(cfg.hidden_dim, 8, cfg.hidden_dim * 4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.num_encoder_layers)
        dec_layer = nn.TransformerDecoderLayer(cfg.hidden_dim, 8, cfg.hidden_dim * 4, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.num_decoder_layers)
        self.query_score = nn.Linear(cfg.hidden_dim, 1)
        self.query_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.cls = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.box = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(inplace=True), nn.Linear(cfg.hidden_dim, 4))
        self.quality = nn.Linear(cfg.hidden_dim, 1)

    def flatten_features(self, feats: Dict[str, Tensor]) -> Tensor:
        seqs = []
        for level, k in enumerate(["p3", "p4", "p5"]):
            f = feats[k]
            b, c, h, w = f.shape
            pos = self.pos(h, w, f.device, f.dtype).unsqueeze(0).expand(b, -1, -1)
            seqs.append(f.flatten(2).transpose(1, 2) + pos + self.level_embed[level].view(1, 1, -1))
        return torch.cat(seqs, dim=1)

    def forward(self, feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        memory = self.encoder(self.flatten_features(feats))
        b, n, c = memory.shape
        k = min(self.cfg.num_queries, n)
        query_idx = self.query_score(memory).squeeze(-1).topk(k, dim=1).indices
        queries = torch.gather(memory, 1, query_idx.unsqueeze(-1).expand(-1, -1, c))
        hs = self.decoder(self.query_proj(queries), memory)
        boxes = sanitize_boxes_xyxy(box_xywh_to_xyxy(torch.sigmoid(self.box(hs))), self.cfg.min_box_size)
        return {"logits": self.cls(hs), "boxes": boxes, "quality": self.quality(hs), "tokens": hs}


class EvidenceRouter(nn.Module):
    def __init__(self, cfg: HybridCXRConfig):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.hidden_dim + 4, cfg.router_hidden_dim), nn.ReLU(inplace=True), nn.Linear(cfg.router_hidden_dim, cfg.router_hidden_dim), nn.ReLU(inplace=True), nn.Linear(cfg.router_hidden_dim, 4))

    def forward(self, global_feat: Tensor, image_stats: Tensor) -> Tensor:
        return self.net(torch.cat((global_feat, image_stats), dim=-1)).softmax(dim=-1)


class DFineBBoxRefiner(nn.Module):
    def __init__(self, cfg: HybridCXRConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.bbox_bins < 4 or cfg.bbox_bins % 2 != 0:
            raise ValueError("D-FINE bbox_bins must be an even integer >= 4")
        half = cfg.bbox_bins // 2
        positive = torch.expm1(torch.linspace(0.0, math.log(1.20), half + 1))[1:]
        bins = torch.cat((-positive.flip(0), positive))
        self.register_buffer("bins", bins / bins.abs().max() * 0.20)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(cfg.hidden_dim + 4, cfg.hidden_dim), nn.ReLU(inplace=True), nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(inplace=True))
            for _ in range(cfg.dfine_layers)
        ])
        self.dist_heads = nn.ModuleList([nn.Linear(cfg.hidden_dim, 4 * cfg.bbox_bins) for _ in range(cfg.dfine_layers)])
        self.quality_heads = nn.ModuleList([nn.Linear(cfg.hidden_dim, 1) for _ in range(cfg.dfine_layers)])

    def forward(self, roi_feat: Tensor, boxes: Tensor) -> Dict[str, Tensor]:
        refined = boxes
        boxes_per_layer = []
        dist_per_layer = []
        quality_per_layer = []
        for layer, dist_head, q_head in zip(self.layers, self.dist_heads, self.quality_heads):
            h = layer(torch.cat((roi_feat, refined), dim=-1))
            dist = dist_head(h).view(*refined.shape[:-1], 4, self.cfg.bbox_bins)
            prob = dist.softmax(dim=-1)
            delta = (prob * self.bins.view(1, 1, 1, -1)).sum(dim=-1)
            refined = sanitize_boxes_xyxy(refined + delta, self.cfg.min_box_size)
            boxes_per_layer.append(refined)
            dist_per_layer.append(dist)
            quality_per_layer.append(q_head(h))
        return {"boxes": boxes_per_layer[-1], "boxes_per_layer": boxes_per_layer, "bbox_dist": dist_per_layer[-1], "dist_per_layer": dist_per_layer, "quality": quality_per_layer[-1], "quality_per_layer": quality_per_layer}


class HybridCXRDetector(nn.Module):
    def __init__(self, cfg: HybridCXRConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = CXRBackbone(cfg)
        self.neck = MultiScaleBiFPNPAN(self.backbone.out_channels, cfg.hidden_dim)
        self.yolo_cxr = YOLOCXRSmallLesionHead(cfg.hidden_dim, cfg.num_classes)
        self.nsec_yolo = NSECYOLOHead(cfg.hidden_dim, cfg.num_classes)
        self.anatomy_unet_fpn = UNetFPNAnatomyHead(cfg.hidden_dim, cfg.num_classes)
        self.rt_detr = RTDETRHead(cfg)
        self.router = EvidenceRouter(cfg)
        self.dfine = DFineBBoxRefiner(cfg)
        self.initialize_weights()


    def initialize_weights(self) -> None:
        """Accuracy-first initialization for sparse CXR detection.

        - Kaiming init for conv features.
        - Xavier init for linear/query layers.
        - Low class/objectness priors so early training is not flooded by false positives.
        - RT-DETR boxes start near image center with moderate size instead of giant 0.5x0.5 boxes.
        - D-FINE distribution heads start as identity refiners, then learn corrections.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        prior = 0.01
        prior_bias = math.log(prior / (1.0 - prior))
        for dense in (self.yolo_cxr.head, self.nsec_yolo.head, self.anatomy_unet_fpn.head):
            for cls_layer in dense.cls:
                nn.init.constant_(cls_layer.bias, prior_bias)
            for obj_layer in dense.obj:
                nn.init.constant_(obj_layer.bias, prior_bias)
            for box_layer in dense.box:
                nn.init.zeros_(box_layer.bias)

        nn.init.constant_(self.rt_detr.cls.bias, prior_bias)
        nn.init.constant_(self.rt_detr.quality.bias, prior_bias)
        final_box = self.rt_detr.box[-1]
        nn.init.zeros_(final_box.weight)
        with torch.no_grad():
            final_box.bias.copy_(torch.tensor([0.0, 0.0, -2.0, -2.0]))

        for dist_head in self.dfine.dist_heads:
            nn.init.zeros_(dist_head.weight)
            nn.init.zeros_(dist_head.bias)
        for quality_head in self.dfine.quality_heads:
            nn.init.constant_(quality_head.bias, prior_bias)


    @staticmethod
    def dense_to_candidates(head_out: List[Dict[str, Tensor]], topk: int) -> Dict[str, Tensor]:
        logits_all, boxes_all, scores_all = [], [], []
        for out in head_out:
            weighted = out["logits"].sigmoid() * out["objectness"].sigmoid().unsqueeze(-1)
            logits_all.append(out["logits"])
            boxes_all.append(out["boxes"])
            scores_all.append(weighted.max(dim=-1).values)
        logits = torch.cat(logits_all, dim=1)
        boxes = torch.cat(boxes_all, dim=1)
        scores = torch.cat(scores_all, dim=1)
        k = min(topk, scores.size(1))
        idx = scores.topk(k, dim=1).indices
        return {"boxes": boxes.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4)), "logits": logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, logits.size(-1))), "scores": scores.gather(1, idx)}

    @staticmethod
    def image_stats(x: Tensor) -> Tensor:
        flat = x.flatten(1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True)
        low = flat.quantile(0.05, dim=1, keepdim=True)
        high = flat.quantile(0.95, dim=1, keepdim=True)
        return torch.cat((mean, std, low, high), dim=1)

    @staticmethod
    def roi_features(feat: Tensor, boxes: Tensor) -> Tensor:
        b, n, _ = boxes.shape
        if roi_align is None:
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            return pooled.unsqueeze(1).expand(-1, n, -1)
        _, _, h, w = feat.shape
        batch_idx = torch.arange(b, device=feat.device, dtype=feat.dtype).view(b, 1, 1).expand(b, n, 1)
        scale = boxes.new_tensor([w - 1, h - 1, w - 1, h - 1]).view(1, 1, 4)
        rois = torch.cat((batch_idx, boxes * scale), dim=-1).reshape(-1, 5)
        aligned = roi_align(feat, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True)
        return aligned.flatten(1).view(b, n, -1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        feats = self.neck(self.backbone(x))
        yolo_raw = self.yolo_cxr(feats)
        nsec_raw = self.nsec_yolo(feats)
        anatomy_raw = self.anatomy_unet_fpn(feats)
        detr_raw = self.rt_detr(feats)
        yolo = self.dense_to_candidates(yolo_raw, self.cfg.topk_per_head)
        nsec = self.dense_to_candidates(nsec_raw, self.cfg.topk_per_head)
        anatomy = self.dense_to_candidates(anatomy_raw, self.cfg.topk_per_head)
        detr_scores = detr_raw["logits"].sigmoid().max(dim=-1).values * detr_raw["quality"].sigmoid().squeeze(-1)
        detr = {"boxes": detr_raw["boxes"], "logits": detr_raw["logits"], "scores": detr_scores}
        global_feat = F.adaptive_avg_pool2d(feats["p3"], 1).flatten(1)
        router_weights = self.router(global_feat, self.image_stats(x))
        boxes = torch.cat((yolo["boxes"], nsec["boxes"], anatomy["boxes"], detr["boxes"]), dim=1)
        logits = torch.cat((yolo["logits"], nsec["logits"], anatomy["logits"], detr["logits"]), dim=1)
        scores = torch.cat((yolo["scores"], nsec["scores"], anatomy["scores"], detr["scores"]), dim=1)
        counts = (yolo["boxes"].size(1), nsec["boxes"].size(1), anatomy["boxes"].size(1), detr["boxes"].size(1))
        head_weight = torch.cat(tuple(router_weights[:, i:i + 1].expand(-1, count) for i, count in enumerate(counts)), dim=1)
        scores = scores * head_weight
        roi_feat = self.roi_features(feats["p3"], boxes)
        refined = self.dfine(roi_feat, boxes)
        return {"boxes": refined["boxes"], "logits": logits, "scores": scores, "router_weights": router_weights, "bbox_quality": refined["quality"].squeeze(-1), "bbox_dist": refined["bbox_dist"], "boxes_per_layer": refined["boxes_per_layer"], "raw": {"yolo_cxr": yolo_raw, "nsec_yolo": nsec_raw, "anatomy_unet_fpn": anatomy_raw, "rt_detr": detr_raw}}


class HungarianMatcher(nn.Module):
    def __init__(self, cls_cost: float = 1.0, l1_cost: float = 5.0, giou_cost: float = 2.0):
        super().__init__()
        self.cls_cost = cls_cost
        self.l1_cost = l1_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        pred_boxes = outputs["boxes"]
        pred_logits = outputs["logits"].sigmoid()
        matches = []
        for b, target in enumerate(targets):
            tgt_boxes = target["boxes"].to(pred_boxes.device).float()
            tgt_labels = target["labels"].to(pred_boxes.device).long()
            if tgt_boxes.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=pred_boxes.device)
                matches.append((empty, empty))
                continue
            cost = self.cls_cost * (-pred_logits[b][:, tgt_labels]) + self.l1_cost * torch.cdist(pred_boxes[b], tgt_boxes, p=1) + self.giou_cost * (-generalized_box_iou(pred_boxes[b], tgt_boxes))
            if linear_sum_assignment is not None:
                rows, cols = linear_sum_assignment(cost.detach().cpu())
                matches.append((torch.as_tensor(rows, dtype=torch.long, device=pred_boxes.device), torch.as_tensor(cols, dtype=torch.long, device=pred_boxes.device)))
            else:
                rows, cols, used_r, used_c = [], [], set(), set()
                flat = cost.flatten().argsort()
                n_pred, n_tgt = cost.shape
                for item in flat.tolist():
                    r, c = divmod(item, n_tgt)
                    if r not in used_r and c not in used_c:
                        rows.append(r); cols.append(c); used_r.add(r); used_c.add(c)
                        if len(cols) == n_tgt:
                            break
                matches.append((torch.as_tensor(rows, dtype=torch.long, device=pred_boxes.device), torch.as_tensor(cols, dtype=torch.long, device=pred_boxes.device)))
        return matches


class HybridDetectionLoss(nn.Module):
    def __init__(self, cfg: HybridCXRConfig):
        super().__init__()
        self.cfg = cfg
        self.matcher = HungarianMatcher(cfg.cls_loss_weight, cfg.box_l1_weight, cfg.giou_weight)

    def sigmoid_focal_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        alpha = self.cfg.focal_alpha * target + (1.0 - self.cfg.focal_alpha) * (1.0 - target)
        return (alpha * (1.0 - pt).pow(self.cfg.focal_gamma) * ce).mean()

    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        matches = self.matcher(outputs, targets)
        logits, boxes, quality = outputs["logits"], outputs["boxes"], outputs["bbox_quality"]
        cls_target = torch.zeros_like(logits)
        pred_boxes, tgt_boxes, q_pred, q_tgt = [], [], [], []
        for b, (pred_idx, tgt_idx) in enumerate(matches):
            if pred_idx.numel() == 0:
                continue
            labels = targets[b]["labels"].to(logits.device).long()[tgt_idx]
            target_boxes = targets[b]["boxes"].to(boxes.device).float()[tgt_idx]
            cls_target[b, pred_idx, labels] = 1.0
            p_boxes = boxes[b, pred_idx]
            pred_boxes.append(p_boxes)
            tgt_boxes.append(target_boxes)
            q_pred.append(quality[b, pred_idx])
            q_tgt.append(box_iou(p_boxes, target_boxes)[0].diag().detach())
        loss_cls = self.sigmoid_focal_loss(logits, cls_target)
        if pred_boxes:
            pred = torch.cat(pred_boxes, dim=0)
            tgt = torch.cat(tgt_boxes, dim=0)
            loss_l1 = F.l1_loss(pred, tgt)
            loss_giou = (1.0 - generalized_box_iou(pred, tgt).diag()).mean()
            loss_quality = F.binary_cross_entropy_with_logits(torch.cat(q_pred, dim=0), torch.cat(q_tgt, dim=0))
        else:
            zero = logits.sum() * 0.0
            loss_l1 = zero; loss_giou = zero; loss_quality = zero
        loss_distill = logits.sum() * 0.0
        if "boxes_per_layer" in outputs and len(outputs["boxes_per_layer"]) > 1:
            final = outputs["boxes_per_layer"][-1].detach()
            for mid in outputs["boxes_per_layer"][:-1]:
                loss_distill = loss_distill + F.smooth_l1_loss(mid, final)
            loss_distill = loss_distill / (len(outputs["boxes_per_layer"]) - 1)
        total = self.cfg.cls_loss_weight * loss_cls + self.cfg.box_l1_weight * loss_l1 + self.cfg.giou_weight * loss_giou + self.cfg.quality_weight * loss_quality + self.cfg.dfine_distill_weight * loss_distill
        return {"loss": total, "loss_cls": loss_cls.detach(), "loss_l1": loss_l1.detach(), "loss_giou": loss_giou.detach(), "loss_quality": loss_quality.detach(), "loss_dfine_distill": loss_distill.detach()}


def create_model(num_classes: int = 15, image_channels: int = 1, **kwargs) -> HybridCXRDetector:
    cfg = HybridCXRConfig(num_classes=num_classes, image_channels=image_channels, **kwargs)
    return HybridCXRDetector(cfg)


def create_pretrainer(image_channels: int = 1, **kwargs) -> BarlowCXRPretrainer:
    cfg = HybridCXRConfig(image_channels=image_channels, **kwargs)
    return BarlowCXRPretrainer(cfg)


def create_loss(num_classes: int = 15, image_channels: int = 1, **kwargs) -> HybridDetectionLoss:
    cfg = HybridCXRConfig(num_classes=num_classes, image_channels=image_channels, **kwargs)
    return HybridDetectionLoss(cfg)


if __name__ == "__main__":
    torch.manual_seed(7)
    model = create_model(num_classes=15, image_channels=1, base_channels=16, hidden_dim=64, num_queries=24, num_decoder_layers=2, num_encoder_layers=1, topk_per_head=12, bbox_bins=16, dfine_layers=2)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print("boxes", tuple(out["boxes"].shape))
    print("logits", tuple(out["logits"].shape))
    print("scores", tuple(out["scores"].shape))
    print("router_weights", out["router_weights"].detach())
    targets = [
        {"boxes": torch.tensor([[0.20, 0.20, 0.38, 0.42], [0.55, 0.50, 0.72, 0.76]]), "labels": torch.tensor([1, 4])},
        {"boxes": torch.tensor([[0.10, 0.30, 0.24, 0.46]]), "labels": torch.tensor([3])},
    ]
    losses = create_loss(num_classes=15, image_channels=1)(out, targets)
    print("loss", float(losses["loss"].detach()))
    pretrainer = create_pretrainer(image_channels=1, base_channels=8, projector_dim=256, projector_hidden_dim=128)
    bt = pretrainer(torch.randn(4, 1, 128, 128), torch.randn(4, 1, 128, 128))
    print("barlow_loss", float(bt["loss"].detach()))
