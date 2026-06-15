"""
Explicit BarlowTwins-CXR pretraining and lesion detector for VinDr-CXR.

Scientific basis:
- Zbontar et al., Barlow Twins: Self-Supervised Learning via Redundancy
  Reduction, ICML 2021.
- Sheng et al., BarlowTwins-CXR: enhancing chest X-ray abnormality localization
  in heterogeneous data with cross-domain self-supervised learning, BMC Medical
  Informatics and Decision Making, 2024.

The encoder, projector, cross-correlation loss, FPN, detection heads, training,
evaluation, and visualization are all written explicitly in this file.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torchvision.ops import roi_align
except Exception:  # pragma: no cover
    roi_align = None

from hybrid_cxr_detector import HybridCXRConfig, HybridDetectionLoss
from train_vindr_hybrid import DEFAULT_DATA_ROOT, DETECT_NUM_CLASSES, CLASS_NAMES, VinDrDetectionDataset, ClassBalancedDetectionBatchSampler, detection_collate, evaluate_map50, move_targets, save_checkpoint, seed_everything, resolve_data_root, resolve_training_device, apply_gpu_training_preset, build_optimizer, set_warmup_cosine_lr, ModelEMA, save_detection_visualizations, save_training_checkpoint, resolve_resume_path, load_training_checkpoint


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled and device.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda")


def apply_weight_initialization(model: nn.Module, mode: str) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if mode == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif mode == "orthogonal":
                nn.init.orthogonal_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            if mode == "orthogonal":
                nn.init.orthogonal_(module.weight)
            elif mode == "kaiming":
                nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def build_tunable_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return build_optimizer(model, args)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        opt = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return opt


def xywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1).clamp(0, 1)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualCXRBlock(nn.Module):
    """Basic residual block used by both SSL pretraining and detection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x + self.bn2(self.conv2(self.conv1(x))))


class ResNetBottleneck(nn.Module):
    """ResNet-50 bottleneck block used by BarlowTwins-CXR before Faster R-CNN fine-tuning."""

    expansion = 4

    def __init__(self, in_ch: int, bottleneck_ch: int, stride: int = 1):
        super().__init__()
        out_ch = bottleneck_ch * self.expansion
        self.conv1 = ConvBNReLU(in_ch, bottleneck_ch, 1, 1)
        self.conv2 = ConvBNReLU(bottleneck_ch, bottleneck_ch, 3, stride)
        self.conv3 = nn.Conv2d(bottleneck_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))
        return F.relu(x + residual, inplace=True)


class ExplicitBarlowEncoder(nn.Module):
    """ResNet-50-style encoder: SSL pretraining first, Faster R-CNN/FPN detection second."""

    def __init__(self, image_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(ConvBNReLU(image_channels, c, 7, 2), nn.MaxPool2d(3, stride=2, padding=1))
        self.stage2 = self.make_stage(c, c, blocks=3, stride=1)
        self.stage3 = self.make_stage(c * 4, c * 2, blocks=4, stride=2)
        self.stage4 = self.make_stage(c * 8, c * 4, blocks=6, stride=2)
        self.stage5 = self.make_stage(c * 16, c * 8, blocks=3, stride=2)
        self.out_channels = {"p2": c * 4, "p3": c * 8, "p4": c * 16, "p5": c * 32}
        self.embedding_dim = c * 32

    def make_stage(self, in_ch: int, bottleneck_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResNetBottleneck(in_ch, bottleneck_ch, stride)]
        out_ch = bottleneck_ch * ResNetBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(ResNetBottleneck(out_ch, bottleneck_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        x = self.stem(image)
        p2 = self.stage2(x)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}

    def global_embedding(self, image: Tensor) -> Tensor:
        p5 = self.forward(image)["p5"]
        return F.adaptive_avg_pool2d(p5, 1).flatten(1)


class ExplicitBarlowProjector(nn.Module):
    """High-dimensional projector required by Barlow Twins redundancy reduction."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        return self.linear3(x)


class ExplicitBarlowLoss(nn.Module):
    """Cross-correlation matrix should become identity: invariant diagonal, decorrelated off-diagonal."""

    def __init__(self, lambda_offdiag: float = 5e-3):
        super().__init__()
        self.lambda_offdiag = lambda_offdiag

    def off_diagonal(self, matrix: Tensor) -> Tensor:
        n = matrix.shape[0]
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1_normalized = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
        z2_normalized = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)
        cross_correlation = (z1_normalized.T @ z2_normalized) / z1.shape[0]
        invariance_loss = torch.diagonal(cross_correlation).sub(1.0).pow(2).sum()
        redundancy_loss = self.off_diagonal(cross_correlation).pow(2).sum()
        return invariance_loss + self.lambda_offdiag * redundancy_loss


class ExplicitBarlowPretrainer(nn.Module):
    def __init__(self, image_channels: int, base_channels: int, projector_hidden: int, projector_dim: int, lambda_offdiag: float = 5e-3):
        super().__init__()
        self.encoder = ExplicitBarlowEncoder(image_channels, base_channels)
        self.projector = ExplicitBarlowProjector(self.encoder.embedding_dim, projector_hidden, projector_dim)
        self.loss_fn = ExplicitBarlowLoss(lambda_offdiag)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, view1: Tensor, view2: Tensor) -> Dict[str, Tensor]:
        embedding1 = self.encoder.global_embedding(view1)
        embedding2 = self.encoder.global_embedding(view2)
        projection1 = self.projector(embedding1)
        projection2 = self.projector(embedding2)
        return {"loss": self.loss_fn(projection1, projection2), "z1": projection1, "z2": projection2}


class ExplicitFPN(nn.Module):
    """FPN exposes exactly how SSL encoder features are reused for detection."""

    def __init__(self, in_channels: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.lat2 = nn.Conv2d(in_channels["p2"], hidden_dim, 1)
        self.lat3 = nn.Conv2d(in_channels["p3"], hidden_dim, 1)
        self.lat4 = nn.Conv2d(in_channels["p4"], hidden_dim, 1)
        self.lat5 = nn.Conv2d(in_channels["p5"], hidden_dim, 1)
        self.out2 = ConvBNReLU(hidden_dim, hidden_dim)
        self.out3 = ConvBNReLU(hidden_dim, hidden_dim)
        self.out4 = ConvBNReLU(hidden_dim, hidden_dim)
        self.out5 = ConvBNReLU(hidden_dim, hidden_dim)

    def forward(self, feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        p5 = self.lat5(feats["p5"])
        p4 = self.lat4(feats["p4"]) + F.interpolate(p5, size=feats["p4"].shape[-2:], mode="nearest")
        p3 = self.lat3(feats["p3"]) + F.interpolate(p4, size=feats["p3"].shape[-2:], mode="nearest")
        p2 = self.lat2(feats["p2"]) + F.interpolate(p3, size=feats["p2"].shape[-2:], mode="nearest")
        return {"p2": self.out2(p2), "p3": self.out3(p3), "p4": self.out4(p4), "p5": self.out5(p5)}


class ExplicitDetectionHead(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.shared = ConvBNReLU(channels, channels)
        self.cls = nn.Conv2d(channels, num_classes, 1)
        self.obj = nn.Conv2d(channels, 1, 1)
        self.box = nn.Conv2d(channels, 4, 1)

    def decode(self, raw_box: Tensor) -> Tensor:
        _, _, h, w = raw_box.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=raw_box.device, dtype=raw_box.dtype), torch.arange(w, device=raw_box.device, dtype=raw_box.dtype), indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
        scale = torch.tensor([w, h], device=raw_box.device, dtype=raw_box.dtype).view(1, 2, 1, 1)
        center = (raw_box[:, :2].sigmoid() + grid) / scale
        size = (raw_box[:, 2:].sigmoid().pow(2) * 2.0) / scale
        return xywh_to_xyxy(torch.cat([center, size], dim=1).permute(0, 2, 3, 1)).flatten(1, 2)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.shared(x)
        return {"logits": self.cls(x).flatten(2).transpose(1, 2), "objectness": self.obj(x).flatten(2).transpose(1, 2).squeeze(-1), "boxes": self.decode(self.box(x))}


def sanitize_boxes(boxes: Tensor, eps: float = 1e-4) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    x1 = x1.clamp(0.0, 1.0)
    y1 = y1.clamp(0.0, 1.0)
    x2 = torch.maximum(x2.clamp(0.0, 1.0), x1 + eps).clamp(0.0, 1.0)
    y2 = torch.maximum(y2.clamp(0.0, 1.0), y1 + eps).clamp(0.0, 1.0)
    return torch.stack((x1, y1, x2, y2), dim=-1)


class ExplicitRPNHead(nn.Module):
    """Faster R-CNN region proposal head: shared conv, objectness, and box regression."""

    def __init__(self, channels: int):
        super().__init__()
        self.shared = ConvBNReLU(channels, channels)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.box = nn.Conv2d(channels, 4, 1)

    def decode(self, raw_box: Tensor) -> Tensor:
        _, _, h, w = raw_box.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=raw_box.device, dtype=raw_box.dtype), torch.arange(w, device=raw_box.device, dtype=raw_box.dtype), indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
        scale = torch.tensor([w, h], device=raw_box.device, dtype=raw_box.dtype).view(1, 2, 1, 1)
        center = (raw_box[:, :2].sigmoid() + grid) / scale
        size = (raw_box[:, 2:].sigmoid().pow(2) * 2.0) / scale
        return sanitize_boxes(xywh_to_xyxy(torch.cat([center, size], dim=1).permute(0, 2, 3, 1)).flatten(1, 2))

    def forward(self, feature: Tensor) -> Dict[str, Tensor]:
        x = self.shared(feature)
        return {"objectness": self.objectness(x).flatten(2).squeeze(1), "boxes": self.decode(self.box(x))}


class ExplicitFastRCNNHead(nn.Module):
    """Two-layer ROI head for Faster R-CNN classification and proposal refinement."""

    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels * 2)
        self.fc2 = nn.Linear(channels * 2, channels * 2)
        self.cls = nn.Linear(channels * 2, num_classes)
        self.box_delta = nn.Linear(channels * 2, 4)
        self.quality = nn.Linear(channels * 2, 1)

    def forward(self, roi_features: Tensor) -> Dict[str, Tensor]:
        pooled = F.adaptive_avg_pool2d(roi_features, 1).flatten(1)
        x = F.relu(self.fc1(pooled), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return {"logits": self.cls(x), "box_delta": self.box_delta(x), "quality": self.quality(x).squeeze(-1)}


def apply_box_delta(proposals: Tensor, delta: Tensor) -> Tensor:
    x1, y1, x2, y2 = proposals.unbind(-1)
    px = (x1 + x2) * 0.5
    py = (y1 + y2) * 0.5
    pw = (x2 - x1).clamp_min(1e-4)
    ph = (y2 - y1).clamp_min(1e-4)
    dx, dy, dw, dh = delta.tanh().unbind(-1)
    cx = px + dx * pw * 0.5
    cy = py + dy * ph * 0.5
    ww = pw * torch.exp(dw.clamp(-1.5, 1.5))
    hh = ph * torch.exp(dh.clamp(-1.5, 1.5))
    return sanitize_boxes(torch.stack((cx - ww * 0.5, cy - hh * 0.5, cx + ww * 0.5, cy + hh * 0.5), dim=-1))


class ExplicitBarlowCXRDetector(nn.Module):
    """BarlowTwins-CXR detector: SSL ResNet encoder, FPN, RPN proposals, Fast R-CNN ROI head."""

    def __init__(self, cfg: HybridCXRConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = ExplicitBarlowEncoder(cfg.image_channels, cfg.base_channels)
        self.fpn = ExplicitFPN(self.backbone.out_channels, cfg.hidden_dim)
        self.rpn_p2 = ExplicitRPNHead(cfg.hidden_dim)
        self.rpn_p3 = ExplicitRPNHead(cfg.hidden_dim)
        self.rpn_p4 = ExplicitRPNHead(cfg.hidden_dim)
        self.roi_head = ExplicitFastRCNNHead(cfg.hidden_dim, cfg.num_classes)
        self.initialize_detection_heads()

    def initialize_detection_heads(self) -> None:
        prior_bias = float(np.log(0.01 / 0.99))
        for rpn in [self.rpn_p2, self.rpn_p3, self.rpn_p4]:
            nn.init.constant_(rpn.objectness.bias, prior_bias)
            nn.init.zeros_(rpn.box.bias)
        nn.init.constant_(self.roi_head.cls.bias, prior_bias)
        nn.init.constant_(self.roi_head.quality.bias, prior_bias)
        nn.init.zeros_(self.roi_head.box_delta.bias)

    def load_barlow_encoder(self, checkpoint: str, device: torch.device) -> None:
        if not checkpoint:
            return
        state = torch.load(checkpoint, map_location=device, weights_only=False).get("model", {})
        encoder_state = {key.replace("encoder.", ""): value for key, value in state.items() if key.startswith("encoder.")}
        missing, unexpected = self.backbone.load_state_dict(encoder_state, strict=False)
        print(f"loaded Barlow encoder; missing={len(missing)} unexpected={len(unexpected)}")

    def collect_proposals(self, pyramid: Dict[str, Tensor]) -> Dict[str, Tensor]:
        rpn_outputs = [self.rpn_p2(pyramid["p2"]), self.rpn_p3(pyramid["p3"]), self.rpn_p4(pyramid["p4"])]
        boxes = torch.cat([o["boxes"] for o in rpn_outputs], dim=1)
        scores = torch.cat([o["objectness"].sigmoid() for o in rpn_outputs], dim=1)
        center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
        lung_prior = -((center[..., 0] - 0.5).pow(2) / 0.25 + (center[..., 1] - 0.48).pow(2) / 0.35)
        scores = scores + lung_prior * 1e-6
        k = min(self.cfg.topk_per_head, scores.shape[1])
        idx = scores.topk(k, dim=1).indices
        boxes = boxes.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4))
        scores = scores.gather(1, idx)
        return {"boxes": boxes, "scores": scores}

    def roi_features_for_image(self, pyramid: Dict[str, Tensor], boxes: Tensor, batch_index: int) -> Tensor:
        areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sqrt()
        level_names = torch.where(areas < 0.12, 0, torch.where(areas < 0.25, 1, 2))
        roi_features = boxes.new_zeros((boxes.shape[0], pyramid["p2"].shape[1], 7, 7))
        for level_id, level in enumerate(["p2", "p3", "p4"]):
            keep = torch.nonzero(level_names == level_id, as_tuple=False).flatten()
            if keep.numel() == 0:
                continue
            level = ["p2", "p3", "p4"][int(level_id)]
            feat = pyramid[level][batch_index : batch_index + 1]
            h, w = feat.shape[-2:]
            scaled = boxes[keep] * boxes.new_tensor([w, h, w, h])
            if roi_align is not None:
                pooled = roi_align(feat, [scaled], output_size=(7, 7), spatial_scale=1.0, aligned=True)
            else:
                cx = ((scaled[:, 0] + scaled[:, 2]) * 0.5).long().clamp(0, w - 1)
                cy = ((scaled[:, 1] + scaled[:, 3]) * 0.5).long().clamp(0, h - 1)
                pooled = feat[0, :, cy, cx].permute(1, 0).view(keep.numel(), feat.shape[1], 1, 1).expand(-1, -1, 7, 7)
            roi_features[keep] = pooled
        return roi_features

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        encoder_features = self.backbone(image)
        pyramid = self.fpn(encoder_features)
        proposals = self.collect_proposals(pyramid)
        batch_logits, batch_boxes, batch_quality, batch_scores = [], [], [], []
        for b in range(image.shape[0]):
            proposal_boxes = proposals["boxes"][b]
            proposal_scores = proposals["scores"][b]
            roi_features = self.roi_features_for_image(pyramid, proposal_boxes, b)
            roi_out = self.roi_head(roi_features)
            refined_boxes = apply_box_delta(proposal_boxes, roi_out["box_delta"])
            cls_scores = roi_out["logits"].sigmoid().max(dim=-1).values
            quality = roi_out["quality"] + proposal_scores.clamp(1e-4, 1 - 1e-4).logit()
            batch_logits.append(roi_out["logits"])
            batch_boxes.append(refined_boxes)
            batch_quality.append(quality)
            batch_scores.append(cls_scores * quality.sigmoid())
        boxes = torch.stack(batch_boxes, dim=0)
        logits = torch.stack(batch_logits, dim=0)
        quality = torch.stack(batch_quality, dim=0)
        scores = torch.stack(batch_scores, dim=0)
        return {"boxes": boxes, "logits": logits, "scores": scores, "bbox_quality": quality, "boxes_per_layer": [proposals["boxes"], boxes]}


def make_loader(args, split: str, ssl_pair: bool = False):
    max_images = args.max_images if split == "train" else args.val_max_images
    ds = VinDrDetectionDataset(args.data_root, args.image_size, split=split, val_fraction=args.val_fraction, seed=args.seed, max_images=max_images, normal_only=args.normal_only if ssl_pair else False, positive_only=args.positive_only if (split == "train" and not ssl_pair) else False, ssl_pair=ssl_pair, lung_crop=args.lung_crop, test_fraction=args.test_fraction, use_jpg_cache=args.use_jpg_cache, jpg_quality=args.jpg_quality)
    collate = (lambda batch: (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]))) if ssl_pair else detection_collate
    kwargs = {"batch_size": args.batch_size, "shuffle": split == "train", "num_workers": args.num_workers, "collate_fn": collate, "pin_memory": torch.cuda.is_available(), "drop_last": ssl_pair and split == "train"}
    if split == "train" and not ssl_pair and args.balanced_batches and not args.positive_only:
        kwargs.pop("batch_size")
        kwargs.pop("shuffle")
        kwargs.pop("drop_last")
        kwargs["batch_sampler"] = ClassBalancedDetectionBatchSampler(ds, args.batch_size, args.seed, args.positive_fraction)
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def save(args, model, name: str) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.output_dir / name)
    print(f"saved {args.output_dir / name}")


def train_pretrain(args, device):
    model = ExplicitBarlowPretrainer(1, args.base_channels, args.projector_hidden_dim, args.projector_dim, args.barlow_lambda).to(device)
    if args.weight_init != "default":
        apply_weight_initialization(model, args.weight_init)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    train_model = torch.compile(model) if args.compile_model and hasattr(torch, "compile") else model
    loader = make_loader(args, "train", ssl_pair=True)
    opt = build_tunable_optimizer(model, args)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    effective_steps_per_epoch = min(len(loader), args.steps_per_epoch) if args.steps_per_epoch else len(loader)
    total_steps = max(args.epochs * max(effective_steps_per_epoch, 1), 1)
    step = 0
    start_epoch = 0
    last_loss = float("inf")
    resume_path = resolve_resume_path(args, "barlow_pretrain_last.pt")
    if resume_path is not None and resume_path.exists():
        ckpt = load_training_checkpoint(resume_path, model, opt, scaler, device)
        start_epoch = int(ckpt.get("epoch", 0))
        step = int(ckpt.get("step", 0))
        last_loss = -float(ckpt.get("best_metric", -last_loss))
    model.train()
    for epoch in range(start_epoch, args.epochs):
        for epoch_step, (view1, view2) in enumerate(loader):
            set_warmup_cosine_lr(opt, step, total_steps, args.warmup_steps, args.min_lr_ratio)
            opt.zero_grad(set_to_none=True)
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            if args.channels_last and device.type == "cuda":
                view1 = view1.contiguous(memory_format=torch.channels_last)
                view2 = view2.contiguous(memory_format=torch.channels_last)
            with autocast_context(device, amp_enabled):
                out = train_model(view1, view2)
            last_loss = out["loss"].detach().item()
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            print(f"barlow pretrain epoch={epoch} step={step} loss={last_loss:.4f}")
            step += 1
            if args.checkpoint_every_steps and step % args.checkpoint_every_steps == 0:
                save_training_checkpoint(args, model, opt, scaler, epoch, step, -last_loss, "barlow_pretrain_last.pt")
            if args.max_steps and step >= args.max_steps:
                save(args, model, "barlow_pretrain_smoke.pt"); return -last_loss
            if args.steps_per_epoch and epoch_step + 1 >= args.steps_per_epoch:
                break
        save_training_checkpoint(args, model, opt, scaler, epoch + 1, step, -last_loss, "barlow_pretrain_last.pt")
        if args.save_every_epoch:
            save_training_checkpoint(args, model, opt, scaler, epoch + 1, step, -last_loss, f"barlow_pretrain_epoch_{epoch + 1:03d}.pt")
    save(args, model, "barlow_pretrain_final.pt")
    return -last_loss


def train_detect(args, device):
    cfg = HybridCXRConfig(num_classes=DETECT_NUM_CLASSES, image_channels=1, base_channels=args.base_channels, hidden_dim=args.hidden_dim, topk_per_head=args.topk_per_head, cls_loss_weight=args.cls_loss_weight, box_l1_weight=args.box_l1_weight, giou_weight=args.giou_weight, quality_weight=args.quality_weight)
    model = ExplicitBarlowCXRDetector(cfg).to(device)
    model.load_barlow_encoder(args.pretrained_barlow, device)
    if args.weight_init != "default" and not args.pretrained_barlow:
        apply_weight_initialization(model, args.weight_init)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    train_model = torch.compile(model) if args.compile_model and hasattr(torch, "compile") else model
    loss_fn = HybridDetectionLoss(cfg).to(device)
    train_loader = make_loader(args, "train")
    val_loader = make_loader(args, "val")
    opt = build_tunable_optimizer(model, args)
    effective_steps_per_epoch = min(len(train_loader), args.steps_per_epoch) if args.steps_per_epoch else len(train_loader)
    total_steps = max(args.epochs * max(effective_steps_per_epoch, 1), 1)
    ema = None if args.no_ema else ModelEMA(model, args.ema_decay)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)
    step = 0
    start_epoch = 0
    best_map = -1.0
    resume_path = resolve_resume_path(args, "barlow_detector_last.pt")
    if resume_path is not None and resume_path.exists():
        ckpt = load_training_checkpoint(resume_path, model, opt, scaler, device, ema)
        start_epoch = int(ckpt.get("epoch", 0))
        step = int(ckpt.get("step", 0))
        best_map = float(ckpt.get("best_metric", best_map))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for epoch_step, (images, targets) in enumerate(train_loader):
            set_warmup_cosine_lr(opt, step, total_steps, args.warmup_steps, args.min_lr_ratio)
            opt.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            if args.channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            moved_targets = move_targets(targets, device)
            with autocast_context(device, amp_enabled):
                losses = loss_fn(train_model(images), moved_targets)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            box_count = sum(int(target["boxes"].shape[0]) for target in moved_targets)
            print(f"barlow detect epoch={epoch} step={step} boxes={box_count} loss={losses['loss'].detach().item():.8f}")
            step += 1
            if args.checkpoint_every_steps and step % args.checkpoint_every_steps == 0:
                save_training_checkpoint(args, model, opt, scaler, epoch, step, best_map, "barlow_detector_last.pt", ema)
            if args.max_steps and step >= args.max_steps:
                save(args, ema.ema if ema else model, "barlow_detector_smoke.pt")
                eval_model = ema.ema if ema else model
                return evaluate_map50(eval_model, val_loader, device, args.map_steps)
            if args.steps_per_epoch and epoch_step + 1 >= args.steps_per_epoch:
                break
        eval_model = ema.ema if ema else model
        map50 = evaluate_map50(eval_model, val_loader, device, args.map_steps)
        print(f"val epoch={epoch} mAP50={map50:.4f}")
        if map50 > best_map:
            best_map = map50; save(args, eval_model, "barlow_detector_best.pt")
        save_training_checkpoint(args, model, opt, scaler, epoch + 1, step, best_map, "barlow_detector_last.pt", ema)
        if args.save_every_epoch:
            save_training_checkpoint(args, model, opt, scaler, epoch + 1, step, best_map, f"barlow_detector_epoch_{epoch + 1:03d}.pt", ema)
    save(args, ema.ema if ema else model, "barlow_detector_final.pt")
    if args.visualize:
        visualize(ema.ema if ema else model, args, device)
    return best_map


@torch.no_grad()
def visualize(model, args, device):
    save_detection_visualizations(model, args, device, "barlow_cxr", (60, 220, 255))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["pretrain", "detect"], default="detect")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-dir", type=Path, default=Path("runs/barlow_twins_cxr"))
    p.add_argument("--pretrained-barlow", type=str, default="")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--val-max-images", type=int, default=128)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--normal-only", action="store_true")
    p.add_argument("--positive-only", action="store_true")
    p.add_argument("--balanced-batches", action="store_true", default=True)
    p.add_argument("--no-balanced-batches", dest="balanced_batches", action="store_false")
    p.add_argument("--positive-fraction", type=float, default=0.5)
    p.add_argument("--lung-crop", action="store_true", default=True)
    p.add_argument("--no-lung-crop", dest="lung_crop", action="store_false")
    p.add_argument("--use-jpg-cache", action="store_true", default=True)
    p.add_argument("--no-jpg-cache", dest="use_jpg_cache", action="store_false")
    p.add_argument("--jpg-quality", type=int, default=95)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--gpu-preset", choices=["auto", "low", "medium", "high", "none"], default="auto")
    p.add_argument("--resume", type=str, default="", help="Resume from an explicit training checkpoint path.")
    p.add_argument("--auto-resume", action="store_true", default=True, help="Resume from the latest checkpoint in output-dir if it exists.")
    p.add_argument("--no-auto-resume", dest="auto_resume", action="store_false", help="Start from scratch even if a last checkpoint exists.")
    p.add_argument("--save-every-epoch", action="store_true", help="Keep a numbered checkpoint for every finished epoch.")
    p.add_argument("--checkpoint-every-steps", type=int, default=100, help="Save last checkpoint every N optimizer steps; 0 disables step checkpoints.")
    p.add_argument("--steps-per-epoch", type=int, default=150, help="Limit optimizer steps per epoch; use 0 for full dataset.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--backbone-lr-mult", type=float, default=0.2)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--ema-decay", type=float, default=0.9998)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--topk-per-head", type=int, default=300)
    p.add_argument("--projector-dim", type=int, default=8192)
    p.add_argument("--projector-hidden-dim", type=int, default=2048)
    p.add_argument("--barlow-lambda", type=float, default=5e-3)
    p.add_argument("--weight-init", choices=["default", "kaiming", "xavier", "orthogonal"], default="default")
    p.add_argument("--cls-loss-weight", type=float, default=1.0)
    p.add_argument("--box-l1-weight", type=float, default=5.0)
    p.add_argument("--giou-weight", type=float, default=2.0)
    p.add_argument("--quality-weight", type=float, default=0.5)
    p.add_argument("--map-steps", type=int, default=50)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--vis-images", type=int, default=8)
    p.add_argument("--score-thresh", type=float, default=0.15)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--random-search", action="store_true")
    p.add_argument("--search-trials", type=int, default=12)
    p.add_argument("--search-max-steps", type=int, default=80)
    p.add_argument("--search-seed", type=int, default=123)
    p.add_argument("--quick-test", action="store_true", help="Run a short 5-epoch test with 64 train steps per epoch.")
    p.add_argument("--quick-epochs", type=int, default=5)
    p.add_argument("--quick-steps", type=int, default=64)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    args.data_root = resolve_data_root(str(args.data_root))
    args.max_steps = 0
    apply_gpu_training_preset(args, "barlow")
    if args.quick_test:
        args.epochs = min(max(args.quick_epochs, 5), 10)
        args.steps_per_epoch = min(max(args.quick_steps, 60), 70)
        args.val_max_images = min(args.val_max_images, 128)
        args.map_steps = min(args.map_steps, 10)
        args.output_dir = args.output_dir / "quick_test"
    if args.smoke:
        args.epochs = 1; args.image_size = 128; args.batch_size = 2; args.max_images = 4; args.val_max_images = 4
        args.base_channels = 8; args.hidden_dim = 32; args.topk_per_head = 8; args.projector_dim = 256; args.projector_hidden_dim = 128
        args.max_steps = 1; args.steps_per_epoch = 1; args.warmup_steps = 1; args.positive_only = True
        if args.phase == "pretrain": args.normal_only = True
    return args


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def write_search_report(rows: List[Dict], out_dir: Path, title: str, metric_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with (out_dir / "random_search_results.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    best = max(rows, key=lambda r: float(r["score"]))
    (out_dir / "best_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    lines = [f"# {title}", "", f"Best {metric_name}: {best['score']:.6f}", "", "| trial | score | phase | lr | opt | batch | base | hidden | wd | init | lambda |", "|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|"]
    for row in sorted(rows, key=lambda r: float(r["score"]), reverse=True):
        lines.append(f"| {row['trial']} | {row['score']:.6f} | {row['phase']} | {row['lr']:.2e} | {row['optimizer']} | {row['batch_size']} | {row['base_channels']} | {row['hidden_dim']} | {row['weight_decay']:.2e} | {row['weight_init']} | {row['barlow_lambda']:.3g} |")
    (out_dir / "random_search_report.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot([r["trial"] for r in rows], [r["score"] for r in rows], marker="o")
        plt.xlabel("trial")
        plt.ylabel(metric_name)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / "random_search_scores.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"plot skipped: {exc}")


def random_search(args) -> None:
    rng = random.Random(args.search_seed)
    rows = []
    root = args.output_dir / f"random_search_{args.phase}"
    device = resolve_training_device(args.device, args.require_cuda)
    for trial in range(args.search_trials):
        trial_args = copy.deepcopy(args)
        trial_args.random_search = False
        trial_args.output_dir = root / f"trial_{trial:03d}"
        trial_args.epochs = max(1, args.epochs)
        trial_args.max_steps = args.search_max_steps
        trial_args.map_steps = min(args.map_steps, 10)
        trial_args.visualize = False
        trial_args.lr = log_uniform(rng, 3e-5, 8e-4)
        trial_args.weight_decay = log_uniform(rng, 1e-6, 5e-4)
        trial_args.optimizer = rng.choice(["adamw", "adam", "sgd"])
        trial_args.batch_size = rng.choice([2, 4, 6])
        trial_args.base_channels = rng.choice([32, 48, 64])
        trial_args.hidden_dim = rng.choice([96, 128, 192])
        trial_args.topk_per_head = rng.choice([200, 300, 450])
        trial_args.projector_hidden_dim = rng.choice([1024, 2048, 3072])
        trial_args.projector_dim = rng.choice([4096, 8192])
        trial_args.barlow_lambda = rng.choice([0.003, 0.005, 0.008, 0.01])
        trial_args.weight_init = rng.choice(["default", "kaiming", "xavier"])
        trial_args.cls_loss_weight = rng.choice([0.75, 1.0, 1.25])
        trial_args.box_l1_weight = rng.choice([4.0, 5.0, 6.0])
        trial_args.giou_weight = rng.choice([1.5, 2.0, 2.5])
        trial_args.quality_weight = rng.choice([0.25, 0.5, 0.75])
        print(f"random-search trial={trial} phase={trial_args.phase} optimizer={trial_args.optimizer} lr={trial_args.lr:.2e}")
        score = train_pretrain(trial_args, device) if trial_args.phase == "pretrain" else train_detect(trial_args, device)
        fields = ["phase", "lr", "weight_decay", "optimizer", "batch_size", "base_channels", "hidden_dim", "topk_per_head", "projector_hidden_dim", "projector_dim", "barlow_lambda", "weight_init", "cls_loss_weight", "box_l1_weight", "giou_weight", "quality_weight"]
        rows.append({k: getattr(trial_args, k) for k in fields} | {"trial": trial, "score": float(score or 0.0)})
        write_search_report(rows, root, "Barlow Twins Random Search", "negative_loss" if args.phase == "pretrain" else "mAP50")


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_training_device(args.device, args.require_cuda)
    print(f"device={device} phase={args.phase} data={args.data_root}")
    if args.random_search:
        random_search(args)
    elif args.phase == "pretrain":
        train_pretrain(args, device)
    else:
        train_detect(args, device)


if __name__ == "__main__":
    main()
