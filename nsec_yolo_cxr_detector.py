"""
Explicit NSEC-YOLO style CXR lesion detector for VinDr-CXR.

Scientific basis:
- Zhang et al., NSEC-YOLO: Real-time lesion detection on chest X-ray with
  adaptive noise suppression and global perception aggregation, Journal of
  Radiation Research and Applied Sciences, 2025.

This file intentionally writes the model architecture out line-by-line instead
of hiding it behind imported model classes. Shared imports are only for dataset,
training loop utilities, and the generic detection loss/evaluator.
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

from hybrid_cxr_detector import HybridCXRConfig, HybridDetectionLoss, box_iou, generalized_box_iou
from train_vindr_hybrid import (
    DEFAULT_DATA_ROOT,
    DETECT_NUM_CLASSES,
    CLASS_NAMES,
    VinDrDetectionDataset,
    detection_collate,
    evaluate_map50,
    move_targets,
    save_checkpoint,
    seed_everything,
    build_optimizer,
    set_warmup_cosine_lr,
    ModelEMA,
    save_detection_visualizations,
)


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
        elif isinstance(module, nn.BatchNorm2d):
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


def box_xywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def sanitize_boxes(boxes: Tensor, eps: float = 1e-4) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    x1 = x1.clamp(0, 1)
    y1 = y1.clamp(0, 1)
    x2 = torch.maximum(x2.clamp(0, 1), x1 + eps).clamp(0, 1)
    y2 = torch.maximum(y2.clamp(0, 1), y1 + eps).clamp(0, 1)
    return torch.stack((x1, y1, x2, y2), dim=-1)


class ConvBNAct(nn.Module):
    """One visible conv block: Conv2d -> BatchNorm -> SiLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResidualRefConv(nn.Module):
    """YOLO-CXR/RefConv-like local feature refinement, written explicitly."""

    def __init__(self, channels: int):
        super().__init__()
        self.local_3x3 = ConvBNAct(channels, channels, kernel=3)
        self.refocus_1x1 = ConvBNAct(channels, channels, kernel=1)
        self.merge = ConvBNAct(channels * 2, channels, kernel=1)

    def forward(self, x: Tensor) -> Tensor:
        local = self.local_3x3(x)
        refocus = self.refocus_1x1(x)
        merged = self.merge(torch.cat([local, refocus], dim=1))
        return x + merged


class ExplicitCXRBackbone(nn.Module):
    """CXR backbone. Returns P2/P3/P4/P5 so small lesions keep high-res P2."""

    def __init__(self, image_channels: int, base_channels: int):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(ConvBNAct(image_channels, c, 7, 2), ConvBNAct(c, c, 3, 1))
        self.down2 = ConvBNAct(c, c * 2, 3, 2)
        self.refine2 = nn.Sequential(ResidualRefConv(c * 2), ResidualRefConv(c * 2))
        self.down3 = ConvBNAct(c * 2, c * 4, 3, 2)
        self.refine3 = nn.Sequential(ResidualRefConv(c * 4), ResidualRefConv(c * 4))
        self.down4 = ConvBNAct(c * 4, c * 8, 3, 2)
        self.refine4 = nn.Sequential(ResidualRefConv(c * 8), ResidualRefConv(c * 8))
        self.down5 = ConvBNAct(c * 8, c * 8, 3, 2)
        self.refine5 = nn.Sequential(ResidualRefConv(c * 8), ResidualRefConv(c * 8))
        self.out_channels = {"p2": c * 2, "p3": c * 4, "p4": c * 8, "p5": c * 8}

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        x = self.stem(image)
        p2 = self.refine2(self.down2(x))
        p3 = self.refine3(self.down3(p2))
        p4 = self.refine4(self.down4(p3))
        p5 = self.refine5(self.down5(p4))
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


class ExplicitFPN(nn.Module):
    """YOLO-style FPN+PAN neck with visible top-down and bottom-up fusion."""

    def __init__(self, in_channels: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.p2_lateral = nn.Conv2d(in_channels["p2"], hidden_dim, 1)
        self.p3_lateral = nn.Conv2d(in_channels["p3"], hidden_dim, 1)
        self.p4_lateral = nn.Conv2d(in_channels["p4"], hidden_dim, 1)
        self.p5_lateral = nn.Conv2d(in_channels["p5"], hidden_dim, 1)
        self.p2_out = ResidualRefConv(hidden_dim)
        self.p3_out = ResidualRefConv(hidden_dim)
        self.p4_out = ResidualRefConv(hidden_dim)
        self.p5_out = ResidualRefConv(hidden_dim)
        self.p2_down = ConvBNAct(hidden_dim, hidden_dim, 3, 2)
        self.p3_down = ConvBNAct(hidden_dim, hidden_dim, 3, 2)
        self.p4_down = ConvBNAct(hidden_dim, hidden_dim, 3, 2)
        self.pan3 = ResidualRefConv(hidden_dim)
        self.pan4 = ResidualRefConv(hidden_dim)
        self.pan5 = ResidualRefConv(hidden_dim)

    def forward(self, feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        p5 = self.p5_lateral(feats["p5"])
        p4 = self.p4_lateral(feats["p4"]) + F.interpolate(p5, size=feats["p4"].shape[-2:], mode="nearest")
        p3 = self.p3_lateral(feats["p3"]) + F.interpolate(p4, size=feats["p3"].shape[-2:], mode="nearest")
        p2 = self.p2_lateral(feats["p2"]) + F.interpolate(p3, size=feats["p2"].shape[-2:], mode="nearest")
        p2 = self.p2_out(p2)
        p3 = self.p3_out(p3)
        p4 = self.p4_out(p4)
        p5 = self.p5_out(p5)
        p3 = self.pan3(p3 + self.p2_down(p2))
        p4 = self.pan4(p4 + self.p3_down(p3))
        p5 = self.pan5(p5 + self.p4_down(p4))
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


class AdaptiveNoiseSuppression(nn.Module):
    """NSEC idea: separate smooth anatomy from residual noise, then learn a gate."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise_smooth = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.smooth_bn = nn.BatchNorm2d(channels)
        self.gate_conv1 = nn.Conv2d(channels * 3, channels, 1)
        self.gate_conv2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        smooth = self.smooth_bn(self.depthwise_smooth(x))
        residual = x - smooth
        local_variance = F.avg_pool2d(residual.pow(2), kernel_size=5, stride=1, padding=2)
        gate_input = torch.cat([x, smooth, local_variance], dim=1)
        gate = F.silu(self.gate_conv1(gate_input))
        gate = torch.sigmoid(self.gate_conv2(gate))
        return gate * x + (1.0 - gate) * smooth


class GlobalPerceptionAggregation(nn.Module):
    """NSEC idea: global CXR context modulates local features before detection."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels, 1)
        self.fc2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        context = self.pool(x)
        context = F.silu(self.fc1(context))
        context = torch.sigmoid(self.fc2(context))
        return x * (1.0 + context)


class DenseDetectionHead(nn.Module):
    """YOLO-style dense head. Predicts class logits, objectness, and boxes."""

    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.shared = ResidualRefConv(channels)
        self.cls = nn.Conv2d(channels, num_classes, 1)
        self.obj = nn.Conv2d(channels, 1, 1)
        self.box = nn.Conv2d(channels, 4, 1)

    def decode_boxes(self, raw_box: Tensor) -> Tensor:
        _, _, h, w = raw_box.shape
        y, x = torch.meshgrid(torch.arange(h, device=raw_box.device, dtype=raw_box.dtype), torch.arange(w, device=raw_box.device, dtype=raw_box.dtype), indexing="ij")
        grid = torch.stack([x, y], dim=0).unsqueeze(0)
        scale = torch.tensor([w, h], device=raw_box.device, dtype=raw_box.dtype).view(1, 2, 1, 1)
        center_xy = (raw_box[:, 0:2].sigmoid() + grid) / scale
        size_wh = (raw_box[:, 2:4].sigmoid().pow(2) * 2.0) / scale
        boxes = torch.cat([center_xy, size_wh], dim=1).permute(0, 2, 3, 1)
        boxes = box_xywh_to_xyxy(boxes).flatten(1, 2)
        return sanitize_boxes(boxes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.shared(x)
        return {
            "logits": self.cls(x).flatten(2).transpose(1, 2),
            "objectness": self.obj(x).flatten(2).transpose(1, 2).squeeze(-1),
            "boxes": self.decode_boxes(self.box(x)),
        }


def accurate_eiou_loss(pred: Tensor, target: Tensor) -> Tensor:
    """AccurEIOU-style box loss: IoU, center distance, side length, and aspect penalties."""
    iou = box_iou(pred, target)[0].diag().clamp(0.0, 1.0)
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    pw, ph = (px2 - px1).clamp_min(1e-6), (py2 - py1).clamp_min(1e-6)
    tw, th = (tx2 - tx1).clamp_min(1e-6), (ty2 - ty1).clamp_min(1e-6)
    pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
    tcx, tcy = (tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5
    enc_x1, enc_y1 = torch.minimum(px1, tx1), torch.minimum(py1, ty1)
    enc_x2, enc_y2 = torch.maximum(px2, tx2), torch.maximum(py2, ty2)
    cw, ch = (enc_x2 - enc_x1).clamp_min(1e-6), (enc_y2 - enc_y1).clamp_min(1e-6)
    center_penalty = ((pcx - tcx).pow(2) + (pcy - tcy).pow(2)) / (cw.pow(2) + ch.pow(2)).clamp_min(1e-6)
    width_penalty = (pw - tw).pow(2) / cw.pow(2).clamp_min(1e-6)
    height_penalty = (ph - th).pow(2) / ch.pow(2).clamp_min(1e-6)
    aspect_penalty = (torch.atan(tw / th) - torch.atan(pw / ph)).pow(2) * (4.0 / (np.pi ** 2))
    return (1.0 - iou + center_penalty + width_penalty + height_penalty + aspect_penalty).mean()


class NSECYOLOLoss(HybridDetectionLoss):
    """NSEC-YOLO training criterion with focal classification and AccurEIOU localization."""

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
            loss_accureiou = accurate_eiou_loss(pred, tgt)
            loss_quality = F.binary_cross_entropy_with_logits(torch.cat(q_pred, dim=0), torch.cat(q_tgt, dim=0))
        else:
            zero = logits.sum() * 0.0
            loss_l1 = zero; loss_accureiou = zero; loss_quality = zero
        total = self.cfg.cls_loss_weight * loss_cls + self.cfg.box_l1_weight * loss_l1 + self.cfg.giou_weight * loss_accureiou + self.cfg.quality_weight * loss_quality
        return {"loss": total, "loss_cls": loss_cls.detach(), "loss_l1": loss_l1.detach(), "loss_giou": loss_accureiou.detach(), "loss_quality": loss_quality.detach(), "loss_dfine_distill": logits.sum().detach() * 0.0}


class ExplicitNSECYOLO(nn.Module):
    """Full model: Backbone -> FPN -> NSEC modules -> dense detection heads."""

    def __init__(self, cfg: HybridCXRConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = ExplicitCXRBackbone(cfg.image_channels, cfg.base_channels)
        self.fpn = ExplicitFPN(self.backbone.out_channels, cfg.hidden_dim)
        self.noise_p2 = AdaptiveNoiseSuppression(cfg.hidden_dim)
        self.noise_p3 = AdaptiveNoiseSuppression(cfg.hidden_dim)
        self.noise_p4 = AdaptiveNoiseSuppression(cfg.hidden_dim)
        self.noise_p5 = AdaptiveNoiseSuppression(cfg.hidden_dim)
        self.context_p2 = GlobalPerceptionAggregation(cfg.hidden_dim)
        self.context_p3 = GlobalPerceptionAggregation(cfg.hidden_dim)
        self.context_p4 = GlobalPerceptionAggregation(cfg.hidden_dim)
        self.context_p5 = GlobalPerceptionAggregation(cfg.hidden_dim)
        self.head_p2 = DenseDetectionHead(cfg.hidden_dim, cfg.num_classes)
        self.head_p3 = DenseDetectionHead(cfg.hidden_dim, cfg.num_classes)
        self.head_p4 = DenseDetectionHead(cfg.hidden_dim, cfg.num_classes)
        self.head_p5 = DenseDetectionHead(cfg.hidden_dim, cfg.num_classes)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        prior = 0.01
        prior_bias = float(np.log(prior / (1.0 - prior)))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for head in [self.head_p2, self.head_p3, self.head_p4, self.head_p5]:
            nn.init.constant_(head.cls.bias, prior_bias)
            nn.init.constant_(head.obj.bias, prior_bias)
            nn.init.zeros_(head.box.bias)

    def collect_topk(self, outs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        logits = torch.cat([o["logits"] for o in outs], dim=1)
        boxes = torch.cat([o["boxes"] for o in outs], dim=1)
        obj = torch.cat([o["objectness"] for o in outs], dim=1).sigmoid()
        scores = (logits.sigmoid() * obj.unsqueeze(-1)).max(dim=-1).values
        if self.training:
            return {"boxes": boxes, "logits": logits, "scores": scores}
        center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
        lung_prior = -((center[..., 0] - 0.5).pow(2) / 0.25 + (center[..., 1] - 0.48).pow(2) / 0.35)
        scores = scores + lung_prior * 1e-6
        k = min(self.cfg.topk_per_head, scores.shape[1])
        idx = scores.topk(k, dim=1).indices
        return {
            "boxes": boxes.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4)),
            "logits": logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, logits.shape[-1])),
            "scores": scores.gather(1, idx),
        }

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        raw_feats = self.backbone(image)
        feats = self.fpn(raw_feats)
        p2 = self.context_p2(self.noise_p2(feats["p2"]))
        p3 = self.context_p3(self.noise_p3(feats["p3"]))
        p4 = self.context_p4(self.noise_p4(feats["p4"]))
        p5 = self.context_p5(self.noise_p5(feats["p5"]))
        out_p2 = self.head_p2(p2)
        out_p3 = self.head_p3(p3)
        out_p4 = self.head_p4(p4)
        out_p5 = self.head_p5(p5)
        cand = self.collect_topk([out_p2, out_p3, out_p4, out_p5])
        quality_logit = cand["scores"].clamp(1e-4, 1 - 1e-4).logit()
        return {"boxes": cand["boxes"], "logits": cand["logits"], "scores": cand["scores"], "bbox_quality": quality_logit, "boxes_per_layer": [cand["boxes"]]}


def make_loader(args, split: str) -> DataLoader:
    max_images = args.max_images if split == "train" else args.val_max_images
    ds = VinDrDetectionDataset(args.data_root, args.image_size, split=split, val_fraction=args.val_fraction, seed=args.seed, max_images=max_images, positive_only=args.positive_only if split == "train" else False, lung_crop=args.lung_crop, test_fraction=args.test_fraction, use_jpg_cache=args.use_jpg_cache, jpg_quality=args.jpg_quality)
    kwargs = {"batch_size": args.batch_size, "shuffle": split == "train", "num_workers": args.num_workers, "collate_fn": detection_collate, "pin_memory": torch.cuda.is_available()}
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


@torch.no_grad()
def visualize(model: nn.Module, args, device: torch.device) -> None:
    save_detection_visualizations(model, args, device, "nsec_yolo", (0, 255, 80))


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HybridCXRConfig(num_classes=DETECT_NUM_CLASSES, image_channels=1, base_channels=args.base_channels, hidden_dim=args.hidden_dim, topk_per_head=args.topk_per_head, cls_loss_weight=args.cls_loss_weight, box_l1_weight=args.box_l1_weight, giou_weight=args.giou_weight, quality_weight=args.quality_weight)
    model = ExplicitNSECYOLO(cfg).to(device)
    if args.weight_init != "default":
        apply_weight_initialization(model, args.weight_init)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    train_model = torch.compile(model) if args.compile_model and hasattr(torch, "compile") else model
    loss_fn = NSECYOLOLoss(cfg).to(device)
    train_loader = make_loader(args, "train")
    val_loader = make_loader(args, "val")
    opt = build_tunable_optimizer(model, args)
    total_steps = max(args.epochs * max(len(train_loader), 1), 1)
    ema = None if args.no_ema else ModelEMA(model, args.ema_decay)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)
    step = 0
    best_map = -1.0
    for epoch in range(args.epochs):
        model.train()
        for images, targets in train_loader:
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
            if step % args.log_every == 0:
                print(f"nsec epoch={epoch} step={step} loss={losses['loss'].detach().item():.4f}")
            step += 1
            if args.max_steps and step >= args.max_steps:
                save_checkpoint(args, ema.ema if ema else model, "nsec_yolo_smoke.pt")
                eval_model = ema.ema if ema else model
                return evaluate_map50(eval_model, val_loader, device, args.map_steps)
        eval_model = ema.ema if ema else model
        map50 = evaluate_map50(eval_model, val_loader, device, args.map_steps)
        print(f"val epoch={epoch} mAP50={map50:.4f}")
        if map50 > best_map:
            best_map = map50
            save_checkpoint(args, eval_model, "nsec_yolo_best.pt")
    save_checkpoint(args, ema.ema if ema else model, "nsec_yolo_final.pt")
    if args.visualize:
        visualize(ema.ema if ema else model, args, device)
    return best_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-dir", type=Path, default=Path("runs/nsec_yolo"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--image-size", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--val-max-images", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--positive-only", action="store_true")
    p.add_argument("--lung-crop", action="store_true", default=True)
    p.add_argument("--no-lung-crop", dest="lung_crop", action="store_false")
    p.add_argument("--use-jpg-cache", action="store_true", default=True)
    p.add_argument("--no-jpg-cache", dest="use_jpg_cache", action="store_false")
    p.add_argument("--jpg-quality", type=int, default=95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
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
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--topk-per-head", type=int, default=300)
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
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    args.max_steps = 0
    if args.smoke:
        args.epochs = 1; args.image_size = 128; args.batch_size = 2; args.max_images = 4; args.val_max_images = 4
        args.base_channels = 8; args.hidden_dim = 32; args.topk_per_head = 8; args.max_steps = 1; args.warmup_steps = 1; args.log_every = 1; args.positive_only = True
    return args


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def write_search_report(rows: List[Dict], out_dir: Path, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    csv_path = out_dir / "random_search_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    best = max(rows, key=lambda r: float(r["score"]))
    (out_dir / "best_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    lines = [f"# {title}", "", f"Best score: {best['score']:.6f}", "", "| trial | score | lr | opt | batch | base | hidden | topk | wd | init |", "|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for row in sorted(rows, key=lambda r: float(r["score"]), reverse=True):
        lines.append(f"| {row['trial']} | {row['score']:.6f} | {row['lr']:.2e} | {row['optimizer']} | {row['batch_size']} | {row['base_channels']} | {row['hidden_dim']} | {row['topk_per_head']} | {row['weight_decay']:.2e} | {row['weight_init']} |")
    (out_dir / "random_search_report.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["trial"] for r in rows]
        ys = [r["score"] for r in rows]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("trial")
        plt.ylabel("mAP50")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / "random_search_scores.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"plot skipped: {exc}")


def random_search(args) -> None:
    rng = random.Random(args.search_seed)
    rows = []
    root = args.output_dir / "random_search"
    for trial in range(args.search_trials):
        trial_args = copy.deepcopy(args)
        trial_args.random_search = False
        trial_args.output_dir = root / f"trial_{trial:03d}"
        trial_args.epochs = max(1, args.epochs)
        trial_args.max_steps = args.search_max_steps
        trial_args.map_steps = min(args.map_steps, 10)
        trial_args.visualize = False
        trial_args.lr = log_uniform(rng, 5e-5, 8e-4)
        trial_args.weight_decay = log_uniform(rng, 1e-6, 5e-4)
        trial_args.optimizer = rng.choice(["adamw", "adam", "sgd"])
        trial_args.batch_size = rng.choice([2, 4, 6])
        trial_args.base_channels = rng.choice([24, 32, 48])
        trial_args.hidden_dim = rng.choice([96, 128, 192])
        trial_args.topk_per_head = rng.choice([200, 300, 450])
        trial_args.weight_init = rng.choice(["default", "kaiming", "xavier"])
        trial_args.cls_loss_weight = rng.choice([0.75, 1.0, 1.25])
        trial_args.box_l1_weight = rng.choice([4.0, 5.0, 6.0])
        trial_args.giou_weight = rng.choice([1.5, 2.0, 2.5])
        trial_args.quality_weight = rng.choice([0.25, 0.5, 0.75])
        print(f"random-search trial={trial} optimizer={trial_args.optimizer} lr={trial_args.lr:.2e}")
        score = float(train(trial_args) or 0.0)
        rows.append({k: getattr(trial_args, k) for k in ["lr", "weight_decay", "optimizer", "batch_size", "base_channels", "hidden_dim", "topk_per_head", "weight_init", "cls_loss_weight", "box_l1_weight", "giou_weight", "quality_weight"]} | {"trial": trial, "score": score})
        write_search_report(rows, root, "NSEC-YOLO Random Search")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.random_search:
        random_search(parsed)
    else:
        train(parsed)
