"""
Explicit DETR/RT-DETR style CXR lesion detector for VinDr-CXR.

Scientific basis:
- Carion et al., End-to-End Object Detection with Transformers, ECCV 2020.
- Zhao et al., DETRs Beat YOLOs on Real-time Object Detection, CVPR 2024.

The complete detector is written in this file: CNN backbone, FPN, positional
encoding, transformer encoder/decoder, query selection, prediction heads,
training, evaluation, and visualization.
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

from hybrid_cxr_detector import HybridCXRConfig, HybridDetectionLoss
from train_vindr_hybrid import DEFAULT_DATA_ROOT, DETECT_NUM_CLASSES, CLASS_NAMES, VinDrDetectionDataset, detection_collate, evaluate_map50, move_targets, save_checkpoint, seed_everything, build_optimizer, set_warmup_cosine_lr, ModelEMA, save_detection_visualizations


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
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
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


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.norm2(self.conv2(self.conv1(x)))
        return F.silu(x + residual)


class ExplicitDETRBackbone(nn.Module):
    """CNN feature extractor. P3/P4/P5 become transformer input sequences."""

    def __init__(self, image_channels: int, base_channels: int):
        super().__init__()
        c = base_channels
        self.stem = ConvNormAct(image_channels, c, stride=2)
        self.stage2 = nn.Sequential(ConvNormAct(c, c * 2, stride=2), ResidualBlock(c * 2))
        self.stage3 = nn.Sequential(ConvNormAct(c * 2, c * 4, stride=2), ResidualBlock(c * 4), ResidualBlock(c * 4))
        self.stage4 = nn.Sequential(ConvNormAct(c * 4, c * 8, stride=2), ResidualBlock(c * 8), ResidualBlock(c * 8))
        self.stage5 = nn.Sequential(ConvNormAct(c * 8, c * 8, stride=2), ResidualBlock(c * 8))
        self.out_channels = {"p3": c * 4, "p4": c * 8, "p5": c * 8}

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        x = self.stem(image)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return {"p3": p3, "p4": p4, "p5": p5}


class ExplicitHybridEncoder(nn.Module):
    """RT-DETR-inspired: project each scale, then fuse high-level context."""

    def __init__(self, in_channels: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.p3_proj = nn.Conv2d(in_channels["p3"], hidden_dim, 1)
        self.p4_proj = nn.Conv2d(in_channels["p4"], hidden_dim, 1)
        self.p5_proj = nn.Conv2d(in_channels["p5"], hidden_dim, 1)
        self.p3_fuse = ConvNormAct(hidden_dim, hidden_dim)
        self.p4_fuse = ConvNormAct(hidden_dim, hidden_dim)
        self.p5_fuse = ConvNormAct(hidden_dim, hidden_dim)

    def forward(self, feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        p5 = self.p5_proj(feats["p5"])
        p4 = self.p4_proj(feats["p4"]) + F.interpolate(p5, size=feats["p4"].shape[-2:], mode="nearest")
        p3 = self.p3_proj(feats["p3"]) + F.interpolate(p4, size=feats["p3"].shape[-2:], mode="nearest")
        return {"p3": self.p3_fuse(p3), "p4": self.p4_fuse(p4), "p5": self.p5_fuse(p5)}


class SinePositionEncoding(nn.Module):
    """Adds explicit spatial position because transformer attention is permutation-invariant."""

    def __init__(self, hidden_dim: int, temperature: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, height: int, width: int, device, dtype) -> Tensor:
        y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype), torch.arange(width, device=device, dtype=dtype), indexing="ij")
        y = y / max(height - 1, 1)
        x = x / max(width - 1, 1)
        base = max(self.hidden_dim // 4, 1)
        dim_t = torch.arange(base, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / base)
        pos_x = x[..., None] / dim_t
        pos_y = y[..., None] / dim_t
        pos = torch.cat([pos_x.sin(), pos_x.cos(), pos_y.sin(), pos_y.cos()], dim=-1)[..., : self.hidden_dim]
        if pos.shape[-1] < self.hidden_dim:
            pos = F.pad(pos, (0, self.hidden_dim - pos.shape[-1]))
        return pos.flatten(0, 1)


class ExplicitDETR(nn.Module):
    """Full NMS-free detector with visible query selection and prediction heads."""

    def __init__(self, cfg: HybridCXRConfig):
        super().__init__()
        self.cfg = cfg
        self.token_pool_sizes = {"p3": 32, "p4": 16, "p5": 8}
        self.backbone = ExplicitDETRBackbone(cfg.image_channels, cfg.base_channels)
        self.hybrid_encoder = ExplicitHybridEncoder(self.backbone.out_channels, cfg.hidden_dim)
        self.position = SinePositionEncoding(cfg.hidden_dim)
        self.level_embedding = nn.Parameter(torch.randn(3, cfg.hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_dim, nhead=8, dim_feedforward=cfg.hidden_dim * 4, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)

        self.decoder_layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=cfg.hidden_dim, nhead=8, dim_feedforward=cfg.hidden_dim * 4, dropout=0.1, batch_first=True) for _ in range(cfg.num_decoder_layers)]
        )
        self.object_queries = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        self.class_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.box_hidden = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.box_output = nn.Linear(cfg.hidden_dim, 4)
        self.localization_quality = nn.Linear(cfg.hidden_dim, 1)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        prior = 0.01
        prior_bias = float(np.log(prior / (1.0 - prior)))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.class_head.bias, prior_bias)
        nn.init.constant_(self.localization_quality.bias, prior_bias)
        nn.init.zeros_(self.box_output.weight)
        with torch.no_grad():
            self.box_output.bias.copy_(torch.tensor([0.0, 0.0, -2.0, -2.0]))

    def predict_from_queries(self, queries: Tensor) -> Dict[str, Tensor]:
        logits = self.class_head(queries)
        raw_boxes = self.box_output(F.relu(self.box_hidden(queries)))
        boxes = xywh_to_xyxy(raw_boxes.sigmoid())
        quality = self.localization_quality(queries).squeeze(-1)
        scores = logits.sigmoid().max(dim=-1).values * quality.sigmoid()
        return {"boxes": boxes, "logits": logits, "scores": scores, "bbox_quality": quality}

    def feature_map_to_sequence(self, feat: Tensor, level: int) -> Tensor:
        batch, _, height, width = feat.shape
        content = feat.flatten(2).transpose(1, 2)
        position = self.position(height, width, feat.device, feat.dtype).unsqueeze(0).expand(batch, -1, -1)
        return content + position + self.level_embedding[level].view(1, 1, -1)

    def pool_for_global_attention(self, feat: Tensor, level_name: str) -> Tensor:
        max_size = self.token_pool_sizes[level_name]
        height, width = feat.shape[-2:]
        if height <= max_size and width <= max_size:
            return feat
        return F.adaptive_avg_pool2d(feat, (min(height, max_size), min(width, max_size)))

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        # 1. CNN extracts multi-scale lesion features.
        raw_features = self.backbone(image)

        # 2. RT-DETR-style hybrid encoder fuses cross-scale semantic context.
        features = self.hybrid_encoder(raw_features)

        # 3. Convert P3/P4/P5 maps into token sequences with spatial positions.
        p3_tokens = self.feature_map_to_sequence(self.pool_for_global_attention(features["p3"], "p3"), level=0)
        p4_tokens = self.feature_map_to_sequence(self.pool_for_global_attention(features["p4"], "p4"), level=1)
        p5_tokens = self.feature_map_to_sequence(self.pool_for_global_attention(features["p5"], "p5"), level=2)
        memory_input = torch.cat([p3_tokens, p4_tokens, p5_tokens], dim=1)

        # 4. Transformer encoder lets every token reason about global CXR context.
        memory = self.transformer_encoder(memory_input)

        # 5. DETR paper: a fixed set of learned object queries asks the image tokens for objects.
        queries = self.object_queries.weight.unsqueeze(0).expand(image.shape[0], -1, -1)

        # 6. Decoder layers are explicit so auxiliary losses can supervise every refinement step.
        layer_outputs = []
        decoded_queries = queries
        for decoder_layer in self.decoder_layers:
            decoded_queries = decoder_layer(decoded_queries, memory)
            layer_outputs.append(self.predict_from_queries(decoded_queries))

        # 7. Final layer is the normal prediction; previous layers are DETR auxiliary predictions.
        final = layer_outputs[-1]
        final["boxes_per_layer"] = [o["boxes"] for o in layer_outputs]
        final["aux_outputs"] = layer_outputs[:-1]
        return final


class DETRSetCriterion(HybridDetectionLoss):
    """DETR set criterion: Hungarian final loss plus auxiliary decoder-layer losses."""

    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        losses = super().forward(outputs, targets)
        aux_outputs = outputs.get("aux_outputs", [])
        if not aux_outputs:
            return losses
        total = losses["loss"]
        aux_cls = losses["loss_cls"] * 0.0
        aux_l1 = losses["loss_l1"] * 0.0
        aux_giou = losses["loss_giou"] * 0.0
        aux_quality = losses["loss_quality"] * 0.0
        for aux in aux_outputs:
            aux_losses = super().forward({**aux, "boxes_per_layer": [aux["boxes"]]}, targets)
            total = total + 0.5 * aux_losses["loss"]
            aux_cls = aux_cls + aux_losses["loss_cls"]
            aux_l1 = aux_l1 + aux_losses["loss_l1"]
            aux_giou = aux_giou + aux_losses["loss_giou"]
            aux_quality = aux_quality + aux_losses["loss_quality"]
        denom = max(len(aux_outputs), 1)
        losses["loss"] = total
        losses["loss_cls"] = (losses["loss_cls"] + aux_cls / denom).detach()
        losses["loss_l1"] = (losses["loss_l1"] + aux_l1 / denom).detach()
        losses["loss_giou"] = (losses["loss_giou"] + aux_giou / denom).detach()
        losses["loss_quality"] = (losses["loss_quality"] + aux_quality / denom).detach()
        return losses


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
    save_detection_visualizations(model, args, device, "detr", (255, 190, 0))


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HybridCXRConfig(num_classes=DETECT_NUM_CLASSES, image_channels=1, base_channels=args.base_channels, hidden_dim=args.hidden_dim, num_queries=args.num_queries, num_decoder_layers=args.num_decoder_layers, num_encoder_layers=args.num_encoder_layers, cls_loss_weight=args.cls_loss_weight, box_l1_weight=args.box_l1_weight, giou_weight=args.giou_weight, quality_weight=args.quality_weight)
    model = ExplicitDETR(cfg).to(device)
    if args.weight_init != "default":
        apply_weight_initialization(model, args.weight_init)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    train_model = torch.compile(model) if args.compile_model and hasattr(torch, "compile") else model
    loss_fn = DETRSetCriterion(cfg).to(device)
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
                print(f"detr epoch={epoch} step={step} loss={losses['loss'].detach().item():.4f}")
            step += 1
            if args.max_steps and step >= args.max_steps:
                save_checkpoint(args, ema.ema if ema else model, "detr_smoke.pt")
                eval_model = ema.ema if ema else model
                return evaluate_map50(eval_model, val_loader, device, args.map_steps)
        eval_model = ema.ema if ema else model
        map50 = evaluate_map50(eval_model, val_loader, device, args.map_steps)
        print(f"val epoch={epoch} mAP50={map50:.4f}")
        if map50 > best_map:
            best_map = map50
            save_checkpoint(args, eval_model, "detr_best.pt")
    save_checkpoint(args, ema.ema if ema else model, "detr_final.pt")
    if args.visualize:
        visualize(ema.ema if ema else model, args, device)
    return best_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-dir", type=Path, default=Path("runs/detr_cxr"))
    p.add_argument("--epochs", type=int, default=80)
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
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--ema-decay", type=float, default=0.9998)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--grad-clip", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-queries", type=int, default=300)
    p.add_argument("--num-decoder-layers", type=int, default=6)
    p.add_argument("--num-encoder-layers", type=int, default=3)
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
        args.base_channels = 8; args.hidden_dim = 32; args.num_queries = 8; args.num_decoder_layers = 1; args.max_steps = 1; args.warmup_steps = 1; args.log_every = 1; args.positive_only = True
    return args


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def write_search_report(rows: List[Dict], out_dir: Path, title: str) -> None:
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
    lines = [f"# {title}", "", f"Best score: {best['score']:.6f}", "", "| trial | score | lr | opt | batch | base | hidden | queries | enc | dec | wd | init |", "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in sorted(rows, key=lambda r: float(r["score"]), reverse=True):
        lines.append(f"| {row['trial']} | {row['score']:.6f} | {row['lr']:.2e} | {row['optimizer']} | {row['batch_size']} | {row['base_channels']} | {row['hidden_dim']} | {row['num_queries']} | {row['num_encoder_layers']} | {row['num_decoder_layers']} | {row['weight_decay']:.2e} | {row['weight_init']} |")
    (out_dir / "random_search_report.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot([r["trial"] for r in rows], [r["score"] for r in rows], marker="o")
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
        trial_args.lr = log_uniform(rng, 3e-5, 5e-4)
        trial_args.weight_decay = log_uniform(rng, 1e-6, 5e-4)
        trial_args.optimizer = rng.choice(["adamw", "adam", "sgd"])
        trial_args.batch_size = rng.choice([2, 4])
        trial_args.base_channels = rng.choice([24, 32, 40])
        trial_args.hidden_dim = rng.choice([128, 192, 256])
        trial_args.num_queries = rng.choice([200, 300, 400])
        trial_args.num_encoder_layers = rng.choice([2, 3, 4])
        trial_args.num_decoder_layers = rng.choice([4, 6])
        trial_args.weight_init = rng.choice(["default", "xavier", "kaiming"])
        trial_args.cls_loss_weight = rng.choice([0.75, 1.0, 1.25])
        trial_args.box_l1_weight = rng.choice([4.0, 5.0, 6.0])
        trial_args.giou_weight = rng.choice([1.5, 2.0, 2.5])
        trial_args.quality_weight = rng.choice([0.25, 0.5, 0.75])
        print(f"random-search trial={trial} optimizer={trial_args.optimizer} lr={trial_args.lr:.2e}")
        score = float(train(trial_args) or 0.0)
        fields = ["lr", "weight_decay", "optimizer", "batch_size", "base_channels", "hidden_dim", "num_queries", "num_encoder_layers", "num_decoder_layers", "weight_init", "cls_loss_weight", "box_l1_weight", "giou_weight", "quality_weight"]
        rows.append({k: getattr(trial_args, k) for k in fields} | {"trial": trial, "score": score})
        write_search_report(rows, root, "DETR Random Search")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.random_search:
        random_search(parsed)
    else:
        train(parsed)
