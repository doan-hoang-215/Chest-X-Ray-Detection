"""
Train the hybrid CXR detector on the local VinDr-CXR style dataset.

Examples:
  python train_vindr_hybrid.py --smoke
  python train_vindr_hybrid.py --phase detect --epochs 30 --image-size 768
  python train_vindr_hybrid.py --phase pretrain --epochs 20 --normal-only
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from hybrid_cxr_detector import create_loss, create_model, create_pretrainer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / ("data_jpg" if (PROJECT_ROOT / "data_jpg").exists() else "data")
NO_FINDING_CLASS_ID = 14
DETECT_NUM_CLASSES = 14
CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
    "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_data_root(path: str | None) -> Path:
    root = Path(path) if path else DEFAULT_DATA_ROOT
    if not root.exists() and root.name == "data" and (root.parent / "data_jpg").exists():
        root = root.parent / "data_jpg"
    if not root.exists():
        raise FileNotFoundError(f"data root not found: {root}")
    if not (root / "train.csv").exists():
        raise FileNotFoundError(f"missing train.csv under {root}")
    if not (root / "train").exists():
        raise FileNotFoundError(f"missing train image folder under {root}")
    return root


def read_dicom_image(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    try:
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    except Exception:
        arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = np.percentile(arr, (0.5, 99.5))
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return arr.astype(np.float32)


def read_jpg_image(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read JPG image: {path}")
    return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)


def read_train_image(data_root: Path, image_id: str, use_jpg_cache: bool = True, jpg_quality: int = 95) -> np.ndarray:
    """Read train image through a normalized JPG cache to avoid repeated DICOM decode."""
    jpg_path = data_root / "train_jpg" / f"{image_id}.jpg"
    if use_jpg_cache and jpg_path.exists():
        return read_jpg_image(jpg_path)
    train_jpg_path = data_root / "train" / f"{image_id}.jpg"
    if train_jpg_path.exists():
        return read_jpg_image(train_jpg_path)
    train_jpeg_path = data_root / "train" / f"{image_id}.jpeg"
    if train_jpeg_path.exists():
        return read_jpg_image(train_jpeg_path)
    flat_jpg_path = data_root / f"{image_id}.jpg"
    if flat_jpg_path.exists():
        return read_jpg_image(flat_jpg_path)
    flat_jpeg_path = data_root / f"{image_id}.jpeg"
    if flat_jpeg_path.exists():
        return read_jpg_image(flat_jpeg_path)
    dicom_path = data_root / "train" / f"{image_id}.dicom"
    img = read_dicom_image(dicom_path)
    if use_jpg_cache:
        jpg_path.parent.mkdir(parents=True, exist_ok=True)
        img8 = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", img8, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        if ok:
            jpg_path.write_bytes(encoded.tobytes())
    return img


def resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32)


def infer_lung_crop(image: np.ndarray, pad_fraction: float = 0.08) -> Tuple[int, int, int, int]:
    """Conservative CXR foreground crop; keeps padding so peripheral lesions survive."""
    img8 = (image.clip(0.0, 1.0) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(img8, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() > 127:
        mask = 255 - mask
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if components <= 1:
        return 0, 0, image.shape[1], image.shape[0]
    keep = np.argsort(stats[1:, cv2.CC_STAT_AREA])[-2:] + 1
    ys, xs = np.where(np.isin(labels, keep))
    if xs.size == 0 or ys.size == 0:
        return 0, 0, image.shape[1], image.shape[0]
    h, w = image.shape[:2]
    pad = int(round(max(h, w) * pad_fraction))
    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad + 1, w)
    y2 = min(int(ys.max()) + pad + 1, h)
    if (x2 - x1) < 0.45 * w or (y2 - y1) < 0.45 * h:
        return 0, 0, w, h
    return x1, y1, x2, y2


def normalize_boxes(rows: pd.DataFrame, height: int, width: int, crop: Tuple[int, int, int, int] | None = None) -> Tuple[Tensor, Tensor]:
    if crop is None:
        crop = (0, 0, width, height)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop
    crop_w = max(crop_x2 - crop_x1, 1)
    crop_h = max(crop_y2 - crop_y1, 1)
    boxes: List[List[float]] = []
    labels: List[int] = []
    for row in rows.itertuples(index=False):
        class_id = int(row.class_id)
        if class_id == NO_FINDING_CLASS_ID:
            continue
        if pd.isna(row.x_min) or pd.isna(row.y_min) or pd.isna(row.x_max) or pd.isna(row.y_max):
            continue
        x1 = max((float(row.x_min) - crop_x1) / crop_w, 0.0)
        y1 = max((float(row.y_min) - crop_y1) / crop_h, 0.0)
        x2 = min((float(row.x_max) - crop_x1) / crop_w, 1.0)
        y2 = min((float(row.y_max) - crop_y1) / crop_h, 1.0)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(class_id)
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def cxr_aug_pair(image: Tensor) -> Tuple[Tensor, Tensor]:
    """CXR-safe SSL views: preserve lung anatomy while varying contrast/noise."""

    def random_resized_crop(x: Tensor) -> Tensor:
        _, h, w = x.shape
        scale = 0.70 + 0.30 * random.random()
        ratio = 0.92 + 0.16 * random.random()
        crop_h = max(1, min(h, int(round(h * scale / (ratio ** 0.5)))))
        crop_w = max(1, min(w, int(round(w * scale * (ratio ** 0.5)))))
        top = 0 if h == crop_h else random.randint(0, h - crop_h)
        left = 0 if w == crop_w else random.randint(0, w - crop_w)
        crop = x[:, top : top + crop_h, left : left + crop_w].unsqueeze(0)
        return F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)

    def aug(x: Tensor) -> Tensor:
        x = random_resized_crop(x)
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        if random.random() < 0.35:
            arr = (x.squeeze(0).numpy().clip(0.0, 1.0) * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            x = torch.from_numpy(clahe.apply(arr).astype(np.float32) / 255.0).unsqueeze(0)
        gamma = 0.85 + 0.30 * random.random()
        gain = 0.90 + 0.20 * random.random()
        bias = -0.05 + 0.10 * random.random()
        x = (x.clamp_min(1e-6).pow(gamma) * gain + bias).clamp(0.0, 1.0)
        if random.random() < 0.30:
            x = F.avg_pool2d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        if random.random() < 0.35:
            x = (x + torch.randn_like(x) * 0.015).clamp(0.0, 1.0)
        return x
    return aug(image.clone()), aug(image.clone())


def image_detection_labels(rows: pd.DataFrame) -> set[int]:
    labels = {int(c) for c in rows["class_id"].tolist() if int(c) != NO_FINDING_CLASS_ID}
    return labels if labels else {NO_FINDING_CLASS_ID}


def split_train_val_test_ids(groups: Dict[str, pd.DataFrame], val_fraction: float, test_fraction: float, seed: int) -> Dict[str, List[str]]:
    """Iterative multilabel-stratified split over all diseases and No finding."""
    val_fraction = min(max(float(val_fraction), 0.0), 0.4)
    test_fraction = min(max(float(test_fraction), 0.0), 0.4)
    if val_fraction + test_fraction >= 0.8:
        raise ValueError("val_fraction + test_fraction must be < 0.8")
    ids = sorted(groups.keys())
    rng = random.Random(seed)
    labels_by_id = {image_id: image_detection_labels(groups[image_id]) for image_id in ids}
    normal_ids = [image_id for image_id in ids if labels_by_id[image_id] == {NO_FINDING_CLASS_ID}]
    abnormal_ids = [image_id for image_id in ids if labels_by_id[image_id] != {NO_FINDING_CLASS_ID}]
    split_names = ("train", "val", "test")
    fractions = {"train": 1.0 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    target_sizes = {"val": int(round(len(ids) * val_fraction)), "test": int(round(len(ids) * test_fraction))}
    target_sizes["train"] = len(ids) - target_sizes["val"] - target_sizes["test"]
    split_ids = {"train": [], "val": [], "test": []}
    target_normal = {"val": int(round(len(normal_ids) * val_fraction)), "test": int(round(len(normal_ids) * test_fraction))}
    target_normal["train"] = len(normal_ids) - target_normal["val"] - target_normal["test"]
    normal_order = normal_ids[:]
    rng.shuffle(normal_order)
    cursor = 0
    for split in split_names:
        take = target_normal[split]
        split_ids[split].extend(normal_order[cursor : cursor + take])
        cursor += take
    target_abnormal = {split: target_sizes[split] - target_normal[split] for split in split_names}
    remaining_size = target_sizes.copy()
    for split in split_names:
        remaining_size[split] = target_abnormal[split]
    label_totals = defaultdict(int)
    label_to_ids = defaultdict(set)
    for image_id in abnormal_ids:
        labels = labels_by_id[image_id]
        for label in labels:
            label_totals[label] += 1
            label_to_ids[label].add(image_id)
    remaining_label_need = {
        split: {label: label_totals[label] * target_abnormal[split] / max(len(abnormal_ids), 1) for label in label_totals}
        for split in split_names
    }
    unassigned = set(abnormal_ids)
    while unassigned:
        active_labels = [(len(label_to_ids[label] & unassigned), label) for label in label_totals if label_to_ids[label] & unassigned]
        if active_labels:
            _, rare_label = min(active_labels)
            candidates = list(label_to_ids[rare_label] & unassigned)
        else:
            candidates = list(unassigned)
        rng.shuffle(candidates)
        for image_id in candidates:
            if image_id not in unassigned:
                continue
            available = [split for split in split_names if remaining_size[split] > 0]
            if not available:
                available = ["train"]
            labels = labels_by_id[image_id]

            def assignment_score(split: str) -> Tuple[float, float, float]:
                rare_need = remaining_label_need[split].get(rare_label, 0.0)
                all_label_need = sum(max(remaining_label_need[split].get(label, 0.0), 0.0) for label in labels)
                size_need = remaining_size[split] / max(target_sizes[split], 1)
                return rare_need, all_label_need, size_need

            chosen = max(available, key=assignment_score)
            split_ids[chosen].append(image_id)
            remaining_size[chosen] -= 1
            for label in labels:
                remaining_label_need[chosen][label] -= 1.0
            unassigned.remove(image_id)
    for split in split_ids:
        split_ids[split].sort()
    return split_ids


class VinDrDetectionDataset(Dataset):
    def __init__(self, data_root: Path, image_size: int = 512, split: str = "train", val_fraction: float = 0.1, seed: int = 42, max_images: int | None = None, normal_only: bool = False, ssl_pair: bool = False, positive_only: bool = False, lung_crop: bool = True, test_fraction: float = 0.1, use_jpg_cache: bool = True, jpg_quality: int = 95):
        self.data_root = data_root
        self.image_size = image_size
        self.split = split
        self.ssl_pair = ssl_pair
        self.lung_crop = lung_crop
        self.use_jpg_cache = use_jpg_cache
        self.jpg_quality = jpg_quality
        self.df = pd.read_csv(data_root / "train.csv")
        self.groups = {image_id: group.copy() for image_id, group in self.df.groupby("image_id")}
        split_ids = split_train_val_test_ids(self.groups, val_fraction, test_fraction, seed)
        if split not in split_ids:
            raise ValueError(f"split must be one of {sorted(split_ids)}, got {split}")
        ids = split_ids[split]
        if normal_only:
            ids = [i for i in ids if not (self.groups[i]["class_id"] != NO_FINDING_CLASS_ID).any()]
        if positive_only:
            ids = [i for i in ids if (self.groups[i]["class_id"] != NO_FINDING_CLASS_ID).any()]
        if max_images is not None:
            ids = ids[:max_images]
        self.image_ids = ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        img = read_train_image(self.data_root, image_id, self.use_jpg_cache, self.jpg_quality)
        height, width = img.shape[:2]
        crop = infer_lung_crop(img) if self.lung_crop else (0, 0, width, height)
        boxes, labels = normalize_boxes(self.groups[image_id], height, width, crop)
        x1, y1, x2, y2 = crop
        img = img[y1:y2, x1:x2]
        img = resize_image(img, self.image_size)
        image = torch.from_numpy(img).unsqueeze(0)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id}
        if self.ssl_pair:
            return cxr_aug_pair(image)
        return image, target


def detection_collate(batch):
    return torch.stack([item[0] for item in batch], dim=0), [item[1] for item in batch]


def ssl_collate(batch):
    return torch.stack([item[0] for item in batch], dim=0), torch.stack([item[1] for item in batch], dim=0)


def make_loader(args, split: str, ssl_pair: bool = False, normal_only: bool = False) -> DataLoader:
    max_images = args.max_images if split == "train" else min(args.max_images or 999999, args.val_max_images)
    ds = VinDrDetectionDataset(args.data_root, args.image_size, split, args.val_fraction, args.seed, max_images, normal_only, ssl_pair, args.positive_only if not ssl_pair else False, getattr(args, "lung_crop", True), getattr(args, "test_fraction", 0.1), getattr(args, "use_jpg_cache", True), getattr(args, "jpg_quality", 95))
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": split == "train",
        "num_workers": args.num_workers,
        "collate_fn": ssl_collate if ssl_pair else detection_collate,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": ssl_pair and split == "train",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


@torch.no_grad()
def save_detection_visualizations(model: torch.nn.Module, args, device: torch.device, prefix: str, color: Tuple[int, int, int]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = VinDrDetectionDataset(args.data_root, args.image_size, split="val", val_fraction=args.val_fraction, seed=args.seed, max_images=args.vis_images, lung_crop=getattr(args, "lung_crop", True), test_fraction=getattr(args, "test_fraction", 0.1), use_jpg_cache=getattr(args, "use_jpg_cache", True), jpg_quality=getattr(args, "jpg_quality", 95))
    rows = []
    model.eval()
    for i in range(len(ds)):
        image, target = ds[i]
        out = model(image.unsqueeze(0).to(device))
        probs = out["logits"].sigmoid()[0] * out["scores"][0].unsqueeze(-1)
        scores, labels = probs.max(dim=-1)
        keep = scores > args.score_thresh
        base = cv2.cvtColor((image[0].numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        h, w = base.shape[:2]
        pred = base.copy()
        gt = base.copy()
        both = base.copy()
        heat = np.zeros((h, w), dtype=np.float32)
        for box, score, label in zip(out["boxes"][0][keep].cpu(), scores[keep].cpu(), labels[keep].cpu()):
            x1, y1, x2, y2 = (box.numpy() * np.array([w, h, w, h])).astype(int)
            cls = int(label)
            rows.append({"image_id": target["image_id"], "class_id": cls, "class_name": CLASS_NAMES[cls], "score": float(score), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            cv2.rectangle(pred, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(both, (x1, y1), (x2, y2), color, 2)
            cv2.putText(pred, f"{cls}:{float(score):.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            cx, cy = max(min((x1 + x2) // 2, w - 1), 0), max(min((y1 + y2) // 2, h - 1), 0)
            cv2.circle(heat, (cx, cy), max(4, int(min(w, h) * 0.018)), float(score), -1)
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = (box.numpy() * np.array([w, h, w, h])).astype(int)
            cv2.rectangle(gt, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.rectangle(both, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(gt, f"GT {int(label)}", (x1, min(h - 4, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(w, h) * 0.025)
        heat = (heat / max(float(heat.max()), 1e-6) * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat_overlay = cv2.addWeighted(base, 0.65, heat_color, 0.35, 0)
        panel = np.vstack([np.hstack([base, pred]), np.hstack([gt, both]), np.hstack([heat_overlay, base])])
        cv2.imwrite(str(args.output_dir / f"{prefix}_diagnostic_{i}.jpg"), panel)
    if rows:
        pd.DataFrame(rows).to_csv(args.output_dir / f"{prefix}_predictions.csv", index=False)
    print(f"saved diagnostic visualizations to {args.output_dir}")


def move_targets(targets: List[Dict], device: torch.device) -> List[Dict[str, Tensor]]:
    return [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device), "image_id": t["image_id"]} for t in targets]


def load_pretrained_backbone(model, checkpoint_path: str, device: torch.device) -> None:
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    backbone_state = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            backbone_state[key.replace("backbone.", "")] = value
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"loaded pretrained backbone from {checkpoint_path}; missing={len(missing)} unexpected={len(unexpected)}")


def save_checkpoint(args, model, name: str) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / name
    safe_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    torch.save({"model": model.state_dict(), "args": safe_args}, path)
    print(f"saved {path}")




class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9998):
        import copy
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            src = model_state[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                v.copy_(src)


def build_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    backbone_decay, backbone_no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith("backbone")
        is_no_decay = name.endswith("bias") or "bn" in name.lower() or "norm" in name.lower() or "level_embed" in name
        if is_backbone and is_no_decay:
            backbone_no_decay.append(param)
        elif is_backbone:
            backbone_decay.append(param)
        elif is_no_decay:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay, "lr": args.lr, "initial_lr": args.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": args.lr, "initial_lr": args.lr},
        {"params": backbone_decay, "weight_decay": args.weight_decay, "lr": args.lr * args.backbone_lr_mult, "initial_lr": args.lr * args.backbone_lr_mult},
        {"params": backbone_no_decay, "weight_decay": 0.0, "lr": args.lr * args.backbone_lr_mult, "initial_lr": args.lr * args.backbone_lr_mult},
    ]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def set_warmup_cosine_lr(optimizer: torch.optim.Optimizer, step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> None:
    if warmup_steps > 0 and step < warmup_steps:
        scale = float(step + 1) / float(warmup_steps)
    else:
        denom = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
        scale = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale


def train_pretrain(args, device: torch.device) -> None:
    loader = make_loader(args, "train", ssl_pair=True, normal_only=args.normal_only)
    model = create_pretrainer(image_channels=1, base_channels=args.base_channels, projector_dim=args.projector_dim, projector_hidden_dim=args.projector_hidden_dim).to(device)
    opt = build_optimizer(model, args)
    total_steps = max(args.epochs * max(len(loader), 1), 1)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for v1, v2 in loader:
            out = model(v1.to(device), v2.to(device))
            set_warmup_cosine_lr(opt, step, total_steps, args.warmup_steps, args.min_lr_ratio)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if step % args.log_every == 0:
                print(f"pretrain epoch={epoch} step={step} loss={float(out['loss'].detach()):.4f}")
            step += 1
            if args.max_steps and step >= args.max_steps:
                save_checkpoint(args, model, "pretrainer_smoke.pt")
                return
    save_checkpoint(args, model, "pretrainer_final.pt")


@torch.no_grad()
def validate(model, loss_fn, loader, device: torch.device, max_steps: int) -> float:
    model.eval()
    losses = []
    for step, (images, targets) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        out = model(images.to(device))
        losses.append(float(loss_fn(out, move_targets(targets, device))["loss"].detach()))
    return float(np.mean(losses)) if losses else float("inf")




def average_precision(tp: np.ndarray, fp: np.ndarray, total_gt: int) -> float:
    if total_gt == 0:
        return float("nan")
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(total_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    idx = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))


@torch.no_grad()
def evaluate_map50(model, loader, device: torch.device, max_steps: int, score_thresh: float = 0.01) -> float:
    model.eval()
    preds = {c: [] for c in range(DETECT_NUM_CLASSES)}
    gt_count = {c: 0 for c in range(DETECT_NUM_CLASSES)}
    matched = defaultdict(set)
    for step, (images, targets) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        out = model(images.to(device))
        probs = out["logits"].sigmoid() * out["scores"].unsqueeze(-1)
        pred_scores, pred_labels = probs.max(dim=-1)
        pred_boxes = out["boxes"].detach().cpu()
        pred_scores = pred_scores.detach().cpu()
        pred_labels = pred_labels.detach().cpu()
        for b, target in enumerate(targets):
            image_id = target["image_id"]
            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu()
            for cls in gt_labels.tolist():
                if 0 <= cls < DETECT_NUM_CLASSES:
                    gt_count[cls] += 1
            keep = pred_scores[b] >= score_thresh
            for box, score, label in zip(pred_boxes[b][keep], pred_scores[b][keep], pred_labels[b][keep]):
                cls = int(label)
                same = torch.where(gt_labels == cls)[0]
                is_tp = 0
                if same.numel() > 0:
                    ious = box_iou_torch(box.view(1, 4), gt_boxes[same])[0].view(-1)
                    best_iou, best_pos = torch.max(ious, dim=0)
                    gt_index = int(same[int(best_pos)])
                    key = (image_id, gt_index)
                    if float(best_iou) >= 0.5 and key not in matched[cls]:
                        matched[cls].add(key)
                        is_tp = 1
                preds[cls].append((float(score), is_tp))
    aps = []
    for cls in range(DETECT_NUM_CLASSES):
        items = sorted(preds[cls], key=lambda x: x[0], reverse=True)
        if not items:
            continue
        tp = np.array([x[1] for x in items], dtype=np.float32)
        fp = 1.0 - tp
        ap = average_precision(tp, fp, gt_count[cls])
        if not math.isnan(ap):
            aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def box_iou_torch(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp_min(1e-7), union


def train_detector(args, device: torch.device) -> None:
    train_loader = make_loader(args, "train")
    val_loader = make_loader(args, "val")
    model = create_model(num_classes=DETECT_NUM_CLASSES, image_channels=1, base_channels=args.base_channels, hidden_dim=args.hidden_dim, num_queries=args.num_queries, num_decoder_layers=args.num_decoder_layers, num_encoder_layers=args.num_encoder_layers, topk_per_head=args.topk_per_head, bbox_bins=args.bbox_bins, dfine_layers=args.dfine_layers).to(device)
    load_pretrained_backbone(model, args.pretrained_backbone, device)
    loss_fn = create_loss(num_classes=DETECT_NUM_CLASSES, image_channels=1).to(device)
    opt = build_optimizer(model, args)
    total_steps = max(args.epochs * max(len(train_loader), 1), 1)
    ema = None if args.no_ema else ModelEMA(model, args.ema_decay)
    best_val = math.inf
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for images, targets in train_loader:
            out = model(images.to(device))
            losses = loss_fn(out, move_targets(targets, device))
            set_warmup_cosine_lr(opt, step, total_steps, args.warmup_steps, args.min_lr_ratio)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if ema is not None:
                ema.update(model)
            if step % args.log_every == 0:
                print(f"detect epoch={epoch} step={step} loss={float(losses['loss'].detach()):.4f} cls={float(losses['loss_cls']):.4f} l1={float(losses['loss_l1']):.4f} giou={float(losses['loss_giou']):.4f}")
            step += 1
            if args.max_steps and step >= args.max_steps:
                save_checkpoint(args, model, "detector_smoke.pt")
                return
        eval_model = ema.ema if ema is not None else model
        val_loss = validate(eval_model, loss_fn, val_loader, device, args.val_steps)
        map50 = evaluate_map50(eval_model, val_loader, device, args.map_steps)
        print(f"val epoch={epoch} loss={val_loss:.4f} mAP50={map50:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args, eval_model, "detector_best.pt")
    save_checkpoint(args, ema.ema if ema is not None else model, "detector_final.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--output-dir", type=str, default="runs/vindr_hybrid")
    p.add_argument("--phase", choices=["detect", "pretrain"], default="detect")
    p.add_argument("--pretrained-backbone", type=str, default="")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--normal-only", action="store_true")
    p.add_argument("--positive-only", action="store_true")
    p.add_argument("--lung-crop", action="store_true", default=True)
    p.add_argument("--no-lung-crop", dest="lung_crop", action="store_false")
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--use-jpg-cache", action="store_true", default=True)
    p.add_argument("--no-jpg-cache", dest="use_jpg_cache", action="store_false")
    p.add_argument("--jpg-quality", type=int, default=95)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--val-max-images", type=int, default=128)
    p.add_argument("--val-steps", type=int, default=20)
    p.add_argument("--map-steps", type=int, default=10)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.2)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--ema-decay", type=float, default=0.9998)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--num-decoder-layers", type=int, default=3)
    p.add_argument("--num-encoder-layers", type=int, default=1)
    p.add_argument("--topk-per-head", type=int, default=64)
    p.add_argument("--bbox-bins", type=int, default=32)
    p.add_argument("--dfine-layers", type=int, default=3)
    p.add_argument("--projector-dim", type=int, default=2048)
    p.add_argument("--projector-hidden-dim", type=int, default=1024)
    args = p.parse_args()
    if args.smoke:
        args.image_size = 160
        args.batch_size = 2
        args.epochs = 1
        args.max_steps = 1
        args.max_images = 8
        args.val_max_images = 4
        args.positive_only = True
        args.val_steps = 1
        args.map_steps = 1
        args.base_channels = 8
        args.hidden_dim = 32
        args.num_queries = 8
        args.num_decoder_layers = 1
        args.topk_per_head = 4
        args.bbox_bins = 8
        args.dfine_layers = 1
        args.log_every = 1
        args.warmup_steps = 1
        args.no_ema = False
    args.data_root = resolve_data_root(args.data_root)
    args.output_dir = Path(args.output_dir)
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} data_root={args.data_root}")
    if args.phase == "pretrain":
        train_pretrain(args, device)
    else:
        train_detector(args, device)


if __name__ == "__main__":
    main()
