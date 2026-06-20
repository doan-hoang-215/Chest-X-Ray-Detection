"""
NSEC-YOLO: Real-time Lesion Detection on Chest X-ray
Implemented from paper:
  "NSEC-YOLO: Real-time lesion detection on chest X-ray with adaptive
   noise suppression and global perception aggregation"
  Journal of Radiation Research and Applied Sciences, 2025

Dataset: NIH ChestX-ray14 (chỉ dùng 880 ảnh có Bounding Box)
  - BBox_List_2017.csv   : ground-truth bounding boxes
  - Data_Entry_2017.csv  : metadata / image size gốc
  - 8 classes: Atelectasis, Cardiomegaly, Effusion, Infiltrate,
               Mass, Nodule, Pneumonia, Pneumothorax

Architecture (built on YOLOv7x):
  1. ANS Module     — Adaptive Noise Suppression (Eq. 1-7)
  2. GPAdetect Head — Global Perceptual Aggregation (Eq. 8-14)
  3. AccurEIOU Loss — Combined CIOU + EIOU (Eq. 15-23)

RESUME TRAINING: Hỗ trợ tiếp tục train từ checkpoint đã lưu.
  Sử dụng: train(cfg, resume_path="path/to/best_nsec_yolo.pth")
"""

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.ops as ops


# ──────────────────────────────────────────────
# 0. CONFIG
# ──────────────────────────────────────────────
class Config:
    # ── Paths (chỉnh lại theo máy bạn) ───────────────────────────────────
    DATA_ROOT    = r"C:\FPT\SU26\DPL302m\archive (2)"
    BBOX_CSV     = r"C:\FPT\SU26\DPL302m\archive (2)\BBox_List_2017.csv"
    DATA_ENTRY   = r"C:\FPT\SU26\DPL302m\archive (2)\Data_Entry_2017.csv"
    SAVE_DIR     = r"C:\FPT\SU26\DPL302m\archive (2)\checkpoints_nsec_2"

    # ── Model ─────────────────────────────────────────────────────────────
    IMG_SIZE     = 224       # paper dùng 640×640
    NUM_CLASSES  = 8         # 8 classes có bbox trong NIH dataset
    BATCH_SIZE   = 4         # giảm nếu VRAM không đủ (paper: 32 trên V100)
    EPOCHS       = 50       # paper: 240 epochs
    LR           = 0.01      # paper: SGD lr=0.01
    WEIGHT_DECAY = 0.0005    # paper: 0.0005
    NUM_WORKERS  = 2         # Windows: 5

    # Split từ 880 ảnh có bbox
    TRAIN_RATIO  = 0.70
    VAL_RATIO    = 0.15
    TEST_RATIO   = 0.15
    RANDOM_SEED  = 42

    # AccurEIOU aspect ratio stability threshold (paper: 0.01)
    RATIO_THRESH = 0.01

    # Loss weights
    LAMBDA_BOX   = 0.05
    LAMBDA_OBJ   = 1.0
    LAMBDA_CLS   = 0.5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 8 classes có bounding box trong NIH ChestX-ray14
    CLASSES = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    ]

    # Default YOLO anchors (pixels ở input 640×640)
    ANCHORS = [
        [[10, 13], [16, 30], [33, 23]],          # P3 stride 8  (small)
        [[30, 61], [62, 45], [59, 119]],          # P4 stride 16 (medium)
        [[116, 90], [156, 198], [373, 326]],      # P5 stride 32 (large)
    ]

    @classmethod
    def setup(cls):
        Path(cls.SAVE_DIR).mkdir(parents=True, exist_ok=True)
        if cls.DEVICE == "cuda":
            torch.cuda.set_device(0)
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] {name}  ({mem:.1f} GB VRAM)")
            torch.backends.cudnn.benchmark = True
        else:
            print("[Device] CPU  (không tìm thấy GPU)")
        print(f"[Config] IMG_SIZE={cls.IMG_SIZE} | "
              f"NUM_CLASSES={cls.NUM_CLASSES} | EPOCHS={cls.EPOCHS}")


# ──────────────────────────────────────────────
# 1. DATASET  (NIH ChestX-ray14 — BBox only)
# ──────────────────────────────────────────────
def make_splits(bbox_csv, seed=42, train_ratio=0.70, val_ratio=0.15):
    """
    Đọc BBox_List_2017.csv, chia 880 ảnh thành train/val/test
    theo tỉ lệ 70/15/15, stratified theo label đầu tiên của mỗi ảnh.
    """
    df = pd.read_csv(bbox_csv)
    df.columns = ["Image Index", "Finding Label", "x", "y", "w", "h",
                  "u6", "u7", "u8"]

    img_label = (
        df.groupby("Image Index")["Finding Label"]
        .first()
        .reset_index()
    )
    images = img_label["Image Index"].to_numpy()
    labels = img_label["Finding Label"].to_numpy()

    test_size = 1.0 - train_ratio
    train_imgs, temp_imgs, _, temp_labels = train_test_split(
        images, labels, test_size=test_size,
        stratify=labels, random_state=seed,
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.5,
        stratify=temp_labels, random_state=seed,
    )
    print(f"[Split] Train={len(train_imgs)} | "
          f"Val={len(val_imgs)} | Test={len(test_imgs)}")
    return set(train_imgs), set(val_imgs), set(test_imgs)


class NIHBBoxDataset(Dataset):
    """
    NIH ChestX-ray14 Dataset — chỉ dùng 880 ảnh có Bounding Box.

    BBox_List_2017.csv format:
        Image Index, Finding Label, x, y, w, h  (tọa độ trong ảnh GỐC ~2500px)
    Data_Entry_2017.csv: chứa kích thước ảnh gốc (OriginalImage[Width, Height])

    __getitem__ trả về:
        img_t  : (3, IMG_SIZE, IMG_SIZE) tensor
        boxes  : (N, 4) tensor  [x1, y1, x2, y2] normalized [0,1]
        labels : (N,)   tensor  class index
        img_id : filename string
    """

    def __init__(self, bbox_csv, data_entry_csv, image_dir,
                 image_set, img_size=640, mode="train"):
        self.img_size  = img_size
        self.mode      = mode
        self.classes   = Config.CLASSES
        self.image_dir = image_dir

        # ── Build filename → full path (đệ quy qua images_001..012) ──────
        print(f"[Dataset] Scanning {image_dir} ...")
        self.path_index = {}
        for root, dirs, files in os.walk(image_dir):
            for fname in files:
                if fname.lower().endswith(".png"):
                    self.path_index[fname] = os.path.join(root, fname)
        print(f"[Dataset] {len(self.path_index)} images found on disk.")

        # ── Đọc kích thước ảnh gốc từ Data_Entry ────────────────────────
        de = pd.read_csv(data_entry_csv)
        de.columns = de.columns.str.strip()
        # Cột: 'OriginalImage[Width' và 'Height]'
        self.orig_size = {
            row["Image Index"]: (
                int(row["OriginalImage[Width"]),
                int(row["Height]"]),
            )
            for _, row in de.iterrows()
        }

        # ── Đọc BBox CSV ─────────────────────────────────────────────────
        df = pd.read_csv(bbox_csv)
        df.columns = ["Image Index", "Finding Label", "x", "y", "w", "h",
                      "u6", "u7", "u8"]

        # Lọc theo split
        df = df[df["Image Index"].isin(image_set)].reset_index(drop=True)

        # Chỉ giữ ảnh có trên disk
        mask   = df["Image Index"].isin(self.path_index)
        n_miss = (~mask).sum()
        if n_miss > 0:
            print(f"[Dataset] Warning: {n_miss} rows không thấy trên disk.")
        df = df[mask].reset_index(drop=True)

        # Gom bbox theo image (1 ảnh có thể có nhiều bbox)
        self.image_names = df["Image Index"].unique().tolist()
        self.ann_dict    = defaultdict(list)
        for _, row in df.iterrows():
            self.ann_dict[row["Image Index"]].append({
                "label": row["Finding Label"].strip(),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "w": float(row["w"]),
                "h": float(row["h"]),
            })

        self.transform = self._build_transform(mode, img_size)
        print(f"[Dataset] {mode}: {len(self.image_names)} images, "
              f"{len(df)} bbox entries.")

    def _build_transform(self, mode, size):
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if mode == "train":
            return transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.path_index[img_name]
        img      = Image.open(img_path).convert("RGB")

        # Kích thước ảnh gốc để normalize bbox
        orig_w, orig_h = self.orig_size.get(img_name, img.size)

        img_t = self.transform(img)

        # Convert bbox: (x, y, w, h) trong ảnh gốc → (x1,y1,x2,y2) normalized
        anns   = self.ann_dict.get(img_name, [])
        boxes, labels = [], []
        for ann in anns:
            x1 = ann["x"] / orig_w
            y1 = ann["y"] / orig_h
            x2 = (ann["x"] + ann["w"]) / orig_w
            y2 = (ann["y"] + ann["h"]) / orig_h
            x1, y1, x2, y2 = (
                max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)),
                max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2)),
            )
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                cls_id = (self.classes.index(ann["label"])
                          if ann["label"] in self.classes else 0)
                labels.append(cls_id)

        boxes  = torch.tensor(boxes,  dtype=torch.float32) \
                 if boxes  else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long) \
                 if labels else torch.zeros((0,),   dtype=torch.long)

        return img_t, boxes, labels, img_name


def collate_fn(batch):
    """Custom collate vì mỗi ảnh có số bbox khác nhau."""
    imgs, boxes, labels, ids = zip(*batch)
    return torch.stack(imgs), list(boxes), list(labels), list(ids)


# ──────────────────────────────────────────────
# 2. BUILDING BLOCKS  (YOLOv7 style)
# ──────────────────────────────────────────────
class CBS(nn.Module):
    """Conv + BatchNorm + SiLU."""
    def __init__(self, in_c, out_c, k=1, s=1, p=None):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ELAN(nn.Module):
    """
    Efficient Layer Aggregation Network (YOLOv7).
    Concatenate nhiều branch để tăng gradient flow và feature reuse.
    """
    def __init__(self, in_c, out_c, mid_c):
        super().__init__()
        self.cv1 = CBS(in_c,  mid_c, 1)
        self.cv2 = CBS(in_c,  mid_c, 1)
        self.cv3 = CBS(mid_c, mid_c, 3)
        self.cv4 = CBS(mid_c, mid_c, 3)
        self.cv5 = CBS(mid_c, mid_c, 3)
        self.cv6 = CBS(mid_c, mid_c, 3)
        self.cv7 = CBS(mid_c * 4, out_c, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)
        return self.cv7(torch.cat([x1, x2, x4, x6], dim=1))


class MP(nn.Module):
    """MaxPool downsampling block (YOLOv7 style)."""
    def __init__(self, in_c):
        super().__init__()
        half     = in_c // 2
        self.m   = nn.MaxPool2d(2, 2)
        self.cv1 = CBS(in_c, half, 1)
        self.cv2 = CBS(in_c, half, 1)
        self.cv3 = CBS(half, half, 3, 2)

    def forward(self, x):
        return torch.cat([self.m(self.cv1(x)), self.cv3(self.cv2(x))], dim=1)


# ──────────────────────────────────────────────
# 3. ANS MODULE  (Section 3.2)
# ──────────────────────────────────────────────
class ANS(nn.Module):
    """
    Adaptive Noise Suppression Module.

    Input M_in ∈ R^(C×H×W):
      M1 = Conv3×3(M_in)           — same modality
      M2 = Conv5×5(M_in)           → Channel Attention (BN scale γ)
      M3 = Conv7×7(M_in)           → Spatial Attention (pixel norm λ)

    Channel Attention (Eq. 2, 4):
      W_γ = γ_i / Σγ_j
      Mc  = sigmoid(W_γ · BN(M2))

    Spatial Attention (Eq. 3, 5):
      W_λ = λ_i / Σλ_j
      Ms  = sigmoid(W_λ · BN_s(M3))

    Fusion (Eq. 7):
      M_out = ReLU(BN(Conv3×3(M1 ⊗ Mc ⊗ Ms)))

    Loss regularization (Eq. 6):
      Loss += p·Σg(γ) + p·Σg(λ)   [L1 penalty]
    """

    def __init__(self, channels, penalty=1e-4):
        super().__init__()
        self.penalty = penalty

        # 3 nhánh conv với kernel khác nhau
        self.conv3 = CBS(channels, channels, 3)   # M1
        self.conv5 = CBS(channels, channels, 5)   # M2
        self.conv7 = CBS(channels, channels, 7)   # M3

        # Channel attention: BN scale factor γ
        self.bn_ch   = nn.BatchNorm2d(channels)
        self.w_gamma = nn.Parameter(torch.ones(channels))

        # Spatial attention: pixel normalization λ
        self.bn_sp    = nn.BatchNorm2d(channels)
        self.w_lambda = nn.Parameter(torch.ones(channels))

        # Fusion conv (Eq. 7)
        self.fuse_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.fuse_bn   = nn.BatchNorm2d(channels)
        self.fuse_act  = nn.ReLU(inplace=True)

    def forward(self, x):
        M1 = self.conv3(x)
        M2 = self.conv5(x)
        M3 = self.conv7(x)

        # Channel Attention: Mc = sigmoid(W_γ · BN(M2))  (Eq. 4)
        W_g = (self.w_gamma / (self.w_gamma.sum() + 1e-6)).view(1, -1, 1, 1)
        Mc  = torch.sigmoid(W_g * self.bn_ch(M2))

        # Spatial Attention: Ms = sigmoid(W_λ · BN_s(M3))  (Eq. 5)
        W_l = (self.w_lambda / (self.w_lambda.sum() + 1e-6)).view(1, -1, 1, 1)
        Ms  = torch.sigmoid(W_l * self.bn_sp(M3))

        # Fusion: M_out = ReLU(BN(Conv3×3(M1 ⊗ Mc ⊗ Ms)))  (Eq. 7)
        fused = M1 * Mc * Ms
        return self.fuse_act(self.fuse_bn(self.fuse_conv(fused)))

    def l1_penalty(self):
        """L1 regularization trên γ và λ (Eq. 6)."""
        return self.penalty * (self.w_gamma.abs().sum()
                               + self.w_lambda.abs().sum())


# ──────────────────────────────────────────────
# 4. GPA MODULE  (Section 3.3)
# ──────────────────────────────────────────────
class GPA(nn.Module):
    """
    Global Perceptual Aggregation module.
    Inspired by ResNeXt + SENet. Cardinality=4, reduction r=16.

    Flow:
      Conv (F_tr) → GAP (Squeeze, Eq.10) →
      4× Excitation branches (Eq.11) →
      Element-wise product s1⊗s2⊗s3⊗s4 (Eq.12) →
      Scale: X̃_c = s_c · μ_c (Eq.13) →
      Residual: GPA = X ⊕ X̃_c (Eq.14)
    """

    def __init__(self, in_c, out_c, cardinality=4, reduction=16):
        super().__init__()
        mid = max(out_c // reduction, 4)

        self.conv = CBS(in_c, out_c, 1)           # F_tr (Eq. 8-9)
        self.gap  = nn.AdaptiveAvgPool2d(1)        # Squeeze (Eq. 10)

        # 4× Excitation branches (Eq. 11): FC→ReLU→FC→Sigmoid
        self.excitations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_c, mid),
                nn.ReLU(inplace=True),
                nn.Linear(mid, out_c),
                nn.Sigmoid(),
            )
            for _ in range(cardinality)
        ])

        # Shortcut connection (Eq. 14)
        self.shortcut = (CBS(in_c, out_c, 1)
                         if in_c != out_c else nn.Identity())

    def forward(self, x):
        identity = self.shortcut(x)
        u        = self.conv(x)                         # F_tr

        # Squeeze (Eq. 10)
        z = self.gap(u).flatten(1)                      # (B, C)

        # 4× Excitation → element-wise product (Eq. 11-12)
        s = self.excitations[0](z)
        for branch in self.excitations[1:]:
            s = s * branch(z)
        s = s.unsqueeze(-1).unsqueeze(-1)               # (B, C, 1, 1)

        # Scale (Eq. 13) + Residual (Eq. 14)
        return identity + u * s


class GPADetectHead(nn.Module):
    """
    GPAdetect: Detection head với GPA module (Section 3.3).
    Thay thế detection head gốc của YOLOv7x.
    Output: (B, num_anchors × (5 + num_classes), H, W)
    """

    def __init__(self, in_c, num_classes, num_anchors=3):
        super().__init__()
        out_c    = num_anchors * (5 + num_classes)
        self.gpa = GPA(in_c, in_c)
        self.pred = nn.Conv2d(in_c, out_c, 1)
        nn.init.normal_(self.pred.weight, 0, 0.01)
        nn.init.constant_(self.pred.bias, 0)

    def forward(self, x):
        return self.pred(self.gpa(x))


# ──────────────────────────────────────────────
# 5. ACCUREIOU LOSS  (Section 3.4)
# ──────────────────────────────────────────────
def accur_eiou_loss(pred, target, thresh=0.01, eps=1e-7):
    """
    AccurEIOU Loss (Eq. 22-23).

    Kết hợp CIOU + EIOU:
    - Khi Δratio > thresh: dùng CIOU aspect ratio term (global adjustment)
    - Khi Δratio ≤ thresh: dùng EIOU edge terms (local fine-tune)

    Args:
        pred, target: (N, 4) tensors in [x1,y1,x2,y2] format
    Returns:
        loss  : scalar
        iou   : mean IoU (for logging)
    """
    px1, py1, px2, py2 = pred[...,0],   pred[...,1],   pred[...,2],   pred[...,3]
    gx1, gy1, gx2, gy2 = target[...,0], target[...,1], target[...,2], target[...,3]

    pw, ph = px2 - px1 + eps, py2 - py1 + eps
    gw, gh = gx2 - gx1 + eps, gy2 - gy1 + eps

    # Intersection
    ix1 = torch.max(px1, gx1);  iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2);  iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    union = pw * ph + gw * gh - inter + eps
    iou   = inter / union

    # Enclosing box
    enc_x1 = torch.min(px1, gx1);  enc_y1 = torch.min(py1, gy1)
    enc_x2 = torch.max(px2, gx2);  enc_y2 = torch.max(py2, gy2)
    wc     = enc_x2 - enc_x1 + eps
    hc     = enc_y2 - enc_y1 + eps
    c2     = wc**2 + hc**2 + eps      # diagonal² of enclosing box

    # Centroid distance
    pcx = (px1 + px2) / 2;  pcy = (py1 + py2) / 2
    gcx = (gx1 + gx2) / 2;  gcy = (gy1 + gy2) / 2
    rho2 = (pcx - gcx)**2 + (pcy - gcy)**2

    # CIOU aspect ratio consistency v (Eq. 17-18)
    v = (4 / math.pi**2) * (
        torch.atan(gw / gh) - torch.atan(pw / ph)
    )**2
    with torch.no_grad():
        alpha = v / ((1 - iou) + v + eps)

    # EIOU edge terms (Eq. 21)
    rho2_w = (pw - gw)**2
    rho2_h = (ph - gh)**2

    # Δratio (Eq. 23)
    delta_ratio = (pw / ph - gw / gh).abs()

    # Chuyển mode: CIOU khi chưa ổn định, EIOU khi đã ổn định
    stable = (delta_ratio <= thresh).float()

    loss = (
        1 - iou
        + rho2 / c2
        + (1 - stable) * alpha * v
        + stable * (rho2_w / (wc**2 + eps)
                    + rho2_h / (hc**2 + eps))
    )
    return loss.mean(), iou.detach().mean()


# ──────────────────────────────────────────────
# 6. BACKBONE  (YOLOv7x-style)
# ──────────────────────────────────────────────
class Backbone(nn.Module):
    """
    YOLOv7x backbone: CSPNet structure với ELAN blocks.
    Output: P3 (stride 8), P4 (stride 16), P5 (stride 32).
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            CBS(3,   32,  3, 1),
            CBS(32,  64,  3, 2),   # /2  → 320
            CBS(64,  64,  3, 1),
        )
        self.stage1 = nn.Sequential(
            CBS(64,  128, 3, 2),   # /2  → 160
            ELAN(128, 256, 64),
        )
        self.stage2 = nn.Sequential(
            MP(256),               # /2  → 80   (P3)
            ELAN(256, 512, 128),
        )
        self.stage3 = nn.Sequential(
            MP(512),               # /2  → 40   (P4)
            ELAN(512, 768, 192),
        )
        self.stage4 = nn.Sequential(
            MP(768),               # /2  → 20   (P5)
            ELAN(768, 1024, 256),
        )

    def forward(self, x):
        x  = self.stem(x)
        x  = self.stage1(x)
        p3 = self.stage2(x)    # (B, 512,  80, 80)
        p4 = self.stage3(p3)   # (B, 768,  40, 40)
        p5 = self.stage4(p4)   # (B, 1024, 20, 20)
        return p3, p4, p5


# ──────────────────────────────────────────────
# 7. NECK  (PAN + ANS modules)
# ──────────────────────────────────────────────
class Neck(nn.Module):
    """
    PAN Neck với ANS tại mỗi feature fusion stage (Fig. 1 của paper).
    ANS được đặt trước mỗi merge operation để suppress background noise.
    """

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # ── Top-down path ──────────────────────────────────────────────
        self.reduce_p5   = CBS(1024, 512, 1)
        self.ans_td_p4   = ANS(512 + 768)             # ANS trước merge P4
        self.merge_td_p4 = ELAN(512 + 768, 512, 128)

        self.reduce_p4   = CBS(512, 256, 1)
        self.ans_td_p3   = ANS(256 + 512)             # ANS trước merge P3
        self.merge_td_p3 = ELAN(256 + 512, 256, 64)

        # ── Bottom-up path ─────────────────────────────────────────────
        self.ans_bu_p4   = ANS(256 + 512)             # ANS trước merge P4 BU
        self.merge_bu_p4 = ELAN(256 + 512, 512, 128)

        self.ans_bu_p5   = ANS(512 + 1024)            # ANS trước merge P5 BU
        self.merge_bu_p5 = ELAN(512 + 1024, 1024, 256)

    def forward(self, p3, p4, p5):
        # ── Top-down ──
        p5r = self.reduce_p5(p5)                       # 512 ch
        cat_p4 = torch.cat([self.up(p5r), p4], 1)      # (512+768) ch
        p4_td  = self.merge_td_p4(self.ans_td_p4(cat_p4))

        p4r    = self.reduce_p4(p4_td)                 # 256 ch
        cat_p3 = torch.cat([self.up(p4r), p3], 1)      # (256+512) ch
        p3_out = self.merge_td_p3(self.ans_td_p3(cat_p3))  # 256 ch, s8

        # ── Bottom-up ──
        cat_p4b = torch.cat([F.avg_pool2d(p3_out, 2), p4_td], 1)
        p4_out  = self.merge_bu_p4(self.ans_bu_p4(cat_p4b))  # 512 ch, s16

        cat_p5b = torch.cat([F.avg_pool2d(p4_out, 2), p5], 1)
        p5_out  = self.merge_bu_p5(self.ans_bu_p5(cat_p5b))  # 1024 ch, s32

        return p3_out, p4_out, p5_out


# ──────────────────────────────────────────────
# 8. NSEC-YOLO FULL MODEL  (Fig. 1)
# ──────────────────────────────────────────────
class NSECYOLO(nn.Module):
    """
    NSEC-YOLO Full Model.

    Input  : (B, 3, 640, 640)
    Output : 3 raw prediction feature maps
      P3: (B, na*(5+nc), 80, 80)  — small lesions
      P4: (B, na*(5+nc), 40, 40)  — medium lesions
      P5: (B, na*(5+nc), 20, 20)  — large lesions
    """

    def __init__(self, num_classes=8, num_anchors=3):
        super().__init__()
        self.nc = num_classes
        self.na = num_anchors

        self.backbone = Backbone()
        self.neck     = Neck()

        # GPAdetect heads (Section 3.3)
        self.head_p3 = GPADetectHead(256,  num_classes, num_anchors)
        self.head_p4 = GPADetectHead(512,  num_classes, num_anchors)
        self.head_p5 = GPADetectHead(1024, num_classes, num_anchors)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head_p3(p3), self.head_p4(p4), self.head_p5(p5)

    def get_ans_modules(self):
        """Trả về danh sách ANS modules để tính L1 penalty."""
        return [m for m in self.modules() if isinstance(m, ANS)]


# ──────────────────────────────────────────────
# 9. LOSS FUNCTION
# ──────────────────────────────────────────────
class NSECLoss(nn.Module):
    """
    NSEC-YOLO Loss (Section 3.4):
      total = λ_box·L_box(AccurEIOU)
            + λ_obj·L_obj(BCE)
            + λ_cls·L_cls(BCE)
            + Σ ANS_L1_penalty   (Eq. 6)
    """

    def __init__(self, num_classes, anchors_cfg, img_size=640):
        super().__init__()
        self.nc       = num_classes
        self.na       = 3
        self.img_size = img_size
        self.anchors  = [
            torch.tensor(a, dtype=torch.float32)
            for a in anchors_cfg
        ]
        self.strides  = [8, 16, 32]
        self.bce      = nn.BCEWithLogitsLoss()

    def forward(self, preds, gt_boxes, gt_labels, ans_modules=None):
        """
        preds     : list of 3 tensors (P3, P4, P5)
        gt_boxes  : list of (N_i, 4) tensors per image [x1y1x2y2 normalized]
        gt_labels : list of (N_i,) tensors per image
        """
        device    = preds[0].device
        loss_box  = torch.zeros(1, device=device)
        loss_obj  = torch.zeros(1, device=device)
        loss_cls  = torch.zeros(1, device=device)
        n_pos     = 0

        for si, (pred, stride) in enumerate(zip(preds, self.strides)):
            B, _, fH, fW = pred.shape
            na           = self.na
            anchors      = self.anchors[si].to(device)   # (na, 2)

            # Reshape: (B, na, fH, fW, 5+nc)
            pred = pred.view(B, na, 5 + self.nc, fH, fW)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()

            # Objectness target
            tobj = torch.zeros(B, na, fH, fW, device=device)

            pred_boxes_all = []
            gt_boxes_all   = []
            pred_cls_all   = []
            gt_cls_all     = []

            for b in range(B):
                boxes  = gt_boxes[b].to(device)    # (N, 4) normalized
                labels = gt_labels[b].to(device)   # (N,)
                if len(boxes) == 0:
                    continue

                # Convert GT boxes → feature map scale
                scale = torch.tensor([fW, fH, fW, fH],
                                     dtype=torch.float32, device=device)
                boxes_fm = boxes * scale           # (N, 4) in fmap coords

                for gi in range(len(boxes_fm)):
                    box = boxes_fm[gi]
                    cx  = int(((box[0] + box[2]) / 2).clamp(0, fW-1))
                    cy  = int(((box[1] + box[3]) / 2).clamp(0, fH-1))

                    # Chọn anchor tốt nhất (IoU với GT w/h)
                    gt_wh   = (box[2:] - box[:2]).unsqueeze(0)   # (1, 2)
                    anch_wh = anchors / stride                     # (na, 2)
                    mn      = torch.min(gt_wh, anch_wh)
                    mx      = torch.max(gt_wh, anch_wh)
                    iou_a   = mn.prod(1) / (mx.prod(1) + 1e-7)
                    best_a  = int(iou_a.argmax())

                    # Mark objectness
                    tobj[b, best_a, cy, cx] = 1.0

                    # Decode predicted box
                    raw          = pred[b, best_a, cy, cx]
                    anch_wh_best = anch_wh[best_a]
                    pred_xy      = torch.sigmoid(raw[:2])
                    pred_wh      = torch.exp(raw[2:4].clamp(-4, 4)) * anch_wh_best
                    pred_x1      = (cx + pred_xy[0] - pred_wh[0] / 2) / fW
                    pred_y1      = (cy + pred_xy[1] - pred_wh[1] / 2) / fH
                    pred_x2      = pred_x1 + pred_wh[0] / fW
                    pred_y2      = pred_y1 + pred_wh[1] / fH

                    pred_boxes_all.append(
                        torch.stack([pred_x1, pred_y1, pred_x2, pred_y2])
                    )
                    gt_boxes_all.append(boxes[gi])

                    # Classification
                    pred_cls_all.append(raw[5:])
                    gt_c = torch.zeros(self.nc, device=device)
                    gt_c[labels[gi]] = 1.0
                    gt_cls_all.append(gt_c)
                    n_pos += 1

            # Objectness loss (BCE over all grid cells)
            loss_obj += self.bce(pred[..., 4], tobj)

            if pred_boxes_all:
                pb = torch.stack(pred_boxes_all)
                gb = torch.stack(gt_boxes_all)
                bl, _ = accur_eiou_loss(pb, gb, thresh=Config.RATIO_THRESH)
                loss_box += bl
                loss_cls += self.bce(
                    torch.stack(pred_cls_all),
                    torch.stack(gt_cls_all),
                )

        # Scale losses
        total = (Config.LAMBDA_BOX * loss_box
                 + Config.LAMBDA_OBJ * loss_obj
                 + Config.LAMBDA_CLS * loss_cls)

        # ANS L1 regularization (Eq. 6)
        if ans_modules:
            for ans in ans_modules:
                total = total + ans.l1_penalty()

        return total, loss_box.detach(), loss_obj.detach(), loss_cls.detach()


# ──────────────────────────────────────────────
# 10. TRAINING
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion,
                    device, scaler=None):
    model.train()
    ans_modules = model.get_ans_modules()
    t_loss = b_loss = o_loss = c_loss = 0.0

    for i, (imgs, boxes, labels, _) in enumerate(loader):
        imgs = imgs.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast(device_type="cuda"):
                preds = model(imgs)
                loss, bl, ol, cl = criterion(preds, boxes, labels, ans_modules)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss, bl, ol, cl = criterion(preds, boxes, labels, ans_modules)
            loss.backward()
            optimizer.step()

        t_loss += loss.item()
        b_loss += bl.item()
        o_loss += ol.item()
        c_loss += cl.item()

        if (i + 1) % 10 == 0:
            print(f"  [Batch {i+1:3d}/{len(loader)}] "
                  f"loss={loss.item():.4f} "
                  f"box={bl.item():.4f} "
                  f"obj={ol.item():.4f} "
                  f"cls={cl.item():.4f}")

    n = len(loader)
    return t_loss/n, b_loss/n, o_loss/n, c_loss/n


def train(cfg, resume_path=None):
    """
    Train NSEC-YOLO.

    Args:
        cfg         : Config object
        resume_path : (str | None) Đường dẫn tới file .pth để resume training.
                      Ví dụ: "C:\\...\\checkpoints_nsec_2\\best_nsec_yolo.pth"
                      Nếu None → train từ đầu.
    """
    cfg.setup()

    # ── Split 880 ảnh bbox thành train/val/test ──────────────────────────
    train_imgs, val_imgs, test_imgs = make_splits(
        cfg.BBOX_CSV, seed=cfg.RANDOM_SEED,
        train_ratio=cfg.TRAIN_RATIO, val_ratio=cfg.VAL_RATIO,
    )

    train_ds = NIHBBoxDataset(cfg.BBOX_CSV, cfg.DATA_ENTRY, cfg.DATA_ROOT,
                               train_imgs, cfg.IMG_SIZE, mode="train")
    val_ds   = NIHBBoxDataset(cfg.BBOX_CSV, cfg.DATA_ENTRY, cfg.DATA_ROOT,
                               val_imgs,   cfg.IMG_SIZE, mode="val")
    test_ds  = NIHBBoxDataset(cfg.BBOX_CSV, cfg.DATA_ENTRY, cfg.DATA_ROOT,
                               test_imgs,  cfg.IMG_SIZE, mode="test")

    pin = (cfg.DEVICE == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=pin, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=pin, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=pin, collate_fn=collate_fn)

    model     = NSECYOLO(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    criterion = NSECLoss(cfg.NUM_CLASSES, cfg.ANCHORS, cfg.IMG_SIZE)

    # SGD optimizer (paper: SGD, momentum=0.937, Nesterov)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.LR,
        momentum=0.937, weight_decay=cfg.WEIGHT_DECAY, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR * 0.01,
    )
    scaler    = torch.amp.GradScaler() if cfg.DEVICE == "cuda" else None
    if scaler:
        print("[GPU] AMP (Automatic Mixed Precision) enabled")

    # ── RESUME: load checkpoint nếu có ───────────────────────────────────
    start_epoch = 1
    best_loss   = float("inf")

    if resume_path and Path(resume_path).exists():
        print(f"\n[Resume] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=cfg.DEVICE)

        # 1. Load model weights
        model.load_state_dict(ckpt["model_state"])
        print("  → Loaded model weights")

        # 2. Load optimizer state (giữ đúng momentum / adaptive lr)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print("  → Loaded optimizer state")
        else:
            print("  → optimizer_state không có trong checkpoint, "
                  "optimizer sẽ reset")

        # 3. Advance scheduler đến đúng vị trí đã train
        saved_epoch = ckpt.get("epoch", 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR * 0.01,
            last_epoch=saved_epoch   # ← chỉ cần thêm tham số này
        )
        print(f"  → Scheduler advanced {saved_epoch} steps")

        # 4. Tiếp tục từ epoch tiếp theo
        start_epoch = saved_epoch + 1
        best_loss   = ckpt.get("loss", float("inf"))

        print(f"  → Resuming from epoch {start_epoch} / {cfg.EPOCHS}  "
              f"(best_loss so far = {best_loss:.4f})\n")
    elif resume_path:
        print(f"[Resume] Warning: '{resume_path}' không tồn tại. "
              f"Bắt đầu train từ đầu.\n")
    else:
        print("[Train] Không có checkpoint → train từ đầu.\n")

    best_ckpt = Path(cfg.SAVE_DIR) / "best_nsec_yolo.pth"

    print(f"[Train] Epochs {start_epoch}→{cfg.EPOCHS} | "
          f"Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}\n")

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        loss, bl, ol, cl = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.DEVICE, scaler
        )
        scheduler.step()

        mem_str = ""
        if cfg.DEVICE == "cuda":
            mem_str = f" | VRAM={torch.cuda.memory_reserved()/1e9:.1f}GB"

        print(f"Epoch {epoch:3d}/{cfg.EPOCHS} | "
              f"Loss={loss:.4f} box={bl:.4f} obj={ol:.4f} cls={cl:.4f} | "
              f"LR={scheduler.get_last_lr()[0]:.6f}{mem_str}")

        # Lưu best model
        if loss < best_loss:
            best_loss = loss
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss":            best_loss,
            }, best_ckpt)
            print(f"  → Saved best model (loss={best_loss:.4f})")

        # Lưu checkpoint định kỳ mỗi 20 epoch
        if epoch % 20 == 0:
            ckpt_path = Path(cfg.SAVE_DIR) / f"epoch_{epoch}.pth"
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss":            loss,
            }, ckpt_path)
            print(f"  → Periodic checkpoint saved: {ckpt_path}")

    return model, test_ds


# ──────────────────────────────────────────────
# 11. INFERENCE + NMS
# ──────────────────────────────────────────────
@torch.no_grad()
def detect(model, image_path, device,
           conf_thresh=0.01, iou_thresh=0.45, img_size=640):
    """
    Inference trên 1 ảnh X-quang.
    Trả về list of dicts: {label, conf, bbox=[x1,y1,x2,y2] trong pixels}
    """
    model.eval()
    img          = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    x     = tf(img).unsqueeze(0).to(device)
    preds = model(x)

    all_boxes, all_scores, all_labels_idx = [], [], []
    strides = [8, 16, 32]

    for si, (pred, stride) in enumerate(zip(preds, strides)):
        anchors = torch.tensor(Config.ANCHORS[si],
                               dtype=torch.float32, device=device)
        B, _, fH, fW = pred.shape
        na   = len(anchors)
        pred = pred.view(B, na, 5 + Config.NUM_CLASSES, fH, fW)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()

        for ai in range(na):
            anch_wh = anchors[ai] / stride
            obj_s   = torch.sigmoid(pred[0, ai, :, :, 4])   # (fH, fW)
            mask    = obj_s > conf_thresh
            if not mask.any():
                continue

            ys, xs = mask.nonzero(as_tuple=True)
            for yi, xi in zip(ys, xs):
                raw      = pred[0, ai, yi, xi]
                pred_xy  = torch.sigmoid(raw[:2])
                pred_wh  = torch.exp(raw[2:4].clamp(-4, 4)) * anch_wh
                obj_conf = torch.sigmoid(raw[4]).item()
                cls_conf, cls_id = torch.sigmoid(raw[5:]).max(0)
                score = obj_conf * cls_conf.item()
                if score < conf_thresh:
                    continue

                cx = (xi + pred_xy[0]) / fW
                cy = (yi + pred_xy[1]) / fH
                w  = pred_wh[0] / fW
                h  = pred_wh[1] / fH

                x1 = (cx - w/2).clamp(0,1).item() * orig_w
                y1 = (cy - h/2).clamp(0,1).item() * orig_h
                x2 = (cx + w/2).clamp(0,1).item() * orig_w
                y2 = (cy + h/2).clamp(0,1).item() * orig_h

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
                all_labels_idx.append(int(cls_id))

    if not all_boxes:
        return []

    boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    keep     = ops.nms(boxes_t, scores_t, iou_thresh)

    return [{
        "label": Config.CLASSES[all_labels_idx[k]],
        "conf":  round(all_scores[k], 4),
        "bbox":  [round(v, 1) for v in all_boxes[k]],
    } for k in keep]


def visualize(image_path, results, gt_boxes=None, save_path=None):
    """
    Vẽ kết quả detection lên ảnh.
    gt_boxes: dict {label: (x,y,w,h)} từ BBox_List_2017 (tùy chọn)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img = np.array(Image.open(image_path).convert("RGB"))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")
    ax.set_title("NSEC-YOLO — NIH ChestX-ray14", fontsize=11)
    ax.axis("off")

    cmap   = plt.cm.tab10(np.linspace(0, 1, Config.NUM_CLASSES))
    colors = {cls: cmap[i][:3] for i, cls in enumerate(Config.CLASSES)}

    # Predicted boxes (solid)
    for det in results:
        x1, y1, x2, y2 = det["bbox"]
        color = colors.get(det["label"], (1,0,0))
        ax.add_patch(patches.Rectangle(
            (x1,y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor="none",
        ))
        ax.text(x1, max(y1-6, 0),
                f"{det['label']} {det['conf']:.2f}",
                fontsize=7, color=color, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, pad=1))

    # Ground-truth boxes (dashed green)
    if gt_boxes:
        for label, (gx, gy, gw, gh) in gt_boxes.items():
            ax.add_patch(patches.Rectangle(
                (gx, gy), gw, gh,
                linewidth=2, edgecolor="lime",
                facecolor="none", linestyle="--",
            ))
            ax.text(gx, gy - 6, f"GT: {label}",
                    fontsize=7, color="lime", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# 12. MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config()

    # ── RESUME PATH ──────────────────────────────────────────────────────
    # Đặt đường dẫn tới checkpoint muốn resume.
    # Đặt thành None nếu muốn train từ đầu.
    RESUME_CKPT = r"C:\FPT\SU26\DPL302m\archive (2)\checkpoints_nsec_2\best_nsec_yolo.pth"

    # ── TRAIN (tiếp tục từ epoch đã lưu) ─────────────────────────────────
    model, test_ds = train(cfg, resume_path=RESUME_CKPT)

    # ── LOAD BEST CHECKPOINT ─────────────────────────────────────────────
    best_ckpt = Path(cfg.SAVE_DIR) / "best_nsec_yolo.pth"
    ckpt      = torch.load(best_ckpt, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(cfg.DEVICE)
    print(f"\nLoaded best model from epoch {ckpt['epoch']} "
          f"(loss={ckpt['loss']:.4f})")

    # ── INFERENCE + VISUALIZATION trên ảnh đầu tiên của test set ─────────
    sample_name = test_ds.image_names[0]
    sample_path = test_ds.path_index[sample_name]
    print(f"\n[Demo] {sample_name}")

    results = detect(model, sample_path, cfg.DEVICE,
                     conf_thresh=0.25, iou_thresh=0.45)

    print("=== Detection Results ===")
    if results:
        for r in results:
            print(f"  {r['label']:<22} conf={r['conf']:.3f}  "
                  f"bbox={r['bbox']}")
    else:
        print("  Không phát hiện lesion nào trên ngưỡng 0.25")

    # Lấy GT bbox của ảnh này để so sánh
    gt_boxes_vis = {
        ann["label"]: (ann["x"], ann["y"], ann["w"], ann["h"])
        for ann in test_ds.ann_dict.get(sample_name, [])
    }

    out_path = Path(cfg.SAVE_DIR) / f"result_{sample_name}"
    visualize(sample_path, results,
              gt_boxes=gt_boxes_vis, save_path=str(out_path))