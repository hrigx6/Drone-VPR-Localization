"""
boston_finetune.py — Fine-tune EXP-08 on DJI drone frames paired with Mapbox satellite tiles.

Pulls Mapbox satellite tiles at each GPS coordinate in pairs.json, then fine-tunes
EXP-08 backbone blocks 9-12 with triplet loss (drone anchor, sat positive, sat negative).

Usage (from project root):
    PYTHONPATH=scripts python scripts/boston_finetune.py --pairs dataset/frames/pairs.json
    PYTHONPATH=scripts python scripts/boston_finetune.py --pairs dataset/frames/pairs.json \\
        --epochs 10 --lr 3e-6

Requires MAPBOX_TOKEN in a .env file at the project root.
Output: models/exp09/dinov2_finetuned.pth + models/exp09/training_log.json
"""

import os
import sys
import json
import math
import time
import random
import argparse
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, DEVICE

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
FINETUNE_DIR   = PROJECT_ROOT / "data/boston/finetune_tiles"
EXP09_DIR      = PROJECT_ROOT / "models/exp09"

BATCH_SIZE     = 32
NUM_WORKERS    = 4
MARGIN         = 0.2
SAVE_EVERY     = 2
MIN_NEG_DIST_M = 100.0      # negative GPS must be > this far from anchor

IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]


# ── geo helpers ────────────────────────────────────────────────────────────────

def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R    = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert GPS to Web Mercator tile (x, y)."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


# ── Mapbox tile download ────────────────────────────────────────────────────────

def download_tile(
    lat:      float,
    lon:      float,
    zoom:     int,
    out_path: Path,
    token:    str,
) -> bool:
    """Download Mapbox satellite tile covering (lat, lon) at given zoom."""
    x, y = lat_lon_to_tile(lat, lon, zoom)
    url  = (
        f"https://api.mapbox.com/v4/mapbox.satellite"
        f"/{zoom}/{x}/{y}@2x.jpg90"
        f"?access_token={token}"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            out_path.write_bytes(r.content)
            return True
        print(f"  HTTP {r.status_code} for tile ({x},{y})")
        return False
    except requests.RequestException as e:
        print(f"  Request error: {e}")
        return False


def download_finetune_tiles(pairs: list[dict], token: str, zoom: int = 18) -> list[dict]:
    """
    Download a Mapbox satellite tile for each pair.
    Skips tiles that already exist on disk.

    Adds 'sat_path' key to each pair dict (in-place) for pairs with a tile.
    Returns the updated pairs list (only entries where tile is available).
    """
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped    = 0
    failed     = 0
    valid_pairs: list[dict] = []

    print(f"Downloading satellite tiles for {len(pairs)} GPS points...")
    for pair in tqdm(pairs, desc="Downloading tiles"):
        lat, lon = pair["lat"], pair["lon"]
        fname    = f"tile_{lat:.6f}_{lon:.6f}.jpg"
        out_path = FINETUNE_DIR / fname

        if out_path.exists():
            skipped += 1
        else:
            ok = download_tile(lat, lon, zoom, out_path, token)
            if ok:
                downloaded += 1
                time.sleep(0.1)   # Mapbox rate limit: 600 req/min
            else:
                failed += 1
                continue

        pair["sat_path"] = str(out_path)
        valid_pairs.append(pair)

    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped} (already on disk)")
    print(f"  Failed     : {failed}")
    print(f"  Valid pairs: {len(valid_pairs)}")
    return valid_pairs


# ── augmentation ───────────────────────────────────────────────────────────────

class RandomRot90:
    """Rotate image by a random multiple of 90°."""
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle)


def get_drone_transform(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_sat_transform(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        RandomRot90(),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── triplet dataset ────────────────────────────────────────────────────────────

class BostonTripletDataset(Dataset):
    """
    Triplets for fine-tuning:
        anchor   = DJI drone frame (ColorJitter only)
        positive = Mapbox satellite tile at same GPS
        negative = Mapbox satellite tile from a different GPS (> MIN_NEG_DIST_M away)
    """
    def __init__(self, pairs: list[dict]):
        self.pairs      = pairs
        self.drone_tfm  = get_drone_transform()
        self.sat_tfm    = get_sat_transform()

        # Pre-compute valid negatives for each sample (>MIN_NEG_DIST_M away)
        # Stored as index lists; sampled randomly at __getitem__ time
        lats = np.array([p["lat"] for p in pairs])
        lons = np.array([p["lon"] for p in pairs])

        self.neg_pool: list[list[int]] = []
        for i, p in enumerate(pairs):
            valid = [
                j for j in range(len(pairs))
                if j != i and haversine_meters(p["lat"], p["lon"], lats[j], lons[j]) > MIN_NEG_DIST_M
            ]
            self.neg_pool.append(valid)

        n_no_neg = sum(1 for pool in self.neg_pool if not pool)
        if n_no_neg > 0:
            print(f"  WARNING: {n_no_neg} pairs have no valid negatives > {MIN_NEG_DIST_M}m "
                  "— they will be skipped.")

        print(f"  BostonTripletDataset: {len(pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]

        anchor   = self._load(PROJECT_ROOT / pair["frame_path"], self.drone_tfm)
        positive = self._load(pair["sat_path"],                  self.sat_tfm)

        neg_pool = self.neg_pool[idx]
        if not neg_pool:
            # fallback: use any other pair's tile
            neg_idx = (idx + 1) % len(self.pairs)
        else:
            neg_idx = random.choice(neg_pool)

        negative = self._load(self.pairs[neg_idx]["sat_path"], self.sat_tfm)

        return anchor, positive, negative

    @staticmethod
    def _load(path, tfm):
        return tfm(Image.open(path).convert("RGB"))


# ── model setup ────────────────────────────────────────────────────────────────

def load_exp08():
    """Load DINOv2WithHead with EXP-08 fine-tuned backbone."""
    model = build_model("dinov2_vits14")
    ckpt  = PROJECT_ROOT / "models/exp08/dinov2_finetuned.pth"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        if any(k.startswith("backbone.") for k in state):
            model.load_state_dict(state)
        else:
            model.backbone.load_state_dict(state)
        print(f"  Loaded EXP-08 from {ckpt.relative_to(PROJECT_ROOT)}")
    else:
        print("  WARNING: EXP-08 checkpoint not found, using pretrained backbone")
    return model


def setup_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Freeze all backbone params; selectively unfreeze last 4 blocks
    with discriminative LR (blocks[-4:-2] → lr, blocks[-2:] → 2×lr).
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    # blocks[-4:-2] = indices 8,9  — deeper, lower LR
    for block in model.backbone.blocks[-4:-2]:
        for param in block.parameters():
            param.requires_grad = True

    # blocks[-2:] = indices 10,11 — shallower, higher LR
    for block in model.backbone.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.backbone.parameters())
    print(f"  Trainable backbone params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    deep_params    = [p for b in model.backbone.blocks[-4:-2] for p in b.parameters()]
    shallow_params = [p for b in model.backbone.blocks[-2:]   for p in b.parameters()]

    return torch.optim.AdamW([
        {"params": deep_params,    "lr": lr},
        {"params": shallow_params, "lr": lr * 2.0},
    ])


def triplet_loss(
    anchor:   torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin:   float = MARGIN,
) -> torch.Tensor:
    dist_pos = 1.0 - F.cosine_similarity(anchor, positive)
    dist_neg = 1.0 - F.cosine_similarity(anchor, negative)
    return F.relu(dist_pos - dist_neg + margin).mean()


# ── training ───────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     torch.nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.cuda.amp.GradScaler,
    epoch:     int,
) -> tuple[float, float]:
    model.train()
    total_loss    = 0.0
    total_batches = 0
    hard_batches  = 0

    for anchors, positives, negatives in tqdm(loader, desc=f"Epoch {epoch}"):
        anchors   = anchors.to(DEVICE)
        positives = positives.to(DEVICE)
        negatives = negatives.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(DEVICE):
            emb_a = F.normalize(model.backbone(anchors),   p=2, dim=1)
            emb_p = F.normalize(model.backbone(positives), p=2, dim=1)
            emb_n = F.normalize(model.backbone(negatives), p=2, dim=1)

            # in-batch hard negative mining (hardest satellite per anchor)
            sim      = emb_a @ emb_p.T                       # [B, B]
            eye_mask = torch.eye(sim.size(0), dtype=torch.bool, device=DEVICE)
            sim      = sim.masked_fill(eye_mask, -1.0)
            hard_idx = sim.argmax(dim=1)
            emb_n_hard = emb_p[hard_idx]

            # 50/50 mix: in-batch hard vs dataset random
            mask   = torch.rand(emb_n.size(0), device=DEVICE) > 0.5
            emb_n  = torch.where(mask.unsqueeze(1), emb_n_hard, emb_n)

            loss = triplet_loss(emb_a, emb_p, emb_n)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item()
        total_batches += 1
        if loss.item() > 0:
            hard_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    hard_pct = 100.0 * hard_batches / max(1, total_batches)
    return avg_loss, hard_pct


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune EXP-08 on DJI + Mapbox pairs → EXP-09."
    )
    parser.add_argument("--pairs",  type=Path, default=PROJECT_ROOT / "dataset/frames/pairs.json",
                        help="Path to pairs.json")
    parser.add_argument("--epochs", type=int,   default=5,  help="Training epochs (default: 5)")
    parser.add_argument("--lr",     type=float, default=5e-6, help="Base learning rate (default: 5e-6)")
    args = parser.parse_args()

    if not args.pairs.exists():
        raise FileNotFoundError(f"pairs.json not found: {args.pairs}")

    # ── load MAPBOX_TOKEN ─────────────────────────────────────────────────────
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("MAPBOX_TOKEN")
    if not token:
        raise EnvironmentError(
            "MAPBOX_TOKEN not set. Add MAPBOX_TOKEN=<your_token> to .env "
            "at the project root."
        )

    EXP09_DIR.mkdir(parents=True, exist_ok=True)

    # ── load pairs ────────────────────────────────────────────────────────────
    with open(args.pairs) as f:
        pairs: list[dict] = json.load(f)
    print(f"Loaded {len(pairs)} frames from {args.pairs.relative_to(PROJECT_ROOT)}")

    # ── download satellite tiles ──────────────────────────────────────────────
    pairs = download_finetune_tiles(pairs, token, zoom=18)
    if not pairs:
        raise RuntimeError("No valid pairs after tile download — cannot fine-tune.")

    # ── dataset / loader ──────────────────────────────────────────────────────
    print("\nBuilding triplet dataset...")
    dataset = BostonTripletDataset(pairs)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    print(f"  Batches per epoch: {len(loader)}")

    # ── model + optimizer ─────────────────────────────────────────────────────
    print("\nLoading EXP-08 model...")
    model = load_exp08().to(DEVICE)

    print("\nSetting up optimizer (freeze blocks 0-7, train 8-11)...")
    optimizer = setup_optimizer(model, args.lr)
    scaler    = torch.amp.GradScaler(DEVICE)

    warmup  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, args.epochs // 3)
    )
    cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.epochs // 3)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[max(1, args.epochs // 3)],
    )

    print(f"\nFine-tuning EXP-08 → EXP-09")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {args.epochs}")
    print(f"  LR     : {args.lr:.1e} / {args.lr*2:.1e}  (deep / shallow)")
    print(f"  Pairs  : {len(pairs)}")

    log: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        avg_loss, hard_pct = train_one_epoch(model, loader, optimizer, scaler, epoch)
        scheduler.step()

        lr_deep    = optimizer.param_groups[0]["lr"]
        lr_shallow = optimizer.param_groups[1]["lr"]
        entry = {
            "epoch"     : epoch,
            "loss"      : round(avg_loss, 6),
            "hard_pct"  : round(hard_pct, 1),
            "lr_deep"   : lr_deep,
            "lr_shallow": lr_shallow,
        }
        log.append(entry)
        print(f"Epoch {epoch:2d} | loss={avg_loss:.4f} | hard={hard_pct:.1f}%"
              f" | lr={lr_deep:.1e}/{lr_shallow:.1e}")

        if epoch % SAVE_EVERY == 0:
            ckpt = EXP09_DIR / f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.backbone.state_dict(), ckpt)
            print(f"  Checkpoint → {ckpt.relative_to(PROJECT_ROOT)}")

    # ── save final model ──────────────────────────────────────────────────────
    final_ckpt = EXP09_DIR / "dinov2_finetuned.pth"
    torch.save(model.backbone.state_dict(), final_ckpt)
    print(f"\nFinal model → {final_ckpt.relative_to(PROJECT_ROOT)}")

    log_path = EXP09_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log → {log_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
