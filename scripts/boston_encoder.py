"""
boston_encoder.py — Encode all Boston z18 satellite tiles with EXP-08 + TTA.

Usage (from project root):
    PYTHONPATH=scripts python scripts/boston_encoder.py

Outputs:
    models/boston_gallery_embeddings.npy  [N, 384]
    models/boston_gallery_names.npy       [N] tile stems
    models/boston_gps.json                {stem: {lat, lon}}
"""

import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, DEVICE

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data/boston/tiles_z18"
MODELS_DIR    = PROJECT_ROOT / "models"
BATCH_SIZE    = 32
NUM_WORKERS   = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model():
    """Load DINOv2WithHead, then graft EXP-08 backbone weights."""
    model = build_model("dinov2_vits14")  # pretrained backbone + random head
    ckpt  = PROJECT_ROOT / "models/exp08/dinov2_finetuned.pth"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        # exp08 checkpoint is backbone-only (no backbone.* prefix)
        if any(k.startswith("backbone.") for k in state):
            model.load_state_dict(state)
        else:
            model.backbone.load_state_dict(state)
        print(f"  Loaded EXP-08 backbone from {ckpt.relative_to(PROJECT_ROOT)}")
    else:
        print("  WARNING: EXP-08 checkpoint not found, using pretrained backbone")
    model.eval()
    return model


@torch.no_grad()
def tta_encode(model, imgs: torch.Tensor) -> torch.Tensor:
    """
    TTA: encode at 0/90/180/270°, average, L2 normalize.
    imgs  : [B, 3, H, W] on DEVICE
    returns: [B, 384] L2-normalized float32
    """
    emb_sum = torch.zeros(imgs.size(0), model.backbone.embed_dim, device=DEVICE)
    for angle in [0, 90, 180, 270]:
        rotated = TF.rotate(imgs, angle) if angle > 0 else imgs
        with torch.amp.autocast(DEVICE):
            emb = model.backbone(rotated)
        emb_sum += F.normalize(emb.float(), p=2, dim=1)
    return F.normalize(emb_sum, p=2, dim=1)


class TileDataset(Dataset):
    def __init__(self, tile_paths: list[Path], transform):
        self.tile_paths = tile_paths
        self.transform  = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        img = Image.open(self.tile_paths[idx]).convert("RGB")
        return self.transform(img)


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # ── metadata ──────────────────────────────────────────────────────────────
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {meta_path}. "
            "Run scripts/boston_tile_downloader.py first."
        )
    with open(meta_path) as f:
        metadata: dict = json.load(f)

    # ── tile paths ────────────────────────────────────────────────────────────
    tile_paths = sorted(DATA_DIR.glob("*.jpg"))
    print(f"Tiles found : {len(tile_paths)}  in {DATA_DIR.relative_to(PROJECT_ROOT)}")
    if not tile_paths:
        raise ValueError("No .jpg tiles found. Run boston_tile_downloader.py first.")

    stem_list = [p.stem for p in tile_paths]

    # GPS lookup from metadata (keyed by filename stem)
    gps_from_meta = {
        Path(fname).stem: {"lat": info["lat"], "lon": info["lon"]}
        for fname, info in metadata.items()
    }

    # ── model ─────────────────────────────────────────────────────────────────
    print("\nLoading EXP-08 model...")
    model = load_model()

    # ── dataloader ────────────────────────────────────────────────────────────
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = TileDataset(tile_paths, transform)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    print(f"\nEncoding {len(tile_paths)} tiles  "
          f"(batch={BATCH_SIZE}, TTA=4×, device={DEVICE})...")

    all_embeddings: list[np.ndarray] = []

    for imgs in tqdm(loader, desc="Encoding tiles"):
        imgs = imgs.to(DEVICE)
        embs = tta_encode(model, imgs).cpu().numpy()
        all_embeddings.append(embs)

    all_embeddings_np = np.concatenate(all_embeddings, axis=0)  # [N, 384]

    gps_out = {stem: gps_from_meta.get(stem, {"lat": None, "lon": None})
               for stem in stem_list}

    # ── save ──────────────────────────────────────────────────────────────────
    emb_path   = MODELS_DIR / "boston_gallery_embeddings.npy"
    names_path = MODELS_DIR / "boston_gallery_names.npy"
    gps_path   = MODELS_DIR / "boston_gps.json"

    np.save(emb_path,   all_embeddings_np)
    np.save(names_path, np.array(stem_list))
    with open(gps_path, "w") as f:
        json.dump(gps_out, f, indent=2)

    print(f"\nSaved:")
    print(f"  {emb_path.relative_to(PROJECT_ROOT)}   shape={all_embeddings_np.shape}")
    print(f"  {names_path.relative_to(PROJECT_ROOT)}  {len(stem_list)} names")
    print(f"  {gps_path.relative_to(PROJECT_ROOT)}    {len(gps_out)} GPS entries")
    print(f"\nNext: PYTHONPATH=scripts python scripts/boston_index.py")


if __name__ == "__main__":
    main()
