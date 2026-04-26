"""
boston_validate.py — Zero-shot validation of EXP-08 on DJI drone frames vs Boston gallery.

Usage (from project root):
    PYTHONPATH=scripts python scripts/boston_validate.py --pairs dataset/frames/pairs.json
    PYTHONPATH=scripts python scripts/boston_validate.py --pairs dataset/frames/pairs.json --threshold 0.60
"""

import sys
import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, DEVICE
from boston_query import load_boston_index

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RESULTS_DIR   = PROJECT_ROOT / "results"
BATCH_SIZE    = 32
NUM_WORKERS   = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    R    = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def load_model():
    model = build_model("dinov2_vits14")
    ckpt  = PROJECT_ROOT / "models/exp08/dinov2_finetuned.pth"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
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
    """TTA: encode at 0/90/180/270°, average, L2 normalize. Returns [B, 384]."""
    emb_sum = torch.zeros(imgs.size(0), model.backbone.embed_dim, device=DEVICE)
    for angle in [0, 90, 180, 270]:
        rotated = TF.rotate(imgs, angle) if angle > 0 else imgs
        with torch.amp.autocast(DEVICE):
            emb = model.backbone(rotated)
        emb_sum += F.normalize(emb.float(), p=2, dim=1)
    return F.normalize(emb_sum, p=2, dim=1)


class DroneFrameDataset(Dataset):
    def __init__(self, pairs: list[dict], transform):
        self.pairs     = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p         = self.pairs[idx]
        img_path  = PROJECT_ROOT / p["frame_path"]
        img       = Image.open(img_path).convert("RGB")
        return self.transform(img), idx


def print_summary(
    results:   list[dict],
    threshold: float,
    total_in_pairs: int,
) -> None:
    confident = [r for r in results if r["confident"]]
    errors_c  = sorted([r["error_m"] for r in confident])

    uncertain_n = len([r for r in results if not r["confident"]])

    def pct_within(errors, d):
        if not errors:
            return 0.0
        return 100.0 * sum(1 for e in errors if e <= d) / len(errors)

    def percentile(data, p):
        if not data:
            return float("nan")
        idx = max(0, int(math.ceil(len(data) * p / 100)) - 1)
        return data[idx]

    sep = "=" * 56
    print(f"\n{sep}")
    print("  BOSTON ZERO-SHOT VALIDATION SUMMARY")
    print(sep)
    print(f"  Pairs loaded               : {total_in_pairs}")
    print(f"  Frames processed           : {len(results)}")
    print(f"  Confident (score≥{threshold:.2f})   : "
          f"{len(confident)} ({100*len(confident)/max(1,len(results)):.1f}%)")
    print(f"  Uncertain (below threshold): "
          f"{uncertain_n} ({100*uncertain_n/max(1,len(results)):.1f}%)")
    if confident:
        mean_e   = sum(errors_c) / len(errors_c)
        median_e = percentile(errors_c, 50)
        p75_e    = percentile(errors_c, 75)
        p90_e    = percentile(errors_c, 90)
        print(f"\n  GPS error (confident only):")
        print(f"    Median : {median_e:7.1f} m")
        print(f"    Mean   : {mean_e:7.1f} m")
        print(f"    p75    : {p75_e:7.1f} m")
        print(f"    p90    : {p90_e:7.1f} m")
        print(f"\n  Recall @ distance (confident):")
        for d in [5, 10, 20, 50, 100]:
            print(f"    within {d:>3} m : {pct_within(errors_c, d):5.1f}%")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Validate EXP-08 on DJI drone frames against Boston FAISS index."
    )
    parser.add_argument("--pairs",     type=Path, default=PROJECT_ROOT / "dataset/frames/pairs.json",
                        help="Path to pairs.json (default: dataset/frames/pairs.json)")
    parser.add_argument("--threshold", type=float, default=0.58,
                        help="Min top-1 score to mark frame as confident (default: 0.58)")
    args = parser.parse_args()

    if not args.pairs.exists():
        raise FileNotFoundError(f"pairs.json not found: {args.pairs}")

    RESULTS_DIR.mkdir(exist_ok=True)

    # ── load pairs ────────────────────────────────────────────────────────────
    with open(args.pairs) as f:
        pairs: list[dict] = json.load(f)
    print(f"Loaded {len(pairs)} frames from {args.pairs}")

    # ── load model & index ────────────────────────────────────────────────────
    print("\nLoading EXP-08 model...")
    model = load_model()

    print("\nLoading Boston FAISS index...")
    index, names, gps_lookup = load_boston_index()
    print(f"  Index size : {index.ntotal} vectors")

    # ── encode frames ─────────────────────────────────────────────────────────
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = DroneFrameDataset(pairs, transform)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    print(f"\nEncoding {len(pairs)} drone frames with TTA...")

    # collect (embedding, original_idx) in dataset order
    all_embeddings = np.zeros((len(pairs), model.backbone.embed_dim), dtype=np.float32)
    all_order      = []

    for imgs, idxs in tqdm(loader, desc="Encoding"):
        imgs = imgs.to(DEVICE)
        embs = tta_encode(model, imgs).cpu().numpy()
        for emb, idx in zip(embs, idxs.tolist()):
            all_embeddings[idx] = emb
            all_order.append(idx)

    # ── batch query ───────────────────────────────────────────────────────────
    print(f"\nQuerying Boston index (k=2)...")
    q_float              = all_embeddings.astype(np.float32)
    scores_all, idx_all  = index.search(q_float, k=2)   # k=2 for margin

    # ── build per-frame results ───────────────────────────────────────────────
    results: list[dict] = []

    for i, pair in enumerate(pairs):
        top1_idx   = idx_all[i, 0]
        top2_idx   = idx_all[i, 1]
        top1_score = float(scores_all[i, 0])
        top2_score = float(scores_all[i, 1])
        margin     = top1_score - top2_score

        top1_name  = str(names[top1_idx])
        top1_gps   = gps_lookup.get(top1_name, {})
        pred_lat   = top1_gps.get("lat")
        pred_lon   = top1_gps.get("lon")

        true_lat = pair["lat"]
        true_lon = pair["lon"]

        if pred_lat is not None and pred_lon is not None:
            error_m = haversine_meters(true_lat, true_lon, pred_lat, pred_lon)
        else:
            error_m = float("nan")

        confident = top1_score >= args.threshold

        results.append({
            "frame_path" : pair["frame_path"],
            "true_lat"   : true_lat,
            "true_lon"   : true_lon,
            "pred_lat"   : pred_lat,
            "pred_lon"   : pred_lon,
            "error_m"    : round(error_m, 2) if not math.isnan(error_m) else None,
            "top1_score" : round(top1_score, 6),
            "top2_score" : round(top2_score, 6),
            "margin"     : round(margin, 6),
            "confident"  : confident,
            "alt"        : pair.get("rel_alt"),
        })

    # ── save results ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "boston_zeroshot.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results → {out_path.relative_to(PROJECT_ROOT)}")

    print_summary(results, args.threshold, len(pairs))


if __name__ == "__main__":
    main()
