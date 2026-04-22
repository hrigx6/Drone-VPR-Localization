import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from dataloader import FlatImageDataset, get_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS = 4
DATA_ROOT = Path("data/university1652")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def load_dinov2(model_name="dinov2_vits14"):
    print(f"Loading {model_name} on {DEVICE}...")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    ckpt  = Path("models/exp08/dinov2_finetuned.pth")
    if ckpt.exists():
        print(f"  Loading fine-tuned weights from {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    else:
        print("  No checkpoint found, using zero-shot")
    model.eval()
    model.to(DEVICE)
    print(f"  Embedding dim: {model.embed_dim}")
    return model


def extract(model, root_dir, split_name):
    """
    Extract DINOv2 CLS token embeddings for all images in root_dir.
    Returns:
        embeddings : np.array of shape [N, embed_dim], L2 normalized
        ids        : list of N building ID strings
        paths      : list of N image path strings
    """
    dataset = FlatImageDataset(root_dir, transform=get_transform())
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,          # keep order for evaluation
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    all_embeddings = []
    all_ids        = []
    all_paths      = []

    print(f"\nExtracting features: {split_name} ({len(dataset)} images)...")

    with torch.no_grad():
        for imgs, ids, paths in tqdm(loader, desc=split_name):
            imgs = imgs.to(DEVICE)

            # TTA: average over 4 rotations, renormalize
            emb_sum = torch.zeros(imgs.size(0), model.embed_dim, device=DEVICE)
            for angle in [0, 90, 180, 270]:
                rotated = TF.rotate(imgs, angle) if angle > 0 else imgs
                emb_sum += F.normalize(model(rotated), p=2, dim=1)
            embeddings = F.normalize(emb_sum, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend(ids)
            all_paths.extend(paths)

    # stack all batches into one array
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Done. Shape: {all_embeddings.shape}")
    return all_embeddings, all_ids, all_paths


def save(embeddings, ids, paths, name):
    np.save(MODELS_DIR / f"{name}_embeddings.npy", embeddings)
    np.save(MODELS_DIR / f"{name}_ids.npy", np.array(ids))
    np.save(MODELS_DIR / f"{name}_paths.npy", np.array(paths))
    print(f"  Saved {name} → models/{name}_embeddings.npy")


if __name__ == "__main__":
    model = load_dinov2("dinov2_vits14")

    # --- Gallery: satellite images (your reference database) ---
    gallery_emb, gallery_ids, gallery_paths = extract(
        model,
        DATA_ROOT / "test/gallery_satellite",
        "gallery_satellite"
    )
    save(gallery_emb, gallery_ids, gallery_paths, "gallery")

    # --- Queries: drone images (what you want to localize) ---
    query_emb, query_ids, query_paths = extract(
        model,
        DATA_ROOT / "test/query_drone",
        "query_drone"
    )
    save(query_emb, query_ids, query_paths, "query")

    print("\nAll done.")
    print(f"  Gallery: {gallery_emb.shape}")
    print(f"  Query  : {query_emb.shape}")
    print(f"\nNext step: run scripts/build_index.py to build FAISS index")
