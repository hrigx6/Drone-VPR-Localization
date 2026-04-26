"""
boston_index.py — Build FAISS flat inner-product index from Boston gallery embeddings.

Usage (from project root):
    PYTHONPATH=scripts python scripts/boston_index.py

Input:  models/boston_gallery_embeddings.npy
Output: models/boston_gallery.index
"""

import sys
import faiss
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build IndexFlatIP (exact cosine search for L2-normalized vectors).
    Higher score = more similar.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


if __name__ == "__main__":
    emb_path   = MODELS_DIR / "boston_gallery_embeddings.npy"
    names_path = MODELS_DIR / "boston_gallery_names.npy"
    index_path = MODELS_DIR / "boston_gallery.index"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {emb_path}. "
            "Run scripts/boston_encoder.py first."
        )

    print("Loading gallery embeddings...")
    gallery_emb   = np.load(emb_path)
    gallery_names = np.load(names_path)
    print(f"  Shape : {gallery_emb.shape}")
    print(f"  dtype : {gallery_emb.dtype}")
    print(f"  Tiles : {len(gallery_names)}")

    print("\nBuilding FAISS IndexFlatIP...")
    index = build_index(gallery_emb)
    print(f"  Index size : {index.ntotal} vectors")
    print(f"  Dimension  : {index.d}")

    print("\nSaving index...")
    faiss.write_index(index, str(index_path))
    print(f"  Saved → {index_path.relative_to(PROJECT_ROOT)}")

    print("\nSanity check — querying first gallery vector against itself...")
    test_vec = gallery_emb[0:1].astype(np.float32)
    scores, indices = index.search(test_vec, k=3)
    print(f"  Top-3 indices : {indices[0]}")
    print(f"  Top-3 scores  : {scores[0].round(4)}")
    print(f"  Top-1 name    : {gallery_names[indices[0][0]]}")
    print(f"  Top-1 score   : {scores[0][0]:.6f}  (should be ~1.0)")

    assert scores[0][0] > 0.999, "Top-1 self-match score below 0.999 — check embeddings"
    print("\nOK — index ready.")
    print(f"\nNext: PYTHONPATH=scripts python scripts/boston_query.py")
