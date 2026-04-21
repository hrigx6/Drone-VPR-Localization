import faiss
import numpy as np
from pathlib import Path

MODELS_DIR = Path("models")


def build_index(embeddings):
    """
    Build a FAISS flat inner product index.

    IndexFlatIP = exact search using dot product (inner product).
    Since embeddings are L2 normalized, dot product == cosine similarity.
    Higher score = more similar.

    For 951 vectors this is instant.
    For millions of vectors you'd switch to IndexIVFFlat (approximate).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


if __name__ == "__main__":
    print("Loading gallery embeddings...")
    gallery_emb = np.load(MODELS_DIR / "gallery_embeddings.npy")
    gallery_ids = np.load(MODELS_DIR / "gallery_ids.npy")
    print(f"  Shape: {gallery_emb.shape}")
    print(f"  dtype: {gallery_emb.dtype}")

    print("\nBuilding FAISS index...")
    index = build_index(gallery_emb)
    print(f"  Index size: {index.ntotal} vectors")
    print(f"  Dimension : {index.d}")

    print("\nSaving index...")
    faiss.write_index(index, str(MODELS_DIR / "gallery.index"))
    print("  Saved → models/gallery.index")

    print("\nQuick sanity check — querying first gallery vector against itself...")
    test_vec = gallery_emb[0:1].astype(np.float32)
    scores, indices = index.search(test_vec, k=3)
    print(f"  Top-3 indices : {indices[0]}")
    print(f"  Top-3 scores  : {scores[0].round(4)}")
    print(f"  Top-1 ID      : {gallery_ids[indices[0][0]]}")
    print("  (index 0 should be top match with score ~1.0 — same vector)")
