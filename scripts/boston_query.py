"""
boston_query.py — FAISS query interface for the Boston satellite gallery.

Importable API:
    from boston_query import load_boston_index, query_boston, batch_query_boston

Usage (from project root):
    PYTHONPATH=scripts python scripts/boston_query.py
"""

import sys
import json
import faiss
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"


def load_boston_index() -> tuple[faiss.IndexFlatIP, np.ndarray, dict]:
    """
    Load the Boston FAISS index, tile names, and GPS lookup.

    Returns
    -------
    index      : faiss.IndexFlatIP
    names      : np.ndarray[str]  shape [N] — tile stem names
    gps_lookup : dict  {stem: {lat, lon}}
    """
    index_path = MODELS_DIR / "boston_gallery.index"
    names_path = MODELS_DIR / "boston_gallery_names.npy"
    gps_path   = MODELS_DIR / "boston_gps.json"

    for p in (index_path, names_path, gps_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p.relative_to(PROJECT_ROOT)} not found. "
                "Run boston_encoder.py then boston_index.py first."
            )

    index      = faiss.read_index(str(index_path))
    names      = np.load(names_path)
    with open(gps_path) as f:
        gps_lookup: dict = json.load(f)

    return index, names, gps_lookup


def query_boston(
    embedding:  np.ndarray,
    index:      faiss.IndexFlatIP,
    names:      np.ndarray,
    gps_lookup: dict,
    k:          int = 5,
) -> list[dict]:
    """
    Query FAISS index with a single L2-normalized embedding.

    Parameters
    ----------
    embedding : np.ndarray shape [384] — L2-normalized query vector

    Returns
    -------
    List of k dicts, each with:
        rank       : 1 = best match
        tile_name  : tile stem (e.g. 'tile_18_38583_49143')
        score      : cosine similarity (higher = better)
        lat, lon   : GPS centre of matched tile
    """
    q = embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(q, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        name = str(names[idx])
        gps  = gps_lookup.get(name, {})
        results.append({
            "rank"      : rank + 1,
            "tile_name" : name,
            "score"     : float(score),
            "lat"       : gps.get("lat"),
            "lon"       : gps.get("lon"),
        })
    return results


def batch_query_boston(
    embeddings: np.ndarray,
    index:      faiss.IndexFlatIP,
    k:          int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized query for N embeddings at once.

    Parameters
    ----------
    embeddings : np.ndarray shape [N, 384]

    Returns
    -------
    scores  : np.ndarray [N, k]
    indices : np.ndarray [N, k]
    """
    q = embeddings.astype(np.float32)
    scores, indices = index.search(q, k)
    return scores, indices


if __name__ == "__main__":
    print("Loading Boston index...")
    index, names, gps_lookup = load_boston_index()
    print(f"  Index size : {index.ntotal} vectors, dim={index.d}")
    print(f"  GPS entries: {len(gps_lookup)}")

    print("\nTesting with first gallery embedding...")
    gallery_emb = np.load(MODELS_DIR / "boston_gallery_embeddings.npy")
    test_emb    = gallery_emb[0]
    true_name   = str(names[0])

    results = query_boston(test_emb, index, names, gps_lookup, k=5)

    print(f"\nQuery tile : {true_name}")
    print("Top-5 matches:")
    for r in results:
        match = "✓" if r["tile_name"] == true_name else " "
        print(f"  {match} rank {r['rank']}: {r['tile_name']}  "
              f"score={r['score']:.4f}  "
              f"lat={r['lat']}  lon={r['lon']}")

    print("\nBatch query test (first 10 embeddings)...")
    batch_emb               = gallery_emb[:10]
    batch_scores, batch_idx = batch_query_boston(batch_emb, index, k=3)
    print(f"  scores shape  : {batch_scores.shape}")
    print(f"  indices shape : {batch_idx.shape}")
    print(f"  Top-1 scores  : {batch_scores[:, 0].round(4)}")
