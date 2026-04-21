import faiss
import json
import numpy as np
from pathlib import Path

MODELS_DIR  = Path("models")
CONFIGS_DIR = Path("configs")


def load_index():
    index        = faiss.read_index(str(MODELS_DIR / "gallery.index"))
    gallery_ids  = np.load(MODELS_DIR / "gallery_ids.npy")
    gallery_paths= np.load(MODELS_DIR / "gallery_paths.npy")
    with open(CONFIGS_DIR / "gps_index.json") as f:
        gps_index = json.load(f)
    return index, gallery_ids, gallery_paths, gps_index


def query(embedding, index, gallery_ids, gallery_paths, gps_index, k=5):
    """
    Query FAISS index with a single embedding.

    Args:
        embedding   : np.array shape [384] — L2 normalized query vector
        k           : number of top matches to return

    Returns list of dicts, each with:
        rank        : 1 = best match
        building_id : e.g. '0234'
        score       : cosine similarity (higher = better)
        lat, lon    : GPS coordinates
        img_path    : path to matched satellite image
    """
    # FAISS expects shape [1, dim]
    q = embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(q, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        building_id = str(gallery_ids[idx])
        gps         = gps_index.get(building_id, {})
        results.append({
            "rank"        : rank + 1,
            "building_id" : building_id,
            "score"       : float(score),
            "lat"         : gps.get("lat"),
            "lon"         : gps.get("lon"),
            "img_path"    : str(gallery_paths[idx]),
        })
    return results


def batch_query(embeddings, index, gallery_ids, gallery_paths, gps_index, k=5):
    """
    Query FAISS index with many embeddings at once.
    Much faster than calling query() in a loop.

    Args:
        embeddings : np.array shape [N, 384]

    Returns:
        all_scores  : np.array [N, k]
        all_indices : np.array [N, k]
    """
    q = embeddings.astype(np.float32)
    all_scores, all_indices = index.search(q, k)
    return all_scores, all_indices


if __name__ == "__main__":
    print("Loading index and assets...")
    index, gallery_ids, gallery_paths, gps_index = load_index()
    print(f"  Index size : {index.ntotal} vectors")

    print("\nLoading first query embedding...")
    query_emb  = np.load(MODELS_DIR / "query_embeddings.npy")
    query_ids  = np.load(MODELS_DIR / "query_ids.npy")

    # test single query
    test_emb   = query_emb[0]
    true_id    = str(query_ids[0])
    results    = query(test_emb, index, gallery_ids, gallery_paths, gps_index, k=5)

    print(f"\nQuery building ID : {true_id}")
    print(f"Top-5 matches:")
    for r in results:
        match = "✓" if r["building_id"] == true_id else "✗"
        print(f"  {match} rank {r['rank']}: building {r['building_id']}  "
              f"score={r['score']:.4f}  "
              f"lat={r['lat']:.4f}  lon={r['lon']:.4f}")
