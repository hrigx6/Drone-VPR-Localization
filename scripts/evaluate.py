import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2
from query import load_index, batch_query

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def haversine_meters(lat1, lon1, lat2, lon2):
    R    = 6371000
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a    = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def evaluate(k_values=[1, 5, 10]):
    print("Loading assets...")
    index, gallery_ids, gallery_paths, gps_index = load_index()

    query_emb  = np.load(MODELS_DIR / "query_embeddings.npy")
    query_ids  = np.load(MODELS_DIR / "query_ids.npy")
    print(f"  Queries  : {len(query_emb)}")
    print(f"  Gallery  : {index.ntotal}")

    max_k = max(k_values)
    print(f"\nRunning batch search (top-{max_k})...")
    all_scores, all_indices = batch_query(
        query_emb, index, gallery_ids, gallery_paths, gps_index, k=max_k
    )

    # ── per-query evaluation ──────────────────────────────────────────────────
    recalls     = {k: 0 for k in k_values}
    gps_errors  = []
    top1_scores = []
    correct_at1 = []
    failed_gps  = 0

    for i in tqdm(range(len(query_emb)), desc="Evaluating"):
        true_id = str(query_ids[i])
        top_ids = [str(gallery_ids[all_indices[i][j]]) for j in range(max_k)]
        score   = float(all_scores[i][0])

        for k in k_values:
            if true_id in top_ids[:k]:
                recalls[k] += 1

        top1_scores.append(score)
        correct_at1.append(top_ids[0] == true_id)

        true_gps = gps_index.get(true_id)
        pred_gps = gps_index.get(top_ids[0])
        if true_gps and pred_gps:
            gps_errors.append(haversine_meters(
                true_gps["lat"], true_gps["lon"],
                pred_gps["lat"], pred_gps["lon"]
            ))
        else:
            failed_gps += 1

    n           = len(query_emb)
    gps_errors  = np.array(gps_errors)
    top1_scores = np.array(top1_scores)
    correct_at1 = np.array(correct_at1)

    # ── standard results ─────────────────────────────────────────────────────
    results = {
        "n_queries"          : n,
        "n_gallery"          : index.ntotal,
        "recalls"            : {f"R@{k}": round(recalls[k]/n*100, 2) for k in k_values},
        "gps_error_median_m" : round(float(np.median(gps_errors)), 1),
        "gps_error_mean_m"   : round(float(np.mean(gps_errors)), 1),
        "gps_error_p75_m"    : round(float(np.percentile(gps_errors, 75)), 1),
        "failed_gps_lookup"  : failed_gps,
    }

    # ── threshold analysis ────────────────────────────────────────────────────
    thresholds = np.arange(0.40, 1.00, 0.02)
    thresh_rows = []

    for t in thresholds:
        mask     = top1_scores >= t
        coverage = mask.sum() / n * 100
        if mask.sum() == 0:
            continue
        precision = correct_at1[mask].mean() * 100
        errs      = gps_errors[mask[:len(gps_errors)]]
        gps_p50   = float(np.median(errs)) if len(errs) else float("nan")
        gps_p75   = float(np.percentile(errs, 75)) if len(errs) else float("nan")
        thresh_rows.append({
            "threshold" : round(float(t), 2),
            "coverage"  : round(coverage, 1),
            "precision" : round(precision, 1),
            "gps_p50_m" : round(gps_p50, 1),
            "gps_p75_m" : round(gps_p75, 1),
        })

    # find lowest threshold that gives >= 95% precision
    recommended = next(
        (r for r in thresh_rows if r["precision"] >= 95.0), None
    )
    results["threshold_analysis"]   = thresh_rows
    results["recommended_threshold"] = recommended

    # ── print ─────────────────────────────────────────────────────────────────
    print("\n" + "="*45)
    print("  EVALUATION RESULTS (fine-tuned DINOv2 v8)")
    print("="*45)
    for k, v in results["recalls"].items():
        print(f"  {k:6s} : {v:.2f}%")
    print(f"  GPS error median : {results['gps_error_median_m']:>10.1f} m")
    print(f"  GPS error mean   : {results['gps_error_mean_m']:>10.1f} m")
    print(f"  GPS error p75    : {results['gps_error_p75_m']:>10.1f} m")
    print("="*45)

    print("\n  THRESHOLD ANALYSIS (top-1 cosine similarity)")
    print(f"  {'Thresh':>6}  {'Coverage':>8}  {'Precision':>9}  {'GPS p50':>8}  {'GPS p75':>8}")
    print("  " + "-"*50)
    for r in thresh_rows:
        marker = " ◄" if recommended and r["threshold"] == recommended["threshold"] else ""
        print(f"  {r['threshold']:>6.2f}  {r['coverage']:>7.1f}%  "
              f"{r['precision']:>8.1f}%  {r['gps_p50_m']:>7.1f}m  "
              f"{r['gps_p75_m']:>7.1f}m{marker}")

    if recommended:
        print(f"\n  Recommended threshold : {recommended['threshold']}")
        print(f"  At this threshold     : {recommended['coverage']:.1f}% of queries answered")
        print(f"                          {recommended['precision']:.1f}% precision")
        print(f"                          {recommended['gps_p75_m']:.1f}m GPS p75")
    print("="*45)

    # ── save ──────────────────────────────────────────────────────────────────
    out = RESULTS_DIR / "eval_finetuned_v8.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")

    return results


if __name__ == "__main__":
    evaluate()
