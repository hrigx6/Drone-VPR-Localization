"""
Run from project root:
    PYTHONPATH=scripts python scripts/visualize.py
Outputs four plots to results/plots/.
"""
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from math import radians, sin, cos, sqrt, atan2
import faiss

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

EVAL_FILES = {
    "Zero-shot":            RESULTS_DIR / "eval_zeroshot.json",
    "FT v1\n(full, 1e-4)":  RESULTS_DIR / "eval_finetuned.json",
    "FT v2\n(frozen, 1e-5)": RESULTS_DIR / "eval_finetuned_v2.json",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a  = sin(dp/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_img(path, size=224):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size))
        arr = np.array(img)
        # pad to square so all subplots are the same shape
        h, w = arr.shape[:2]
        pad  = np.zeros((size, size, 3), dtype=np.uint8)
        pad[:h, :w] = arr
        return pad
    except Exception:
        return np.zeros((size, size, 3), dtype=np.uint8)


# ── shared assets (loaded once) ───────────────────────────────────────────────

def load_assets():
    print("Loading assets...")
    index         = faiss.read_index(str(MODELS_DIR / "gallery.index"))
    gallery_ids   = np.load(MODELS_DIR / "gallery_ids.npy")
    gallery_paths = np.load(MODELS_DIR / "gallery_paths.npy")
    query_emb     = np.load(MODELS_DIR / "query_embeddings.npy").astype("float32")
    query_ids     = np.load(MODELS_DIR / "query_ids.npy")
    query_paths   = np.load(MODELS_DIR / "query_paths.npy")
    with open(Path("configs") / "gps_index.json") as f:
        gps_index = json.load(f)
    print(f"  Gallery: {index.ntotal}  |  Queries: {len(query_emb)}")
    return index, gallery_ids, gallery_paths, query_emb, query_ids, query_paths, gps_index


def compute_top_k(index, query_emb, k=10):
    print(f"Running FAISS search (top-{k})...")
    _, indices = index.search(query_emb, k)
    return indices   # [N, k]


# ── Plot 1: Recall@k grouped bar chart ───────────────────────────────────────

def plot_recall_bars():
    labels, r1, r5, r10 = [], [], [], []
    for label, path in EVAL_FILES.items():
        if not path.exists():
            print(f"  Missing {path}, skipping")
            continue
        with open(path) as f:
            d = json.load(f)
        labels.append(label)
        r1.append(d["recalls"]["R@1"])
        r5.append(d["recalls"]["R@5"])
        r10.append(d["recalls"]["R@10"])

    x     = np.arange(len(labels))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (vals, label, color) in enumerate(zip([r1, r5, r10], ["R@1", "R@5", "R@10"], colors)):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall@k — Drone→Satellite Cross-View Localization")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    out = PLOTS_DIR / "1_recall_bars.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


# ── Plot 2: Training curves ───────────────────────────────────────────────────

def plot_training_curves():
    log_path = MODELS_DIR / "training_log.json"
    if not log_path.exists():
        print(f"  {log_path} not found, skipping training curves")
        return

    with open(log_path) as f:
        log = json.load(f)

    epochs   = [e["epoch"]    for e in log]
    losses   = [e["loss"]     for e in log]
    hard_pct = [e["hard_pct"] for e in log]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)   # restore right spine for second axis

    l1, = ax1.plot(epochs, losses,   "o-",  color="#2196F3", linewidth=2, label="Loss")
    l2, = ax2.plot(epochs, hard_pct, "s--", color="#F44336", linewidth=2, label="Hard%")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg Loss",  color="#2196F3")
    ax2.set_ylabel("Hard triplets (%)", color="#F44336")
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2.tick_params(axis="y", labelcolor="#F44336")
    ax1.set_xticks(epochs)
    ax1.set_title("Training Dynamics — FT v2 (LR=1e-5, last 2 blocks unfrozen)")

    ax1.legend([l1, l2], ["Loss", "Hard%"], loc="upper right")
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    out = PLOTS_DIR / "2_training_curves.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


# ── Plot 3: GPS error CDF ─────────────────────────────────────────────────────

def plot_gps_cdf(top_k_indices, query_ids, gallery_ids, gps_index):
    # compute per-query top-1 GPS error
    errors_m = []
    for i in range(len(query_ids)):
        true_id = str(query_ids[i])
        pred_id = str(gallery_ids[top_k_indices[i, 0]])
        tg = gps_index.get(true_id)
        pg = gps_index.get(pred_id)
        if tg and pg:
            errors_m.append(haversine_m(tg["lat"], tg["lon"], pg["lat"], pg["lon"]))
        else:
            errors_m.append(np.nan)

    errors_m = np.array(errors_m)
    valid    = errors_m[~np.isnan(errors_m)]
    valid_km = valid / 1000

    sorted_e = np.sort(valid_km)
    cdf_pct  = np.arange(1, len(sorted_e) + 1) / len(errors_m) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_e, cdf_pct, color="#2196F3", linewidth=2)

    # annotate p50 / p75 / p90
    for pct, color, ls in [(50, "#FF9800", "--"), (75, "#4CAF50", "-."), (90, "#F44336", ":")]:
        val = np.percentile(sorted_e, pct)
        # find CDF value at this x
        cdf_at_pct = np.searchsorted(sorted_e, val) / len(errors_m) * 100
        ax.axvline(val, color=color, linestyle=ls, alpha=0.8,
                   label=f"p{pct} = {val:.1f} km  ({cdf_at_pct:.0f}% queries)")

    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("GPS Error (km, symlog scale)")
    ax.set_ylabel("Queries within error (%)")
    ax.set_title("GPS Error CDF — FT v2 Fine-tuned Model")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    out = PLOTS_DIR / "3_gps_cdf.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


# ── Plot 4: Qualitative retrieval grid ───────────────────────────────────────

def plot_retrieval_grid(top_k_indices, query_ids, query_paths,
                        gallery_ids, gallery_paths, n_rows=8, top_k=5):
    true_ids   = np.array([str(qid) for qid in query_ids])
    pred_top1  = np.array([str(gallery_ids[top_k_indices[i, 0]]) for i in range(len(query_ids))])
    correct_at1 = (pred_top1 == true_ids)

    gid_to_path = {str(gid): str(gp) for gid, gp in zip(gallery_ids, gallery_paths)}

    rng = random.Random(42)
    correct_idx = np.where(correct_at1)[0].tolist()
    wrong_idx   = np.where(~correct_at1)[0].tolist()
    selected = (
        rng.sample(correct_idx, min(n_rows // 2, len(correct_idx))) +
        rng.sample(wrong_idx,   min(n_rows - n_rows // 2, len(wrong_idx)))
    )
    rng.shuffle(selected)
    selected = selected[:n_rows]

    cell = 2.4
    fig = plt.figure(figsize=(cell * (top_k + 1), cell * n_rows + 0.6))
    gs  = gridspec.GridSpec(n_rows, top_k + 1, figure=fig, hspace=0.06, wspace=0.04,
                            top=0.96, bottom=0.01, left=0.06, right=0.99)

    for row, qi in enumerate(selected):
        true_id = str(query_ids[qi])

        # query image
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(load_img(str(query_paths[qi])))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(f"ID {true_id}", fontsize=7, rotation=0,
                      labelpad=30, va="center")
        if row == 0:
            ax.set_title("Query\n(drone)", fontsize=8, pad=3)

        # top-k gallery matches
        for col in range(top_k):
            pred_id = str(gallery_ids[top_k_indices[qi, col]])
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(load_img(gid_to_path.get(pred_id, "")))
            ax.set_xticks([])
            ax.set_yticks([])
            color = "#4CAF50" if pred_id == true_id else "#F44336"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)
            if row == 0:
                ax.set_title(f"Top-{col + 1}", fontsize=8, pad=3)

    green_patch = mpatches.Patch(color="#4CAF50", label="Correct match")
    red_patch   = mpatches.Patch(color="#F44336", label="Wrong match")
    fig.legend(handles=[green_patch, red_patch], loc="upper center",
               ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.55, 0.995))
    fig.suptitle("Qualitative Retrieval — Drone Query vs Satellite Top-5",
                 fontsize=10, x=0.55, y=1.015)

    out = PLOTS_DIR / "4_retrieval_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # plots 1 + 2 need no FAISS — run them immediately
    print("\n── Plot 1: Recall bars ─────────────────────────")
    plot_recall_bars()

    print("\n── Plot 2: Training curves ─────────────────────")
    plot_training_curves()

    # plots 3 + 4 need FAISS search
    index, gallery_ids, gallery_paths, query_emb, query_ids, query_paths, gps_index = load_assets()
    top_k_indices = compute_top_k(index, query_emb, k=10)

    print("\n── Plot 3: GPS error CDF ───────────────────────")
    plot_gps_cdf(top_k_indices, query_ids, gallery_ids, gps_index)

    print("\n── Plot 4: Retrieval grid ──────────────────────")
    plot_retrieval_grid(top_k_indices, query_ids, query_paths,
                        gallery_ids, gallery_paths)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
