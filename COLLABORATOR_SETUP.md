# Collaborator Setup Guide

How to get this project running on a new machine from scratch.
Takes about 20-30 minutes end to end.

---

## Prerequisites

- Ubuntu 20.04, 22.04, or 24.04
- NVIDIA GPU with 8GB+ VRAM (tested: RTX 4060 Laptop)
- NVIDIA drivers installed (`nvidia-smi` should work)
- Git installed

---

## Step 1 — Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda --version
```

---

## Step 2 — SSH key setup (isolated from your other git accounts)

This repo uses a separate GitHub account `hrigx6`. The setup below keeps it
completely isolated — your existing global git identity is untouched.

```bash
# generate dedicated SSH key
ssh-keygen -t ed25519 -C "vpr-project" -f ~/.ssh/id_vpr

# add host alias to ~/.ssh/config
cat >> ~/.ssh/config << 'EOF'

Host github-vpr
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_vpr
EOF

# add public key to GitHub (hrigx6 account)
# Settings → SSH and GPG keys → New SSH key → paste this:
cat ~/.ssh/id_vpr.pub

# test
ssh -T git@github-vpr
# expected: Hi hrigx6! You've successfully authenticated...
```

**To switch back to your other work:** just `cd` to your other project.
The VPR identity only activates inside this repo folder.

---

## Step 3 — Clone and set repo identity

```bash
mkdir -p ~/workspace/vpr
cd ~/workspace/vpr
git clone git@github-vpr:hrigx6/Drone-VPR-Localization.git
cd Drone-VPR-Localization

# repo-level identity (does not affect your global git config)
git config user.name "hrigx6"
git config user.email "hrigsuryawanshi@gmail.com"

# pull model weights (stored via Git LFS)
git lfs pull
```

---

## Step 4 — Create conda environment

```bash
conda env create -f environment.yml
conda activate vpr

# verify
python -c "import torch; import faiss; print('torch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

If CUDA returns False:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 5 — Download University-1652 dataset

Request access at: `https://github.com/layumi/University1652-Baseline`

```bash
mkdir -p ~/workspace/vpr/data
unzip University-Release.zip -d ~/workspace/vpr/data/

# symlink into repo (keeps large data outside git)
cd ~/workspace/vpr/Drone-VPR-Localization
ln -s ~/workspace/vpr/data/University-Release data/university1652

# verify
ls data/university1652/
# should show: readme.txt  test  train
```

Also download the GPS KML files (link labeled [Latitude and Longitude] in the dataset README):
```bash
unzip <kml_file>.zip -d ~/workspace/vpr/data/university1652-gps/
PYTHONPATH=scripts python scripts/parse_gps.py
# saves → configs/gps_index.json
```

---

## Step 6 — Run the pipeline

All scripts require `PYTHONPATH=scripts` prefix due to a ROS/Python conflict.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# feature extraction (uses EXP-08 weights from git lfs)
PYTHONPATH=scripts python scripts/extract_features.py

# build FAISS index
PYTHONPATH=scripts python scripts/build_index.py

# evaluate (should reproduce R@1=87.54%)
PYTHONPATH=scripts python scripts/evaluate.py
```

---

## Step 7 — Boston pipeline (optional, needs Mapbox token)

```bash
# add your Mapbox token
echo "MAPBOX_TOKEN=your_token_here" > .env

# Boston tiles already committed to data/boston/tiles_z18/
# just encode and index them
PYTHONPATH=scripts python scripts/boston_encoder.py
PYTHONPATH=scripts python scripts/boston_index.py
```

---

## What each script does

| Script | Purpose |
|---|---|
| `model.py` | DINOv2WithHead architecture — backbone + projection head |
| `dataloader.py` | Loads University-1652 images with transforms |
| `train_dataset.py` | Triplet dataset with rotation augmentation + hard negative mining |
| `train.py` | Fine-tunes DINOv2 — discriminative LR, warmup, cosine schedule |
| `extract_features.py` | Runs DINOv2 + TTA on all images, saves embeddings |
| `build_index.py` | Builds FAISS IndexFlatIP from gallery embeddings |
| `query.py` | Single/batch query against University-1652 FAISS index |
| `evaluate.py` | Computes Recall@1/5/10, GPS error, threshold analysis |
| `parse_gps.py` | Parses KML files → gps_index.json |
| `visualize.py` | Generates ablation plots and retrieval grid |
| `boston_tile_downloader.py` | Downloads Mapbox satellite tiles for Boston area |
| `boston_encoder.py` | Encodes Boston tiles with EXP-08 model + TTA |
| `boston_index.py` | Builds Boston FAISS index |
| `boston_query.py` | Queries Boston index, returns lat/lon directly |
| `boston_validate.py` | Validates pipeline on DJI footage vs GPS ground truth |

---

## Why we do things this way

**Why freeze the backbone?**
DINOv2 was pretrained on 142M images. Fine-tuning all layers with a high LR
overwrites that knowledge (catastrophic forgetting). Freezing most layers and
only updating the last 4 transformer blocks preserves pretrained features while
adapting to drone-satellite matching. This single decision caused a 7.5×
improvement in Recall@1.

**Why triplet loss over InfoNCE?**
InfoNCE needs large batches (256+) to shine. On our dataset size (37,854 images,
batch=128) triplet loss with hard negative mining provides stronger gradient
signal. InfoNCE consistently underperformed across EXP-05 and EXP-06.

**Why TTA (test-time augmentation)?**
Satellite tiles have no canonical orientation — north can point any direction.
Averaging embeddings from 4 rotations (0/90/180/270°) makes the model
orientation-invariant at inference with zero retraining cost.

**Why PYTHONPATH=scripts?**
ROS Humble installs a `scripts` Python package that conflicts with our local
`scripts/` folder. The prefix forces Python to use our local scripts first.

**Why zoom 18 for Boston tiles?**
At 61m altitude (200ft Boston limit), the DJI Mini 5 Pro covers ~106m×106m
per frame. Zoom 18 tiles are 76m×76m — the closest scale match. This gives
the best retrieval performance. Zoom 20 tiles (19m) are used on-demand for
SuperGlue refinement after retrieval.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA: False` after killed process | `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` |
| `ImportError: cannot import name X from scripts` | Add `PYTHONPATH=scripts` prefix |
| Training killed by screen sleep | `gsettings set org.gnome.desktop.session idle-delay 0` |
| Git push blocked (secret detected) | Move API keys to `.env`, never hardcode |
| Out of memory during training | Reduce `BATCH_SIZE` in `train.py`, enable AMP |
