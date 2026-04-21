# Collaborator Setup Guide

Complete setup guide to reproduce the project from scratch on a new machine.
Follow these steps in order — do not skip any.

---

## Prerequisites

- Ubuntu 20.04 or 22.04
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060 Laptop)
- NVIDIA drivers installed
- Git installed
- Internet access

---

## Step 1 — Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda --version  # should print conda 24.x.x or higher
```

---

## Step 2 — Set up SSH key for this repo

This project uses a separate GitHub account (hrigx6) isolated from any
other git identity on your machine. Your global git config is not affected.

```bash
# generate a new SSH key for this project
ssh-keygen -t ed25519 -C "vpr-project" -f ~/.ssh/id_vpr

# add SSH host alias
cat >> ~/.ssh/config << 'EOF'

# VPR project account
Host github-vpr
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_vpr
EOF

# copy public key and add to GitHub (hrigx6 account)
cat ~/.ssh/id_vpr.pub
# go to github.com → Settings → SSH keys → New SSH key → paste it

# test connection
ssh -T git@github-vpr
# expected: Hi hrigx6! You've successfully authenticated...
```

---

## Step 3 — Clone the repo

```bash
mkdir -p ~/workspace/vpr
cd ~/workspace/vpr
git clone git@github-vpr:hrigx6/Drone-VPR-Localization.git
cd Drone-VPR-Localization

# set repo-level git identity (does not affect global config)
git config user.name "hrigx6"
git config user.email "hrigsuryawanshi@gmail.com"

# verify — should show both global and local identity
git config --list | grep user
```

---

## Step 4 — Create conda environment

```bash
cd ~/workspace/vpr/Drone-VPR-Localization
conda env create -f environment.yml
conda activate vpr

# verify key packages
python -c "import torch; import faiss; import timm; import cv2; print('all good')"

# verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# should print: CUDA: True
```

If CUDA returns False, reinstall PyTorch with CUDA support:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 5 — Download datasets

### University-1652
Request download access from:
`https://github.com/layumi/University1652-Baseline`

Once downloaded, extract outside the repo:
```bash
mkdir -p ~/workspace/vpr/data
unzip University-Release.zip -d ~/workspace/vpr/data/

# create symlink inside repo
cd ~/workspace/vpr/Drone-VPR-Localization
ln -s ~/workspace/vpr/data/University-Release data/university1652

# verify structure
ls data/university1652/
# should show: readme.txt  test  train
```

### GPS metadata (KML files)
Download the "Latitude and Longitude" KML zip from the University-1652
GitHub README (Google Drive link labeled [Latitude and Longitude]).

Extract to:
```bash
mkdir -p ~/workspace/vpr/data/university1652-gps
unzip <downloaded_file>.zip -d ~/workspace/vpr/data/
```

Parse GPS into lookup JSON:
```bash
# update the KML_DIR path in scripts/parse_gps.py to match your extraction path
PYTHONPATH=scripts python scripts/parse_gps.py
# should print: Parsed: 1652 buildings, Missing: 0
# saves → configs/gps_index.json
```

---

## Step 6 — Extract features (zero-shot baseline)

```bash
conda activate vpr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=scripts python scripts/extract_features.py
# downloads DINOv2 weights on first run (~85MB)
# saves → models/gallery_embeddings.npy, query_embeddings.npy etc.
```

---

## Step 7 — Build FAISS index

```bash
PYTHONPATH=scripts python scripts/build_index.py
# saves → models/gallery.index
```

---

## Step 8 — Run zero-shot evaluation (baseline)

```bash
PYTHONPATH=scripts python scripts/evaluate.py
# saves → results/eval_zeroshot.json
# expected: R@1 ~9%, GPS median ~687km (before fine-tuning)
```

---

## Step 9 — Fine-tune DINOv2

```bash
PYTHONPATH=scripts python scripts/train.py
# trains for 10 epochs, ~1 hour on RTX 4060
# saves → models/dinov2_finetuned.pth
# saves checkpoints every 2 epochs → models/checkpoint_epoch_N.pth
```

To resume if interrupted:
```bash
PYTHONPATH=scripts python scripts/train.py
# automatically detects latest checkpoint and resumes
```

---

## Step 10 — Evaluate fine-tuned model

Re-extract features with fine-tuned weights, rebuild index, evaluate:
```bash
PYTHONPATH=scripts python scripts/extract_features.py
PYTHONPATH=scripts python scripts/build_index.py

# update output filename in evaluate.py from eval_zeroshot to eval_finetuned
PYTHONPATH=scripts python scripts/evaluate.py
# saves → results/eval_finetuned.json
```

---

## Project structure
---

## Project structure
Drone-VPR-Localization/
├── configs/
│   └── gps_index.json        # GPS lookup for all 1652 buildings
├── data/
│   └── university1652 → symlink to University-Release
├── models/                   # gitignored — generated locally
│   ├── gallery_embeddings.npy
│   ├── gallery_ids.npy
│   ├── gallery_paths.npy
│   ├── query_embeddings.npy
│   ├── query_ids.npy
│   ├── query_paths.npy
│   ├── gallery.index
│   └── dinov2_finetuned.pth
├── notebooks/
├── results/
│   ├── eval_zeroshot.json    # R@1=9.34%, GPS=687km
│   └── eval_finetuned.json   # R@1=TBD after training
├── scripts/
│   ├── dataloader.py         # image loading + transforms
│   ├── extract_features.py   # DINOv2 feature extraction
│   ├── build_index.py        # FAISS index builder
│   ├── query.py              # retrieval pipeline
│   ├── evaluate.py           # Recall@k + GPS error metrics
│   ├── train_dataset.py      # triplet dataset for fine-tuning
│   ├── train.py              # fine-tuning script
│   └── parse_gps.py          # KML → GPS JSON parser
├── environment.yml           # conda environment
├── README.md                 # project overview
└── COLLABORATOR_SETUP.md     # this file

---

## Known issues

- ROS Humble conflict: if ROS is installed, prefix all script runs with
  `PYTHONPATH=scripts` to avoid the ROS `scripts` package conflict
- CUDA not available after killed processes: run
  `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` to reset
- Lid close suspends training: edit `/etc/systemd/logind.conf`,
  set `HandleLidSwitch=ignore`, restart `systemd-logind`
- For long training runs use tmux:
  `tmux new-session -s training` → run training → `Ctrl+B D` to detach

---

## Current results

| Model | R@1 | R@5 | R@10 | GPS median |
|---|---|---|---|---|
| Zero-shot DINOv2 | 9.34% | 19.65% | 25.74% | 687km |
| Fine-tuned (v1) | 8.88% | 20.38% | 27.39% | 870km |
| Fine-tuned (v2) | TBD | TBD | TBD | TBD |

