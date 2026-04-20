# Drone VPR Localization

GPS-denied drone localization using Visual Place Recognition (VPR).
Matches drone camera frames against a georeferenced satellite image database
using DINOv2 embeddings + FAISS retrieval.

## Project structure
Drone-VPR-Localization/
├── configs/          # config files (encoder, FAISS params etc)
├── data/             # dataset — not committed, download separately
├── models/           # saved embeddings and FAISS index
├── notebooks/        # demo and exploration notebooks
├── results/          # evaluation outputs, plots
├── scripts/          # all runnable Python scripts
└── environment.yml   # conda environment — single source of truth

## Setting up on a new machine (without affecting other work)

### 1. Generate a separate SSH key for this project

```bash
ssh-keygen -t ed25519 -C "vpr-project" -f ~/.ssh/id_vpr
```

### 2. Add SSH host alias

```bash
cat >> ~/.ssh/config << 'SSHEOF'

# VPR project account
Host github-vpr
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_vpr
SSHEOF
```

### 3. Add the public key to GitHub (hrigx6 account)

```bash
cat ~/.ssh/id_vpr.pub
```

Go to github.com → Settings → SSH and GPG keys → New SSH key → paste it.

### 4. Test connection

```bash
ssh -T git@github-vpr
# Expected: Hi hrigx6! You've successfully authenticated...
```

### 5. Clone the repo

```bash
mkdir -p ~/workspace/vpr
cd ~/workspace/vpr
git clone git@github-vpr:hrigx6/Drone-VPR-Localization.git
cd Drone-VPR-Localization
```

### 6. Set repo-level git identity (does not affect global config)

```bash
git config user.name "hrigx6"
git config user.email "hrigsuryawanshi@gmail.com"
```

### 7. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate vpr
```

### 8. Verify setup

```bash
python -c "import torch; import faiss; print('Setup complete')"
```

## Switching back to your other work

Nothing to undo. Just `cd` to your other project — your global git identity
and SSH key are completely untouched. The vpr identity only activates inside
this repo folder.

To switch conda env back:
```bash
conda activate <your-other-env>
```

## Dataset

Download University-1652 from:
https://github.com/layumi/University1652-Baseline

Place it at `data/university1652/`. This folder is gitignored — each machine
needs its own local copy.

## Roadmap

- [ ] Phase 1 — Data: dataloader for University-1652
- [ ] Phase 2 — Features: DINOv2 embedding extraction + FAISS index
- [ ] Phase 3 — Query: retrieval pipeline + pose estimation
- [ ] Phase 4 — Eval: Recall@1, Recall@5, median localization error
- [ ] Phase 5 — Real world: DJI flight validation
