import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from train_dataset import TripletDroneDataset

EXP_NAME   = "exp08"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = Path(f"models/{EXP_NAME}")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

BATCH_SIZE  = 32
NUM_EPOCHS  = 20
WARMUP_EPOCHS = 3
LR_HIGH     = 1e-5    # blocks 11-12 (last 2) — closer to output
LR_LOW      = 5e-6    # blocks 9-10  (next 2) — deeper, more conservative
MARGIN      = 0.2
NUM_WORKERS = 4
SAVE_EVERY  = 2


def load_dinov2():
    print(f"Loading DINOv2 on {DEVICE}...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.to(DEVICE)
    return model


def triplet_loss(anchor, positive, negative, margin=MARGIN):
    dist_pos = 1 - F.cosine_similarity(anchor, positive)
    dist_neg = 1 - F.cosine_similarity(anchor, negative)
    return F.relu(dist_pos - dist_neg + margin).mean()


def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    total_loss    = 0
    total_batches = 0
    hard_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for anchors, positives, negatives, building_ids in pbar:
        anchors   = anchors.to(DEVICE)
        positives = positives.to(DEVICE)
        negatives = negatives.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            emb_a = F.normalize(model(anchors),   p=2, dim=1)
            emb_p = F.normalize(model(positives), p=2, dim=1)
            emb_n = F.normalize(model(negatives), p=2, dim=1)

            # in-batch HNM — find hardest satellite per anchor
            sim  = emb_a @ emb_p.T
            same = torch.tensor(
                [[a == b for b in building_ids] for a in building_ids],
                device=DEVICE,
            )
            sim  = sim.masked_fill(same, -1.0)
            hard_idx   = sim.argmax(dim=1)
            emb_n_hard = emb_p[hard_idx]

            # 50/50 mix: hard vs random (stabilises early training)
            mask  = torch.rand(emb_n.size(0), device=DEVICE) > 0.5
            emb_n = torch.where(mask.unsqueeze(1), emb_n_hard, emb_n)

            loss = triplet_loss(emb_a, emb_p, emb_n)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item()
        total_batches += 1
        if loss.item() > 0:
            hard_batches += 1

        pbar.set_postfix({
            "loss"  : f"{loss.item():.4f}",
            "hard%" : f"{100*hard_batches/total_batches:.1f}",
        })

    return total_loss / total_batches, 100 * hard_batches / total_batches


def main():
    print(f"Device     : {DEVICE}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Epochs     : {NUM_EPOCHS}  (warmup: {WARMUP_EPOCHS})")
    print(f"LR high/low: {LR_HIGH} / {LR_LOW}")

    print("\nBuilding dataset...")
    dataset = TripletDroneDataset("data/university1652/train")
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Batches per epoch: {len(loader)}")

    model = load_dinov2()

    # freeze all, then selectively unfreeze 4 blocks with discriminative LRs
    for param in model.parameters():
        param.requires_grad = False

    for block in model.blocks[-4:-2]:    # blocks 9-10 — lower LR
        for param in block.parameters():
            param.requires_grad = True

    for block in model.blocks[-2:]:      # blocks 11-12 — higher LR
        for param in block.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    deep_params     = [p for b in model.blocks[-4:-2] for p in b.parameters()]
    shallow_params  = [p for b in model.blocks[-2:]   for p in b.parameters()]

    optimizer = torch.optim.AdamW([
        {"params": deep_params,    "lr": LR_LOW},
        {"params": shallow_params, "lr": LR_HIGH},
    ])
    scaler = torch.amp.GradScaler("cuda")

    warmup   = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )

    start_epoch = 1
    log = []
    latest_ckpt = None

    for e in range(NUM_EPOCHS, 0, -1):
        ckpt = MODELS_DIR / f"checkpoint_epoch_{e}.pth"
        if ckpt.exists():
            latest_ckpt = ckpt
            start_epoch = e + 1
            break

    if latest_ckpt:
        print(f"\nResuming from {latest_ckpt}...")
        model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
    else:
        print("\nStarting fresh training...")

    print(f"Starting from epoch {start_epoch}")
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        avg_loss, hard_pct = train_one_epoch(model, loader, optimizer, scaler, epoch)
        scheduler.step()

        lr_deep    = optimizer.param_groups[0]["lr"]
        lr_shallow = optimizer.param_groups[1]["lr"]
        entry = {
            "epoch"    : epoch,
            "loss"     : round(avg_loss, 6),
            "hard_pct" : round(hard_pct, 1),
            "lr_deep"  : lr_deep,
            "lr_shallow": lr_shallow,
        }
        log.append(entry)
        print(f"Epoch {epoch:2d} | loss={avg_loss:.4f} | hard={hard_pct:.1f}% "
              f"| lr_deep={lr_deep:.2e} | lr_shallow={lr_shallow:.2e}")

        if epoch % SAVE_EVERY == 0:
            ckpt = MODELS_DIR / f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  Checkpoint saved → {ckpt}")

    torch.save(model.state_dict(), MODELS_DIR / "dinov2_finetuned.pth")
    print(f"\nFinal model saved → {MODELS_DIR}/dinov2_finetuned.pth")

    with open(MODELS_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved → {MODELS_DIR}/training_log.json")


if __name__ == "__main__":
    main()
