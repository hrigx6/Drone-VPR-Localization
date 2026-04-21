import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from train_dataset import TripletDroneDataset

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE  = 32
NUM_EPOCHS  = 5
LR          = 1e-5
MARGIN      = 0.3
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
    loss = F.relu(dist_pos - dist_neg + margin)
    return loss.mean()


def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    total_loss    = 0
    total_batches = 0
    hard_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for anchors, positives, negatives, _ in pbar:
        anchors   = anchors.to(DEVICE)
        positives = positives.to(DEVICE)
        negatives = negatives.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            emb_a = F.normalize(model(anchors),   p=2, dim=1)
            emb_p = F.normalize(model(positives), p=2, dim=1)
            emb_n = F.normalize(model(negatives), p=2, dim=1)
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
            "hard%" : f"{100*hard_batches/total_batches:.1f}"
        })

    return total_loss / total_batches, 100 * hard_batches / total_batches


def main():
    print(f"Device     : {DEVICE}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Epochs     : {NUM_EPOCHS}")
    print(f"LR         : {LR}")

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

    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    start_epoch = 1
    log = []
    latest_ckpt=None

    for e in range(NUM_EPOCHS, 0, -1):
        ckpt = MODELS_DIR / f"checkpoint_epoch_{e}.pth"
        if ckpt.exists():
            latest_ckpt = ckpt
            start_epoch = e + 1
            break

    if latest_ckpt:
        print(f"\nResuming from {latest_ckpt}...")
        model.load_state_dict(torch.load(latest_ckpt))
    else:
        print("\nStarting fresh training...")

    print(f"Starting from epoch {start_epoch}")
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        avg_loss, hard_pct = train_one_epoch(model, loader, optimizer, scaler, epoch)
        scheduler.step(avg_loss)

        lr = optimizer.param_groups[0]["lr"]
        entry = {
            "epoch"    : epoch,
            "loss"     : round(avg_loss, 6),
            "hard_pct" : round(hard_pct, 1),
            "lr"       : lr,
        }
        log.append(entry)
        print(f"Epoch {epoch:2d} | loss={avg_loss:.4f} | hard={hard_pct:.1f}% | lr={lr:.2e}")

        if epoch % SAVE_EVERY == 0:
            ckpt = MODELS_DIR / f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  Checkpoint saved → {ckpt}")

    torch.save(model.state_dict(), MODELS_DIR / "dinov2_finetuned.pth")
    print(f"\nFinal model saved → models/dinov2_finetuned.pth")

    with open(MODELS_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved → models/training_log.json")


if __name__ == "__main__":
    main()
