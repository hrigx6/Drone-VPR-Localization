
---

### EXP-07: InfoNCE + larger model attempt
**Date:** 2026-04-21

**Results:**
R@1: 72.97%

**Observations:** TBD — fill in config details

---

### EXP-08: triplet + 4 unfrozen blocks + warmup + 20 epochs
**Date:** 2026-04-21

**Config:**
lr_backbone:  5e-6 (blocks 9-10) / 1e-5 (blocks 11-12)
epochs:       20
batch_size:   32
frozen:       blocks 0-8, trainable blocks 9-12
warmup:       linear 3 epochs then CosineAnnealingLR
loss:         triplet, 50/50 hard/random negatives
augmentation: satellite: rotation+colorjitter+hflip
drone: colorjitter+zoom crop
TTA:          4 rotations averaged

**Results:**
R@1:        87.54%
R@5:        96.11%
R@10:       97.51%
GPS median: 0m
GPS p75:    0m
Threshold:  0.58 → 73.6% coverage, 95.5% precision

**Observations:**
- Biggest single gain in series (+14.57 over EXP-06)
- Warmup prevented early instability with 4 unfrozen blocks
- Triplet beats InfoNCE consistently on this dataset size
- Production model for Boston
