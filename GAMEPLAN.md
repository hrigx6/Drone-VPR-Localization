# Project Game Plan

Current status and what comes next.

---

## Where we are

**Model:** EXP-08, R@1=87.54% on University-1652
**Boston database:** 943 satellite tiles downloaded and indexed (JP/Roxbury/Northeastern)
**Next:** Real-world validation with DJI Mini 5 Pro over Boston

---

## What we built so far

```
Phase 0 — Model development (COMPLETE):
  ✓ University-1652 dataset pipeline
  ✓ DINOv2 feature extraction + FAISS retrieval
  ✓ 8 training experiments, R@1 improved 9% → 87.54%
  ✓ Threshold analysis: 0.58 → 95.5% precision, 73.6% coverage
  ✓ Best model: EXP-08 (triplet + 4 unfrozen blocks + warmup + TTA)

Phase 1 — Boston preparation (COMPLETE):
  ✓ 943 Boston satellite tiles downloaded (zoom 18, JP/Roxbury/Northeastern)
  ✓ Tiles encoded with EXP-08 model
  ✓ Boston FAISS index built
  ✓ boston_validate.py ready for DJI footage
```

---

## Immediate next steps

### Pre-flight (do before going to Boston)

```
□ Write frame_extractor.py
    extracts frames from DJI .MP4 at 1 frame/2s
    parses GPS from .SRT subtitle file
    outputs pairs.json with (frame_path, lat, lon, altitude, timestamp)

□ Write boston_finetune.py
    pulls Mapbox tile at exact DJI frame GPS coordinate
    creates (DJI frame, satellite tile) training pairs
    fine-tunes EXP-08 on those pairs → EXP-09

□ Test boston_validate.py on a short test video
    confirm SRT parsing works with Mini 5 Pro format
    confirm frame extraction pipeline is clean
```

---

### Flight plan

**Flight 1 — Calibration + zero-shot test:**
```
Location:  Northeastern University campus
           Center: ~42.3398°N, -71.0892°W
Area:      400m × 400m lawnmower grid
Altitude:  61m (200ft — Boston legal limit)
Speed:     6-8 m/s
Pattern:   E-W rows, step south between rows
Duration:  ~5-6 minutes
Settings:  4K/30fps, gimbal -90°, Video Captions ON
Output:    ~150-200 usable frames after 1/2s subsampling

Purpose:
  First 70% of frames → zero-shot validation
                         EXP-08 + Boston FAISS → measure GPS error
  All frames → fine-tuning data source
               pull API tiles at exact frame GPS → EXP-09 pairs
```

**Flight 2 — Validation:**
```
Location:  Same area, offset lawnmower pattern
           Rows offset by 25m from flight 1 rows
Altitude:  61m
Duration:  ~5 minutes
Purpose:   Clean validation with EXP-09 model
           No data leakage — different drone path
```

**Optional Flight 3 — Altitude comparison:**
```
Location:  Same area
Altitude:  45m (~150ft)
Purpose:   Altitude robustness analysis
           How does accuracy change at lower altitude?
           Publishable data point
```

**Before every flight checklist:**
```
□ DJI Fly → Camera Settings → Video Captions → ON
□ Gimbal angle: -90° (straight down, nadir)
□ Resolution: 4K/30fps
□ Check LAANC authorization in DJI Fly app
□ Battery fully charged
□ SD card has enough space (~2GB per flight)
```

---

### Post-flight analysis

```
□ Extract frames + GPS from Flight 1 footage
□ Run zero-shot validation (EXP-08 + Boston index)
    → measure R@1, median GPS error, p75 on real DJI frames
    → this is the zero-shot Boston result

□ Pull API tiles at exact frame GPS coordinates
    → ~300-400 (DJI frame, satellite tile) pairs

□ EXP-09: fine-tune EXP-08 on Boston pairs
    LR=5e-6, 5-10 epochs, zoom augmentation
    ~30 minutes training

□ Rebuild Boston FAISS index with EXP-09 weights

□ Run validation on Flight 2 footage with EXP-09
    → final Boston validation results
```

---

### Refinement (after basic validation works)

```
□ SuperGlue integration
    After FAISS retrieval → pull zoom 20 tile at matched GPS
    SuperGlue drone frame ↔ zoom 20 tile → pixel offset → precise GPS
    Target: sub-5m accuracy

□ EKF fusion
    VPR GPS fixes (5Hz) + DJI IMU/odometry
    Smooth trajectory, handle uncertain frames
    Target: continuous reliable localization

□ Confidence threshold deployment
    threshold=0.58 → answer 73.6% of frames
    reject 26.4% as uncertain → EKF fills gaps
    never output a confident wrong answer
```

---

## Accuracy targets

```
Stage                          Target GPS error
──────────────────────────────────────────────
Zero-shot (EXP-08, Boston)     < 50m median
Fine-tuned (EXP-09, Boston)    < 20m median
+ SuperGlue refinement          < 5m median
+ EKF fusion                    < 2m continuous
```

---

## Potential publication

**Target venues:** IROS, ICRA, RA-L
**Angle:** First systematic real-hardware validation of consumer drone VPR
           localization at regulated altitudes (61m), no RTK, no LiDAR

**What we need for submission:**
```
✓ Ablation study (EXP-00 through EXP-08)
✓ Real hardware validation (Boston flights)
□ Comparison to baselines (NetVLAD, AnyLoc on same data)
□ Second flight location (generalization)
□ Altitude robustness curve (45m vs 61m)
□ EKF integration results
```

---

## Long-term extensions

```
Progressive unfreezing    → unlock middle DINOv2 layers carefully
Larger model (ViT-B)      → 768-dim embeddings, more capacity
Multi-city training       → generalize beyond Boston
Seasonal robustness       → summer vs winter satellite tiles
Edge deployment           → Jetson Orin, real-time onboard inference
```
