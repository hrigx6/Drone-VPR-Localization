"""
frame_extractor.py — Extract GPS-tagged frames from DJI drone footage.

Usage:
    python scripts/frame_extractor.py [--mp4 PATH] [--srt PATH] [--sample-n N]

Auto-detects dataset/*.mp4 and dataset/*.SRT if paths are not given.
Output: dataset/frames/frame_XXXXX.jpg + dataset/frames/pairs.json
"""

import sys
import re
import json
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# PYTHONPATH=scripts compatibility
sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
FRAMES_DIR   = DATASET_DIR / "frames"

SAMPLE_EVERY_N_FRAMES = 60   # 1 frame per 2 s at 30 fps
MIN_ALT_M             = 20.0 # skip takeoff / landing
MAX_DELTA_M           = 15.0 # position jump → mark as turn

# ── SRT parsing ────────────────────────────────────────────────────────────────

_RE_FRAME = re.compile(r'FrameCnt:\s*(\d+)')
_RE_TS    = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)')
_RE_LAT   = re.compile(r'\[latitude:\s*([-\d.]+)\]')
_RE_LON   = re.compile(r'\[longitude:\s*([-\d.]+)\]')
_RE_ALT   = re.compile(r'\[rel_alt:\s*([\d.]+)\s+abs_alt:\s*([\d.]+)\]')


def parse_srt(srt_path: Path) -> dict[int, dict]:
    """Return {frame_cnt: entry} for every block in the SRT file."""
    text   = srt_path.read_text(encoding="utf-8")
    blocks = re.split(r'\n\s*\n', text.strip())

    entries: dict[int, dict] = {}
    for block in blocks:
        fc  = _RE_FRAME.search(block)
        ts  = _RE_TS.search(block)
        lat = _RE_LAT.search(block)
        lon = _RE_LON.search(block)
        alt = _RE_ALT.search(block)

        if not (fc and ts and lat and lon and alt):
            continue

        frame_cnt = int(fc.group(1))
        entries[frame_cnt] = {
            "frame_cnt": frame_cnt,
            "timestamp": ts.group(1),
            "lat":       float(lat.group(1)),
            "lon":       float(lon.group(1)),
            "rel_alt":   float(alt.group(1)),
            "abs_alt":   float(alt.group(2)),
        }
    return entries


# ── Geo helpers ────────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    R    = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


# ── Frame extraction ───────────────────────────────────────────────────────────

def _nearest_entry(srt_entries: dict, frame_cnt: int, tolerance: int) -> dict | None:
    """Return the closest SRT entry within ±tolerance frames, or None."""
    if frame_cnt in srt_entries:
        return srt_entries[frame_cnt]
    nearest = min(srt_entries, key=lambda k: abs(k - frame_cnt))
    return srt_entries[nearest] if abs(nearest - frame_cnt) <= tolerance else None


def extract_frames(
    mp4_path:    Path,
    srt_entries: dict,
    frames_dir:  Path,
    sample_n:    int = SAMPLE_EVERY_N_FRAMES,
) -> tuple[list[dict], int, float, int]:
    """
    Extract every sample_n-th frame from mp4_path.

    Returns
    -------
    records       : list of output dicts (altitude-filtered, stable annotated)
    total_frames  : frame count reported by cv2
    fps           : frames per second reported by cv2
    sampled_total : how many sampled frames had a matching SRT entry
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    tolerance    = sample_n // 2

    sample_indices = list(range(0, total_frames, sample_n))

    records: list[dict] = []
    prev_lat: float | None = None
    prev_lon: float | None = None
    sampled_total = 0

    with tqdm(total=len(sample_indices), desc="Extracting frames", unit="frame") as pbar:
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                pbar.update(1)
                continue

            # center-crop 16:9 → 1:1 to match square satellite tiles
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            top  = (h - min_dim) // 2
            left = (w - min_dim) // 2
            frame = frame[top:top + min_dim, left:left + min_dim]

            # SRT is 1-indexed; video frames are 0-indexed
            entry = _nearest_entry(srt_entries, frame_idx + 1, tolerance)
            if entry is None:
                pbar.update(1)
                continue

            sampled_total += 1

            # ── altitude filter (skip entirely) ───────────────────────────────
            if entry["rel_alt"] < MIN_ALT_M:
                pbar.update(1)
                continue

            lat, lon = entry["lat"], entry["lon"]

            # ── turn detection (keep frame, mark stable=False) ────────────────
            stable = True
            if prev_lat is not None:
                dist = haversine_m(prev_lat, prev_lon, lat, lon)
                if dist > MAX_DELTA_M:
                    stable = False

            # ── save JPEG ─────────────────────────────────────────────────────
            frame_name = f"frame_{frame_idx:05d}.jpg"
            frame_path = frames_dir / frame_name
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            records.append({
                "frame_path": str(frame_path.relative_to(PROJECT_ROOT)),
                "lat":        lat,
                "lon":        lon,
                "rel_alt":    entry["rel_alt"],
                "abs_alt":    entry["abs_alt"],
                "timestamp":  entry["timestamp"],
                "stable":     stable,
            })

            prev_lat, prev_lon = lat, lon
            pbar.update(1)

    cap.release()
    return records, total_frames, fps, sampled_total


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(
    records:       list[dict],
    total_frames:  int,
    fps:           float,
    n_sampled:     int,
    n_alt_pass:    int,
) -> None:
    stable   = [r for r in records if r["stable"]]
    lats     = [r["lat"]     for r in records]
    lons     = [r["lon"]     for r in records]
    alts     = [r["rel_alt"] for r in records]

    sep = "=" * 52
    print(f"\n{sep}")
    print("  FRAME EXTRACTION SUMMARY")
    print(sep)
    print(f"  Total frames in video      : {total_frames:>7,}  ({fps:.2f} fps)")
    print(f"  Frames sampled (1/{SAMPLE_EVERY_N_FRAMES:<2})       : {n_sampled:>7,}")
    print(f"  After altitude filter      : {n_alt_pass:>7,}  (rel_alt >= {MIN_ALT_M:.0f} m)")
    print(f"  Frames marked stable       : {len(stable):>7,}  (delta <= {MAX_DELTA_M:.0f} m)")
    if lats:
        print(f"  GPS lat range              :  {min(lats):.6f} → {max(lats):.6f}")
        print(f"  GPS lon range              :  {min(lons):.6f} → {max(lons):.6f}")
        gps_span = haversine_m(min(lats), min(lons), max(lats), max(lons))
        print(f"  GPS diagonal span          :  {gps_span:.1f} m")
        print(f"  Altitude range (rel_alt)   :  {min(alts):.1f} m → {max(alts):.1f} m")
    print(sep)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract GPS-tagged frames from DJI drone footage."
    )
    parser.add_argument("--mp4",      type=Path, default=None,
                        help="Path to .mp4 file (auto-detected from dataset/ if omitted)")
    parser.add_argument("--srt",      type=Path, default=None,
                        help="Path to .SRT file (auto-detected from dataset/ if omitted)")
    parser.add_argument("--sample-n", type=int,  default=SAMPLE_EVERY_N_FRAMES,
                        help=f"Sample every N frames (default: {SAMPLE_EVERY_N_FRAMES})")
    args = parser.parse_args()

    # ── auto-detect input files ───────────────────────────────────────────────
    mp4_path: Path = args.mp4
    srt_path: Path = args.srt

    if mp4_path is None:
        mp4_files = sorted(DATASET_DIR.glob("*.mp4"))
        if not mp4_files:
            raise FileNotFoundError(f"No .mp4 found in {DATASET_DIR}")
        mp4_path = mp4_files[0]

    if srt_path is None:
        srt_files = sorted(DATASET_DIR.glob("*.SRT")) + sorted(DATASET_DIR.glob("*.srt"))
        if not srt_files:
            raise FileNotFoundError(f"No .SRT/.srt found in {DATASET_DIR}")
        srt_path = srt_files[0]

    print(f"Video : {mp4_path.relative_to(PROJECT_ROOT)}")
    print(f"SRT   : {srt_path.relative_to(PROJECT_ROOT)}")
    print(f"Output: {FRAMES_DIR.relative_to(PROJECT_ROOT)}/")

    # ── parse SRT ─────────────────────────────────────────────────────────────
    print("\nParsing SRT...", flush=True)
    srt_entries = parse_srt(srt_path)
    print(f"  {len(srt_entries):,} SRT entries parsed")

    # ── extract frames ────────────────────────────────────────────────────────
    records, total_frames, fps, n_sampled = extract_frames(
        mp4_path, srt_entries, FRAMES_DIR, args.sample_n
    )
    n_alt_pass = len(records)

    # ── write pairs.json ──────────────────────────────────────────────────────
    pairs_path = FRAMES_DIR / "pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} records → {pairs_path.relative_to(PROJECT_ROOT)}")

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(records, total_frames, fps, n_sampled, n_alt_pass)


if __name__ == "__main__":
    main()
