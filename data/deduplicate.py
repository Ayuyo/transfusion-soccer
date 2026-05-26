"""
data/deduplicate.py

Removes near-duplicate frames from the extracted dataset using SIFT feature
matching (Lecture 4: Local Features and Fitting) combined with a Laplacian
sharpness filter (Lecture 4: Laplacian of Gaussians) to select the best frame
among duplicates.

Algorithm:
    1. For each game group in pairs.json, sort frames by timestamp.
    2. For each adjacent pair of frames, extract SIFT descriptors and match
       them using a ratio test (Lowe's ratio test, as taught with SIFT).
    3. Estimate a homography via RANSAC to count geometric inliers.
    4. If inlier_count >= inlier_threshold → frames are near-duplicates.
    5. Among duplicates, retain the frame with the higher Laplacian variance
       (sharper image).
    6. Write a deduplicated pairs.json for each split.

Usage:
    python data/deduplicate.py \
        --processed_dir /content/drive/MyDrive/soccernet-caption/processed \
        --splits train valid test \
        --inlier_threshold 80 \
        --lowe_ratio 0.75
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SIFT-based frame deduplication")
    parser.add_argument("--processed_dir", type=str,
                        default="/content/drive/MyDrive/soccernet-caption/processed",
                        help="Output directory from extract_frames.py")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "valid", "test"],
                        choices=["train", "valid", "test", "challenge"])
    parser.add_argument("--inlier_threshold", type=int, default=80,
                        help="RANSAC inlier count above which two frames are "
                             "considered near-duplicates (default: 80)")
    parser.add_argument("--lowe_ratio", type=float, default=0.75,
                        help="Lowe's ratio test threshold for SIFT matching "
                             "(default: 0.75, as in the original SIFT paper)")
    parser.add_argument("--min_matches", type=int, default=10,
                        help="Minimum number of raw matches needed before "
                             "running RANSAC (default: 10)")
    return parser.parse_args()


def laplacian_variance(image_path: str) -> float:
    """
    Compute sharpness of an image using Laplacian variance.
    Higher variance = sharper image (Lecture 4: Laplacian of Gaussians).
    A blurry image has a low Laplacian response; a sharp image has high variance
    in its second-order derivatives.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


def compute_sift_descriptors(image_path: str):
    """
    Load image and compute SIFT keypoints and descriptors.
    Returns (keypoints, descriptors) or (None, None) on failure.

    SIFT pipeline (Lecture 4):
      - DoG scale-space extrema detection
      - Keypoint localisation and orientation assignment
      - 128-dim descriptor from 4x4 orientation histograms (8 bins each)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def count_inliers(kp1, desc1, kp2, desc2, lowe_ratio: float, min_matches: int) -> int:
    """
    Match SIFT descriptors using a BFMatcher with Lowe's ratio test, then
    estimate a homography via RANSAC and return the inlier count.

    Lowe's ratio test (from the original SIFT paper):
        A match is kept only if the nearest-neighbour distance is less than
        lowe_ratio * second-nearest-neighbour distance. This filters ambiguous
        matches where two descriptors are similarly close.

    Returns 0 if matching fails or too few matches found.
    """
    if desc1 is None or desc2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_L2)
    try:
        raw_matches = bf.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return 0

    # Lowe's ratio test
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < min_matches:
        return 0

    # Extract matched keypoint coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC homography to count geometrically consistent inliers
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return 0

    return int(mask.sum())


def deduplicate_game_group(pairs: list[dict], args) -> list[dict]:
    """
    Given a list of pairs sorted by timestamp for one game+half,
    remove near-duplicates using SIFT matching + Laplacian sharpness selection.

    Strategy:
        - Maintain a list of 'kept' pairs.
        - For each new candidate frame, compare it against the most recently
          kept frame (adjacent comparison — sufficient since frames are sorted
          by time and near-duplicates are temporally local).
        - If inlier count >= threshold → duplicate → keep the sharper one.
        - Otherwise → distinct frame → keep it.
    """
    if not pairs:
        return []

    # Sort by timestamp within this group
    pairs = sorted(pairs, key=lambda x: x["timestamp_sec"])

    kept = [pairs[0]]
    prev_kp, prev_desc = compute_sift_descriptors(pairs[0]["frame_path"])
    prev_sharpness = laplacian_variance(pairs[0]["frame_path"])

    for pair in pairs[1:]:
        curr_kp, curr_desc = compute_sift_descriptors(pair["frame_path"])
        curr_sharpness = laplacian_variance(pair["frame_path"])

        inliers = count_inliers(
            prev_kp, prev_desc,
            curr_kp, curr_desc,
            lowe_ratio=args.lowe_ratio,
            min_matches=args.min_matches,
        )

        if inliers >= args.inlier_threshold:
            # Near-duplicate: keep the sharper frame
            if curr_sharpness > prev_sharpness:
                # Replace last kept frame with the sharper current one
                kept[-1] = pair
                prev_kp, prev_desc = curr_kp, curr_desc
                prev_sharpness = curr_sharpness
            # else: keep the already-retained frame, discard current
        else:
            # Distinct frame: keep it and advance the comparison window
            kept.append(pair)
            prev_kp, prev_desc = curr_kp, curr_desc
            prev_sharpness = curr_sharpness

    return kept


# ── Main ──────────────────────────────────────────────────────────────────────

def process_split(split: str, processed_dir: Path, args):
    split_dir  = processed_dir / split
    pairs_file = split_dir / "pairs.json"

    if not pairs_file.exists():
        print(f"  [!] pairs.json not found at {pairs_file} — skipping.")
        return

    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"  Loaded {len(pairs)} pairs from '{split}'")

    # Group pairs by (game, half) so we only compare within the same sequence
    groups = defaultdict(list)
    for pair in pairs:
        key = (pair["game"], pair["half"])
        groups[key].append(pair)

    print(f"  Processing {len(groups)} game-half groups...")

    deduplicated = []
    total_removed = 0

    for (game, half), group_pairs in tqdm(groups.items(), desc=f"  {split}", unit="group"):
        before = len(group_pairs)
        kept = deduplicate_game_group(group_pairs, args)
        after = len(kept)
        total_removed += before - after
        deduplicated.extend(kept)

    # Save deduplicated pairs
    out_file = split_dir / "pairs_deduped.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(deduplicated, f, indent=2, ensure_ascii=False)

    retention_pct = 100 * len(deduplicated) / max(len(pairs), 1)
    print(f"  Before: {len(pairs)} | After: {len(deduplicated)} | "
          f"Removed: {total_removed} ({100 - retention_pct:.1f}%) | "
          f"Saved → {out_file}")


def main():
    args = parse_args()
    processed_dir = Path(args.processed_dir)

    print(f"SIFT Deduplication")
    print(f"  inlier_threshold : {args.inlier_threshold}")
    print(f"  lowe_ratio       : {args.lowe_ratio}")
    print(f"  min_matches      : {args.min_matches}")

    for split in args.splits:
        print(f"\nProcessing split: {split}")
        process_split(split, processed_dir, args)

    print("\nDeduplication complete.")
    print("Next step: run data/dataset.py to verify the PyTorch Dataset loads correctly.")


if __name__ == "__main__":
    main()
