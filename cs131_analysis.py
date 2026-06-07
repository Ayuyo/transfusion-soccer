"""
cs131_analysis.py

Features computed per image:
    1. Canny edge density
    2. Sobel gradient magnitude - mean and standard deviation
    3. SIFT keypoint count
    4. Harris corner count

For each image we already have a per-sample BLEU score from evaluate.py
(predictions.json). We compute Pearson correlations between each CV feature
and BLEU, and produce scatter plots + a summary table.

Usage:
    python cs131_analysis.py \
        --predictions /content/drive/MyDrive/transfusion-soccer-eval/predictions.json \
        --output_dir  /content/drive/MyDrive/transfusion-soccer-eval/cs131_analysis

Outputs (in output_dir):
    features.csv             — per-image features + BLEU score
    correlations.txt         — Pearson r and p-value for each feature vs each BLEU-n
    scatter_canny.png        — scatter plot of canny density vs BLEU-1
    scatter_sobel.png        — scatter plot of sobel mean vs BLEU-1
    scatter_sift.png         — scatter plot of SIFT count vs BLEU-1
    scatter_harris.png       — scatter plot of Harris count vs BLEU-1
    examples_grid.png        — sample images annotated with their feature values
"""

import argparse
import json
import csv
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=str, required=True)
    p.add_argument("--output_dir",  type=str,
                   default="/content/drive/MyDrive/transfusion-soccer-eval/cs131_analysis")
    p.add_argument("--canny_low",   type=int, default=50)
    p.add_argument("--canny_high",  type=int, default=150)
    p.add_argument("--harris_k",    type=float, default=0.04)
    p.add_argument("--max_keypoints", type=int, default=500)
    p.add_argument("--resize_to",   type=int, default=512,)
    return p.parse_args()


# CV feature extractors

def canny_edge_density(gray: np.ndarray, low: int, high: int) -> float:
    """
    Returns fraction of pixels classified as edges (0.0 — 1.0).
    Higher density = more visual clutter / textured scenes.
    """
    edges = cv2.Canny(gray, low, high)
    return float((edges > 0).mean())


def sobel_gradient_stats(gray: np.ndarray) -> tuple[float, float]:
    """.
    Returns (mean, std) of gradient magnitude across the image.
    Higher mean = stronger overall structure / contrast.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return float(magnitude.mean()), float(magnitude.std())


def sift_keypoint_count(gray: np.ndarray, max_keypoints: int) -> int:
    """
    Returns number of detected SIFT keypoints.
    Higher count = more distinctive visual features (players, ball, lines, etc).
    """
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints = sift.detect(gray, None)
    return len(keypoints)


def harris_corner_count(gray: np.ndarray, k: float,
                         threshold_ratio: float = 0.01) -> int:
    """
    Returns number of strong Harris corners (above threshold of max response).
    """
    gray_float = np.float32(gray)
    response   = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=k)
    threshold  = threshold_ratio * response.max()
    return int((response > threshold).sum())


# Feature extraction per image

def extract_features(image_path: str, args) -> dict:
    """Compute all four CS 131 features for one image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((args.resize_to, args.resize_to))
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    canny    = canny_edge_density(gray, args.canny_low, args.canny_high)
    s_mean, s_std = sobel_gradient_stats(gray)
    sift_n   = sift_keypoint_count(gray, args.max_keypoints)
    harris_n = harris_corner_count(gray, args.harris_k)

    return {
        "canny_density"   : canny,
        "sobel_mean"      : s_mean,
        "sobel_std"       : s_std,
        "sift_keypoints"  : sift_n,
        "harris_corners"  : harris_n,
    }


# Correlation analysis

def correlate(feature_vals: list, bleu_vals: list) -> tuple[float, float]:
    """Pearson r and two-tailed p-value, robust to NaN or constant arrays."""
    f = np.array(feature_vals, dtype=np.float64)
    b = np.array(bleu_vals,    dtype=np.float64)
    mask = ~(np.isnan(f) | np.isnan(b))
    if mask.sum() < 3 or f[mask].std() == 0 or b[mask].std() == 0:
        return 0.0, 1.0
    r, p = pearsonr(f[mask], b[mask])
    return float(r), float(p)


# Plotting

def scatter_plot(x_vals, y_vals, x_label: str, y_label: str,
                 title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_vals, y_vals, alpha=0.4, s=15)

    # Linear fit overlay
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    if x_arr.std() > 0 and len(x_arr) > 2:
        m, b = np.polyfit(x_arr, y_arr, 1)
        xs = np.linspace(x_arr.min(), x_arr.max(), 100)
        ax.plot(xs, m * xs + b, color="crimson", linewidth=1.5, label=f"fit: y = {m:.3f}x + {b:.3f}")
        ax.legend()

    r, p = correlate(x_vals, y_vals)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}\nPearson r = {r:.3f}, p = {p:.3g}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    print(f"Loading predictions from {args.predictions}")
    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    print(f"  {len(predictions)} samples")

    # Extract features
    print("Computing classical CV features (Canny, Sobel, SIFT, Harris)...")
    features = []
    for p in tqdm(predictions, desc="Features"):
        try:
            feats = extract_features(p["image"], args)
            features.append({
                "image"           : p["image"],
                "action"          : p.get("action", ""),
                "gt"              : p.get("gt", ""),
                "pred"            : p.get("pred", ""),
                "bleu1"           : p.get("bleu1", 0.0),
                "bleu2"           : p.get("bleu2", 0.0),
                "bleu3"           : p.get("bleu3", 0.0),
                "bleu4"           : p.get("bleu4", 0.0),
                "meteor"          : p.get("meteor", 0.0),
                **feats,
            })
        except Exception as e:
            print(f"  [!] Skipping {p['image']}: {e}")
            continue

    # Save CSV
    csv_path = output_dir / "features.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(features[0].keys()))
        writer.writeheader()
        writer.writerows(features)
    print(f"Saved → {csv_path}")

    # Correlation analysis
    feature_names = [
        ("canny_density",  "Canny edge density",        "Lecture 3"),
        ("sobel_mean",     "Sobel gradient mean",       "Lecture 3"),
        ("sobel_std",      "Sobel gradient std",        "Lecture 3"),
        ("sift_keypoints", "SIFT keypoint count",       "Lecture 4"),
        ("harris_corners", "Harris corner count",       "Lecture 4"),
    ]
    bleu_keys = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor"]

    summary_lines = ["=== CS 131 Classical CV Feature Analysis ===\n"]
    summary_lines.append(f"Samples: {len(features)}\n")
    summary_lines.append(f"{'Feature':<25} {'Lecture':<12} " +
                         " ".join(f"{m:>12}" for m in bleu_keys))
    summary_lines.append("-" * (25 + 12 + 13 * len(bleu_keys)))

    for feat_key, feat_label, lecture in feature_names:
        feat_vals = [f[feat_key]    for f in features]
        row = f"{feat_label:<25} {lecture:<12} "
        for bleu_key in bleu_keys:
            bleu_vals = [f[bleu_key] for f in features]
            r, p = correlate(feat_vals, bleu_vals)
            tag  = "*" if p < 0.05 else " "
            row += f"  r={r:+.3f}{tag} "
        summary_lines.append(row)

    summary_lines.append("\n* = p < 0.05 (statistically significant)")
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    corr_path = output_dir / "correlations.txt"
    with open(corr_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved → {corr_path}")

    # Scatter plots vs BLEU-1
    print("\nGenerating scatter plots...")
    for feat_key, feat_label, _ in feature_names:
        feat_vals = [f[feat_key] for f in features]
        bleu_vals = [f["bleu1"]  for f in features]
        out_path  = output_dir / f"scatter_{feat_key}.png"
        scatter_plot(feat_vals, bleu_vals,
                     x_label=feat_label, y_label="BLEU-1",
                     title=f"{feat_label} vs BLEU-1",
                     out_path=out_path)
        print(f"  Saved {out_path.name}")

    #  Example images grid 
    print("\nGenerating example image grid...")
    # Pick 6 examples spanning the BLEU-1 range
    sorted_by_bleu = sorted(features, key=lambda f: f["bleu1"])
    n = len(sorted_by_bleu)
    indices = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]
    examples = [sorted_by_bleu[i] for i in indices]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, ex in zip(axes.flat, examples):
        try:
            img = Image.open(ex["image"]).convert("RGB").resize((400, 400))
            ax.imshow(img)
            ax.set_title(
                f"BLEU-1={ex['bleu1']:.3f}\n"
                f"Canny={ex['canny_density']:.3f}  "
                f"SIFT={ex['sift_keypoints']}  "
                f"Harris={ex['harris_corners']}",
                fontsize=9
            )
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center")
            ax.axis("off")

    fig.suptitle("Test images spanning BLEU-1 range", fontsize=13)
    fig.tight_layout()
    grid_path = output_dir / "examples_grid.png"
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)
    print(f"Saved → {grid_path}")

    print("\nDone. CS 131 analysis complete.")


if __name__ == "__main__":
    main()
