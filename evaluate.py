"""
evaluate.py

For each test image:
    1. Generate predicted commentary using the trained model
    2. Compare to ground-truth commentary using BLEU, CIDEr, METEOR

Outputs:
    output_dir/predictions.json   — list of {image, gt, pred, bleu1..4, meteor, cider}
    output_dir/metrics_summary.txt — corpus-level scores

Usage:
    python evaluate.py \
        --checkpoint /content/drive/MyDrive/transfusion-soccer-checkpoints/checkpoint_step062000.pt \
        --test_json /content/test.json \
        --output_dir /content/drive/MyDrive/transfusion-soccer-eval \
        --max_samples 1000 \
        --temperature 0.2 \
        --cfg_scale 10.0
"""

import argparse
import json
import re
import os
import sys
import contextlib

# Silence loguru INFO messages from the transfusion library
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
except ImportError:
    pass
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from transfusion_pytorch import Transfusion
from tqdm import tqdm

from data.dataset import IMAGE_SIZE, decode_text


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--test_json",   type=str,
                   default="/content/test.json")
    p.add_argument("--output_dir",  type=str,
                   default="/content/drive/MyDrive/transfusion-soccer-eval")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--center_only", action="store_true", default=True)
    p.add_argument("--max_tokens",  type=int,   default=200)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--cfg_scale",   type=float, default=10.0)
    p.add_argument("--seed",        type=int,   default=42)

    # Model config
    p.add_argument("--dim",      type=int, default=512)
    p.add_argument("--depth",    type=int, default=8)
    p.add_argument("--heads",    type=int, default=8)
    p.add_argument("--dim_head", type=int, default=64)

    return p.parse_args()


# Image transform
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# Text cleanup

def clean(text: str) -> str:
    text = text.replace("\ufffd", "")
    text = re.sub(r"\s*98298\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lstrip(".,;:|/\\-").strip()


# Generation

@torch.no_grad()
def generate_commentary(model, vae, image_path: str, device, args) -> str:
    img    = Image.open(image_path).convert("RGB")
    tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(device, dtype=torch.bfloat16)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        latent = vae.encode(tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
        latent = F.avg_pool2d(latent, kernel_size=2)

        output = model.sample(
            prompt=(0, latent[0]),
            max_length=args.max_tokens,
            text_temperature=args.temperature,
            text_min_p=0.1,
            cfg_scale=args.cfg_scale,
        )

    if isinstance(output, list):
        segments = []
        for item in output:
            if isinstance(item, torch.Tensor) and item.dtype in (torch.long, torch.int):
                segments.append(item.flatten())
        if not segments:
            return ""
        tokens = torch.cat(segments)
    else:
        tokens = output

    return clean(decode_text(tokens))


#  Metric computation

def compute_bleu_smoothed(reference: str, hypothesis: str) -> dict:
    """Compute BLEU-1 through BLEU-4 with smoothing for short sentences."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref_tokens = [reference.split()]
    hyp_tokens = hypothesis.split()

    smooth = SmoothingFunction().method1

    return {
        f"bleu{n}": sentence_bleu(
            ref_tokens, hyp_tokens,
            weights=tuple([1.0 / n] * n + [0.0] * (4 - n)),
            smoothing_function=smooth,
        ) for n in (1, 2, 3, 4)
    }


def compute_meteor(reference: str, hypothesis: str) -> float:
    """METEOR with NLTK."""
    try:
        from nltk.translate.meteor_score import single_meteor_score
        return single_meteor_score(reference.split(), hypothesis.split())
    except Exception:
        return 0.0


def compute_cider_corpus(predictions: list[dict]) -> float:
    """CIDEr is corpus-level — needs all predictions and references together."""
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i: [p["gt"]]   for i, p in enumerate(predictions)}
        res = {i: [p["pred"]] for i, p in enumerate(predictions)}
        cider = Cider()
        score, _ = cider.compute_score(gts, res)
        return score
    except Exception as e:
        print(f"  [!] CIDEr computation failed: {e}")
        return 0.0



def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    #  Load NLTK resources
    import nltk
    for resource in ("punkt", "wordnet", "omw-1.4"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    #  Load test pairs
    print(f"Loading test pairs from {args.test_json}")
    with open(args.test_json) as f:
        pairs = json.load(f)

    if args.center_only:
        pairs = [p for p in pairs if p.get("frame_offset", 0) == 0]
        print(f"  Filtered to center frame only: {len(pairs)} pairs")

    # Shuffle deterministically and limit to max_samples
    import random
    random.seed(args.seed)
    random.shuffle(pairs)
    pairs = pairs[:args.max_samples]
    print(f"  Evaluating on {len(pairs)} samples")

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()

    # Load Transfusion
    print("Loading Transfusion model...")
    model = Transfusion(
        num_text_tokens        = 256,
        dim_latent             = 4,
        channel_first_latent   = True,
        modality_default_shape = (32, 32),
        modality_num_dim       = 2,
        add_pos_emb            = True,
        transformer=dict(
            dim     = args.dim,
            depth   = args.depth,
            heads   = args.heads,
            dim_head= args.dim_head,
        )
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded checkpoint step {ckpt.get('step', '?')}")

    # Generate predictions
    print(f"\nGenerating predictions (temp={args.temperature}, cfg={args.cfg_scale})...")
    predictions = []
    # Redirect stderr to suppress inner tqdm progress bars from the library
    devnull = open(os.devnull, "w")
    for pair in tqdm(pairs, desc="Generating"):
        try:
            with contextlib.redirect_stderr(devnull):
                pred = generate_commentary(model, vae, pair["frame_path"], device, args)
            if not pred:
                continue
            predictions.append({
                "image":      pair["frame_path"],
                "gt":         pair["commentary"],
                "pred":       pred,
                "game":       pair["game"],
                "action":     pair["action_label"],
                "half":       pair["half"],
                "timestamp":  pair["action_timestamp_sec"],
            })
        except Exception as e:
            print(f"  [!] Skipping {pair['frame_path']}: {e}")
            continue

    print(f"\nGenerated {len(predictions)} predictions")

    # Compute per-sample metrics
    print("Computing BLEU and METEOR per sample...")
    for p in tqdm(predictions, desc="Scoring"):
        bleu = compute_bleu_smoothed(p["gt"], p["pred"])
        p.update(bleu)
        p["meteor"] = compute_meteor(p["gt"], p["pred"])

    # Compute corpus CIDEr
    print("Computing corpus CIDEr...")
    corpus_cider = compute_cider_corpus(predictions)

    #  Aggregate
    def avg(key):
        vals = [p[key] for p in predictions if key in p]
        return sum(vals) / len(vals) if vals else 0.0

    avg_bleu1   = avg("bleu1")
    avg_bleu2   = avg("bleu2")
    avg_bleu3   = avg("bleu3")
    avg_bleu4   = avg("bleu4")
    avg_meteor  = avg("meteor")

    summary = (
        f"=== TransfusionSports — Evaluation Summary ===\n"
        f"\n"
        f"Checkpoint        : {args.checkpoint}\n"
        f"Step              : {ckpt.get('step', '?')}\n"
        f"Val loss          : {ckpt.get('val_loss', '?')}\n"
        f"Samples evaluated : {len(predictions)}\n"
        f"Temperature       : {args.temperature}\n"
        f"CFG scale         : {args.cfg_scale}\n"
        f"Center frame only : {args.center_only}\n"
        f"\n"
        f"--- Corpus-level metrics ---\n"
        f"BLEU-1   : {avg_bleu1:.4f}\n"
        f"BLEU-2   : {avg_bleu2:.4f}\n"
        f"BLEU-3   : {avg_bleu3:.4f}\n"
        f"BLEU-4   : {avg_bleu4:.4f}\n"
        f"METEOR   : {avg_meteor:.4f}\n"
        f"CIDEr    : {corpus_cider:.4f}\n"
    )
    print("\n" + summary)

    # Save outputs 
    preds_file   = output_dir / "predictions.json"
    summary_file = output_dir / "metrics_summary.txt"

    with open(preds_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Saved → {preds_file}")
    print(f"Saved → {summary_file}")


if __name__ == "__main__":
    main()
