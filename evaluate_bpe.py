"""
evaluate_bpe.py

BPE version of evaluate.py — uses VOCAB_SIZE from dataset_bpe.

Usage:
    python evaluate_bpe.py \
        --checkpoint /content/drive/MyDrive/transfusion-soccer-checkpoints-bpe/checkpoint_step092000.pt \
        --test_json /content/test.json \
        --output_dir /content/drive/MyDrive/transfusion-soccer-eval-bpe \
        --max_samples 2000 \
        --temperature 0.2 \
        --cfg_scale 10.0
"""

import argparse
import json
import re
import os
import sys
import contextlib
from pathlib import Path

# Silence loguru INFO messages from the transfusion library
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
except ImportError:
    pass

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from transfusion_pytorch import Transfusion
from tqdm import tqdm

from data.dataset_bpe import IMAGE_SIZE, decode_text, VOCAB_SIZE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--test_json",   type=str, default="/content/test.json")
    p.add_argument("--output_dir",  type=str,
                   default="/content/drive/MyDrive/transfusion-soccer-eval-bpe")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--center_only", action="store_true", default=True)
    p.add_argument("--max_tokens",  type=int,   default=64)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--cfg_scale",   type=float, default=10.0)
    p.add_argument("--seed",        type=int,   default=42)

    p.add_argument("--dim",      type=int, default=512)
    p.add_argument("--depth",    type=int, default=8)
    p.add_argument("--heads",    type=int, default=8)
    p.add_argument("--dim_head", type=int, default=64)
    return p.parse_args()


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def clean(text: str) -> str:
    """BPE produces clean text — only light whitespace cleanup needed."""
    return re.sub(r"\s+", " ", text).strip()


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


def compute_bleu_smoothed(reference: str, hypothesis: str) -> dict:
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
    try:
        from nltk.translate.meteor_score import single_meteor_score
        return single_meteor_score(reference.split(), hypothesis.split())
    except Exception:
        return 0.0


def compute_cider_corpus(predictions: list[dict]) -> float:
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i: [p["gt"]]   for i, p in enumerate(predictions)}
        res = {i: [p["pred"]] for i, p in enumerate(predictions)}
        cider = Cider()
        score, _ = cider.compute_score(gts, res)
        return score
    except Exception as e:
        print(f"  [!] CIDEr failed: {e}")
        return 0.0


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    import nltk
    for resource in ("punkt", "wordnet", "omw-1.4"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    print(f"Loading test pairs from {args.test_json}")
    with open(args.test_json) as f:
        pairs = json.load(f)

    if args.center_only:
        pairs = [p for p in pairs if p.get("frame_offset", 0) == 0]
        print(f"  Filtered to center frame only: {len(pairs)} pairs")

    import random
    random.seed(args.seed)
    random.shuffle(pairs)
    pairs = pairs[:args.max_samples]
    print(f"  Evaluating on {len(pairs)} samples")

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()

    print("Loading Transfusion (BPE)...")
    model = Transfusion(
        num_text_tokens        = VOCAB_SIZE,
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
    print(f"  Loaded checkpoint step {ckpt.get('step', '?')} val_loss {ckpt.get('val_loss', '?'):.4f}")

    print(f"\nGenerating predictions (temp={args.temperature}, cfg={args.cfg_scale})...")
    predictions = []
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

    print("Computing BLEU and METEOR per sample...")
    for p in tqdm(predictions, desc="Scoring"):
        p.update(compute_bleu_smoothed(p["gt"], p["pred"]))
        p["meteor"] = compute_meteor(p["gt"], p["pred"])

    print("Computing corpus CIDEr...")
    corpus_cider = compute_cider_corpus(predictions)

    def avg(key):
        vals = [p[key] for p in predictions if key in p]
        return sum(vals) / len(vals) if vals else 0.0

    summary = (
        f"=== TransfusionSports (BPE) — Evaluation Summary ===\n\n"
        f"Checkpoint        : {args.checkpoint}\n"
        f"Step              : {ckpt.get('step', '?')}\n"
        f"Val loss          : {ckpt.get('val_loss', '?')}\n"
        f"Samples evaluated : {len(predictions)}\n"
        f"Temperature       : {args.temperature}\n"
        f"CFG scale         : {args.cfg_scale}\n"
        f"Tokenizer         : GPT-2 BPE (vocab {VOCAB_SIZE})\n"
        f"\n"
        f"--- Corpus-level metrics ---\n"
        f"BLEU-1   : {avg('bleu1'):.4f}\n"
        f"BLEU-2   : {avg('bleu2'):.4f}\n"
        f"BLEU-3   : {avg('bleu3'):.4f}\n"
        f"BLEU-4   : {avg('bleu4'):.4f}\n"
        f"METEOR   : {avg('meteor'):.4f}\n"
        f"CIDEr    : {corpus_cider:.4f}\n"
    )
    print("\n" + summary)

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
