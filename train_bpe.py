"""
train.py

Training script for TransfusionSports — image-to-commentary generation
using the Transfusion architecture (lucidrains/transfusion-pytorch).

Pipeline per batch:
    1. Load raw frames (3, 512, 512) in [-1, 1] from DataLoader
    2. Encode frames to latents (4, 64, 64) using frozen SD VAE
    3. Build interleaved sequences: [image_latent (float), commentary (long)]
    4. Forward pass through Transfusion → combined LM + flow-matching loss
    6. Backward + optimizer step

Checkpoints saved to Drive every --save_every steps.

Usage:
    python train.py \
        --train_json  /content/drive/MyDrive/soccernet-caption/processed_v3/train.json \
        --valid_json  /content/drive/MyDrive/soccernet-caption/processed_v3/valid.json \
        --output_dir  /content/drive/MyDrive/transfusion-soccer-checkpoints \
        --batch_size  4 \
        --lr          3e-4 \
        --max_steps   100000 \
        --save_every  1000 \
        --log_every   100
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import AutoencoderKL
from transfusion_pytorch import Transfusion

from data.dataset_bpe import SoccerCommentaryDataset, collate_fn, VOCAB_SIZE



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_json",  type=str,
                   default="/content/drive/MyDrive/soccernet-caption/processed_v3/train.json")
    p.add_argument("--valid_json",  type=str,
                   default="/content/drive/MyDrive/soccernet-caption/processed_v3/valid.json")
    p.add_argument("--output_dir",  type=str,
                   default="/content/drive/MyDrive/transfusion-soccer-checkpoints")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to checkpoint to resume from")

    # Model
    p.add_argument("--dim",         type=int, default=512,
                   help="Transformer hidden dimension (default: 512)")
    p.add_argument("--depth",       type=int, default=8,
                   help="Transformer depth (default: 8)")
    p.add_argument("--heads",       type=int, default=8,
                   help="Attention heads (default: 8)")
    p.add_argument("--dim_head",    type=int, default=64,
                   help="Dimension per head (default: 64)")

    # Training
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--max_steps",   type=int,   default=100000)
    p.add_argument("--save_every",  type=int,   default=1000)
    p.add_argument("--log_every",   type=int,   default=100)
    p.add_argument("--num_workers", type=int,   default=8)
    p.add_argument("--lambda_image",type=float, default=0.1,
                   help="Weight for image flow-matching loss (default: 0.1)")
    p.add_argument("--grad_accum",  type=int,   default=8,
                   help="Gradient accumulation steps (default: 8, effective batch = batch_size * grad_accum)")
    p.add_argument("--use_bf16",    action="store_true", default=True,
                   help="Use bfloat16 mixed precision (recommended for A100)")

    return p.parse_args()



def load_vae(device: torch.device) -> AutoencoderKL:
    """Load frozen Stable Diffusion VAE in bfloat16 for fast inference."""
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()
    print("VAE loaded and frozen (bfloat16).")
    return vae


@torch.no_grad()
def encode_images(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of images to VAE latents using bfloat16 for speed.

    Args:
        images : FloatTensor (B, 3, 512, 512) in [-1, 1]

    Returns:
        latents : FloatTensor (B, 4, 64, 64) in bfloat16
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        posterior = vae.encode(images)
        latents   = posterior.latent_dist.sample()
        latents   = latents * vae.config.scaling_factor
    return latents


def build_sequences(
    image_latents: torch.Tensor,
    text_tensors:  list[torch.Tensor],
) -> list[list]:
    """
    Build interleaved sequences for Transfusion.

    Each sequence: [image_latent (float), commentary_tokens (long)]

    With channel_first_latent=True, the library expects (C, H, W) tensors
    directly. We downsample the VAE latent from 64x64 to 32x32 via avg pooling
    to reduce sequence length from 4096 to 1024 tokens — a 4x speedup.

    Args:
        image_latents : FloatTensor (B, 4, 64, 64)
        text_tensors  : list of B LongTensors, each (L_i,)

    Returns:
        sequences : list of B lists, each [float_tensor, long_tensor]
    """
    # Downsample 2x: (B, 4, 64, 64) → (B, 4, 32, 32)
    image_latents = F.avg_pool2d(image_latents, kernel_size=2)

    B = image_latents.shape[0]
    sequences = []

    for i in range(B):
        sequences.append([image_latents[i], text_tensors[i]])

    return sequences


@torch.no_grad()
def validate(model, vae, loader, device, args, autocast_ctx, max_batches=50):
    """Run validation loop and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for image_tensors, text_tensors in loader:
        if n_batches >= max_batches:
            break

        images = torch.stack(image_tensors).to(device)
        texts  = [t.to(device) for t in text_tensors]

        latents   = encode_images(vae, images.to(torch.bfloat16))
        sequences = build_sequences(latents, texts)

        with autocast_ctx:
            output = model(sequences)
        loss = output[0] if isinstance(output, tuple) else output
        total_loss += loss.item()
        n_batches  += 1

    model.train()
    return total_loss / max(n_batches, 1)


def main():
    args   = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # bfloat16 mixed precision
    dtype   = torch.bfloat16 if args.use_bf16 and torch.cuda.is_available() else torch.float32
    autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype) if args.use_bf16 else torch.nullcontext()
    print(f"Precision: {dtype}")

    print("Loading datasets...")
    train_dataset = SoccerCommentaryDataset(args.train_json)
    valid_dataset = SoccerCommentaryDataset(args.valid_json)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    vae = load_vae(device)

    # VAE encodes 512×512 → 64×64×4 latents (8× spatial downsampling)
    # dim_latent = 4  (VAE channels)
    # modality_default_shape = (64, 64)  (spatial dimensions of latent)
    # modality_num_dim = 2  (2D spatial latent, not 1D sequence)
    # add_pos_emb = True  (needed for 2D latents)
    model = Transfusion(
        num_text_tokens        = VOCAB_SIZE,  # 50257 for GPT-2 BPE
        dim_latent             = 4,
        channel_first_latent   = True,
        modality_default_shape = (32, 32),    # downsampled (4x fewer tokens)
        modality_num_dim       = 2,
        add_pos_emb            = True,
        transformer=dict(
            dim     = args.dim,
            depth   = args.depth,
            heads   = args.heads,
            dim_head= args.dim_head,
        )
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Transfusion parameters: {n_params/1e6:.1f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-5)

    start_step = 0
    if args.resume_from and Path(args.resume_from).exists():
        print(f"Resuming from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        print(f"  Resumed at step {start_step}")

    model.train()
    optimizer.zero_grad()
    step          = start_step
    running_loss  = 0.0
    train_iter    = iter(train_loader)

    print(f"\nStarting training from step {start_step}...")
    print(f"Effective batch size: {args.batch_size * args.grad_accum} "
          f"({args.batch_size} x {args.grad_accum} grad accum steps)")

    while step < args.max_steps:
        # Cycle through dataloader
        try:
            image_tensors, text_tensors = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            image_tensors, text_tensors = next(train_iter)

        images = torch.stack(image_tensors).to(device)
        texts  = [t.to(device) for t in text_tensors]

        # Encode images to VAE latents (in bfloat16)
        latents = encode_images(vae, images.to(torch.bfloat16))

        # Build interleaved sequences
        sequences = build_sequences(latents, texts)

        # Forward pass with bfloat16 autocast + FlashAttention
        with autocast_ctx, sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            output = model(sequences)

        if isinstance(output, tuple):
            loss, breakdown = output
        else:
            loss, breakdown = output, {}

        # Scale loss for gradient accumulation
        loss = loss / args.grad_accum
        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step == 0 and isinstance(breakdown, dict) and breakdown:
            print(f"  DEBUG breakdown keys: {list(breakdown.keys())}")

        running_loss += loss.item() * args.grad_accum  # unscale for logging
        step         += 1

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            running_loss = 0.0
            lr = scheduler.get_last_lr()[0]

            text_loss = breakdown.get('text', 0)
            flow_loss = breakdown.get('flow', 0)
            text_val  = text_loss.item() if hasattr(text_loss, 'item') else float(text_loss)
            flow_val  = flow_loss.item() if hasattr(flow_loss, 'item') else float(flow_loss)

            print(f"step {step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                  f"text={text_val:.4f} flow={flow_val:.4f}")

        if step % args.save_every == 0:
            # Validation
            val_loss = validate(model, vae, valid_loader, device, args, autocast_ctx)
            print(f"  → step {step} | val_loss {val_loss:.4f}")

            ckpt_path = output_dir / f"checkpoint_step{step:06d}.pt"
            torch.save({
                "step":      step,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss":  val_loss,
                "args":      vars(args),
            }, str(ckpt_path))
            print(f"  → Saved checkpoint: {ckpt_path.name}")

    print(f"\nTraining complete at step {step}.")


if __name__ == "__main__":
    main()
