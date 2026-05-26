"""
data/dataset.py

PyTorch Dataset for TransfusionSports.

Each sample is returned as a tuple:
    (image_tensor, text_tensor)

where:
    image_tensor  — float32 (3, 256, 256)  raw pixel values in [0, 1]
                    The VAE encoder is applied inside the model/training loop,
                    NOT here, so the Dataset stays decoupled from the model.
    text_tensor   — uint8 bytes of the commentary string, cast to torch.long
                    Vocabulary size = 256 (byte-level, consistent with
                    lucidrains/transfusion-pytorch reference examples)

The training loop wraps these into the interleaved sequence format that
Transfusion expects:
    [image_latent (float), commentary_tokens (long)]

Usage:
    from data.dataset import SoccerCommentaryDataset, collate_fn
    from torch.utils.data import DataLoader

    dataset = SoccerCommentaryDataset("path/to/processed/train/pairs_deduped.json")
    loader  = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)

    for image_tensors, text_tensors in loader:
        # image_tensors: list of float tensors  (one per sample in batch)
        # text_tensors:  list of long tensors   (one per sample in batch)
        ...
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_SIZE    = 256    # must match VAE input size (sd-vae-ft-mse expects 256x256)
MAX_TEXT_LEN  = 256    # maximum commentary length in bytes; longer strings truncated
MIN_TEXT_LEN  = 4      # discard degenerate captions shorter than this


# ── Image transforms ──────────────────────────────────────────────────────────

# Standard pipeline: PIL → tensor in [0, 1] → normalise to [-1, 1]
# The VAE (stabilityai/sd-vae-ft-mse) expects inputs in [-1, 1].
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),                          # [0, 1], shape (3, H, W)
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),          # → [-1, 1]
])


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def encode_text(text: str, max_len: int = MAX_TEXT_LEN) -> torch.Tensor:
    """
    Byte-level encoding: convert a UTF-8 string to a LongTensor of byte values.

    Vocabulary size = 256 (one entry per possible byte value).
    This is the tokenisation strategy used in the lucidrains reference examples
    and avoids any dependency on an external tokeniser library.

    Args:
        text:    Input commentary string.
        max_len: Maximum number of bytes to keep (truncate if longer).

    Returns:
        LongTensor of shape (L,) where L <= max_len.
    """
    byte_values = list(text.encode("utf-8"))[:max_len]
    return torch.tensor(byte_values, dtype=torch.long)


def decode_text(tokens: torch.Tensor) -> str:
    """
    Inverse of encode_text. Converts a LongTensor of byte values back to a string.
    Invalid byte sequences are replaced with the replacement character (U+FFFD).
    """
    byte_values = tokens.cpu().numpy().astype(np.uint8).tobytes()
    return byte_values.decode("utf-8", errors="replace")


# ── Dataset ───────────────────────────────────────────────────────────────────

class SoccerCommentaryDataset(Dataset):
    """
    Loads image-commentary pairs from a pairs_deduped.json file produced by
    data/extract_frames.py + data/deduplicate.py.

    Each item returns:
        image_tensor : FloatTensor  (3, 256, 256)  in [-1, 1]
        text_tensor  : LongTensor   (L,)            byte-level tokens

    Args:
        pairs_json  : Path to pairs_deduped.json (or pairs.json as fallback).
        transform   : torchvision transform applied to each PIL image.
                      Defaults to IMAGE_TRANSFORM defined above.
        max_text_len: Maximum commentary length in bytes.
        min_text_len: Pairs with commentary shorter than this are skipped.
    """

    def __init__(
        self,
        pairs_json: str | Path,
        transform=None,
        max_text_len: int = MAX_TEXT_LEN,
        min_text_len: int = MIN_TEXT_LEN,
    ):
        self.pairs_json  = Path(pairs_json)
        self.transform   = transform or IMAGE_TRANSFORM
        self.max_text_len = max_text_len
        self.min_text_len = min_text_len

        self.pairs = self._load_and_filter(self.pairs_json)
        print(f"[SoccerCommentaryDataset] Loaded {len(self.pairs)} pairs "
              f"from {self.pairs_json}")

    def _load_and_filter(self, pairs_json: Path) -> list[dict]:
        with open(pairs_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        filtered = []
        skipped_missing = 0
        skipped_short   = 0

        for pair in raw:
            # Skip if frame file is missing from disk
            if not Path(pair["frame_path"]).exists():
                skipped_missing += 1
                continue

            # Skip degenerate commentary strings
            commentary = pair.get("commentary", "").strip()
            if len(commentary.encode("utf-8")) < self.min_text_len:
                skipped_short += 1
                continue

            pair["commentary"] = commentary
            filtered.append(pair)

        if skipped_missing > 0:
            print(f"  [!] Skipped {skipped_missing} pairs with missing frame files.")
        if skipped_short > 0:
            print(f"  [!] Skipped {skipped_short} pairs with commentary < "
                  f"{self.min_text_len} bytes.")

        return filtered

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]

        # ── Load and transform image ──────────────────────────────────────────
        img = Image.open(pair["frame_path"]).convert("RGB")
        image_tensor = self.transform(img)           # FloatTensor (3, 256, 256)

        # ── Encode commentary ─────────────────────────────────────────────────
        text_tensor = encode_text(
            pair["commentary"],
            max_len=self.max_text_len,
        )                                            # LongTensor (L,)

        return image_tensor, text_tensor


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Custom collate function for DataLoader.

    Transfusion does NOT use padded batches — it expects a list of tensors
    of variable length (one per sample), which its internal dataloader handles
    via the interleaved sequence format. We return two parallel lists so the
    training loop can build the interleaved input easily.

    Args:
        batch: List of (image_tensor, text_tensor) tuples from __getitem__.

    Returns:
        image_tensors : list of FloatTensors, each (3, 256, 256)
        text_tensors  : list of LongTensors,  each (L_i,)  [variable length]
    """
    image_tensors = [item[0] for item in batch]
    text_tensors  = [item[1] for item in batch]
    return image_tensors, text_tensors


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    if len(sys.argv) < 2:
        print("Usage: python data/dataset.py <path_to_pairs_deduped.json>")
        sys.exit(1)

    pairs_path = sys.argv[1]
    dataset    = SoccerCommentaryDataset(pairs_path)

    print(f"\nDataset size : {len(dataset)}")

    # Inspect first sample
    img_t, txt_t = dataset[0]
    print(f"Image tensor : shape={img_t.shape}, dtype={img_t.dtype}, "
          f"min={img_t.min():.3f}, max={img_t.max():.3f}")
    print(f"Text tensor  : shape={txt_t.shape}, dtype={txt_t.dtype}")
    print(f"Decoded text : {decode_text(txt_t)!r}")

    # Test DataLoader with collate_fn
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    imgs, txts = next(iter(loader))
    print(f"\nBatch check:")
    print(f"  image batch : {len(imgs)} tensors, first shape = {imgs[0].shape}")
    print(f"  text batch  : {len(txts)} tensors, first shape = {txts[0].shape}")
    print(f"  first text  : {decode_text(txts[0])!r}")
    print("\nDataset OK.")
