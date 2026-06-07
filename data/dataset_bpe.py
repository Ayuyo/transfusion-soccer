"""
data/dataset.py

PyTorch Dataset for TransfusionSports.

Reads train.json / valid.json / test.json produced by split_pairs.py.
Each sample returns:
    image_tensor : FloatTensor (3, 512, 512) normalised to [-1, 1]
    text_tensor  : LongTensor  (L,)          byte-level commentary tokens

The VAE encoding and Transfusion interleaved sequence construction are done
in the training loop, not here.

Usage:
    from data.dataset import SoccerCommentaryDataset, collate_fn
    from torch.utils.data import DataLoader

    dataset = SoccerCommentaryDataset(
        pairs_json="/content/drive/MyDrive/soccernet-caption/processed_v3/train.json"
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    for image_tensors, text_tensors in loader:
        # image_tensors : list of FloatTensors, each (3, 512, 512)
        # text_tensors  : list of LongTensors,  each (L,)
        ...
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SIZE   = 512   # must match VAE input (sd-vae-ft-mse supports up to 512)
MAX_TEXT_LEN = 64    # max commentary length in BPE tokens (~250 characters)
MIN_TEXT_LEN = 2     # discard degenerate captions shorter than 2 tokens


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),
])


import tiktoken
_BPE = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = _BPE.n_vocab  # 50257 for gpt2


def encode_text(text: str, max_len: int = MAX_TEXT_LEN) -> torch.Tensor:
    """
    GPT-2 BPE encoding: string → LongTensor of subword token IDs.
    Vocabulary size = 50257. Much cleaner than byte-level — avoids the
    garbled-word problem caused by invalid UTF-8 sequences.
    """
    ids = _BPE.encode(text)[:max_len]
    return torch.tensor(ids, dtype=torch.long)


def decode_text(tokens: torch.Tensor) -> str:
    """Inverse of encode_text using GPT-2 BPE."""
    ids = tokens.cpu().tolist()
    # Filter to valid vocab range (in case model generates out-of-vocab IDs)
    ids = [i for i in ids if 0 <= i < VOCAB_SIZE]
    return _BPE.decode(ids)



class SoccerCommentaryDataset(Dataset):
    """
    Loads image-commentary pairs from a split JSON file.
    No file existence checks — assumes all frames are present on disk.

    Args:
        pairs_json   : Path to train.json / valid.json / test.json
        transform    : torchvision transform applied to each PIL image
        max_text_len : Maximum commentary length in bytes
        min_text_len : Pairs with commentary shorter than this are skipped
    """

    def __init__(
        self,
        pairs_json:   str | Path,
        transform=None,
        max_text_len: int = MAX_TEXT_LEN,
        min_text_len: int = MIN_TEXT_LEN,
    ):
        self.pairs_json   = Path(pairs_json)
        self.transform    = transform or IMAGE_TRANSFORM
        self.max_text_len = max_text_len
        self.min_text_len = min_text_len
        self.pairs        = self._load(self.pairs_json)
        print(f"[SoccerCommentaryDataset] {len(self.pairs)} pairs from "
              f"{self.pairs_json.name}")

    def _load(self, pairs_json: Path) -> list[dict]:
        with open(pairs_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Only filter out degenerate commentary
        filtered      = []
        skipped_short = 0

        for pair in raw:
            commentary = pair.get("commentary", "").strip()
            if len(_BPE.encode(commentary)) < self.min_text_len:
                skipped_short += 1
                continue
            pair["commentary"] = commentary
            filtered.append(pair)

        if skipped_short:
            print(f"  [!] Skipped {skipped_short} pairs — commentary too short")

        return filtered

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # If a frame is corrupted or unreadable, fall back to the next index so a single bad file doesn't crash training
        for attempt in range(5):
            pair = self.pairs[(idx + attempt) % len(self.pairs)]
            try:
                img          = Image.open(pair["frame_path"]).convert("RGB")
                image_tensor = self.transform(img)
                text_tensor  = encode_text(
                    pair["commentary"],
                    max_len=self.max_text_len,
                )
                return image_tensor, text_tensor
            except Exception as e:
                if attempt == 0:
                    print(f"  [!] Skipping unreadable frame: {pair['frame_path']} ({e})")
                continue

        # never reach here unless 5 consecutive frames are corrupted
        raise RuntimeError(f"5 consecutive frames failed to load starting at idx={idx}")



def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Returns two parallel lists of variable-length tensors.
    Transfusion expects lists not padded stacks.
    """
    image_tensors = [item[0] for item in batch]
    text_tensors  = [item[1] for item in batch]
    return image_tensors, text_tensors


if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    if len(sys.argv) < 2:
        print("Usage: python data/dataset.py <path_to_train.json>")
        sys.exit(1)

    dataset = SoccerCommentaryDataset(sys.argv[1])
    print(f"Dataset size : {len(dataset)}")

    img_t, txt_t = dataset[0]
    print(f"Image tensor : shape={img_t.shape}  dtype={img_t.dtype}  "
          f"min={img_t.min():.3f}  max={img_t.max():.3f}")
    print(f"Text tensor  : shape={txt_t.shape}  dtype={txt_t.dtype}")
    print(f"Commentary   : {decode_text(txt_t)!r}")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    imgs, txts = next(iter(loader))
    print(f"\nBatch check:")
    print(f"  images : {len(imgs)} tensors, first shape = {imgs[0].shape}")
    print(f"  texts  : {len(txts)} tensors, first shape = {txts[0].shape}")
    print(f"  first  : {decode_text(txts[0])!r}")
    print("\nDataset OK.")
