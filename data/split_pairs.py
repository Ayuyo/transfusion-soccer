"""
data/split_pairs.py

Splits pairs.json into train/valid/test at the GAME level

Split sizes (default):
    train : 80%
    valid : 10%
    test  : 10%

Output:
    output_dir/
        train.json
        valid.json
        test.json

Usage:
    python data/split_pairs.py \
        --pairs_json /content/drive/MyDrive/soccernet-caption/processed_v3/pairs.json \
        --output_dir /content/drive/MyDrive/soccernet-caption/processed_v3 \
        --train_frac 0.8 \
        --valid_frac 0.1 \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Game-level train/valid/test split")
    p.add_argument("--pairs_json", type=str,
                   default="/content/drive/MyDrive/soccernet-caption/processed_v3/pairs.json")
    p.add_argument("--output_dir", type=str,
                   default="/content/drive/MyDrive/soccernet-caption/processed_v3")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--valid_frac", type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading pairs.json...")
    with open(args.pairs_json, encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"Total pairs: {len(pairs)}")

    # Group pairs by game
    by_game = defaultdict(list)
    for pair in pairs:
        by_game[pair["game"]].append(pair)

    games = sorted(by_game.keys())
    print(f"Total games: {len(games)}")

    # Shuffle games with fixed seed for reproducibility
    random.seed(args.seed)
    random.shuffle(games)

    # Split games into train/valid/test
    n = len(games)
    n_train = int(n * args.train_frac)
    n_valid = int(n * args.valid_frac)

    train_games = set(games[:n_train])
    valid_games = set(games[n_train:n_train + n_valid])
    test_games  = set(games[n_train + n_valid:])

    # Assign pairs to splits
    train_pairs = []
    valid_pairs = []
    test_pairs  = []

    for game, game_pairs in by_game.items():
        if game in train_games:
            train_pairs.extend(game_pairs)
        elif game in valid_games:
            valid_pairs.extend(game_pairs)
        else:
            test_pairs.extend(game_pairs)

    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_pairs in [
        ("train", train_pairs),
        ("valid", valid_pairs),
        ("test",  test_pairs),
    ]:
        out_file = output_dir / f"{split_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(split_pairs, f, indent=2, ensure_ascii=False)
        print(f"\n{split_name:5s}: {len(split_pairs):7d} pairs  "
              f"({len(split_pairs)/len(pairs)*100:.1f}%)  "
              f"→ {out_file}")

    # Verify no game appears in more than one split
    assert train_games.isdisjoint(valid_games), "Train/valid overlap!"
    assert train_games.isdisjoint(test_games),  "Train/test overlap!"
    assert valid_games.isdisjoint(test_games),  "Valid/test overlap!"
    print("\nSplit integrity verified — no game appears in more than one split.")


if __name__ == "__main__":
    main()
