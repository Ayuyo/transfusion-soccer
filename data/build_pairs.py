"""
data/build_pairs.py

Joins matched_actions.json and extracted_frames.json into the final pairs.json
used for training.

matched_actions.json  — {game, half, action_timestamp_sec, action_label,
                          caption_timestamp_sec, delta_sec, commentary, league, season}
extracted_frames.json — {game, half, action_timestamp_sec, frame_offset, frame_path}

Join key: (game, half, action_timestamp_sec)

Output pairs.json:
    [{
        "frame_path":            "...frames/game_slug_00585000_p0.jpg",
        "commentary":            "Brilliant strike from distance!",
        "game":                  "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
        "league":                "england_epl",
        "season":                "2014-2015",
        "half":                  1,
        "action_label":          "Shots on target",
        "action_timestamp_sec":  585.0,
        "caption_timestamp_sec": 598.0,
        "delta_sec":             13.0,
        "frame_offset":          0
    }, ...]

Usage:
    python data/build_pairs.py \
        --matched_json    /content/drive/MyDrive/soccernet-caption/processed_v3/matched_actions.json \
        --extracted_json  /content/drive/MyDrive/soccernet-caption/processed_v3/extracted_frames.json \
        --output_dir      /content/drive/MyDrive/soccernet-caption/processed_v3
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Join matched actions and extracted frames")
    p.add_argument("--matched_json",   type=str,
                   default="/content/drive/MyDrive/soccernet-caption/processed_v3/matched_actions.json")
    p.add_argument("--extracted_json", type=str,
                   default="/content/drive/MyDrive/soccernet-caption/processed_v3/extracted_frames.json")
    p.add_argument("--output_dir",     type=str,
                   default="/content/drive/MyDrive/soccernet-caption/processed_v3")
    return p.parse_args()


def make_key(entry: dict) -> tuple:
    """Join key: (game, half, action_timestamp_sec)"""
    return (
        entry["game"],
        entry["half"],
        round(entry["action_timestamp_sec"], 3),
    )


def main():
    args = parse_args()

    print("Loading matched_actions.json...")
    with open(args.matched_json, encoding="utf-8") as f:
        matched_actions = json.load(f)

    print("Loading extracted_frames.json...")
    with open(args.extracted_json, encoding="utf-8") as f:
        extracted_frames = json.load(f)

    print(f"  Matched actions : {len(matched_actions)}")
    print(f"  Extracted frames: {len(extracted_frames)}")

    # Index matched actions by join key
    actions_index = {}
    for action in matched_actions:
        key = make_key(action)
        actions_index[key] = action

    # Build pairs by joining on key
    pairs = []
    skipped = 0

    for frame in extracted_frames:
        key = make_key(frame)
        action = actions_index.get(key)

        if action is None:
            skipped += 1
            continue

        # Skip if frame file doesn't exist on disk
        if not Path(frame["frame_path"]).exists():
            skipped += 1
            continue

        pairs.append({
            "frame_path":            frame["frame_path"],
            "commentary":            action["commentary"],
            "game":                  action["game"],
            "league":                action["league"],
            "season":                action["season"],
            "half":                  action["half"],
            "action_label":          action["action_label"],
            "action_timestamp_sec":  action["action_timestamp_sec"],
            "caption_timestamp_sec": action["caption_timestamp_sec"],
            "delta_sec":             action["delta_sec"],
            "frame_offset":          frame["frame_offset"],
        })

    # Save pairs.json
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "pairs.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    # Stats
    print(f"\nTotal pairs      : {len(pairs)}")
    print(f"Skipped          : {skipped}")
    print(f"Unique actions   : {len(pairs) // 13}")
    print(f"Avg delta        : {sum(p['delta_sec'] for p in pairs) / len(pairs):.2f}s")

    # Breakdown by league
    from collections import Counter
    league_counts = Counter(p["league"] for p in pairs)
    print(f"\nPairs by league:")
    for league, count in sorted(league_counts.items(), key=lambda x: -x[1]):
        print(f"  {league:40s} {count:6d}")

    # Breakdown by frame offset
    offset_counts = Counter(p["frame_offset"] for p in pairs)
    print(f"\nPairs by frame offset:")
    for offset in sorted(offset_counts.keys()):
        print(f"  offset {offset:+d} : {offset_counts[offset]:6d}")

    print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    main()
