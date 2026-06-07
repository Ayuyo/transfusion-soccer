"""
data/match_actions_captions.py

For each visible action spotting label in Labels-v2.json, finds the first
caption in Labels-caption.json that occurs within [T, T + window_sec] in
the same half. Saves all matched pairs to matched_actions.json.

Output:
    output_dir/matched_actions.json
    [
        {
            "game":                  "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
            "league":                "england_epl",
            "season":                "2014-2015",
            "half":                  1,
            "action_timestamp_sec":  272.0,
            "action_label":          "Shots on target",
            "caption_timestamp_sec": 285.0,
            "delta_sec":             13.0,
            "commentary":            "Brilliant strike from distance!"
        },
        ...
    ]

Usage:
    python data/match_actions_captions.py \
        --data_dir   /content/drive/MyDrive/soccernet-caption \
        --output_dir /content/drive/MyDrive/soccernet-caption/processed_v3 \
        --window_sec 30.0
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm



def parse_args():
    p = argparse.ArgumentParser(
        description="Match action spotting labels to caption annotations"
    )
    p.add_argument("--data_dir",   type=str,
                   default="/content/drive/MyDrive/soccernet-caption",
                   )
    p.add_argument("--output_dir", type=str,
                   default="/content/drive/MyDrive/soccernet-caption/processed_v3",
                   )
    p.add_argument("--window_sec", type=float, default=30.0,
                   )
    return p.parse_args()



def parse_gametime(game_time: str):
    """
    Parse 'H - MM:SS' into (half: int, seconds: float).
    Returns (None, None) on failure.
    """
    try:
        half_str, ts_str = game_time.split(" - ", 1)
        half = int(half_str.strip())
        parts = [float(x) for x in ts_str.strip().split(":")]
        if len(parts) == 2:
            sec = parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            return None, None
        return half, sec
    except Exception:
        return None, None



def load_actions(labels_v2_path: Path) -> list[dict]:
    """
    Load visible action spotting annotations sorted by (half, timestamp_sec).
    """
    with open(labels_v2_path, encoding="utf-8") as f:
        data = json.load(f)

    actions = []
    for ann in data.get("annotations", []):
        if ann.get("visibility", "visible") != "visible":
            continue
        half, sec = parse_gametime(ann.get("gameTime", ""))
        if half is None:
            continue
        actions.append({
            "half":          half,
            "timestamp_sec": sec,
            "label":         ann.get("label", ""),
        })

    actions.sort(key=lambda x: (x["half"], x["timestamp_sec"]))
    return actions


def load_captions_by_half(labels_caption_path: Path) -> dict[int, list[dict]]:
    """
    Load all shown caption annotations keyed by half.
    Returns {1: [...], 2: [...]} with each list sorted by timestamp_sec.
    Halves are kept separate to avoid cross-half matching bugs.
    """
    with open(labels_caption_path, encoding="utf-8") as f:
        data = json.load(f)

    captions = {1: [], 2: []}

    for ann in data.get("annotations", []):
        if ann.get("visibility", "shown") != "shown":
            continue
        commentary = ann.get("description", ann.get("comment", "")).strip()
        if not commentary:
            continue
        half, sec = parse_gametime(ann.get("gameTime", ""))
        if half not in (1, 2):
            continue
        captions[half].append({
            "timestamp_sec": sec,
            "commentary":    commentary,
        })

    for half in (1, 2):
        captions[half].sort(key=lambda x: x["timestamp_sec"])

    return captions


def find_caption(
    captions_by_half: dict[int, list[dict]],
    half: int,
    action_sec: float,
    window_sec: float,
) -> dict | None:
    """
    Find the first caption in [action_sec, action_sec + window_sec]
    within the given half. Searches only within the correct half.
    """
    for cap in captions_by_half.get(half, []):
        delta = cap["timestamp_sec"] - action_sec
        if delta < 0:
            continue               # caption is before action
        if delta <= window_sec:
            return cap             # first caption within window
        break                      # past window — stop
    return None


def process_game(
    game_dir_video:   Path,
    game_dir_caption: Path,
    window_sec:       float,
    league:           str,
    season:           str,
) -> list[dict]:
    """
    Match all visible actions in one game to captions.
    Returns list of matched pair dicts.
    """
    labels_v2_path      = game_dir_video   / "Labels-v2.json"
    labels_caption_path = game_dir_caption / "Labels-caption.json"

    if not labels_v2_path.exists() or not labels_caption_path.exists():
        return []

    actions          = load_actions(labels_v2_path)
    captions_by_half = load_captions_by_half(labels_caption_path)

    if not actions:
        return []

    matched_pairs = []
    for action in actions:
        caption = find_caption(
            captions_by_half,
            half=action["half"],
            action_sec=action["timestamp_sec"],
            window_sec=window_sec,
        )
        if caption is None:
            continue

        delta = caption["timestamp_sec"] - action["timestamp_sec"]
        matched_pairs.append({
            "game":                  game_dir_caption.name,
            "league":                league,
            "season":                season,
            "half":                  action["half"],
            "action_timestamp_sec":  round(action["timestamp_sec"], 3),
            "action_label":          action["label"],
            "caption_timestamp_sec": round(caption["timestamp_sec"], 3),
            "delta_sec":             round(delta, 3),
            "commentary":            caption["commentary"],
        })

    return matched_pairs


def main():
    args             = parse_args()
    data_dir         = Path(args.data_dir)
    output_dir       = Path(args.output_dir)
    annotations_root = data_dir / "caption-2024"

    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(annotations_root.rglob("Labels-caption.json"))
    print(f"Found {len(label_files)} games")
    print(f"Matching window: [T, T + {args.window_sec}s] (forward only, same half)")

    all_matched = []
    skipped     = 0

    for label_path in tqdm(label_files, desc="Matching", unit="game"):
        game_dir_caption = label_path.parent
        relative         = game_dir_caption.relative_to(annotations_root)

        # relative = league/season/game
        parts  = relative.parts
        league = parts[0] if len(parts) > 0 else "unknown"
        season = parts[1] if len(parts) > 1 else "unknown"

        game_dir_video = data_dir / relative

        if not (game_dir_video / "Labels-v2.json").exists():
            skipped += 1
            continue

        pairs = process_game(
            game_dir_video=game_dir_video,
            game_dir_caption=game_dir_caption,
            window_sec=args.window_sec,
            league=league,
            season=season,
        )
        all_matched.extend(pairs)

    # Save
    out_file = output_dir / "matched_actions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_matched, f, indent=2, ensure_ascii=False)

    # Stats
    print(f"\nTotal matched actions : {len(all_matched)}")
    print(f"Skipped (no Labels-v2): {skipped} games")

    if all_matched:
        deltas    = [p["delta_sec"] for p in all_matched]
        avg_delta = sum(deltas) / len(deltas)
        print(f"Avg caption delay     : {avg_delta:.2f}s")

        # Breakdown by league
        from collections import Counter
        league_counts = Counter(p["league"] for p in all_matched)
        print(f"\nMatches by league:")
        for league, count in sorted(league_counts.items(), key=lambda x: -x[1]):
            print(f"  {league:40s} {count}")

    print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    main()
