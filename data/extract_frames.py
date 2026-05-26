"""
data/extract_frames.py

Extracts keyframes from SoccerNet-Caption videos at annotated event timestamps
and pairs each frame with its corresponding commentary string.

SoccerNet directory structure (two parallel roots):
    Annotations: data_dir/caption-2024/{league}/{season}/{game}/Labels-caption.json
    Videos:      data_dir/{league}/{season}/{game}/1_224p.mkv

Output structure:
    output_dir/
        all/
            frames/
                {game_slug}_{timestamp_ms}_{half}.jpg
                ...
            pairs.json   ←  [{frame_path, commentary, game, half, timestamp_sec}, ...]

Usage (on Colab after mounting Drive):
    python data/extract_frames.py \
        --data_dir  /content/drive/MyDrive/soccernet-caption \
        --output_dir /content/drive/MyDrive/soccernet-caption/processed \
        --context_sec 1.0
"""

import argparse
import json
import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Extract keyframes from SoccerNet-Caption")
    parser.add_argument("--data_dir",   type=str,
                        default="/content/drive/MyDrive/soccernet-caption",
                        help="Root directory of downloaded SoccerNet data")
    parser.add_argument("--output_dir", type=str,
                        default="/content/drive/MyDrive/soccernet-caption/processed",
                        help="Where to save extracted frames and pairs.json")
    parser.add_argument("--context_sec", type=float, default=1.0,
                        help="Seconds before annotated timestamp to extract frame")
    parser.add_argument("--img_size",   type=int,   default=256,
                        help="Resize extracted frames to this square size (default: 256)")
    parser.add_argument("--quality",    type=int,   default=95,
                        help="JPEG quality for saved frames (default: 95)")
    parser.add_argument("--max_games",  type=int,   default=None,
                        help="Optional: limit number of games processed (for testing)")
    return parser.parse_args()


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


def timestamp_to_seconds(timestamp: str) -> float:
    parts = timestamp.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        raise ValueError(f"Unrecognised timestamp format: {timestamp}")


def extract_frame_at(video_path: Path, target_sec: float, img_size: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = int(target_sec * fps)
    target_frame = max(0, min(target_frame, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return frame


def find_video(video_game_dir: Path, half: int) -> Path | None:
    for res in ["224p", "720p"]:
        p = video_game_dir / f"{half}_{res}.mkv"
        if p.exists():
            return p
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Two parallel roots
    annotations_root = data_dir / "caption-2024"   # annotations live here
    videos_root      = data_dir                     # videos live directly under data_dir/league/

    frames_dir = output_dir / "all" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Find all annotation files
    label_files = sorted(annotations_root.rglob("Labels-caption.json"))
    print(f"Found {len(label_files)} annotated games")

    if args.max_games:
        label_files = label_files[:args.max_games]
        print(f"  (limited to {args.max_games} games for testing)")

    pairs = []
    skipped_no_video   = 0
    skipped_bad_frame  = 0
    skipped_no_caption = 0

    for label_path in tqdm(label_files, desc="Extracting frames", unit="game"):
        game_dir_annotation = label_path.parent

        # Derive the video path by stripping "caption-2024" from the annotation path
        # annotation: data_dir/caption-2024/league/season/game/
        # video:      data_dir/league/season/game/
        relative = game_dir_annotation.relative_to(annotations_root)
        video_game_dir = videos_root / relative

        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        events = data.get("annotations", [])
        game_slug = slugify(game_dir_annotation.name)

        for event in events:
            game_time  = event.get("gameTime", "")
            commentary = event.get("description", event.get("comment", "")).strip()

            if not commentary or not game_time or event.get("visibility", "shown") != "shown":
                skipped_no_caption += 1
                continue

            try:
                half_str, ts_str = game_time.split(" - ", 1)
                half = int(half_str.strip())
                event_sec = timestamp_to_seconds(ts_str)
            except (ValueError, AttributeError):
                continue

            target_sec = max(0.0, event_sec - args.context_sec)

            video_path = find_video(video_game_dir, half)
            if video_path is None:
                skipped_no_video += 1
                continue

            frame = extract_frame_at(video_path, target_sec, args.img_size)
            if frame is None:
                skipped_bad_frame += 1
                continue

            ts_ms = int(target_sec * 1000)
            frame_filename = f"{game_slug}_{ts_ms:08d}_{half}.jpg"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])

            pairs.append({
                "frame_path":    str(frame_path),
                "commentary":    commentary,
                "game":          game_dir_annotation.name,
                "half":          half,
                "timestamp_sec": round(target_sec, 3),
            })

    # Save pairs.json
    pairs_file = output_dir / "all" / "pairs.json"
    with open(pairs_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"\nExtracted {len(pairs)} pairs")
    print(f"Skipped: {skipped_no_video} (no video) | "
          f"{skipped_bad_frame} (bad frame) | "
          f"{skipped_no_caption} (no caption)")
    print(f"Saved → {pairs_file}")


if __name__ == "__main__":
    main()
