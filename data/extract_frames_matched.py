"""
data/extract_frames_matched.py

Reads matched_actions.json and extracts 13 consecutive frames centered on
each action timestamp using Decord CPU decoding.

Frames are written to local disk first, then bulk-copied to Drive per game
to avoid slow per-file Drive writes.

Usage:
    python data/extract_frames_matched.py \
        --data_dir     /content/drive/MyDrive/soccernet-caption \
        --matched_json /content/drive/MyDrive/soccernet-caption/processed_v3/matched_actions.json \
        --output_dir   /content/drive/MyDrive/soccernet-caption/processed_v3 \
        --tmp_dir      /content/tmp_videos \
        --fps 25 \
        --n_frames 13 \
        --img_size 512
"""

import argparse
import json
import re
import shutil
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from decord import VideoReader, cpu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     type=str,
                   default="/content/drive/MyDrive/soccernet-caption")
    p.add_argument("--matched_json", type=str,
                   default="/content/drive/MyDrive/soccernet-caption/processed_v3/matched_actions.json")
    p.add_argument("--output_dir",   type=str,
                   default="/content/drive/MyDrive/soccernet-caption/processed_v3")
    p.add_argument("--tmp_dir",      type=str, default="/content/tmp_videos")
    p.add_argument("--fps",          type=int, default=25)
    p.add_argument("--n_frames",     type=int, default=13)
    p.add_argument("--img_size",     type=int, default=512)
    p.add_argument("--quality",      type=int, default=95)
    p.add_argument("--max_games",    type=int, default=None)
    return p.parse_args()


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


def find_video(game_dir: Path, half: int) -> Path | None:
    """Only use 720p — skip if not available."""
    p = game_dir / f"{half}_720p.mkv"
    return p if p.exists() else None


def has_720p(game_dir: Path) -> bool:
    """Check both halves have 720p before copying."""
    return (game_dir / "1_720p.mkv").exists() and \
           (game_dir / "2_720p.mkv").exists()


def copy_videos_to_local(game_dir_drive: Path, tmp_dir: Path) -> Path:
    """Copy only 720p video files to local disk."""
    local = tmp_dir / game_dir_drive.name
    local.mkdir(parents=True, exist_ok=True)
    for half in (1, 2):
        src = game_dir_drive / f"{half}_720p.mkv"
        dst = local / f"{half}_720p.mkv"
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))
    return local


def delete_local(local_dir: Path):
    """Delete local directory to free disk space."""
    if local_dir.exists():
        shutil.rmtree(str(local_dir))


def clamp(indices: list[int], total: int) -> list[int]:
    return [max(0, min(i, total - 1)) for i in indices]



def extract_half(
    video_path:       Path,
    actions:          list[dict],
    local_frames_dir: Path,
    drive_frames_dir: Path,
    game_slug:        str,
    args,
) -> list[dict]:
    """
    Extract frames for all actions in one half using Decord CPU.
    Writes to local disk — Drive path recorded for final pairs.json.
    """
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
    except Exception as e:
        print(f"  [!] Failed to open {video_path.name}: {e}")
        return []

    total   = len(vr)
    half_w  = args.n_frames // 2
    offsets = list(range(-half_w, half_w + 1))
    results = []

    for action in actions:
        action_sec = action["action_timestamp_sec"]
        center     = int(round(action_sec * args.fps))
        indices    = clamp([center + o for o in offsets], total)
        action_ms  = int(action_sec * 1000)

        try:
            batch = vr.get_batch(indices).asnumpy()
        except Exception as e:
            print(f"  [!] get_batch failed at {action_sec:.1f}s: {e}")
            continue

        for offset, frame_rgb in zip(offsets, batch):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.resize(frame_bgr, (args.img_size, args.img_size),
                                   interpolation=cv2.INTER_AREA)

            offset_str     = f"p{offset}" if offset >= 0 else f"m{abs(offset)}"
            frame_filename = f"{game_slug}_{action_ms:08d}_{offset_str}.jpg"

            cv2.imwrite(str(local_frames_dir / frame_filename),
                        frame_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])

            results.append({
                "game":                 action["game"],
                "half":                 action["half"],
                "action_timestamp_sec": action_sec,
                "frame_offset":         offset,
                "frame_path":           str(drive_frames_dir / frame_filename),
            })

    return results


def process_game(
    local_game_dir:   Path,
    actions:          list[dict],
    local_frames_dir: Path,
    drive_frames_dir: Path,
    game_slug:        str,
    args,
) -> list[dict]:
    """Process all actions for one game grouped by half."""
    by_half = defaultdict(list)
    for action in actions:
        by_half[action["half"]].append(action)

    extracted = []
    for half, half_actions in by_half.items():
        video_path = find_video(local_game_dir, half)
        if video_path is None:
            continue
        result = extract_half(
            video_path=video_path,
            actions=half_actions,
            local_frames_dir=local_frames_dir,
            drive_frames_dir=drive_frames_dir,
            game_slug=game_slug,
            args=args,
        )
        extracted.extend(result)

    return extracted


def main():
    args             = parse_args()
    data_dir         = Path(args.data_dir)
    output_dir       = Path(args.output_dir)
    tmp_dir          = Path(args.tmp_dir)
    drive_frames_dir = output_dir / "frames"
    local_frames_dir = Path("/content/tmp_frames")

    drive_frames_dir.mkdir(parents=True, exist_ok=True)
    local_frames_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with open(args.matched_json, encoding="utf-8") as f:
        matched_actions = json.load(f)

    print(f"Loaded {len(matched_actions)} matched actions")
    print(f"Frames per action : {args.n_frames} (offsets -{args.n_frames//2} to +{args.n_frames//2})")
    print(f"Image size        : {args.img_size}x{args.img_size}")
    print(f"Decoder           : decord2 CPU")
    print(f"Frame write       : local disk → bulk copy to Drive per game")

    by_game = defaultdict(list)
    for action in matched_actions:
        by_game[action["game"]].append(action)

    games = sorted(by_game.keys())
    if args.max_games:
        games = games[:args.max_games]
        print(f"(limited to {args.max_games} games)")

    print(f"Games to process  : {len(games)}")

    all_extracted = []

    for game_name in tqdm(games, desc="Extracting", unit="game"):
        actions        = by_game[game_name]
        sample         = actions[0]
        game_dir_drive = data_dir / sample["league"] / sample["season"] / game_name
        game_slug      = slugify(game_name)

        if not game_dir_drive.exists() or not has_720p(game_dir_drive):
            continue

        local_game_dir = copy_videos_to_local(game_dir_drive, tmp_dir)

        try:
            extracted = process_game(
                local_game_dir=local_game_dir,
                actions=actions,
                local_frames_dir=local_frames_dir,
                drive_frames_dir=drive_frames_dir,
                game_slug=game_slug,
                args=args,
            )
            all_extracted.extend(extracted)

            # Bulk copy all frames for this game from local disk to Drive
            for f in local_frames_dir.iterdir():
                shutil.copy2(str(f), str(drive_frames_dir / f.name))

            # Clear local frames dir for next game
            for f in local_frames_dir.iterdir():
                f.unlink()

        finally:
            delete_local(local_game_dir)

    # Save extracted_frames.json
    out_file = output_dir / "extracted_frames.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_extracted, f, indent=2, ensure_ascii=False)

    print(f"\nTotal frames extracted : {len(all_extracted)}")
    print(f"Unique actions         : {len(all_extracted) // args.n_frames}")
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
