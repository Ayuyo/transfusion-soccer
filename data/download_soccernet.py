"""
data/download_soccernet.py

Downloads SoccerNet-Caption annotations and 224p videos to a target directory.
Run this ONCE on Colab and save output to Google Drive so you never re-download.

Usage:
    python data/download_soccernet.py --output_dir /content/drive/MyDrive/soccernet-caption

Requirements:
    pip install SoccerNet
"""

import argparse
import getpass
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader as SNdl


def parse_args():
    parser = argparse.ArgumentParser(description="Download SoccerNet-Caption dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/soccernet-caption"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test", "challenge"],
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="224p",
        choices=["224p", "720p"],
    )
    parser.add_argument(
        "--annotations_only",
        action="store_true",
    )

    parser.add_argument(
        "--video_only",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_dir}")

    downloader = SNdl(LocalDirectory=str(output_dir))

    # Download caption annotations (no password required)
    if not args.video_only:
        print("\n[1/2] Downloading caption annotations (2024 edition)...")
        downloader.downloadDataTask(task="caption-2024")
        print("Annotations downloaded.")
    else:
        print("\n[1/2] Skipping annotations (--video_only flag set).")

    # Download videos (NDA password required)
    if not args.annotations_only:
        print("\n[2/2] Downloading videos (NDA password required)...")
        print("Enter your SoccerNet NDA password (from your registration email):")
        downloader.password = getpass.getpass("Password: ")

        video_file = f"1_{args.resolution}.mkv"  # first half
        video_file_2 = f"2_{args.resolution}.mkv"  # second half

        downloader.downloadGames(
            files=[video_file, video_file_2],
            split=args.splits,
        )
        print(f"Videos downloaded at {args.resolution}.")
    else:
        print("\n[2/2] Skipping video download (--annotations_only flag set).")

    print(f"\nDownload complete. Data saved to: {output_dir}")
    print("Directory structure:")
    for p in sorted(output_dir.rglob("*"))[:20]:
        print(f"  {p.relative_to(output_dir)}")
    if sum(1 for _ in output_dir.rglob("*")) > 20:
        print("  ... (truncated)")


if __name__ == "__main__":
    main()
