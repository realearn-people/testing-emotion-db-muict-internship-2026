#!/usr/bin/env python3
"""
Extract 30ms image frames from all MP4 files in P1-V1-Ravdess.

Output:
  {BASE_DIR}/{category}/{basename}/img/{basename}-img-{end_ms}-{chunk_id:02d}.jpg

FAST VERSION: decodes each video with a SINGLE ffmpeg call using
`-vf fps=1000/30` (one frame every 30ms) instead of spawning a new
ffmpeg process per frame. This is dramatically faster - one decode
pass per clip instead of ~150+ seeks per clip.

- Deletes and redoes img folder if it already exists
- Returns the ACTUAL number of .jpg files saved
"""
import os
import re
import shutil
import subprocess
import sys

BASE_DIR    = "/home/dl-box/users/students/soongswang-kornkanok/P1-V2-Ravdess"
CATEGORIES  = ["Angry", "Calm", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
INTERVAL_MS = 30
FPS_EXPR    = f"1000/{INTERVAL_MS}"  # one frame every 30ms


def process_file(input_path, img_dir, basename):
    # Delete and redo
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir)

    # ffmpeg writes frames as temp-numbered files first, then we rename
    # them to match the required naming convention.
    tmp_pattern = os.path.join(img_dir, "_tmp_frame_%05d.jpg")

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_path,
            "-vf", f"fps={FPS_EXPR}",
            "-vsync", "0",
            "-q:v", "2",          # high JPEG quality
            tmp_pattern,
        ],
        capture_output=True,
    )

    if result.returncode != 0:
        print(f"  ERROR {basename}:\n{result.stderr[-300:].decode()}", file=sys.stderr)
        return 0

    # Collect and rename frames in order: frame N (1-indexed) -> end_ms = N*30
    tmp_files = sorted(
        f for f in os.listdir(img_dir) if re.match(r"_tmp_frame_\d+\.jpg$", f)
    )

    saved_count = 0
    for idx, tmp_name in enumerate(tmp_files, start=1):
        end_ms = idx * INTERVAL_MS
        final_name = f"{basename}-img-{end_ms}-{idx:02d}.jpg"
        os.replace(
            os.path.join(img_dir, tmp_name),
            os.path.join(img_dir, final_name),
        )
        saved_count += 1

    return saved_count


def main():
    total_files    = 0
    total_segments = 0

    for category in CATEGORIES:
        input_dir = os.path.join(BASE_DIR, category)
        mp4_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".mp4"))
        print(f"\n[{category}] {len(mp4_files)} files")

        for i, filename in enumerate(mp4_files, 1):
            basename   = os.path.splitext(filename)[0]
            input_path = os.path.join(input_dir, filename)
            img_dir    = os.path.join(input_dir, basename, "img")

            n = process_file(input_path, img_dir, basename)
            total_segments += n
            total_files    += 1

            if i % 10 == 0 or i == len(mp4_files):
                print(f"  [{i}/{len(mp4_files)}] {filename} -> {n} frames")

    print(f"\nDone. {total_files} videos -> {total_segments} frames total.")


if __name__ == "__main__":
    main()
