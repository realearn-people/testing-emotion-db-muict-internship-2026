#!/usr/bin/env python3
"""
Face-filter & restructure script.

Uses OpenFace's `-fdir` mode to process ALL frames of a clip in a SINGLE
FaceLandmarkImg call (loads the model once per clip, not once per frame).
This is dramatically faster than calling the binary per-image.

For every chunk_id of every clip:
  1. Batch-run OpenFace on the clip's entire img/ folder.
  2. Parse the resulting per-image CSVs to determine face/no-face.
  3. If face detected -> copy img, matching audio, matching spectrogram
     into the NEW directory structure below.
  4. If no face -> skip (do not copy any of the three files).

OLD structure (read from):
  {OLD_BASE}/{category}/{basename}/img/{basename}-img-{end_ms}-{chunk_id:02d}.jpg
  {OLD_BASE}/{category}/{basename}/audio/{basename}-audio-{end_ms}-{chunk_id:02d}.wav
  {OLD_BASE}/{category}/{basename}/spec/{basename}-spec-{end_ms}-{chunk_id:02d}.jpg

NEW structure (written to):
  {NEW_BASE}/Img/{category}/{basename}/{basename}-img-{end_ms}-{chunk_id:02d}.jpg
  {NEW_BASE}/Audio/{category}/{basename}/{basename}-audio-{end_ms}-{chunk_id:02d}.wav
  {NEW_BASE}/Spectrogram/{category}/{basename}/{basename}-spec-{end_ms}-{chunk_id:02d}.jpg
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

# ─── Paths ────────────────────────────────────────────────────────────────
OLD_BASE = "/home/dl-box/users/students/soongswang-kornkanok/P1-V2-Ravdess"
NEW_BASE = "/home/dl-box/users/students/soongswang-kornkanok/P2-V2-Ravdess-FaceFiltered"
CATEGORIES = ["Angry", "Calm", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

# Local-install OpenFace binary (built via install_openface.sh, no sudo/system changes)
OPENFACE_BIN = os.path.expanduser("~/openface_env/OpenFace/build/bin/FaceLandmarkImg")

IMG_PATTERN = re.compile(r"^(.+)-img-(\d+)-(\d+)\.jpg$")


def batch_detect_faces(img_dir: str, img_files: list, tmp_dir: str) -> dict:
    """
    Run OpenFace ONCE on the whole folder (-fdir) instead of once per image.
    Returns {filename: bool_has_face}.
    """
    if not img_files:
        return {}

    result = subprocess.run(
        [
            OPENFACE_BIN,
            "-fdir", img_dir,
            "-out_dir", tmp_dir,
            "-2Dfp",
            "-q",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ERROR running OpenFace on {img_dir}:\n{result.stderr[-300:]}", file=sys.stderr)

    face_map = {}
    for img_file in img_files:
        base = os.path.splitext(img_file)[0]
        csv_path = os.path.join(tmp_dir, f"{base}.csv")
        face_map[img_file] = _csv_has_face(csv_path)

    return face_map


def _csv_has_face(csv_path: str, confidence_threshold: float = 0.0) -> bool:
    """
    Determine whether a face was detected from OpenFace's per-image CSV.
      - If NO face is detected at all, no CSV is written for that image.
      - The CSV's first column is often 'face' (the face index within the
        image, e.g. 0 for the first/only face) — NOT a success flag.
      - A 'success' column is sometimes absent entirely, so we cannot rely on it being present.
      - 'confidence' is reliably present and is the best signal: low/zero
        confidence means the detector didn't actually lock onto a face.
    """
    if not os.path.exists(csv_path):
        return False  # no CSV at all => OpenFace found nothing

    with open(csv_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if len(lines) < 2:
        return False  # header only, no detection row

    header = [h.strip() for h in lines[0].split(",")]
    row = [v.strip() for v in lines[1].split(",")]

    # If a 'success' column is present, respect it first.
    if "success" in header:
        try:
            success_val = row[header.index("success")]
            if success_val in ("0", "0.0"):
                return False
        except (ValueError, IndexError):
            pass

    if "confidence" in header:
        try:
            conf_val = float(row[header.index("confidence")])
            return conf_val > confidence_threshold
        except (ValueError, IndexError):
            pass

    # CSV exists with a data row but no usable confidence/success column
    # -> assume a face was found (a CSV row was written at all).
    return True


def process_clip(category, basename):
    old_img_dir = os.path.join(OLD_BASE, category, basename, "img")
    old_audio_dir = os.path.join(OLD_BASE, category, basename, "audio")
    old_spec_dir = os.path.join(OLD_BASE, category, basename, "spec")

    if not os.path.isdir(old_img_dir):
        return 0, 0

    new_img_dir = os.path.join(NEW_BASE, "Img", category, basename)
    new_audio_dir = os.path.join(NEW_BASE, "Audio", category, basename)
    new_spec_dir = os.path.join(NEW_BASE, "Spectrogram", category, basename)
    os.makedirs(new_img_dir, exist_ok=True)
    os.makedirs(new_audio_dir, exist_ok=True)
    os.makedirs(new_spec_dir, exist_ok=True)

    img_files = sorted(f for f in os.listdir(old_img_dir) if f.endswith(".jpg"))
    if not img_files:
        return 0, 0

    kept = 0
    dropped = 0

    # ONE OpenFace call for the whole clip's frames (fast)
    with tempfile.TemporaryDirectory() as tmp_dir:
        face_map = batch_detect_faces(old_img_dir, img_files, tmp_dir)

        for img_file in img_files:
            m = IMG_PATTERN.match(img_file)
            if not m:
                continue
            fbase, end_ms, chunk_id = m.group(1), m.group(2), m.group(3)

            if not face_map.get(img_file, False):
                # copy audio
                shutil.copy2(old_audio_path, new_audio_dir)
                dropped += 1
                continue

            audio_file = f"{fbase}-audio-{end_ms}-{chunk_id}.wav"
            spec_file = f"{fbase}-spec-{end_ms}-{chunk_id}.jpg"

            old_img_path = os.path.join(old_img_dir, img_file)
            old_audio_path = os.path.join(old_audio_dir, audio_file)
            old_spec_path = os.path.join(old_spec_dir, spec_file)

            shutil.copy2(old_img_path, os.path.join(new_img_dir, img_file))

            if os.path.isfile(old_audio_path):
                shutil.copy2(old_audio_path, os.path.join(new_audio_dir, audio_file))
            else:
                print(f"  WARNING: missing audio for {img_file}", file=sys.stderr)

            if os.path.isfile(old_spec_path):
                shutil.copy2(old_spec_path, os.path.join(new_spec_dir, spec_file))
            else:
                print(f"  WARNING: missing spec for {img_file}", file=sys.stderr)

            kept += 1

    return kept, dropped


def main():
    if not os.path.isfile(OPENFACE_BIN):
        print(f"ERROR: OpenFace binary not found at {OPENFACE_BIN}", file=sys.stderr)
        print("Run install_openface.sh first.", file=sys.stderr)
        sys.exit(1)

    total_kept = 0
    total_dropped = 0

    for category in CATEGORIES:
        cat_dir = os.path.join(OLD_BASE, category)
        if not os.path.isdir(cat_dir):
            continue

        basenames = sorted(
            d for d in os.listdir(cat_dir)
            if os.path.isdir(os.path.join(cat_dir, d)) and not d.startswith(".")
        )
        print(f"\n[{category}] {len(basenames)} clips")

        for i, basename in enumerate(basenames, 1):
            kept, dropped = process_clip(category, basename)
            total_kept += kept
            total_dropped += dropped

            if i % 10 == 0 or i == len(basenames):
                print(f"  [{i}/{len(basenames)}] {basename} -> kept {kept}, dropped {dropped}")

    print(f"\nDone. Kept {total_kept} chunks, dropped {total_dropped} (no face detected).")
    print(f"New dataset at: {NEW_BASE}")


if __name__ == "__main__":
    main()