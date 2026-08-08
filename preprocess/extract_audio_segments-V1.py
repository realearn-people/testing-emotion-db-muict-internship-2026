#!/usr/bin/env python3
"""
Extract exactly 30ms audio segments from all MP4 files in P1-V1-Ravdess.

Output:
  {BASE_DIR}/{category}/{basename}/audio/{basename}-audio-{end_ms}-{chunk_id:02d}.wav

Config:
  - Sample rate  : 48000 Hz
  - Chunk size   : 30ms = 1440 samples
  - Decodes whole clip ONCE (fast), then slices into 30ms chunks in Python
  - Chunk count synced to the number of image frames already extracted
    (image extraction is the ground truth for total chunk count)
"""

import os
import shutil
import struct
import subprocess
import sys

BASE_DIR = "/home/dl-box/users/students/soongswang-kornkanok/P1-V1-Ravdess"
CATEGORIES = ["Negative", "Positive", "Neutral"]
INTERVAL_MS = 30
SAMPLE_RATE = 48000


def _write_wav(filepath, pcm_data, sample_rate):
    """Write raw 16-bit mono PCM data as a WAV file."""
    num_channels = 1
    bytes_per_sample = 2
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    subchunk2_size = len(pcm_data)

    with open(filepath, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + subchunk2_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))             # PCM
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", 16))            # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", subchunk2_size))
        f.write(pcm_data)


def process_file(input_path, audio_dir, basename, img_dir):
    # Wipe any stale files from previous runs (mirrors image script behavior).
    # Without this, leftover .wav files from earlier runs (e.g. when img_dir
    # had a different frame count) can accumulate and outnumber n_images.
    if os.path.exists(audio_dir):
        shutil.rmtree(audio_dir)
    os.makedirs(audio_dir)

    # Sync chunk count with images already extracted (ground truth)
    if not os.path.isdir(img_dir):
        print(f"  WARNING {basename}: img_dir not found ({img_dir}), skipping", file=sys.stderr)
        return 0

    n_images = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    if n_images == 0:
        print(f"  WARNING {basename}: no images in {img_dir}, skipping", file=sys.stderr)
        return 0

    # Decode MP4 -> raw mono 16-bit PCM at 48kHz via ffmpeg (pipe, no temp file)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
        ],
        capture_output=True,
    )

    if result.returncode != 0:
        print(f"  ERROR {basename}:\n{result.stderr[-300:].decode()}", file=sys.stderr)
        return 0

    pcm_data = result.stdout
    samples_per_chunk = (SAMPLE_RATE * INTERVAL_MS) // 1000  # 1440 samples
    bytes_per_chunk = samples_per_chunk * 2                   # 2880 bytes

    n_pcm_chunks = -(-len(pcm_data) // bytes_per_chunk)  # ceil division
    if n_pcm_chunks < n_images:
        print(
            f"  WARNING {basename}: audio has only {n_pcm_chunks} chunks "
            f"but {n_images} images exist - using {n_pcm_chunks}",
            file=sys.stderr,
        )
        n_images = n_pcm_chunks

    chunk_id = 0
    offset = 0

    while offset < len(pcm_data) and chunk_id < n_images:
        chunk_id += 1
        chunk_pcm = pcm_data[offset: offset + bytes_per_chunk]

        wav_path = os.path.join(
            audio_dir,
            f"{basename}-audio-{chunk_id * INTERVAL_MS}-{chunk_id:02d}.wav",
        )
        _write_wav(wav_path, chunk_pcm, SAMPLE_RATE)

        offset += bytes_per_chunk

    return chunk_id


def main():
    total_files = 0
    total_segments = 0

    for category in CATEGORIES:
        input_dir = os.path.join(BASE_DIR, category)

        mp4_files = sorted(
            f for f in os.listdir(input_dir) if f.endswith(".mp4")
        )
        print(f"\n[{category}] {len(mp4_files)} files")

        for i, filename in enumerate(mp4_files, 1):
            basename = os.path.splitext(filename)[0]
            input_path = os.path.join(input_dir, filename)
            audio_dir = os.path.join(input_dir, basename, "audio")
            img_dir   = os.path.join(input_dir, basename, "img")

            n = process_file(input_path, audio_dir, basename, img_dir)
            total_segments += n
            total_files += 1

            if i % 10 == 0 or i == len(mp4_files):
                print(f"  [{i}/{len(mp4_files)}] {filename} -> {n} segments")

    print(f"\nDone. {total_files} videos -> {total_segments} segments total.")


if __name__ == "__main__":
    main()