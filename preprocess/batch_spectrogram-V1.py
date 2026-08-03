"""
Batch Audio -> Spectrogram Converter
=====================================
Structure expected:
  <root>/
    Negative/
      01-01-04-02-01-02-22/
        audio/
          01-01-04-02-01-02-22-audio-1020-34.wav
          ...
        img/
        spec/          <-- will be created by this script
          01-01-04-02-01-02-22-spec-1020-34.jpg
          ...
      ...
    Positive/
      ...
    Neutral/
      ...

Usage:
  python batch_spectrogram.py <root_dir> [--categories Negative Positive Neutral]
"""

import os
import re
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import stft
from scipy.io import wavfile


# ─── STFT Configuration ──────────────────────────────────────────────────────
SR          = 48_000
WINDOW_SIZE = 1400
OVERLAP     = 250
HOP_LENGTH  = WINDOW_SIZE - OVERLAP   # 1150
WINDOW_TYPE = "hann"

X_MAX_SEC   = 0.03
Y_MAX_HZ    = 24_000
# ─────────────────────────────────────────────────────────────────────────────


def load_audio(filepath: str) -> np.ndarray:
    sr, data = wavfile.read(filepath)
    if sr != SR:
        raise ValueError(f"SR mismatch: {filepath} is {sr} Hz (expected {SR} Hz)")

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2_147_483_648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def compute_spectrogram(signal: np.ndarray):
    freqs, times, Zxx = stft(
        signal,
        fs=SR,
        window=WINDOW_TYPE,
        nperseg=WINDOW_SIZE,
        noverlap=OVERLAP,
        boundary="zeros",
        padded=True,
    )
    power = np.abs(Zxx) ** 2
    Sdb   = 10.0 * np.log10(np.maximum(power, 1e-10))
    return freqs, times, Sdb


def save_spectrogram(freqs, times, Sdb, output_path: str):
    time_mask = times <= X_MAX_SEC
    freq_mask = freqs <= Y_MAX_HZ

    t_crop = times[time_mask]
    f_crop = freqs[freq_mask]
    S_crop = Sdb[np.ix_(freq_mask, time_mask)]

    fig, ax = plt.subplots(figsize=(10, 5))
    img = ax.pcolormesh(
        t_crop,
        f_crop / 1_000,
        S_crop,
        shading="auto",
        cmap="inferno",
    )
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label("Power (dB)", fontsize=11)

    ax.set_xlabel("Time (sec)", fontsize=12)
    ax.set_ylabel("Frequency (kHz)", fontsize=12)
    ax.set_xlim(0, X_MAX_SEC)
    ax.set_ylim(0, Y_MAX_HZ / 1_000)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def audio_to_spec_name(audio_filename: str) -> str:
    """
    Convert audio filename to spectrogram filename.
    e.g. '01-01-04-02-01-02-22-audio-1020-34.wav'
      -> '01-01-04-02-01-02-22-spec-1020-34.jpg'
    """
    # Replace '-audio-' with '-spec-' and change extension to .jpg
    name = os.path.splitext(audio_filename)[0]   # strip .wav
    name = name.replace("-audio-", "-spec-")
    return name + ".jpg"


def process_clip_folder(clip_folder: str, stats: dict):
    """
    Process one clip folder (e.g. .../Negative/01-01-04-02-01-02-22/).
    Reads from <clip>/audio/, writes to <clip>/spec/.
    """
    audio_dir = os.path.join(clip_folder, "audio")
    spec_dir  = os.path.join(clip_folder, "spec")

    if not os.path.isdir(audio_dir):
        return  # skip folders that have no audio/ subdir

    os.makedirs(spec_dir, exist_ok=True)

    wav_files = sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(".wav"))
    if not wav_files:
        return

    for wav_file in wav_files:
        wav_path  = os.path.join(audio_dir, wav_file)
        spec_name = audio_to_spec_name(wav_file)
        spec_path = os.path.join(spec_dir, spec_name)

        if os.path.exists(spec_path):
            stats["skipped"] += 1
            continue

        try:
            signal          = load_audio(wav_path)
            freqs, times, S = compute_spectrogram(signal)
            save_spectrogram(freqs, times, S, spec_path)
            stats["done"] += 1
        except Exception as e:
            print(f"  [ERROR] {wav_path}: {e}")
            stats["errors"] += 1


def batch_process(root_dir: str, categories: list):
    stats = {"done": 0, "skipped": 0, "errors": 0}
    total_clips = 0

    for category in categories:
        cat_path = os.path.join(root_dir, category)
        if not os.path.isdir(cat_path):
            print(f"[WARN] Category folder not found: {cat_path}")
            continue

        clip_folders = sorted(
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d))
        )

        print(f"\n[{category}] {len(clip_folders)} clip folders found")

        for i, clip_name in enumerate(clip_folders, 1):
            clip_path = os.path.join(cat_path, clip_name)
            print(f"  ({i}/{len(clip_folders)}) {clip_name}", end="  ", flush=True)
            before = stats["done"]
            process_clip_folder(clip_path, stats)
            added = stats["done"] - before
            print(f"+{added} specs")
            total_clips += 1

    print("\n" + "=" * 50)
    print(f"DONE  — {total_clips} clip folders processed")
    print(f"  Created : {stats['done']}")
    print(f"  Skipped : {stats['skipped']}  (already existed)")
    print(f"  Errors  : {stats['errors']}")


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch audio -> spectrogram converter")
    parser.add_argument("root_dir", help="Root dataset directory (contains Negative/Positive/Neutral)")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["Negative", "Positive", "Neutral"],
        help="Category subfolder names to process (default: Negative Positive Neutral)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"[ERROR] root_dir not found: {args.root_dir}")
        sys.exit(1)

    print(f"Root      : {args.root_dir}")
    print(f"Categories: {args.categories}")
    print(f"STFT      : SR={SR}, W={WINDOW_SIZE}, overlap={OVERLAP}, hop={HOP_LENGTH}")
    print(f"Axes      : x=0-{X_MAX_SEC}s, y=0-{Y_MAX_HZ//1000}kHz")

    batch_process(args.root_dir, args.categories)
