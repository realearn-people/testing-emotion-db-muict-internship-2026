import shutil
from pathlib import Path

import pandas as pd


# =========================
# CONFIG
# =========================

BASE_DIR = Path("/home/dl-box/users/students/soongswang-kornkanok/P3-V2-Ravdess")

face_root = BASE_DIR / "Img"
spec_root = BASE_DIR / "Spectrogram"
audio_root = BASE_DIR / "Audio"

openface_csv = BASE_DIR / "openface_output" / "mouth_filter_result.csv"

# Output silver dataset folder
silver_root = BASE_DIR / "silver_dataset_open_mouth"

silver_img_root = silver_root / "Img"
silver_spec_root = silver_root / "Spectrogram"
silver_audio_root = silver_root / "Audio"

# ใส่ threshold ที่หาได้จาก sample ตรงนี้
BEST_THRESHOLD = 0.35

# True = test only, ไม่ copy จริง
# False = copy จริง
DRY_RUN = False

# True = ลบ silver_dataset_open_mouth เดิมก่อนสร้างใหม่
CLEAR_OLD_OUTPUT = True

image_exts = [".jpg", ".jpeg", ".png"]
spec_exts = [".jpg", ".jpeg", ".png"]
audio_exts = [".wav", ".mp3", ".flac", ".m4a"]


# =========================
# HELPERS
# =========================

def convert_stem(stem, target_type):
    """
    Convert image stem to matching spectrogram/audio stem.

    Example:
      01-xx-img-2460-82  ->  01-xx-spec-2460-82
      01-xx-img-2460-82  ->  01-xx-audio-2460-82
    """
    if "-img-" in stem:
        return stem.replace("-img-", f"-{target_type}-")
    return stem


def find_matching_file(folder: Path, stem: str, exts):
    """
    Find file by stem + extension.

    Example:
      folder / stem.jpg
      folder / stem.wav
    """
    if not folder.exists():
        return None

    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p

        p_upper = folder / f"{stem}{ext.upper()}"
        if p_upper.exists():
            return p_upper

    return None


def copy_file(src: Path, dst: Path):
    if src is None or not src.exists():
        return "not_found"

    if DRY_RUN:
        return "would_copy"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def clear_old_output():
    if silver_root.exists():
        print("Removing old silver dataset folder:")
        print(silver_root)
        shutil.rmtree(silver_root)

    silver_img_root.mkdir(parents=True, exist_ok=True)
    silver_spec_root.mkdir(parents=True, exist_ok=True)
    silver_audio_root.mkdir(parents=True, exist_ok=True)


# =========================
# MAIN
# =========================

if CLEAR_OLD_OUTPUT and not DRY_RUN:
    clear_old_output()
else:
    silver_img_root.mkdir(parents=True, exist_ok=True)
    silver_spec_root.mkdir(parents=True, exist_ok=True)
    silver_audio_root.mkdir(parents=True, exist_ok=True)

if not openface_csv.exists():
    raise FileNotFoundError(f"OpenFace result CSV not found: {openface_csv}")

df = pd.read_csv(openface_csv)
df.columns = df.columns.str.strip()

if "mouth_ratio" not in df.columns:
    raise ValueError("mouth_ratio column not found in OpenFace CSV")

if "folder" not in df.columns:
    raise ValueError("folder column not found in OpenFace CSV")

if "file" not in df.columns:
    raise ValueError("file column not found in OpenFace CSV")

df["mouth_ratio"] = pd.to_numeric(df["mouth_ratio"], errors="coerce")

# Create silver label from threshold
df["silver_mouth_label"] = df["mouth_ratio"].apply(
    lambda x: "closed" if pd.notna(x) and x < BEST_THRESHOLD else "open"
)

df["silver_threshold_used"] = BEST_THRESHOLD

# Keep only open mouth
open_df = df[df["silver_mouth_label"] == "open"].copy()

print("=== Silver Dataset Config ===")
print("BASE_DIR:", BASE_DIR)
print("OpenFace CSV:", openface_csv)
print("Silver output:", silver_root)
print("Threshold:", BEST_THRESHOLD)
print("DRY_RUN:", DRY_RUN)
print()

print("=== Input Summary ===")
print("Total rows:", len(df))
print("Open mouth rows:", len(open_df))
print("Closed mouth rows:", len(df) - len(open_df))
print()


# =========================
# COPY FILES
# =========================

copy_logs = []

for i, (_, row) in enumerate(open_df.iterrows(), start=1):
    folder = str(row["folder"])
    img_stem = str(row["file"])

    spec_stem = convert_stem(img_stem, "spec")
    audio_stem = convert_stem(img_stem, "audio")

    # Source folders
    src_img_folder = face_root / folder
    src_spec_folder = spec_root / folder
    src_audio_folder = audio_root / folder

    # Find source files
    img_src = find_matching_file(src_img_folder, img_stem, image_exts)
    spec_src = find_matching_file(src_spec_folder, spec_stem, spec_exts)
    audio_src = find_matching_file(src_audio_folder, audio_stem, audio_exts)

    # Destination folders keep same relative structure
    dst_img_folder = silver_img_root / folder
    dst_spec_folder = silver_spec_root / folder
    dst_audio_folder = silver_audio_root / folder

    # Destination file paths
    img_dst = dst_img_folder / img_src.name if img_src is not None else None
    spec_dst = dst_spec_folder / spec_src.name if spec_src is not None else None
    audio_dst = dst_audio_folder / audio_src.name if audio_src is not None else None

    # Copy
    img_status = copy_file(img_src, img_dst) if img_dst is not None else "not_found"
    spec_status = copy_file(spec_src, spec_dst) if spec_dst is not None else "not_found"
    audio_status = copy_file(audio_src, audio_dst) if audio_dst is not None else "not_found"

    copy_logs.append({
        "class": row.get("class", ""),
        "folder": folder,
        "file": img_stem,
        "spec_file_stem": spec_stem,
        "audio_file_stem": audio_stem,
        "frame": row.get("frame", ""),
        "mouth_ratio": row.get("mouth_ratio", ""),
        "silver_mouth_label": row.get("silver_mouth_label", ""),
        "silver_threshold_used": BEST_THRESHOLD,

        "img_src": str(img_src) if img_src is not None else "",
        "spec_src": str(spec_src) if spec_src is not None else "",
        "audio_src": str(audio_src) if audio_src is not None else "",

        "img_dst": str(img_dst) if img_dst is not None else "",
        "spec_dst": str(spec_dst) if spec_dst is not None else "",
        "audio_dst": str(audio_dst) if audio_dst is not None else "",

        "img_copy_status": img_status,
        "spec_copy_status": spec_status,
        "audio_copy_status": audio_status,
    })

    if i % 5000 == 0:
        print(f"Processed {i}/{len(open_df)} open-mouth rows...")


# =========================
# SAVE LOG
# =========================

log_df = pd.DataFrame(copy_logs)

silver_csv = silver_root / "silver_mouth_state_dataset_open_only.csv"
log_df.to_csv(silver_csv, index=False)

# Also save full label CSV, including closed rows, for record
all_label_csv = silver_root / "silver_mouth_state_all_predictions.csv"
df.to_csv(all_label_csv, index=False)


# =========================
# SUMMARY
# =========================

print("\n=== Copy Summary ===")
print("Rows in open-mouth silver dataset:", len(log_df))

print("\nImage copy status:")
print(log_df["img_copy_status"].value_counts(dropna=False))

print("\nSpectrogram copy status:")
print(log_df["spec_copy_status"].value_counts(dropna=False))

print("\nAudio copy status:")
print(log_df["audio_copy_status"].value_counts(dropna=False))

print("\nSaved silver dataset folder to:")
print(silver_root)

print("\nSaved open-only CSV to:")
print(silver_csv)

print("\nSaved all-predictions CSV to:")
print(all_label_csv)