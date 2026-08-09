"""File 9-ChatGPT — Metamorphic Robustness (Brightness) Test

Runs the same face-sentiment classifier against the baseline images (1.00x
brightness) AND against brightness-perturbed copies of the same images
(1.02x, 1.06x, 1.12x, 1.13x, 1.19x, 1.22x, 1.27x, 1.28x), then measures
"consistency" = how often the model's predicted label for a perturbed image
matches its predicted label for the corresponding baseline image.

Matching across brightness levels
--------------------------------------------------------------------------
`InferenceJob.job_id` is derived from `label + resolved image path`, so it
differs between brightness levels (each level lives in its own directory).
Cross-level matching is therefore done using `(label, filename)` extracted
from the `face_path` field that `FaceImageDatasetAdapter` puts in every
result record (`InferenceJob.metadata["face_path"]`, flattened onto the
result dict by `InferenceJob.result_base()`).
--------------------------------------------------------------------------
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageEnhance

from lab_pipeline.core import (
    CheckpointRepository,
    OutputInterpreter,
    PipelineConfig,
    RedisJobQueue,
    RunMetadataRepository,
    RunPaths,
    WorkerPipeline,
    build_or_load_manifest,
    setup_logging,
    write_reports,
)
from lab_pipeline.datasets import FaceImageDatasetAdapter
from lab_pipeline.providers import ProviderFactory

# =============================================================================
# EXPERIMENT CONFIGURATION
# Edit these values when changing the experiment.
# =============================================================================

FACE_SUBDIRECTORY = "Img"
# SPECTROGRAM_SUBDIRECTORY = "Spectrogram" # face-only doesn't use spectrogram

# Model
PROVIDER = "openai"
MODEL = "gpt-5-nano" 
TEMPERATURE = 1.0
TOP_P = 1.0
NUM_CTX = 4096  

# Processing
NUM_RUNS = 3
NUM_WORKERS = 100
MAX_CONCURRENCY = 3

# Retry
REQUEST_TIMEOUT = 180.0
REQUEST_RETRIES = 2
RETRY_DELAY = 2.0

# Dataset
TEST_SIZE = 0.20
SEED = 42
MAX_SAMPLES = 0

# Output interpretation
BERT_ENABLED = False
# Unused variable - only read if BERT_ENABLED is True
BERT_MODEL = "typeform/distilbert-base-uncased-mnli"
BERT_THRESHOLD = 0.38
BERT_MIN_MARGIN = 0.03
# Dataset-pairing safety # comment because face-only doesn't require image pairing
# STRICT_PAIRS = False
# ALLOW_POSITIONAL_FALLBACK = False
# Set True only for validation without model execution.
DRY_RUN = False

# --- Metamorphic Relation (MR) test config ---------------------------------
# Baseline is always 1.00 (original, unmodified image).
BASELINE_LEVEL: float = 1.00
FOLLOWUP_LEVELS: List[float] = [1.02, 1.12, 1.13, 1.22]
ALL_LEVELS: List[float] = [BASELINE_LEVEL] + FOLLOWUP_LEVELS

# Where brightness-augmented copies of the dataset images are cached.
# (Created once, reused on subsequent runs unless you delete it.)
AUGMENTED_CACHE_SUBDIR = "_mr_brightness_cache"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}  # must match datasets.py IMAGE_EXTENSIONS

LOGGER = logging.getLogger("slurm_experiment")

SYSTEM_PROMPT = (
    "You are an expert at classifying sentiment from facial expressions.\n"
    "Output format: Respond with exactly one label: Positive, Negative, or "
    "Neutral. Output the label only, no explanation, punctuation, or "
    "additional text."
)

USER_MESSAGE = (
    "Classify the sentiment expressed in this face image."
)


# =============================================================================
# Environment helpers (unchanged)
# =============================================================================

def required_environment(name: str) -> str:
    """Read an environment variable that Slurm must provide."""
    value = os.environ.get(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable is missing: {name}"
        )

    return value


def environment_bool(name: str, default: bool = False) -> bool:
    """Read a Boolean environment variable such as 0/1 or true/false."""
    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def resolve_dataset_root() -> Path:
    return Path(
        required_environment("DATASET_ROOT")
    ).expanduser().resolve()


# =============================================================================
# Brightness augmentation
# =============================================================================

def brightness_dir_for_level(dataset_root: Path, level: float) -> Path:
    """Directory that holds the brightness-adjusted copies for `level`."""
    tag = f"b{level:.2f}"
    return dataset_root / AUGMENTED_CACHE_SUBDIR / tag


def generate_brightness_variant(
    src_dir: Path,
    dst_dir: Path,
    factor: float,
) -> None:
    """Create brightness-adjusted copies of every image in `src_dir` into
    `dst_dir`, preserving filenames (and any subfolder structure), so that
    the same basename maps 1:1 to the corresponding baseline image.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    def _is_hidden(path: Path) -> bool:
        rel_parts = path.relative_to(src_dir).parts
        return path.name.startswith("._") or any(part.startswith(".") for part in rel_parts)

    image_paths = [
        p for p in src_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not _is_hidden(p)
    ]

    if not image_paths:
        raise FileNotFoundError(f"No images found under {src_dir}")

    # Skip regeneration if it looks like it was already done.
    existing = list(dst_dir.rglob("*"))
    if len(existing) >= len(image_paths):
        LOGGER.info(
            "Brightness cache already present for factor=%.2f (%s), skipping regen",
            factor,
            dst_dir,
        )
        return

    LOGGER.info(
        "Generating brightness=%.2f variant for %d images -> %s",
        factor,
        len(image_paths),
        dst_dir,
    )

    for src_path in image_paths:
        rel = src_path.relative_to(src_dir)
        out_path = dst_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src_path) as img:
            img = img.convert("RGB")
            adjusted = ImageEnhance.Brightness(img).enhance(factor)
            adjusted.save(out_path)


def ensure_face_dir_for_level(dataset_root: Path, level: float) -> Path:
    """Return the face directory to use for a given brightness level,
    generating augmented copies on disk if needed (baseline uses the
    original directory unmodified).
    """
    original_face_dir = dataset_root / FACE_SUBDIRECTORY

    if level == BASELINE_LEVEL:
        return original_face_dir

    variant_dir = brightness_dir_for_level(dataset_root, level)
    generate_brightness_variant(original_face_dir, variant_dir, level)
    return variant_dir


# =============================================================================
# Pipeline config / run, parameterized by brightness level
# =============================================================================

def create_config(level: float, face_dir: Path, level_run_dir: Path) -> PipelineConfig:
    paths = RunPaths.from_run_dir(level_run_dir)

    dataset_root = resolve_dataset_root()
    resume = environment_bool("RESUME")

    experiment_name = f"25.gpt5-face[o+c]-no-instruction-MR-b{level:.2f}"

    immutable = {
        "experiment": experiment_name,
        "dataset_root": str(dataset_root),
        "face_dir": str(face_dir),
        "brightness_level": level,
        "provider": PROVIDER,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "num_ctx": NUM_CTX,
        "num_runs": NUM_RUNS,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "max_samples": MAX_SAMPLES,
        "prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }

    return PipelineConfig(
        paths=paths,
        run_id=paths.run_dir.name,
        experiment_name=experiment_name,
        provider=PROVIDER,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_message=USER_MESSAGE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_ctx=NUM_CTX,
        num_runs=NUM_RUNS,
        workers=NUM_WORKERS,
        max_concurrency=MAX_CONCURRENCY,
        request_timeout_s=REQUEST_TIMEOUT,
        request_retries=REQUEST_RETRIES,
        retry_delay_s=RETRY_DELAY,
        redis_host=os.environ.get("REDIS_HOST", "127.0.0.1"),
        redis_port=int(os.environ.get("REDIS_PORT", "6379")),
        redis_db=int(os.environ.get("REDIS_DB", "0")),
        redis_password=os.environ.get("REDIS_PASSWORD"),
        bert_enabled=BERT_ENABLED,
        # Unused kwargs - never read since bert_enabled is False
        bert_model=BERT_MODEL,
        bert_threshold=BERT_THRESHOLD,
        bert_min_margin=BERT_MIN_MARGIN,
        resume=resume,
        immutable_settings=immutable,
    )


def extract_image_key(result: dict) -> str:
    """Identify the underlying photo behind a result record so it can be
    matched to the same photo's baseline record across brightness levels.

    `FaceImageDatasetAdapter` stores the resolved image path in
    `InferenceJob.metadata["face_path"]`, which `result_base()` flattens
    onto the result dict as a top-level "face_path" key. Since brightness
    variants live in a different directory, we key on
    "<ground-truth label>/<filename>" (e.g. "Positive/img_014.jpg"), which
    is identical for the same photo at every brightness level.
    """
    face_path = Path(result["face_path"])
    label = result.get("label", "")
    return f"{label}/{face_path.name}"


def extract_label(result: dict) -> Optional[str]:
    """Predicted sentiment label for a result record.

    Only "completed" runs with a prediction in VALID_LABELS are considered
    usable; failed runs (status == "failed", prediction == "Error") are
    excluded from the consistency comparison.
    """
    if result.get("status") != "completed":
        return None
    prediction = result.get("prediction")
    if prediction not in ("Positive", "Negative", "Neutral"):
        return None
    return str(prediction)


async def run_level(level: float) -> Tuple[Dict[str, str], int]:
    """Run the full pipeline for one brightness level.

    Returns (image_key -> predicted_label map, number of manifest jobs).
    """
    base_run_dir = Path(required_environment("RUN_DIR")).expanduser().resolve()
    level_run_dir = base_run_dir / f"brightness_{level:.2f}"

    dataset_root = resolve_dataset_root()
    face_dir = ensure_face_dir_for_level(dataset_root, level)

    if not face_dir.is_dir():
        raise FileNotFoundError(f"Face directory not found: {face_dir}")

    config = create_config(level, face_dir, level_run_dir)
    config.paths.create()
    setup_logging(config.paths.log_dir)

    metadata = RunMetadataRepository(config)
    metadata.start()
    try:
        dataset = FaceImageDatasetAdapter(face_dir)

        def build_jobs():
            all_jobs = dataset.build()
            train_jobs, test_jobs = dataset.split(all_jobs, TEST_SIZE, SEED)

            if MAX_SAMPLES:
                test_jobs = test_jobs[:MAX_SAMPLES]

            LOGGER.info(
                "[brightness=%.2f] Created manifest: all=%d train=%d test=%d",
                level,
                len(all_jobs),
                len(train_jobs),
                len(test_jobs),
            )
            return test_jobs, len(train_jobs)

        manifest, _ = build_or_load_manifest(
            path=config.paths.timestamp_dir / "manifest.jsonl",
            resume=config.resume,
            build_jobs=build_jobs,
        )

        if DRY_RUN:
            metadata.finish("validated")
            LOGGER.info("[brightness=%.2f] PASS: dry run completed", level)
            return {}, len(manifest)

        checkpoint = CheckpointRepository(
            config.paths,
            resume=config.resume,
            total_jobs=len(manifest),
        )
        completed = checkpoint.completed_job_ids()
        pending = [job for job in manifest if job.job_id not in completed]
        LOGGER.info(
            "[brightness=%.2f] Resume calc: total=%d completed=%d pending=%d",
            level, len(manifest), len(completed), len(pending),
        )

        elapsed = 0.0
        worker_stats = {}
        if pending:
            semaphore = asyncio.Semaphore(config.max_concurrency)
            provider = ProviderFactory.create(
                config.provider,
                host=required_environment("MODEL_HOST"),
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=config.model,
                temperature=config.temperature,
                top_p=config.top_p,
                num_ctx=config.num_ctx,
                timeout_s=config.request_timeout_s,
                retries=config.request_retries,
                retry_delay_s=config.retry_delay_s,
                semaphore=semaphore,
            )
            queue = RedisJobQueue(config)
            interpreter = OutputInterpreter(
                bert_enabled=config.bert_enabled,
                # Unused kwargs - only read inside OutputInterpreter if bert_enabled is True
                bert_model=config.bert_model,
                threshold=config.bert_threshold,
                min_margin=config.bert_min_margin,
            )
            # Dependency injection: the pipeline receives, rather than creates,
            # the provider, Redis queue, checkpoint repository, and interpreter.
            pipeline = WorkerPipeline(
                config=config,
                queue=queue,
                checkpoint=checkpoint,
                provider=provider,
                interpreter=interpreter,
            )
            elapsed, worker_stats = await pipeline.run(pending)

        results = checkpoint.latest_in_manifest_order(manifest)
        write_reports(
            config=config,
            results=results,
            worker_stats=worker_stats,
            pipeline_elapsed_s=elapsed,
            processed_this_attempt=len(pending),
        )

        status = (
            "completed"
            if len(checkpoint.completed_job_ids()) == len(manifest)
            else "incomplete"
        )
        metadata.finish(status)
        LOGGER.info("[brightness=%.2f] %s", level, status.upper())

        # Build image_key -> predicted-label map for the consistency
        # comparison. `results` are the raw dicts saved by CheckpointRepository
        # (one per completed/failed job), each already carrying its own
        # "face_path" and "label" (ground truth) fields.
        key_to_label: Dict[str, str] = {}
        for result in results:
            prediction = extract_label(result)
            if prediction is not None:
                key_to_label[extract_image_key(result)] = prediction

        return key_to_label, len(manifest)

    except Exception as exc:
        metadata.finish("failed", error=str(exc))
        raise


# =============================================================================
# Consistency computation
# =============================================================================

def compute_consistency(
    baseline_map: Dict[str, str],
    level_map: Dict[str, str],
) -> Tuple[float, int, int, List[Tuple[str, str, str]]]:
    """Compare a brightness level's predictions to baseline predictions for
    the same images.

    Returns (consistency_score, n_matched, n_compared, mismatches) where
    mismatches is a list of (image_key, baseline_label, level_label).
    """
    common_keys = sorted(set(baseline_map) & set(level_map))
    n_compared = len(common_keys)
    mismatches: List[Tuple[str, str, str]] = []

    n_matched = 0
    for key in common_keys:
        base_label = baseline_map[key]
        level_label = level_map[key]
        if base_label == level_label:
            n_matched += 1
        else:
            mismatches.append((key, base_label, level_label))

    consistency = (n_matched / n_compared) if n_compared else 0.0
    return consistency, n_matched, n_compared, mismatches


def write_consistency_report(
    run_dir: Path,
    per_level_results: Dict[float, Dict[str, str]],
) -> None:
    baseline_map = per_level_results[BASELINE_LEVEL]

    summary_rows = []
    mismatch_records = []

    for level in FOLLOWUP_LEVELS:
        level_map = per_level_results.get(level, {})
        consistency, n_matched, n_compared, mismatches = compute_consistency(
            baseline_map, level_map
        )
        summary_rows.append(
            {
                "brightness_level": level,
                "n_compared": n_compared,
                "n_matched": n_matched,
                "consistency": round(consistency, 4),
            }
        )
        for image_key, base_label, level_label in mismatches:
            mismatch_records.append(
                {
                    "brightness_level": level,
                    "image": image_key,
                    "baseline_label": base_label,
                    "level_label": level_label,
                }
            )
        LOGGER.info(
            "MR consistency @ brightness=%.2f -> %d/%d matched (%.2f%%)",
            level, n_matched, n_compared, consistency * 100,
        )

    csv_path = run_dir / "mr_brightness_consistency_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["brightness_level", "n_compared", "n_matched", "consistency"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    mismatches_path = run_dir / "mr_brightness_mismatches.jsonl"
    with open(mismatches_path, "w", encoding="utf-8") as f:
        for record in mismatch_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("MR summary written to %s", csv_path)
    LOGGER.info("MR mismatches written to %s", mismatches_path)


# =============================================================================
# Main entry point
# =============================================================================

async def run() -> int:
    base_run_dir = Path(required_environment("RUN_DIR")).expanduser().resolve()
    base_run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(base_run_dir / "logs")

    per_level_results: Dict[float, Dict[str, str]] = {}
    overall_status = 0

    for level in ALL_LEVELS:
        LOGGER.info("=== Starting brightness level %.2f ===", level)
        try:
            key_to_label, n_jobs = await run_level(level)
            per_level_results[level] = key_to_label
            if n_jobs and len(key_to_label) < n_jobs:
                LOGGER.warning(
                    "[brightness=%.2f] Only %d/%d jobs produced a usable label",
                    level, len(key_to_label), n_jobs,
                )
        except Exception:
            LOGGER.exception("[brightness=%.2f] FAILED", level)
            overall_status = 2
            per_level_results[level] = {}

    if BASELINE_LEVEL in per_level_results and per_level_results[BASELINE_LEVEL]:
        write_consistency_report(base_run_dir, per_level_results)
    else:
        LOGGER.error("Baseline (1.00) produced no results; cannot compute consistency.")
        overall_status = 2

    return overall_status


def main() -> int:
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        LOGGER.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
