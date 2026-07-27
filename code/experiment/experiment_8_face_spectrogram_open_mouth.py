"""File 8"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path

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
from lab_pipeline.datasets import FaceSpectrogramDatasetAdapter
from lab_pipeline.providers import ProviderFactory

# =============================================================================
# EXPERIMENT CONFIGURATION
# Edit these values when changing the experiment.
# =============================================================================

FACE_SUBDIRECTORY = "Img"
SPECTROGRAM_SUBDIRECTORY = "Spectrogram"

# Model
PROVIDER = "ollama"
MODEL = "ministral-3:14b"
TEMPERATURE = 0.0
TOP_P = 0.07
NUM_CTX = 4096

# Processing
NUM_RUNS = 3
NUM_WORKERS = 3
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
BERT_MODEL = "typeform/distilbert-base-uncased-mnli"
BERT_THRESHOLD = 0.38
BERT_MIN_MARGIN = 0.03

# Dataset-pairing safety
STRICT_PAIRS = False
ALLOW_POSITIONAL_FALLBACK = False

# Set True only for validation without model execution.
DRY_RUN = False

# =============================================================================

LOGGER = logging.getLogger("slurm_experiment")

SYSTEM_PROMPT = """You are an expert at classifying sentiment by combining facial expression and audio spectrograms.
Instructions: You will receive one face image and one spectrogram image. Examine both.
For the face image: Examine the face in the image and focus on:

* Mouth shape: Observe whether the mouth forms a smile, frown, neutral expression, or lip compression.
* Eyebrow position: Determine whether the eyebrows are raised, furrowed, or relaxed, as these may reflect different emotional states.
* Eye expression: Assess whether the eyes appear wide open, squinted, relaxed, or show noticeable tension.
* Facial muscle tension: Examine tension in the cheeks, forehead, and jaw, including the presence of wrinkles, tightened muscles, or relaxed facial features.
* Facial symmetry: Consider whether the facial expression is symmetrical or shows asymmetrical muscle activation, which may provide additional cues about the expressed emotion.
* Overall facial affect: Integrate the observed facial features to infer the dominant emotional expression while relying only on visible facial characteristics.

For the spectrogram image: Examine the spectrogram as a frequency-domain representation of the audio signal. Base your sentiment judgment only on visual acoustic patterns in the spectrogram, not on speech content or transcript information. Focus on the following features:

* Energy level: Identify regions with high or low intensity in decibel scale. Higher overall energy or sudden energy changes may indicate stronger emotional expression, while lower and more stable energy may suggest neutral or calm content.
* Dominant frequency range: Observe whether the main energy is concentrated in low, mid, or high frequencies. Higher-frequency dominance may be associated with sharper or more excited acoustic patterns, while lower-frequency dominance may indicate calmer or heavier tones.
* Temporal variation: Examine how the energy changes over time. Rapid fluctuations, abrupt bursts, or irregular patterns may indicate stronger emotional variation, whereas smooth and stable patterns may suggest neutral sentiment.
* Harmonic structure: Look for evenly spaced horizontal bands that indicate clear pitch or voiced sound. Strong and regular harmonic structures may reflect stable vocal expression, while weak, broken, or noisy structures may indicate tension, uncertainty, or negative affect.
* High-frequency energy: Check whether energy is visible above approximately 10-15 kHz or mostly confined to lower frequencies. Extended high-frequency energy may indicate sharper, brighter, or more intense sound characteristics.
* Spectral balance: Consider whether the spectrogram shows a balanced distribution of energy or whether it is concentrated in limited frequency regions. Use this information to support the final sentiment classification.

After examining these features from the face image and the spectrogram image, classify the sentiment into one of the following categories: Positive, Neutral, or Negative. Output format: Respond with exactly one label: Positive, Negative, or Neutral. Output the label only, no explanation, punctuation, or additional text."""

USER_MESSAGE = "Classify the sentiment expressed in both face image and spectrogram image."

def resolve_dataset_paths() -> tuple[Path, Path, Path]:
    dataset_root = Path(
        required_environment("DATASET_ROOT")
    ).expanduser().resolve()

    face_dir = dataset_root / FACE_SUBDIRECTORY
    spectrogram_dir = dataset_root / SPECTROGRAM_SUBDIRECTORY

    return dataset_root, face_dir, spectrogram_dir

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


def create_config() -> PipelineConfig:
    run_dir = Path(
        required_environment("RUN_DIR")
    ).expanduser().resolve()

    paths = RunPaths.from_run_dir(run_dir)

    dataset_root, face_dir, spectrogram_dir = resolve_dataset_paths()

    resume = environment_bool("RESUME")

    immutable = {
        "experiment": "8.0-face-spectrogram-open-mouth",
        "face_dir": str(face_dir),
        "spectrogram_dir": str(spectrogram_dir),
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
        experiment_name="8.0-face-spectrogram-open-mouth",
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
        redis_host=os.environ.get(
            "REDIS_HOST",
            "127.0.0.1",
        ),
        redis_port=int(
            os.environ.get("REDIS_PORT", "6379")
        ),
        redis_db=int(
            os.environ.get("REDIS_DB", "0")
        ),
        redis_password=os.environ.get("REDIS_PASSWORD"),
        bert_enabled=BERT_ENABLED,
        bert_model=BERT_MODEL,
        bert_threshold=BERT_THRESHOLD,
        bert_min_margin=BERT_MIN_MARGIN,
        resume=resume,
        immutable_settings=immutable,
    )


async def run() -> int:
    config = create_config()
    config.paths.create()
    setup_logging(config.paths.log_dir)

    dataset_root, face_dir, spectrogram_dir = resolve_dataset_paths()

    if not face_dir.is_dir():
        raise FileNotFoundError(
            f"Face directory not found: {face_dir}"
        )

    if not spectrogram_dir.is_dir():
        raise FileNotFoundError(
            f"Spectrogram directory not found: "
            f"{spectrogram_dir}"
        )

    metadata = RunMetadataRepository(config)
    metadata.start()
    try:
        dataset = FaceSpectrogramDatasetAdapter(
            face_dir.expanduser().resolve(),
            spectrogram_dir.expanduser().resolve(),
            strict_pairs=STRICT_PAIRS,
            allow_positional_fallback=ALLOW_POSITIONAL_FALLBACK,
        )

        def build_jobs():
            all_jobs = dataset.build()

            train_jobs, test_jobs = dataset.split(
                all_jobs,
                TEST_SIZE,
                SEED,
            )

            if MAX_SAMPLES:
                test_jobs = test_jobs[:MAX_SAMPLES]

            LOGGER.info(
                "Created manifest: all=%d train=%d test=%d",
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
            LOGGER.info("PASS: dry run completed")
            return 0

        checkpoint = CheckpointRepository(
            config.paths,
            resume=config.resume,
            total_jobs=len(manifest),
        )
        completed = checkpoint.completed_job_ids()
        pending = [job for job in manifest if job.job_id not in completed]
        LOGGER.info("Resume calculation: total=%d completed=%d pending=%d", len(manifest), len(completed), len(pending))

        elapsed = 0.0
        worker_stats = {}
        if pending:
            semaphore = asyncio.Semaphore(config.max_concurrency)
            provider = ProviderFactory.create(
                config.provider,
                host=required_environment("MODEL_HOST"),
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

        if len(checkpoint.completed_job_ids()) == len(manifest):
            metadata.finish("completed")
            LOGGER.info("PASS: run completed")
            return 0
        metadata.finish("incomplete")
        return 2
    except Exception as exc:
        metadata.finish("failed", error=str(exc))
        raise


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
