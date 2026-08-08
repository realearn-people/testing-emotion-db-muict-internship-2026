"""Shared Redis, checkpoint, worker, and reporting workflow."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .providers import ModelProvider

LOGGER = logging.getLogger("slurm_experiment")
VALID_LABELS = ("Positive", "Negative", "Neutral")


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string, truncated to whole seconds."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_default(value: Any) -> Any:
    """Fallback serializer used by json.dumps() for objects it can't handle natively.

    Converts Path objects to strings and numpy-like scalars (anything with an
    `.item()` method) to plain Python values. Raises TypeError for anything else,
    matching the standard json.dumps `default=` contract.
    """
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def json_dumps(value: Any, *, indent: int | None = None) -> str:
    """Serialize a Python value to a JSON string.

    Thin wrapper around json.dumps() that keeps non-ASCII characters readable
    (ensure_ascii=False) and uses json_default() to handle Path/numpy-like values.
    """
    return json.dumps(value, ensure_ascii=False, default=json_default, indent=indent)


def atomic_write_json(path: Path, value: Any) -> None:
    """Write `value` as JSON to `path` without ever leaving a half-written file.

    Writes to a temporary "<path>.tmp" file first, then atomically renames it
    over the destination. This prevents readers from ever seeing a corrupted
    or partially-written JSON file, even if the process crashes mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_key(value: str) -> str:
    """Sanitize a string so it is safe to use as a Redis key / namespace segment.

    Replaces any run of characters that are not letters, digits, dots, underscores,
    or hyphens with a single underscore, and strips leading/trailing underscores.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    log_dir: Path
    timestamp_dir: Path
    output_dir: Path

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "RunPaths":
        """Build a RunPaths from a base run directory.

        Resolves `run_dir` to an absolute path and derives the standard
        sub-folders ("log", "timestamp", "output") used throughout a run.
        """
        root = run_dir.expanduser().resolve()
        return cls(root, root / "log", root / "timestamp", root / "output")

    def create(self) -> None:
        """Create the run directory and all of its sub-folders if they don't exist yet."""
        for path in (self.run_dir, self.log_dir, self.timestamp_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PipelineConfig:
    paths: RunPaths
    run_id: str
    experiment_name: str
    provider: str
    model: str
    system_prompt: str
    user_message: str
    temperature: float = 0.0
    top_p: float = 0.07
    num_ctx: int = 4096
    num_runs: int = 3
    workers: int = 3
    max_concurrency: int = 3
    request_timeout_s: float = 180.0
    request_retries: int = 2
    retry_delay_s: float = 2.0
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    bert_enabled: bool = True
    bert_model: str = "typeform/distilbert-base-uncased-mnli"
    bert_threshold: float = 0.38
    bert_min_margin: float = 0.03
    resume: bool = False
    immutable_settings: Mapping[str, Any] = field(default_factory=dict)

    def runtime_record(self) -> dict[str, Any]:
        """Return this config as a plain dict, safe to log or save to disk.

        Converts the dataclass to a dict via asdict() and masks the Redis
        password (replacing it with the literal string "<set>" or None) so
        secrets never end up in run.json or log files.
        """
        record = asdict(self)
        record["redis_password"] = "<set>" if self.redis_password else None
        return record


@dataclass(frozen=True)
class InferenceJob:
    job_id: str
    label: str
    image_paths: tuple[Path, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert this job into a JSON-serializable dict (e.g. for the manifest file or Redis)."""
        return {
            "job_id": self.job_id,
            "label": self.label,
            "image_paths": [str(path) for path in self.image_paths],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "InferenceJob":
        """Reconstruct an InferenceJob from a dict previously produced by to_record()."""
        return cls(
            job_id=str(value["job_id"]),
            label=str(value["label"]),
            image_paths=tuple(Path(path) for path in value["image_paths"]),
            metadata=dict(value.get("metadata", {})),
        )

    def result_base(self) -> dict[str, Any]:
        """Build the common result fields (job_id, label, image_paths, metadata) shared
        by every result row, so callers only need to add prediction-specific fields on top.
        """
        return {
            "job_id": self.job_id,
            "label": self.label,
            "image_paths": [str(path) for path in self.image_paths],
            **dict(self.metadata),
        }


@dataclass
class LabelDecision:
    label: str | None
    method: str
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] | None = None


class OutputInterpreter:
    REFUSAL_PHRASES = (
        "i'm not able", "i am not able", "i cannot", "i can't",
        "unable to provide", "cannot assist", "can't assist",
        "not able to provide assistance",
    )

    def __init__(self, *, bert_enabled: bool, bert_model: str, threshold: float, min_margin: float) -> None:
        """Store the BERT/zero-shot-classification settings and set up lazy-loading state.

        The actual classifier model is NOT loaded here; it is only loaded on first
        use (see _get_classifier), so constructing this object is cheap.
        """
        self.bert_enabled = bert_enabled
        self.bert_model = bert_model
        self.threshold = threshold
        self.min_margin = min_margin
        self._classifier: Any = None
        self._load_failed = False
        self._lock = threading.Lock()

    @staticmethod
    def _normalise(text: str) -> str:
        """Clean up raw model output for matching: lowercase, strip HTML-like tags,
        remove punctuation/non-letters, and collapse repeated whitespace.
        """
        cleaned = text.strip().lower().replace("\n", " ")
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _direct(cls, text: str) -> str | None:
        """Try to read the label straight off the text without needing the BERT model.

        Returns "Positive", "Negative", or "Neutral" (capitalised) if the cleaned
        text is exactly one of those words, or matches a simple
        "sentiment is <label>" / "answer: <label>" style pattern. Returns None if
        no direct match is found, so the caller can fall back to the BERT classifier.
        """
        cleaned = cls._normalise(text)
        if cleaned in {"positive", "negative", "neutral"}:
            return cleaned.capitalize()
        patterns = (
            r"^(?:answer|sentiment|label|prediction)\s*(?:is|:)?\s*(positive|negative|neutral)$",
            r"^(?:the\s+)?(?:overall\s+)?(?:visual\s+)?sentiment\s*(?:is|:)?\s*(positive|negative|neutral)$",
        )
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
        return None

    def _get_classifier(self) -> Any:
        """Lazily load and cache the zero-shot-classification (BERT/NLI) pipeline.

        Returns None immediately if BERT is disabled or a previous load attempt
        already failed, so we don't keep retrying a broken model on every call.
        On success the loaded pipeline is cached on self._classifier for reuse.
        """
        if not self.bert_enabled or self._load_failed:
            return None
        if self._classifier is not None:
            return self._classifier
        try:
            from transformers import pipeline
            LOGGER.info("Loading CPU BERT/NLI interpreter: %s", self.bert_model)
            self._classifier = pipeline("zero-shot-classification", model=self.bert_model, device=-1)
            return self._classifier
        except Exception as exc:
            self._load_failed = True
            LOGGER.warning("BERT interpreter unavailable: %s", exc)
            return None

    def interpret(self, raw_output: str | None) -> LabelDecision:
        """Turn one raw model response into a LabelDecision (Positive/Negative/Neutral/None).

        Order of attempts:
          1. Empty output -> LabelDecision(None, "empty_output").
          2. Known refusal phrases (e.g. "I can't assist") -> LabelDecision(None, "refusal_or_unusable_output").
          3. Direct text match via _direct() -> LabelDecision(label, "direct_match").
          4. Otherwise, fall back to the BERT zero-shot classifier (thread-safe via
             self._lock) and apply the configured threshold/margin to decide whether
             the top predicted label is confident enough to keep.
        """
        text = "" if raw_output is None else str(raw_output).strip()
        cleaned = self._normalise(text)
        zero = {label: 0.0 for label in VALID_LABELS}
        if not cleaned:
            return LabelDecision(None, "empty_output", zero)
        if any(phrase in cleaned for phrase in self.REFUSAL_PHRASES):
            return LabelDecision(None, "refusal_or_unusable_output", zero)
        direct = self._direct(text)
        if direct:
            scores = dict(zero)
            scores[direct] = 1.0
            return LabelDecision(direct, "direct_match", scores)

        with self._lock:
            classifier = self._get_classifier()
            if classifier is None:
                return LabelDecision(None, "bert_unavailable", zero)
            try:
                candidates = [
                    "positive visual sentiment",
                    "negative visual sentiment",
                    "neutral visual sentiment",
                ]
                result = classifier(text, candidate_labels=candidates, hypothesis_template="This text describes {}.")
                labels = list(result.get("labels", []))
                scores_raw = [float(value) for value in result.get("scores", [])]
                if not labels or not scores_raw:
                    return LabelDecision(None, "bert_empty_result", zero)
                mapped = dict(zero)
                for candidate, score in zip(labels, scores_raw):
                    for final_label in VALID_LABELS:
                        if candidate.startswith(final_label.lower()):
                            mapped[final_label] = round(score, 4)
                top_score = scores_raw[0]
                second_score = scores_raw[1] if len(scores_raw) > 1 else 0.0
                margin = top_score - second_score
                selected = None
                if top_score >= self.threshold and margin >= self.min_margin:
                    selected = next(
                        (label for label in VALID_LABELS if labels[0].startswith(label.lower())),
                        None,
                    )
                return LabelDecision(
                    selected,
                    "bert_preprocess",
                    mapped,
                    {
                        "top_label": labels[0],
                        "top_score": round(top_score, 4),
                        "second_score": round(second_score, 4),
                        "margin": round(margin, 4),
                    },
                )
            except Exception as exc:
                return LabelDecision(None, "bert_error", zero, {"error": str(exc)})


class RedisJobQueue:
    CLAIM_SCRIPT = """
    local payload = redis.call('LPOP', KEYS[1])
    if payload then
        local job = cjson.decode(payload)
        redis.call('HSET', KEYS[2], job['job_id'], payload)
    end
    return payload
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Connect to Redis and set up the namespaced queue keys for this run.

        The namespace is built from the OS user and the run_id (sanitised via
        safe_key) so that multiple users/runs sharing the same Redis instance
        don't collide. Also registers the Lua CLAIM_SCRIPT used by claim() to
        atomically pop-and-reserve a job.
        """
        try:
            import redis.asyncio as redis_async
        except ImportError as exc:
            raise RuntimeError("Python package 'redis' is required") from exc
        self._redis = redis_async.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=30,
        )
        namespace = f"orca:{safe_key(os.environ.get('USER', 'unknown'))}:{safe_key(config.run_id)}"
        self.pending = f"{namespace}:pending"
        self.processing = f"{namespace}:processing"
        self.completed = f"{namespace}:completed"
        self.failed = f"{namespace}:failed"
        self._claim = self._redis.register_script(self.CLAIM_SCRIPT)

    async def initialise(self, jobs: Sequence[InferenceJob]) -> None:
        """Reset this run's queue in Redis and load it with the given jobs.

        Pings Redis to confirm connectivity, clears any leftover pending/
        processing/completed/failed data from a previous attempt at this
        namespace, then pushes all jobs onto the pending list.
        """
        await self._redis.ping()
        await self._redis.delete(self.pending, self.processing, self.completed, self.failed)
        if jobs:
            await self._redis.rpush(self.pending, *[json_dumps(job.to_record()) for job in jobs])
        LOGGER.info("Redis queue initialised with %d pending jobs", len(jobs))

    async def claim(self) -> InferenceJob | None:
        """Atomically pop the next pending job and mark it as being processed.

        Runs the CLAIM_SCRIPT Lua script so the pop-from-pending +
        add-to-processing happens as a single atomic Redis operation, which is
        safe even with multiple concurrent workers. Returns None once the
        pending queue is empty.
        """
        payload = await self._claim(keys=[self.pending, self.processing], args=[])
        return None if payload is None else InferenceJob.from_record(json.loads(payload))

    async def acknowledge(self, job: InferenceJob, result: Mapping[str, Any]) -> None:
        """Mark a job as successfully completed.

        Atomically removes the job from the "processing" hash and records its
        result in the "completed" hash.
        """
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hdel(self.processing, job.job_id)
            pipe.hset(self.completed, job.job_id, json_dumps(result))
            await pipe.execute()

    async def reject(self, job: InferenceJob, error: str) -> None:
        """Mark a job as failed.

        Atomically removes the job from the "processing" hash and records the
        job plus its error message and failure timestamp in the "failed" hash.
        """
        record = {"job": job.to_record(), "error": error, "failed_at": utc_now()}
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hdel(self.processing, job.job_id)
            pipe.hset(self.failed, job.job_id, json_dumps(record))
            await pipe.execute()

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        await self._redis.aclose()


class CheckpointRepository:
    def __init__(self, paths: RunPaths, *, resume: bool, total_jobs: int) -> None:
        """Set up the on-disk checkpoint/progress files for this run.

        If a checkpoint file already exists and `resume` is False, raises
        FileExistsError to avoid silently overwriting a previous run's results.
        If `resume` is True, loads the existing checkpoint into memory via
        _load() before continuing.
        """
        self.checkpoint_path = paths.timestamp_dir / "results_checkpoint.jsonl"
        self.progress_path = paths.timestamp_dir / "progress.json"
        self.total_jobs = total_jobs
        self._lock = asyncio.Lock()
        self._latest: dict[str, dict[str, Any]] = {}
        if self.checkpoint_path.exists():
            if not resume:
                raise FileExistsError(
                    f"Checkpoint exists: {self.checkpoint_path}. Use --resume or a new run directory."
                )
            self._load()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_progress()

    def _load(self) -> None:
        """Read the existing checkpoint JSONL file back into memory (self._latest),
        keyed by job_id, so a resumed run knows which jobs are already done.
        """
        with self.checkpoint_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    self._latest[str(record["job_id"])] = record
                except Exception as exc:
                    raise ValueError(f"Invalid checkpoint line {line_number}: {exc}") from exc

    def _write_progress(self) -> None:
        """Recompute completed/failed/remaining counts and write them to progress.json,
        giving an at-a-glance view of run progress without parsing the full checkpoint.
        """
        completed = sum(row.get("status") == "completed" for row in self._latest.values())
        failed = sum(row.get("status") == "failed" for row in self._latest.values())
        atomic_write_json(
            self.progress_path,
            {
                "updated_at_utc": utc_now(),
                "total": self.total_jobs,
                "completed": completed,
                "failed": failed,
                "remaining": max(self.total_jobs - completed, 0),
            },
        )

    async def save(self, result: dict[str, Any]) -> None:
        """Persist one job result to the checkpoint file, durably.

        Appends the result as a JSON line, flushes and fsyncs the file so it
        survives a crash, updates the in-memory cache, and refreshes
        progress.json. Guarded by an asyncio.Lock since multiple workers call
        this concurrently.
        """
        async with self._lock:
            with self.checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json_dumps(result) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._latest[str(result["job_id"])] = result
            self._write_progress()

    def completed_job_ids(self) -> set[str]:
        """Return the set of job_ids that have already completed successfully."""
        return {
            job_id for job_id, row in self._latest.items()
            if row.get("status") == "completed"
        }

    def latest_in_manifest_order(self, manifest: Sequence[InferenceJob]) -> list[dict[str, Any]]:
        """Return the checkpointed results for jobs in `manifest`, in the manifest's order.

        Jobs that have no recorded result yet are simply skipped.
        """
        return [self._latest[job.job_id] for job in manifest if job.job_id in self._latest]


class RunMetadataRepository:
    def __init__(self, config: PipelineConfig) -> None:
        """Bind this repository to the run.json file for the given config's run directory."""
        self.path = config.paths.run_dir / "run.json"
        self.config = config

    def _base(self) -> dict[str, Any]:
        """Build the base metadata dict (run id, experiment name, slurm job id,
        full configuration, and immutable settings) shared by fresh and resumed runs.
        """
        return {
            "run_id": self.config.run_id,
            "experiment_name": self.config.experiment_name,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "configuration": self.config.runtime_record(),
            "immutable_settings": dict(self.config.immutable_settings),
        }

    def start(self) -> None:
        """Create or resume run.json at the start of a run.

        On a fresh run: fails if run.json already exists (to avoid clobbering
        a previous run), otherwise writes a new record with status "running".
        On resume: requires run.json to already exist, checks that the
        immutable_settings match the original run (raising ValueError if they
        don't, since changing them mid-resume would be unsafe), and updates
        the record with a new "resumed_at_utc" timestamp.
        """
        if self.config.resume:
            if not self.path.is_file():
                raise FileNotFoundError(f"run.json not found for resume: {self.path}")
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if existing.get("immutable_settings") != dict(self.config.immutable_settings):
                raise ValueError("Resume configuration does not match the original run")
            record = existing
            record["status"] = "running"
            record["resumed_at_utc"] = utc_now()
            record["current_slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
        else:
            if self.path.exists():
                raise FileExistsError(f"run.json already exists: {self.path}")
            record = self._base()
            record.update({"status": "running", "started_at_utc": utc_now()})
        atomic_write_json(self.path, record)

    def finish(self, status: str, *, error: str | None = None) -> None:
        """Update run.json at the end of a run with a final status (e.g. "completed"
        or "failed") and an optional error message, plus an "updated_at_utc" timestamp.
        """
        record = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else self._base()
        record.update({"status": status, "updated_at_utc": utc_now()})
        if error:
            record["error"] = error
        atomic_write_json(self.path, record)


def majority_vote(predictions: Sequence[str | None], fallback: str = "Neutral") -> str:
    """Pick the majority label out of several per-run predictions for the same job.

    Ignores None/invalid entries. Returns `fallback` if there are no valid
    predictions at all, or if the top two labels are tied for first place;
    otherwise returns the single most common valid label.
    """
    valid = [prediction for prediction in predictions if prediction in VALID_LABELS]
    if not valid:
        return fallback
    counts = Counter(valid).most_common(2)
    if len(counts) == 1 or counts[0][1] > counts[1][1]:
        return counts[0][0]
    return fallback


class WorkerPipeline:
    """Shared workflow; dependencies are injected through the constructor."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        queue: RedisJobQueue,
        checkpoint: CheckpointRepository,
        provider: ModelProvider,
        interpreter: OutputInterpreter,
    ) -> None:
        """Wire together the pipeline's dependencies (config, job queue, checkpoint
        store, model provider, output interpreter) and initialise per-run stat
        counters used while workers are running.
        """
        self.config = config
        self.queue = queue
        self.checkpoint = checkpoint
        self.provider = provider
        self.interpreter = interpreter
        self.worker_stats: dict[str, dict[str, Any]] = {}
        self._processed = 0
        self._total_pending = 0
        self._progress_lock = asyncio.Lock()

    async def _process_job(self, worker_id: str, job: InferenceJob) -> dict[str, Any]:
        """Run one inference job to completion and build its result record.

        Fires `config.num_runs` concurrent calls to the model provider for the
        same job (to reduce noise from a single response), interprets each raw
        output into a label via `self.interpreter`, combines them with
        majority_vote(), and returns a result dict containing the final
        prediction plus per-run diagnostics (raw outputs, methods, scores,
        latencies). Raises RuntimeError if every single run failed.
        """
        started = time.perf_counter()
        tasks = [
            self.provider.infer(
                system_prompt=self.config.system_prompt,
                user_message=self.config.user_message,
                image_paths=job.image_paths,
            )
            for _ in range(self.config.num_runs)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        raw_outputs: list[str] = []
        predictions: list[str | None] = []
        methods: list[str] = []
        scores: list[dict[str, float]] = []
        details: list[dict[str, Any] | None] = []
        latencies: list[float] = []
        failed_runs: list[str] = []

        for response in responses:
            if isinstance(response, Exception):
                failed_runs.append(str(response))
                continue
            decision = self.interpreter.interpret(response.raw_output)
            raw_outputs.append(response.raw_output)
            predictions.append(decision.label)
            methods.append(decision.method)
            scores.append(decision.scores)
            details.append(decision.details)
            latencies.append(response.elapsed_ms)

        if not raw_outputs:
            raise RuntimeError("All model runs failed: " + "; ".join(failed_runs))

        result = {
            **job.result_base(),
            "prediction": majority_vote(predictions),
            "predictions_raw": raw_outputs,
            "predictions_preprocessed": predictions,
            "preprocess_methods": methods,
            "preprocess_scores": scores,
            "interpreter_details": details,
            "failed_runs": failed_runs,
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
            "per_run_ms": [round(value, 1) for value in latencies],
            "avg_run_ms": round(sum(latencies) / len(latencies), 1),
            "worker_id": worker_id,
            "status": "completed",
            "completed_at_utc": utc_now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        return result

    async def _worker(self, worker_id: str) -> None:
        """Run one worker's main loop: repeatedly claim a job from the queue,
        process it, checkpoint the result, and acknowledge/reject it in the
        queue, until there are no more pending jobs. Tracks per-worker stats
        (processed/successful/failed counts, wall time, average API latency)
        in `self.worker_stats` for later reporting.
        """
        started = time.perf_counter()
        processed = successful = failed = 0
        total_api_ms = 0.0
        LOGGER.info("[%s] worker started", worker_id)

        while True:
            job = await self.queue.claim()
            if job is None:
                break
            try:
                result = await self._process_job(worker_id, job)
                await self.checkpoint.save(result)
                await self.queue.acknowledge(job, result)
                successful += 1
                total_api_ms += float(result["avg_run_ms"])
                LOGGER.info("[%s] PASS %-8s %s", worker_id, result["prediction"], job.job_id)
            except Exception as exc:
                error = str(exc)
                result = {
                    **job.result_base(),
                    "prediction": "Error",
                    "status": "failed",
                    "error": error,
                    "worker_id": worker_id,
                    "failed_at_utc": utc_now(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                }
                await self.checkpoint.save(result)
                await self.queue.reject(job, error)
                failed += 1
                LOGGER.exception("[%s] FAIL %s", worker_id, job.job_id)

            processed += 1
            async with self._progress_lock:
                self._processed += 1
                LOGGER.info("Progress this attempt: %d/%d", self._processed, self._total_pending)

        elapsed = time.perf_counter() - started
        self.worker_stats[worker_id] = {
            "total_processed": processed,
            "successful": successful,
            "failed": failed,
            "wall_time_s": round(elapsed, 2),
            "avg_api_ms": round(total_api_ms / max(successful, 1), 1),
        }

    async def run(self, pending: Sequence[InferenceJob]) -> tuple[float, dict[str, Any]]:
        """Run the whole pipeline for a batch of pending jobs.

        Loads `pending` into the job queue, then launches `config.workers`
        concurrent `_worker()` coroutines and waits for them all to finish
        (always closing the queue afterward, even on error). Returns the
        total elapsed wall-clock time and the per-worker stats dict.
        """
        self._total_pending = len(pending)
        await self.queue.initialise(pending)
        started = time.perf_counter()
        try:
            await asyncio.gather(*[
                self._worker(f"C{number}")
                for number in range(1, self.config.workers + 1)
            ])
        finally:
            await self.queue.close()
        return time.perf_counter() - started, self.worker_stats


def setup_logging(log_dir: Path) -> None:
    """Configure the module-level LOGGER to write to both stdout and a log file.

    Creates `log_dir` if needed, resets any existing handlers (so this can be
    called safely more than once), and attaches a stream handler plus a file
    handler ("pipeline.log") sharing the same timestamped log format.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def write_manifest(path: Path, jobs: Sequence[InferenceJob]) -> None:
    """Write the full list of jobs for this run to a JSONL manifest file (one job per line).

    This manifest is what allows an exact resume later via load_manifest().
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json_dumps(job.to_record()) + "\n")


def load_manifest(path: Path) -> list[InferenceJob]:
    """Read back the list of jobs from a JSONL manifest file written by write_manifest().

    Raises FileNotFoundError if the manifest doesn't exist, or ValueError if
    any line contains invalid/unparsable JSON (including the offending line number).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Resume manifest not found: {path}")
    jobs: list[InferenceJob] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                jobs.append(InferenceJob.from_record(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid manifest line {line_number}: {exc}") from exc
    return jobs


def build_or_load_manifest(
    *,
    path: Path,
    resume: bool,
    build_jobs: Callable[[], tuple[Sequence[InferenceJob], int]],
) -> tuple[list[InferenceJob], int]:
    """Get the list of jobs to run, either by loading an exact resume manifest
    or by building it fresh.

    If `resume` is True, loads the existing manifest at `path` (so a resumed
    run processes exactly the same jobs as the original) and returns a
    train_count of -1 as a sentinel since it isn't recomputed. Otherwise calls
    `build_jobs()` to construct the jobs and their train_count, writes them to
    `path` as a new manifest, and returns them.
    """
    if resume:
        jobs = load_manifest(path)
        LOGGER.info("Loaded exact resume manifest with %d jobs", len(jobs))
        return jobs, -1
    jobs, train_count = build_jobs()
    write_manifest(path, jobs)
    return list(jobs), train_count


def write_reports(
    *,
    config: PipelineConfig,
    results: Sequence[dict[str, Any]],
    worker_stats: Mapping[str, Mapping[str, Any]],
    pipeline_elapsed_s: float,
    processed_this_attempt: int,
) -> None:
    """Generate all end-of-run report files from the collected results.

    Writes, into `config.paths.output_dir`:
      - results.csv: every result row, via pandas.
      - metrics.json: sample counts, and (if there's at least one successful,
        valid-label result) accuracy, confusion matrix, and a full
        classification report from scikit-learn.
      - timing_per_worker.csv: per-worker processing/timing stats.
      - timing_pipeline.csv: one summary row for the whole pipeline run
        (run id, model, provider, elapsed time, average API latency).
    Raises RuntimeError if pandas is not installed.
    """
    output = config.paths.output_dir
    output.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required") from exc
    pd.DataFrame(results).to_csv(output / "results.csv", index=False)

    successful = [
        row for row in results
        if row.get("status") == "completed" and row.get("prediction") in VALID_LABELS
    ]
    metrics: dict[str, Any] = {
        "records_in_checkpoint": len(results),
        "successful_samples": len(successful),
        "failed_samples": len(results) - len(successful),
    }
    if successful:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        y_true = [str(row["label"]) for row in successful]
        y_pred = [str(row["prediction"]) for row in successful]
        labels = list(VALID_LABELS)
        metrics.update({
            "accuracy": accuracy_score(y_true, y_pred),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, labels=labels, output_dict=True, zero_division=0
            ),
        })
    atomic_write_json(output / "metrics.json", metrics)

    fields = ["worker_id", "total_processed", "successful", "failed", "wall_time_s", "avg_api_ms"]
    with (output / "timing_per_worker.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for worker_id, stats in sorted(worker_stats.items()):
            writer.writerow({"worker_id": worker_id, **stats})

    latencies = [float(row.get("avg_run_ms", 0)) for row in successful if row.get("avg_run_ms")]
    pipeline_row = {
        "run_id": config.run_id,
        "model": config.model,
        "provider": config.provider,
        "processed_this_attempt": processed_this_attempt,
        "pipeline_wall_time_s": round(pipeline_elapsed_s, 2),
        "avg_api_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
    }
    with (output / "timing_pipeline.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pipeline_row))
        writer.writeheader()
        writer.writerow(pipeline_row)
