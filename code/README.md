# test-emotion-recognition-slurm

Multimodal (face + spectrogram) emotion sentiment classification using vision-language models, run as Slurm jobs on a shared GPU cluster. Supports three model providers — a local Ollama server, OpenAI, and Google Gemini — behind a single, reusable pipeline.

Each experiment classifies images (face-only, spectrogram-only, or paired face+spectrogram) into **Positive / Negative / Neutral**, using zero-shot prompting, with checkpointed/resumable execution and automatic CSV/JSON reporting.

> This README was verified directly against the repository contents (`git clone` + inspection) as of the latest commit, rather than written from memory of an earlier project state.

---

## Collaborators

| Name | 
|---|
|Assistant Professor Watanee Jearanaiwongkul |
|Associate Professor Teeradaj Racharak | 
|Rawisara Chantravutikorn |
|Kornkanok Soongswang | 

---

## Datasets

Two datasets are used across the experiments in this repo, both stored on the cluster's shared data volume (not in this git repo):

```
/data1/workspace/students/soongswang-kornkanok/project-01/dataset/
├── P3-V1-Ravdess/
└── silver_dataset_open_mouth/
```

Each dataset provides at least the following, so any experiment script can point at either one (`P3-V1-Ravdess` additionally ships a raw `Audio/` folder — see below):
```
<dataset-name>/
├── Img/
│   ├── Positive/
│   ├── Negative/
│   └── Neutral/
└── Spectrogram/
    ├── Positive/
    ├── Negative/
    └── Neutral/
```
- `Img/<label>/` — face images extracted from video frames, one class-labeled folder per sentiment.
- `Spectrogram/<label>/` — corresponding audio spectrogram images for the same samples, used by the multimodal (`facespec`) and spectrogram-only (`spec`) experiments. Filenames are matched to their paired face image by normalized stem (see `FaceSpectrogramDatasetAdapter` in `lab_pipeline/datasets.py`).

### `P3-V1-Ravdess`
Derived from the **RAVDESS** dataset (Ryerson Audio-Visual Database of Emotional Speech and Song) — a corpus of actors speaking/singing lexically-matched statements with different emotional expressions. This project's `P3-V1` variant combines samples across **both open-mouth and closed-mouth** frames (referenced as the `[o+c]` dataset tag in experiment/job filenames). Used by experiments 1–4 and 9–16.

Unlike the general two-folder layout above, this dataset also keeps the intermediate audio clips it was built from:
```
P3-V1-Ravdess/
├── Img/{Positive,Negative,Neutral}/          # e.g. Img/Negative/01-01-0...jpg (RAVDESS-style filename)
├── Audio/{Positive,Negative,Neutral}/         # intermediate audio clips (not read by the pipeline directly)
└── Spectrogram/{Positive,Negative,Neutral}/
```
(`Audio/` is a preprocessing intermediate — `FaceSpectrogramDatasetAdapter` only ever reads `Img/` and `Spectrogram/`; it exists on disk for reference/reproducibility but isn't consumed by any experiment script.)

**Preparation pipeline** (from the original RAVDESS video/audio source):

*Image extraction:*
1. Each RAVDESS video is split into frames at a **30ms interval** — chosen because an average speaker articulates roughly one phoneme every ~30ms, so this rate captures each distinct mouth/facial shape without excessive redundancy between frames.
2. Frames with no detectable face are discarded. Remaining frames are sorted into `Positive` / `Negative` / `Neutral` folders based on the sentiment class of the source clip.

*Audio → spectrogram extraction:*
1. The corresponding RAVDESS audio is likewise sliced into **30ms segments**, matching the same interval as the image extraction for consistency between the two modalities.
2. Each audio segment is converted to a spectrogram using the **Short-Time Fourier Transform (STFT)** (e.g. via `scipy`), with:
   - Sampling rate: **48 kHz**
   - Window size (W): **1400**
   - Overlap between neighboring segments: **250 samples**
3. Spectrogram magnitude is expressed in **decibels (dB)**.
4. Resulting spectrogram axes: **x-axis 0–0.03 s** (time), **y-axis 0–24 kHz** (frequency, i.e. up to the Nyquist frequency for a 48 kHz sampling rate).

This shared 30ms-interval design is what allows `FaceSpectrogramDatasetAdapter` to pair a face frame with its corresponding spectrogram slice — both were sliced from the same source clip at the same cadence, then matched by filename stem.

### `silver_dataset_open_mouth`
A dataset restricted to **open-mouth-only** frames (referenced as the `[o]` dataset tag in experiment/job filenames). Used by experiments 5–8 and 17–24.

*[Add here: source/provenance of this dataset, why "silver" (e.g. auto-labeled/weakly-supervised vs. a "gold" human-verified set?), sample counts per class, and how "open mouth" frames were selected/filtered from the source video.]*

> Both dataset descriptions above are based on how each dataset is used in the pipeline (folder layout, filename tags, which experiments reference them) rather than on their original creation/labeling process, which I don't have direct knowledge of — please fill in the bracketed notes above with the actual provenance details.

---

## Repository structure

```

.
├── README.md
├── requirements.txt
├── code/
│   ├── experiment/                   # One script per experiment (26 total)
│   │   ├── experiment_1_face[o+c]_no_instruction.py
│   │   ├── experiment_2_face[o+c]_instruction.py
│   │   ├── experiment_3_spec[o+c]_instruction.py
│   │   ├── experiment_4_facespec[o+c]_instruction.py
│   │   ├── experiment_5_face[o]_no_instruction.py
│   │   ├── experiment_6_face[o]_instruction.py
│   │   ├── experiment_7_spec[o]_instruction.py
│   │   ├── experiment_8_facespec[o]_instruction.py
│   │   ├── experiment_9_gpt5_face[o+c]_no_instruction.py
│   │   ├── experiment_10_gpt5_face[o+c]_instruction.py
│   │   ├── experiment_11_gemini_face[o+c]_no_instruction.py
│   │   ├── experiment_12_gpt5_spec[o+c]_instruction.py
│   │   ├── experiment_13_gpt5_facespec[o+c]_instruction.py
│   │   ├── experiment_14_gemini_face[o+c]_instruction.py
│   │   ├── experiment_15_gemini_spec[o+c]_instruction.py
│   │   ├── experiment_16_gemini_facespec[o+c]_instruction.py
│   │   ├── experiment_17_gpt5_face[o]_no_instruction.py
│   │   ├── experiment_18_gpt5_face[o]_instruction.py
│   │   ├── experiment_19_gpt5_spec[o]_instruction.py
│   │   ├── experiment_20_gpt5_facespec[o]_instruction.py
│   │   ├── experiment_21_gemini_face[o]_no_instruction.py
│   │   ├── experiment_22_gemini_face[o]_instruction.py
│   │   ├── experiment_23_gemini_spec[o]_instruction.py
│   │   ├── experiment_24_gemini_facespec[o]_instruction.py
│   │   └── experiment_26_luna_face[o+c]_no_instruction.py
│   └── lab_pipeline/                 # Shared, reusable pipeline package
│       ├── __init__.py
│       ├── core.py                   # 672 lines — pipeline engine (see below)
│       ├── datasets.py                # 338 lines — dataset adapters
│       └── providers.py               # 413 lines — model-provider adapters
└── jobs/
    └── run_<N>_<same-suffix-as-matching-experiment>.slurm   # one per experiment.py
```

Every `run_<N>_...slurm` file has a matching `experiment_<N>_...py` file with the exact same suffix — e.g. `jobs/run_26_luna_face[o+c]_no_instruction.slurm` runs `code/experiment/experiment_26_luna_face[o+c]_no_instruction.py`.

### Filename convention, decoded
```
experiment_<N>_<provider-tag>_<modality>[<dataset-tag>]_<instruction-state>.py
```
| Segment | Meaning |
|---|---|
| `<N>` | Experiment number |
| `<provider-tag>` | Absent = Ollama (the original/baseline provider). `gpt5` = OpenAI `gpt-5-nano`. `gemini` = Google Gemini. `luna` = OpenAI `gpt-5.6-luna`. |
| `<modality>` | `face` (face image only), `spec` (spectrogram image only), `facespec` (paired face + spectrogram). |
| `[<dataset-tag>]` | `[o]` = the **open-mouth-only** dataset (`silver_dataset_open_mouth`). `[o+c]` = the **open+closed-mouth** combined dataset (`P3-V1-Ravdess`). Selects which dataset the job defaults to, via each `.slurm` file's `DATASET_NAME` default. |
| `<instruction-state>` | `no_instruction` = a short, minimal system prompt. `instruction` = the full, detailed cue-based system prompt (mouth shape/eyebrows/eyes/etc. for faces; energy/frequency/harmonics/etc. for spectrograms). |

### Full experiment matrix (provider × model, verified from source)
| # | File suffix | Provider | Model |
|---|---|---|---|
| 1 | `face[o+c]_no_instruction` | ollama | `ministral-3:14b` |
| 2 | `face[o+c]_instruction` | ollama | `ministral-3:14b` |
| 3 | `spec[o+c]_instruction` | ollama | `ministral-3:14b` |
| 4 | `facespec[o+c]_instruction` | ollama | `ministral-3:14b` |
| 5 | `face[o]_no_instruction` | ollama | `ministral-3:14b` |
| 6 | `face[o]_instruction` | ollama | `ministral-3:14b` |
| 7 | `spec[o]_instruction` | ollama | `ministral-3:14b` |
| 8 | `facespec[o]_instruction` | ollama | `ministral-3:14b` |
| 9 | `gpt5_face[o+c]_no_instruction` | openai | `gpt-5-nano` |
| 10 | `gpt5_face[o+c]_instruction` | openai | `gpt-5-nano` |
| 11 | `gemini_face[o+c]_no_instruction` | gemini | `gemini-3.5-flash-lite` |
| 12 | `gpt5_spec[o+c]_instruction` | openai | `gpt-5-nano` |
| 13 | `gpt5_facespec[o+c]_instruction` | openai | `gpt-5-nano` |
| 14 | `gemini_face[o+c]_instruction` | gemini | `gemini-3.5-flash-lite` |
| 15 | `gemini_spec[o+c]_instruction` | gemini | `gemini-3.5-flash-lite` |
| 16 | `gemini_facespec[o+c]_instruction` | gemini | `gemini-3.5-flash-lite` |
| 17 | `gpt5_face[o]_no_instruction` | openai | `gpt-5-nano` |
| 18 | `gpt5_face[o]_instruction` | openai | `gpt-5-nano` |
| 19 | `gpt5_spec[o]_instruction` | openai | `gpt-5-nano` |
| 20 | `gpt5_facespec[o]_instruction` | openai | `gpt-5-nano` |
| 21 | `gemini_face[o]_no_instruction` | gemini | `gemini-3.5-flash-lite` |
| 22 | `gemini_face[o]_instruction` | gemini | `gemini-3.5-flash-lite` |
| 23 | `gemini_spec[o]_instruction` | gemini | `gemini-3.5-flash-lite` |
| 24 | `gemini_facespec[o]_instruction` | gemini | `gemini-3.5-flash-lite` |
| 26 | `luna_face[o+c]_no_instruction` | openai | `gpt-5.6-luna` |

Note: `gpt-5-nano` and `gpt-5.6-luna` are OpenAI's reasoning-family models — they only accept `TEMPERATURE = 1.0` / `TOP_P = 1.0` (custom values throw a `400 unsupported_value` error). Experiment 26 additionally sets `REASONING_EFFORT = "none"` to minimize reasoning-token cost.

---

## Architecture: `lab_pipeline`

Follows four design patterns so that adding a new experiment, dataset modality, or model provider never requires touching the shared pipeline logic:

- **Strategy** — every model provider implements the same `ModelProvider.infer()` interface (in `providers.py`).
- **Adapter** — `OllamaVisionAdapter` / `OpenAIVisionAdapter` / `GeminiVisionAdapter` translate that common interface into each provider's own request format.
- **Factory** — `ProviderFactory.create(provider_name, ...)` builds whichever adapter is configured.
- **Dependency injection** — `WorkerPipeline` receives its provider, queue, checkpoint repository, and output interpreter rather than constructing them itself.

### `lab_pipeline/datasets.py`
| Class | Purpose |
|---|---|
| `FaceSpectrogramDatasetAdapter` | Multimodal: pairs a face image with a spectrogram image by matching normalized filename stems (strips `-img-`/`-spec-` tags). `strict_pairs=True` raises if any file is left unmatched; `allow_positional_fallback=True` allows pairing by sorted position when *no* filename match exists at all for a class (off by default in every experiment — the riskier option). |
| `FaceImageDatasetAdapter` | Single-modality: walks `Img/{Positive,Negative,Neutral}/`, one `InferenceJob` per face image. |
| `SpectrogramImageDatasetAdapter` | Single-modality: same, for `Spectrogram/{Positive,Negative,Neutral}/`. |

All three expose `.build()` (returns all `InferenceJob`s) and `.split(jobs, test_size, seed)` (stratified `sklearn.train_test_split`). Hidden files (`._*` and dotfiles from e.g. macOS zip artifacts) are filtered out automatically.

### `lab_pipeline/providers.py`
| Class | Purpose |
|---|---|
| `OllamaVisionAdapter` | Local Ollama server via the `ollama` SDK's `AsyncClient`. Images sent as a flat base64 `images` list. |
| `OpenAIVisionAdapter` | OpenAI Chat Completions API. Images sent as base64 data-URI `image_url` content blocks. Optional `reasoning_effort` (`"minimal"` / `"none"` / etc.) for `gpt-5`-family reasoning models — omitted from the request entirely if left `None`, so non-reasoning models are unaffected. |
| `GeminiVisionAdapter` | Gemini API via `google-genai`'s `Client`. Images sent as inline `Part.from_bytes`. Optional `thinking_level` (default `"minimal"`) to reduce Gemini 3.x thinking-token cost — note this is a *different* parameter than Gemini 2.5-era's `thinking_budget`; sending the wrong one for a given model generation returns a `400` error. |
| `ProviderFactory.create(provider, ...)` | Picks the adapter class based on the `provider` string (`"ollama"` / `"openai"` / `"gemini"`), and forwards `api_key`, `thinking_level`, `reasoning_effort` through to whichever adapter is constructed. |

Both API-based adapters retry failed requests with exponential backoff, applying a **much longer** backoff specifically when the error text contains `429` / `rate_limit` / `RESOURCE_EXHAUSTED` than for other transient errors.

### `lab_pipeline/core.py`
| Component | Purpose |
|---|---|
| `RunPaths` | The on-disk layout for one run: `run_dir/{log,timestamp,output}/`. |
| `PipelineConfig` | Frozen dataclass holding every experiment setting (provider, model, prompt, sampling params, retry policy, Redis connection, BERT settings, `immutable_settings` for provenance). |
| `InferenceJob` | One unit of work: `job_id`, `label`, `image_paths` (a tuple — length 1 for single-modality, 2 for paired), `metadata`. |
| `OutputInterpreter` | Maps a model's raw free-text answer to `Positive`/`Negative`/`Neutral`, optionally backed by a DistilBERT zero-shot classifier (`BERT_ENABLED`). |
| `RedisJobQueue` | Distributes pending jobs across `NUM_WORKERS` async workers. |
| `CheckpointRepository` | Disk-backed record of completed jobs — enables resume after a crash, Slurm timeout, or provider billing cutoff without redoing finished work. |
| `RunMetadataRepository` | Records run status (`"completed"` / `"incomplete"` / `"failed"` / `"validated"`) and timing to `run.json`. |
| `majority_vote()` | Combines `NUM_RUNS` repeated predictions per job into one final label (ties fall back to `"Neutral"`). |
| `WorkerPipeline` | The worker loop itself: pull job → call provider `NUM_RUNS` times → majority vote → interpret output → checkpoint result. |
| `build_or_load_manifest()` | Builds the test-set job list fresh, or loads an existing `manifest.jsonl` when resuming. |
| `write_reports()` | Writes `output/results.csv`, `output/metrics.json`, `output/timing_per_worker.csv`, `output/timing_pipeline.csv` (see exact contents below). |

---

## Experiment script shape

Every `experiment_*.py` follows the identical structure and differs only in its configuration block:

```python
FACE_SUBDIRECTORY = "Img"                # and/or SPECTROGRAM_SUBDIRECTORY
PROVIDER = "openai"                       # "ollama" | "openai" | "gemini"
MODEL = "gpt-5-nano"
TEMPERATURE = 1.0
TOP_P = 1.0
REASONING_EFFORT = "none"                 # OpenAI reasoning models only
THINKING_LEVEL = "minimal"                # Gemini 3.x only
NUM_RUNS = 3                               # calls per image, majority vote
NUM_WORKERS = 3
MAX_CONCURRENCY = 3
MAX_SAMPLES = 0                            # 0 = full test set; >0 = smoke-test cap
BERT_ENABLED = False
SYSTEM_PROMPT = "..."                      # the actual instruction/prompt
```

then: `resolve_dataset_paths()` → `create_config()` → `run()` (build dataset → resolve/load manifest → resolve pending jobs from checkpoint → `WorkerPipeline.run()` → `write_reports()`) → `main()`.

### Adding a new experiment
Copy the closest existing script and change only what genuinely differs: `PROVIDER`/`MODEL`, `SYSTEM_PROMPT`, the dataset adapter used, and the `"experiment"`/`experiment_name` labels. Everything from `build_jobs()` downward needs no changes.

---

## Slurm scripts (`jobs/*.slurm`)

1. **User configuration** — `USER_NAME`, `PROJECT_NAME`, `DATASET_NAME` (each overridable via environment at submit time; `DATASET_NAME`'s default matches the `[o]`/`[o+c]` tag in the filename).
2. **`DEST_LABEL` auto-derivation** — derived from the matching experiment script's filename (`experiment_<N>_<rest>.py` → `results-<rest>`), used as the destination subfolder under `complete-runs/`. Overridable via an explicit `DEST_LABEL=...`.
3. **Redis check** — pings the Redis service before proceeding.
4. **Reads `PROVIDER`/`MODEL` directly from the `.py` file** via an embedded Python AST parser — the Slurm script never hardcodes which model/provider a given experiment uses.
5. **Model backend setup**:
   - `PROVIDER == "ollama"` → starts a local `ollama serve` process on a free port, waits for `/api/tags` to respond, verifies the model is available.
   - `PROVIDER == "openai"` → sets `MODEL_HOST` to the OpenAI API endpoint, requires `OPENAI_API_KEY`.
   - `PROVIDER == "gemini"` → sets `MODEL_HOST`, requires `GOOGLE_API_KEY`.
6. **conda environment activation** — `conda activate emotion-recognition` before resolving `PYTHON`.
7. **Runs the experiment**: `srun --unbuffered "$PYTHON" "$PROGRAM"` — this is internal to the script; you never type `srun` yourself, only `sbatch`.
8. **On success only** (exit code `0`), copies the completed run directory into `complete-runs/<DEST_LABEL>/`.

---

## Running an experiment

```bash
# 1. Load API keys into the current shell (needed for openai/gemini providers)
source ~/.env.llm

# 2. Submit
DATASET_NAME="<dataset-folder-name>" sbatch "jobs/run_9_gpt5_face[o+c]_no_instruction.slurm"

# 3. Watch
squeue -u <your-username>
tail -f /data1/workspace/students/<user>/project-01/slurm-runs/run-<jobid>-*/log/stdout.log
```

### Smoke-testing before a full run
Set `MAX_SAMPLES` to a small number (e.g. `10`) before running the full dataset, to confirm the provider/prompt/pipeline are wired correctly (and, for paid API providers, to check actual per-call cost) before committing to a run of potentially tens of thousands of images.

### Resuming an interrupted run
Completed results are already saved to the checkpoint on disk even if a job is killed (Slurm time limit, provider billing exhaustion, network failure). Resume the *same* run directory:
```bash
source ~/.env.llm
RESUME_RUN_DIR="/data1/workspace/students/<user>/project-01/slurm-runs/run-<jobid>-<timestamp>" \
  DATASET_NAME="<same-dataset-as-before>" \
  sbatch "jobs/run_9_gpt5_face[o+c]_no_instruction.slurm"
```
This gets a **new** Slurm job ID (expected — IDs are never reused) but reuses the same run directory, only processing jobs that weren't already completed. `output/results.csv` etc. end up containing the merged results from every attempt.

### Concurrency tuning
For a **local Ollama** provider, raising `NUM_WORKERS`/`MAX_CONCURRENCY` has little effect — a single GPU serializes inference regardless of concurrent request count (confirmed empirically: wall time ≈ `jobs × avg_api_ms`, i.e. effectively sequential). For **API providers** with a high rate limit, raising concurrency (e.g. `NUM_WORKERS=100`, as in experiments 17+) gives close to proportional speedup, limited mainly by local CPU (image encoding) rather than the API.

---

## Environment variables reference

| Variable | Set by | Purpose |
|---|---|---|
| `DATASET_ROOT` | Slurm script | Root folder containing `Img/` and/or `Spectrogram/` |
| `RUN_DIR` | Slurm script | Where this run's logs/manifest/output get written |
| `RESUME` | Slurm script | `1` if resuming, else `0` |
| `MODEL_HOST` | Slurm script | Ollama server URL, or the OpenAI/Gemini API endpoint |
| `OPENAI_API_KEY` | Your shell (`.env.llm`), passed through by Slurm | OpenAI authentication |
| `GOOGLE_API_KEY` | Your shell (`.env.llm`), passed through by Slurm | Gemini authentication |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | Slurm script (defaults: `127.0.0.1` / `6379` / `0`) | Redis connection |

`.env.llm` (outside the git repo, `chmod 600`) holds the two API keys and is `source`d manually before each `sbatch` call — never committed to git, never written into any run's output files.

---

## Output files, per run

| File | Contents |
|---|---|
| `timestamp/manifest.jsonl` | The frozen list of test-set jobs for this run |
| `output/results.csv` | Per-job model prediction vs. ground truth (`pd.DataFrame(results)`) |
| `output/metrics.json` | `accuracy`, `confusion_matrix`, `classification_report` (via scikit-learn), plus `records_in_checkpoint`/`successful_samples`/`failed_samples` |
| `output/timing_per_worker.csv` | Per-worker: `total_processed`, `successful`, `failed`, `wall_time_s`, `avg_api_ms` |
| `output/timing_pipeline.csv` | `run_id`, `model`, `provider`, `processed_this_attempt`, `pipeline_wall_time_s`, `avg_api_ms` (`avg_api_ms` is milliseconds, not minutes — divide `pipeline_wall_time_s` by 60 for run time in minutes) |
| `log/stdout.log` / `log/stderr.log` | Full run log |
| `run.json` | Run status, timestamps, immutable config snapshot (including a SHA-256 hash of the system prompt for provenance) |

---

## Known issues / gotchas encountered during development

- **`mimetypes` import missing** in `providers.py` surfaces as `name 'mimetypes' is not defined` inside every job's error, not as an obvious import error — check the top-of-file imports if every job fails identically.
- **`DATASET_NAME` starting with a hyphen** (e.g. a typo) fails the Slurm script's path-name validation regex silently and immediately, before the job even shows up in `squeue`. Check `bootstrap-<jobid>.out/.err` for `fail()` messages when a job seems to vanish instantly.
- **Provider errors mentioning a model is "no longer available"** are account/billing/tier issues, not code bugs — confirm by reproducing the same error with a bare, non-async, text-only SDK call outside the pipeline entirely.
- **Gemini model names and parameter names have changed across generations** during this project (`gemini-2.0-flash` → deprecated; `thinking_budget` → `thinking_level` between the 2.5 and 3.x generations) — always check `client.models.list()` for current availability rather than trusting a model name from documentation or an earlier experiment.
- **`gpt-5-nano` / `gpt-5.6-luna` force `TEMPERATURE=1.0`/`TOP_P=1.0`** — a custom value throws `400 unsupported_value`.
