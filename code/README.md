# File 4 reusable Slurm experiment

This project converts File 4 into a reusable Python pipeline that runs through Slurm.

## Server layout

The active code and Python environment are stored under:

```text
/home/workspace/<linux-account>/<project-name>/
├── .venv/
├── code/
│   ├── experiment_4_multimodal.py
│   └── lab_pipeline/
└── jobs/
    └── run_4_multimodal.slurm
```

Datasets and run results are stored under:

```text
/data1/workspace/students/<user-name>/
```

## User configuration

Open:

```text
jobs/run_4_multimodal.slurm
```

Normally, users only need to change:

```bash
USER_NAME="soongswang-kornkanok"
PROJECT_NAME="template-pun"
DATASET_NAME="P3-V1-Ravdess"
```

The Linux account is detected automatically from the Slurm job.

For example, these values produce:

```text
Code root:
/home/workspace/high-user-01/template-pun

Dataset root:
/data1/workspace/students/soongswang-kornkanok/P3-V1-Ravdess

Run root:
/data1/workspace/students/soongswang-kornkanok/template-pun/slurm-runs
```

## Dataset structure

Experiment 4 expects the selected dataset to contain:

```text
P3-V1-Ravdess/
├── Img/
│   ├── Positive/
│   ├── Negative/
│   └── Neutral/
└── Spectrogram/
    ├── Positive/
    ├── Negative/
    └── Neutral/
```

The Slurm script exports `DATASET_ROOT`, and Python automatically selects:

```text
DATASET_ROOT/Img
DATASET_ROOT/Spectrogram
```

Users do not need to enter these complete paths manually.

## Run structure

Every new Slurm job automatically creates:

```text
/data1/workspace/students/<user-name>/<project-name>/slurm-runs/
└── run-<job-id>-<UTC-timestamp>/
    ├── run.json
    ├── log/
    │   ├── bootstrap.out
    │   ├── bootstrap.err
    │   ├── stdout.log
    │   ├── stderr.log
    │   ├── pipeline.log
    │   └── ollama.log
    ├── timestamp/
    │   ├── manifest.jsonl
    │   ├── results_checkpoint.jsonl
    │   └── progress.json
    └── output/
        ├── results.csv
        ├── metrics.json
        └── timing reports
```

## First validation

To process only three test samples, open:

```text
code/experiment_4_multimodal.py
```

Set:

```python
MAX_SAMPLES = 3
```

Then submit:

```bash
cd /home/workspace/high-user-01/template-pun
sbatch jobs/run_4_multimodal.slurm
```

Each sample contains one face image and one matching spectrogram image.

After the validation succeeds, restore:

```python
MAX_SAMPLES = 0
```

`MAX_SAMPLES = 0` means there is no sample limit, so the program processes the complete test split.

## Full run

After restoring `MAX_SAMPLES = 0`, submit:

```bash
cd /home/workspace/high-user-01/template-pun
sbatch jobs/run_4_multimodal.slurm
```

## Resume

To continue an interrupted run, use its original run directory:

```bash
RESUME_RUN_DIR="/data1/workspace/students/soongswang-kornkanok/template-pun/slurm-runs/run-1234-TIMESTAMP" \
sbatch jobs/run_4_multimodal.slurm
```

The new Slurm job reads the original manifest and checkpoint files. It then places only unfinished samples back into Redis.

Do not change the model, prompt, dataset, split settings, or `MAX_SAMPLES` when resuming the same run.

## Responsibilities

The Slurm script:

* requests CPU, memory, and one GPU;
* derives paths from the user, project, and dataset names;
* creates the run directory;
* checks the Redis system service;
* starts Ollama inside the Slurm allocation;
* exports runtime paths to Python;
* executes the Python experiment.

The Python program:

* selects the `Img/` and `Spectrogram/` folders;
* builds face–spectrogram pairs;
* creates Redis workers;
* calls the configured model;
* saves durable checkpoints;
* resumes unfinished work;
* exports CSV and JSON reports.

## Model and experiment settings

Edit these settings in:

```text
code/experiment_4_multimodal.py
```

Examples:

```python
MODEL = "ministral-3:14b"
NUM_RUNS = 3
NUM_WORKERS = 3
MAX_CONCURRENCY = 3
MAX_SAMPLES = 0
```

The Slurm script reads `PROVIDER` and `MODEL` automatically from the Python file, so these values are not configured twice.

## Design patterns

* **Strategy:** `ModelProvider`
* **Adapter:** `OllamaVisionAdapter`
* **Factory:** `ProviderFactory`
* **Dependency injection:** `WorkerPipeline` receives the provider, Redis queue, checkpoint repository, and output interpreter.

File 5 can later reuse the same `lab_pipeline` package and add only a face-only dataset adapter and experiment entry point.
