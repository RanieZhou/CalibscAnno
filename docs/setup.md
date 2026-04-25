# CalibscAnno setup

This project treats `external/scFoundation/` as upstream reference code. Project code should live in `scripts/`, `docs/`, and future CalibscAnno modules.

## Local environment

Use the local environment for light experiments only: data inspection, split creation, PCA embeddings, baseline classifiers, and result summaries.

```bash
python3 -m venv .venv
source .venv/bin/activate
mkdir -p .cache/matplotlib
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
```

Quick check:

```bash
python -c "import anndata, numpy, pandas, sklearn, scipy; print('local env ok')"
```

## Server environment

Use the server environment for expensive experiments: scFoundation embedding extraction, multi-seed full runs, calibration/rejection sweeps, and external benchmarks.

On the H100 server, install PyTorch according to the server CUDA stack first if the cluster image does not already provide it. Then install the remaining requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
```

Quick check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import scanpy, einops, local_attention; print('server env ok')"
```

## Data policy

Generated data and large local artifacts are ignored by Git:

```text
data/
logs/
outputs/
external/scFoundation/annotation/data/
```

The official annotation archive is kept under `external/scFoundation/annotation/annotation_data.zip`. If `data/` is missing, regenerate project artifacts from the scripts rather than committing large intermediates.

## Experiment metadata

Experiment scripts record lightweight runtime metadata for later reporting:

```text
runtime_seconds
fit_seconds
predict_seconds
peak_memory_mb
started_at_utc
finished_at_utc
```

Baseline metrics are written directly into their result CSV files under `results/tables/`. Embedding runs append one row to `results/tables/embedding_runs.csv`.

`peak_memory_mb` records the peak CPU resident memory of the current process. GPU memory should be recorded separately in server-side PyTorch/scFoundation scripts.

## Git workflow

The current clone uses:

```bash
git remote -v
git status --short --branch
```

Recommended workflow:

```bash
git checkout -b codex/<task-name>
git add <changed-files>
git commit -m "<short message>"
git push -u origin codex/<task-name>
```

The GitHub CLI is optional. If installed, it can be used later for pull requests:

```bash
gh auth login
gh pr create --draft
```
