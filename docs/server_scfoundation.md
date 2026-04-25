# Server scFoundation embedding run

Use this on the H100 server to extract frozen scFoundation cell embeddings. The wrapper keeps project outputs standardized and leaves `external/scFoundation/` unmodified.

## 1. Prepare environment

Install server dependencies:

```bash
python -m pip install -r requirements-server.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Place model checkpoint

Download the scFoundation checkpoint according to the upstream instructions and place it at:

```text
external/scFoundation/model/models/models.ckpt
```

The wrapper checks this path before launching extraction.

## 3. Prepare processed data

If `data/processed/` is missing on the server, regenerate it:

```bash
python scripts/01_prepare_official_h5ad.py
```

Expected inputs:

```text
data/processed/zheng68k/adata.h5ad
data/processed/segerstolpe/adata.h5ad
```

## 4. Dry-run check

Run this first. It validates paths and input shapes but does not launch the model:

```bash
python scripts/09_extract_scfoundation_embeddings.py \
  --datasets zheng68k segerstolpe \
  --dry_run
```

`--dry_run` does not require the checkpoint to exist; it records whether the expected checkpoint path is present.

## 5. Extract embeddings

Full run:

```bash
python scripts/09_extract_scfoundation_embeddings.py \
  --datasets zheng68k segerstolpe \
  --pool_type all \
  --tgthighres t4 \
  --pre_normalized F \
  --version ce
```

Expected standardized outputs:

```text
data/embeddings/zheng68k_scfoundation_all_t4_ce.npy
data/embeddings/segerstolpe_scfoundation_all_t4_ce.npy
```

Each output also has a metadata JSON file with command, runtime, and peak GPU memory from `nvidia-smi` when available.

## 6. Benchmark scFoundation embeddings

After extraction, reuse the existing open-set score benchmark:

```bash
python scripts/06_benchmark_open_set_scores.py \
  --embedding data/embeddings/zheng68k_scfoundation_all_t4_ce.npy \
  --output results/tables/open_set_score_benchmark_scfoundation.csv \
  --summary_output results/tables/open_set_score_benchmark_scfoundation_summary.csv
```

Then create a CalibscAnno-v0 style summary by pointing the summary script to the scFoundation benchmark:

```bash
python scripts/08_build_calibscanno_v0_summary.py \
  --input results/tables/open_set_score_benchmark_scfoundation_summary.csv \
  --output results/tables/calibscanno_v0_main_results_scfoundation.csv
```

## Notes

- The upstream scFoundation script performs cell embedding inference one cell at a time for `output_type=cell`.
- The wrapper standardizes outputs but does not change upstream inference internals.
- Use `--demo` for a 10-cell smoke run.
- Use `--force` to overwrite an existing standardized output.
