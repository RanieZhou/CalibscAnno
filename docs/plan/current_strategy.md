# CalibscAnno current strategy

Updated: 2026-04-25

This document is the current working direction. It supersedes older numeric
claims in `docs/plan/研究方案.md`, while preserving the same broad research
topic: open-set single-cell annotation with unknown-cell rejection.

## 1. Current research question

The paper should answer four questions clearly:

```text
1. What problem exists?
   Closed-set cell annotation methods can be overconfident when query cells
   contain unseen cell types.

2. What do current/simple baselines do?
   MLP confidence, raw prototype distance, and statistical distance scores give
   different unknown-detection behavior. Softmax confidence is especially weak.

3. What do we do?
   Use frozen cell embeddings, then add a lightweight calibrated distance-based
   rejection layer for known-label prediction and unknown-cell detection.

4. What problem does it solve?
   It improves unknown-cell detection while preserving a simple, interpretable
   known-cell assignment path.
```

## 2. Current evidence

Main dataset:

```text
Zheng68K: 6595 cells x 19264 genes, 11 cell types
Holdouts: CD19+ B, CD56+ NK, Dendritic
Protocol: leave-one-cell-type-out open-set evaluation, seeds 0-4
```

The comparison table uses matched split seeds 0-4 for PCA50 and scFoundation.
The older all-result PCA table also contains seed 42 and should be treated as
historical supporting output rather than the paper-facing comparison.

scFoundation extraction on H100:

```text
zheng68k embedding:      6595 x 3072, 105.46 s
segerstolpe embedding:    427 x 3072, 60.29 s
```

The `peak_gpu_memory_mb_nvidia_smi` field in the metadata table is a coarse
wall-clock poll of total GPU memory used during the wrapper process. Treat it
as reporting support, not as a precise per-process profiler.

Paper-facing result tables:

```text
results/tables/embedding_method_comparison.csv
results/tables/paper_claim_snapshot.csv
results/tables/scfoundation_embedding_run_metadata.csv
results/tables/calibscanno_v0_main_results_scfoundation.csv
```

Average unknown AUROC across the three Zheng68K holdouts:

```text
Embedding       MLP maxprob   Raw prototype   class-z prototype   Diag. Mahalanobis
PCA50           0.426         0.562           0.771               0.805
scFoundation    0.391         0.744           0.768               0.860
```

Key interpretation:

```text
1. MLP max probability is weak for unknown detection on both embeddings.
2. scFoundation strongly improves raw prototype distance over PCA.
3. class-z prototype rescues cases where raw prototype is misleading,
   especially Dendritic.
4. Diagonal Mahalanobis is currently the strongest overall rejection score.
```

## 3. Important caveat

The original class-z prototype module is not uniformly better than raw prototype
distance under scFoundation.

Example:

```text
CD56+ NK, scFoundation:
raw prototype AUROC   = 0.888
class-z AUROC         = 0.673
Mahalanobis AUROC     = 0.941
```

Therefore the paper should not claim that class-z is the final universal
solution. A stronger and more defensible framing is:

> CalibscAnno is a calibrated distance-based rejection framework. Prototype
> class-z is an interpretable class-conditional calibration component, while
> diagonal Mahalanobis is the current strongest statistical-distance rejection
> score.

## 4. Recommended method framing

Use this as the current method story:

```text
Frozen embedding encoder
        ↓
Known-label assignment head
        - MLP
        - nearest prototype
        ↓
Unknown rejection scores
        - MLP max probability
        - raw prototype distance
        - class-conditional prototype z-score
        - diagonal Mahalanobis distance
        ↓
Validation-calibrated threshold at fixed known coverage
        ↓
known label or unknown
```

The likely main method for the next paper draft should be:

```text
CalibscAnno-MD: diagonal Mahalanobis calibrated rejection on frozen embeddings
```

The class-z prototype score should remain as:

```text
CalibscAnno-PZ: interpretable prototype z-score ablation/component
```

This avoids overstating a weaker module and lets the strongest measured score
carry the main result.

## 5. Next experiments

Immediate next steps:

```text
1. Add a compact CalibscAnno-v1 summary that treats diagonal Mahalanobis as the
   main method and class-z as an ablation.
2. Generate risk-coverage tables/plots for PCA50 and scFoundation.
3. Add one external validation dataset if runtime/data access is manageable.
4. Only after that, start writing the Results section and figures.
```

Decision rule:

```text
If external validation preserves the pattern that Mahalanobis or calibrated
distance beats MLP confidence and raw prototype on average, proceed to paper
drafting. If not, narrow the claim to Zheng68K/scFoundation-official benchmark
and present the work as a fast technical note-style paper.
```
