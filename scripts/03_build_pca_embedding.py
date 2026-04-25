from pathlib import Path
import argparse
import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from experiment_utils import append_row_csv, peak_memory_mb, runtime_context, timed_block, utc_now_iso


def normalize_log1p(X):
    """
    对非负表达矩阵做 normalize_total + log1p。
    如果发现数据含负值，则认为可能已经处理过，直接返回。
    """
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)

        if X.data.size > 0 and X.data.min() < 0:
            print("Detected negative values in sparse X. Skip normalize/log1p.")
            return X

        cell_sum = np.asarray(X.sum(axis=1)).reshape(-1)
        cell_sum[cell_sum == 0] = 1.0
        scale = 1e4 / cell_sum

        X = sparse.diags(scale).dot(X)
        X.data = np.log1p(X.data)
        return X

    X = np.asarray(X, dtype=np.float32)

    if X.min() < 0:
        print("Detected negative values in dense X. Skip normalize/log1p.")
        return X

    cell_sum = X.sum(axis=1, keepdims=True)
    cell_sum[cell_sum == 0] = 1.0
    X = X / cell_sum * 1e4
    X = np.log1p(X)
    return X


def main():
    run_started_at = utc_now_iso()
    with timed_block() as total_elapsed:
        run_main(run_started_at, total_elapsed)


def run_main(run_started_at, total_elapsed):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zheng68k")
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adata_path = Path("data/processed") / args.dataset / "adata.h5ad"
    out_dir = Path("data/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {adata_path}")
    adata = ad.read_h5ad(adata_path)

    print("adata shape:", adata.shape)

    X = adata.X
    X = normalize_log1p(X)

    print(f"Running TruncatedSVD with n_components={args.n_components}")

    with timed_block() as svd_elapsed:
        svd = TruncatedSVD(
            n_components=args.n_components,
            random_state=args.seed,
        )

        emb = svd.fit_transform(X)

    print("Explained variance ratio sum:", svd.explained_variance_ratio_.sum())

    with timed_block() as scale_elapsed:
        scaler = StandardScaler()
        emb = scaler.fit_transform(emb).astype(np.float32)

    out_path = out_dir / f"{args.dataset}_pca{args.n_components}.npy"
    np.save(out_path, emb)

    print(f"Saved embedding to: {out_path}")
    print("embedding shape:", emb.shape)

    run_row = {
        "run_type": "pca_embedding",
        "dataset": args.dataset,
        "embedding": out_path.name,
        "output_path": str(out_path),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_components": int(args.n_components),
        "seed": int(args.seed),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "svd_seconds": float(svd_elapsed()),
        "scale_seconds": float(scale_elapsed()),
        "runtime_seconds": float(total_elapsed()),
        "peak_memory_mb": peak_memory_mb(),
        "started_at_utc": run_started_at,
        "finished_at_utc": utc_now_iso(),
    }
    run_row.update(runtime_context())

    metadata_path = Path("results/tables/embedding_runs.csv")
    append_row_csv(metadata_path, run_row)
    print(f"Saved run metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
