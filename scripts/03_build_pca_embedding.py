from pathlib import Path
import argparse
import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


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

    svd = TruncatedSVD(
        n_components=args.n_components,
        random_state=args.seed,
    )

    emb = svd.fit_transform(X)

    print("Explained variance ratio sum:", svd.explained_variance_ratio_.sum())

    scaler = StandardScaler()
    emb = scaler.fit_transform(emb).astype(np.float32)

    out_path = out_dir / f"{args.dataset}_pca{args.n_components}.npy"
    np.save(out_path, emb)

    print(f"Saved embedding to: {out_path}")
    print("embedding shape:", emb.shape)


if __name__ == "__main__":
    main()