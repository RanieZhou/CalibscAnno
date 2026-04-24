from pathlib import Path
import pickle
import numpy as np
import pandas as pd

DATA_DIR = Path("external/scFoundation/annotation/data")


def show_obj(name, obj, indent="  "):
    print(f"{indent}type: {type(obj)}")

    if isinstance(obj, np.ndarray):
        print(f"{indent}shape: {obj.shape}")
        print(f"{indent}dtype: {obj.dtype}")
        if obj.size > 0:
            print(f"{indent}first values: {obj.reshape(-1)[:10]}")

    elif isinstance(obj, pd.DataFrame):
        print(f"{indent}shape: {obj.shape}")
        print(f"{indent}columns: {list(obj.columns)[:20]}")
        print(obj.head())

    elif isinstance(obj, dict):
        print(f"{indent}dict keys: {list(obj.keys())[:20]}")
        for k in list(obj.keys())[:10]:
            v = obj[k]
            print(f"{indent}key={k}")
            show_obj(str(k), v, indent + "  ")

    elif isinstance(obj, (list, tuple)):
        print(f"{indent}length: {len(obj)}")
        if len(obj) > 0:
            print(f"{indent}first item:")
            show_obj("first", obj[0], indent + "  ")

    else:
        print(f"{indent}repr: {repr(obj)[:500]}")


def inspect_h5ad(path):
    try:
        import anndata as ad
    except ImportError:
        print("  anndata not installed. Run: pip install anndata")
        return

    adata = ad.read_h5ad(path, backed="r")
    print(f"  h5ad shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)[:50]}")
    print(f"  var columns: {list(adata.var.columns)[:30]}")

    candidate_cols = [
        "cell_type", "celltype", "CellType", "cell_type1",
        "label", "labels", "annotation", "celltypist_cell_label",
        "celltype.l1", "celltype.l2", "celltype.l3"
    ]

    for col in candidate_cols:
        if col in adata.obs.columns:
            print(f"\n  possible label column: {col}")
            print(adata.obs[col].value_counts().head(20))

    print("\n  obs head:")
    print(adata.obs.head())


def main():
    print("DATA_DIR:", DATA_DIR.resolve())

    for path in sorted(DATA_DIR.iterdir()):
        print("\n" + "=" * 100)
        print("FILE:", path.name)
        print("SIZE:", round(path.stat().st_size / 1024 / 1024, 2), "MB")

        if path.suffix == ".npy":
            arr = np.load(path, allow_pickle=True)
            show_obj(path.name, arr)

        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                obj = pickle.load(f)
            show_obj(path.name, obj)

        elif path.suffix == ".h5ad":
            inspect_h5ad(path)

        else:
            print("  skipped")


if __name__ == "__main__":
    main()