from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import anndata as ad


RAW_DIR = Path("external/scFoundation/annotation/data")
OUT_DIR = Path("data/processed")


DATASETS = {
    "segerstolpe": "celltypist_0806_seg.h5ad",
    "zheng68k": "celltypist_0806_zheng68k.h5ad",
}


def prepare_dataset(name: str, filename: str):
    input_path = RAW_DIR / filename
    output_dir = OUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {name}")
    print(f"Input: {input_path}")

    adata = ad.read_h5ad(input_path)

    required_cols = ["true_labels", "true_strlabels"]
    for col in required_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing required column: {col}")

    labels_df = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "label_id": adata.obs["true_labels"].astype(int).values,
        "cell_type": adata.obs["true_strlabels"].astype(str).values,
    })

    class_counts = (
        labels_df
        .groupby(["label_id", "cell_type"])
        .size()
        .reset_index(name="n_cells")
        .sort_values(["label_id"])
    )

    label_names = (
        class_counts
        .sort_values("label_id")["cell_type"]
        .values
    )

    # 保存一份标准 h5ad
    adata.write_h5ad(output_dir / "adata.h5ad")

    # 保存标签表
    labels_df.to_csv(output_dir / "labels.csv", index=False)

    # 保存类别名
    np.save(output_dir / "label_names.npy", label_names)

    # 保存类别数量统计
    class_counts.to_csv(output_dir / "class_counts.csv", index=False)

    print(f"Output: {output_dir}")
    print(f"adata shape: {adata.shape}")
    print("\nClass counts:")
    print(class_counts)


def main():
    for name, filename in DATASETS.items():
        prepare_dataset(name, filename)


if __name__ == "__main__":
    main()