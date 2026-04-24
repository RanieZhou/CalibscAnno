from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def safe_name(name: str) -> str:
    return (
        name.replace("+", "plus")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def create_closed_set_split(labels_df, output_path, seed=42):
    indices = np.arange(len(labels_df))
    y = labels_df["label_id"].values

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )

    temp_y = y[temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=2 / 3,
        random_state=seed,
        stratify=temp_y,
    )

    np.savez(
        output_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        seed=seed,
    )

    print(f"Saved closed-set split to: {output_path}")
    print(f"train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")


def create_open_set_split(labels_df, holdout_cell_type, output_path, seed=42):
    labels = labels_df["cell_type"].astype(str).values
    label_ids = labels_df["label_id"].values
    all_indices = np.arange(len(labels_df))

    unknown_mask = labels == holdout_cell_type
    known_mask = ~unknown_mask

    known_indices = all_indices[known_mask]
    unknown_indices = all_indices[unknown_mask]

    known_y = label_ids[known_indices]

    train_idx, temp_idx = train_test_split(
        known_indices,
        test_size=0.3,
        random_state=seed,
        stratify=known_y,
    )

    temp_y = label_ids[temp_idx]

    val_idx, known_test_idx = train_test_split(
        temp_idx,
        test_size=2 / 3,
        random_state=seed,
        stratify=temp_y,
    )

    # open-set test = known test + all unknown cells
    test_idx = np.concatenate([known_test_idx, unknown_indices])

    is_unknown_test = labels[test_idx] == holdout_cell_type

    np.savez(
        output_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        known_test_idx=known_test_idx,
        unknown_test_idx=unknown_indices,
        is_unknown_test=is_unknown_test,
        holdout_cell_type=holdout_cell_type,
        seed=seed,
    )

    print(f"Saved open-set split to: {output_path}")
    print(f"holdout unknown: {holdout_cell_type}")
    print(f"train known: {len(train_idx)}")
    print(f"val known: {len(val_idx)}")
    print(f"known test: {len(known_test_idx)}")
    print(f"unknown test: {len(unknown_indices)}")
    print(f"open-set test total: {len(test_idx)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zheng68k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=str, default="CD56+ NK")
    args = parser.parse_args()

    processed_dir = Path("data/processed") / args.dataset
    labels_path = processed_dir / "labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    labels_df = pd.read_csv(labels_path)

    split_dir = Path("data/splits") / args.dataset
    split_dir.mkdir(parents=True, exist_ok=True)

    closed_path = split_dir / f"closed_set_seed{args.seed}.npz"
    create_closed_set_split(labels_df, closed_path, seed=args.seed)

    holdout_safe = safe_name(args.holdout)
    open_path = split_dir / f"open_set_holdout_{holdout_safe}_seed{args.seed}.npz"
    create_open_set_split(
        labels_df,
        holdout_cell_type=args.holdout,
        output_path=open_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()