from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize


def load_dataset(dataset: str, embedding_path: Path):
    emb = np.load(embedding_path)
    labels_df = pd.read_csv(Path("data/processed") / dataset / "labels.csv")
    y = labels_df["label_id"].astype(int).values
    label_lookup = (
        labels_df[["label_id", "cell_type"]]
        .drop_duplicates()
        .sort_values("label_id")
        .set_index("label_id")["cell_type"]
        .to_dict()
    )

    if len(emb) != len(labels_df):
        raise ValueError(f"Embedding rows {len(emb)} != labels rows {len(labels_df)}")

    return emb, y, label_lookup


def top2_margin(values: np.ndarray):
    if values.shape[1] == 1:
        return values[:, 0], np.zeros(values.shape[0]), values[:, 0]
    top2 = np.partition(values, kth=-2, axis=1)[:, -2:]
    second = top2[:, 0]
    first = top2[:, 1]
    return first, second, first - second


def build_prototype_state(X_train, y_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_norm = normalize(X_train_scaled)
    X_val_norm = normalize(X_val_scaled)
    X_test_norm = normalize(X_test_scaled)

    classes = np.unique(y_train)
    prototypes = []
    for c in classes:
        proto = X_train_norm[y_train == c].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        prototypes.append(proto)
    prototypes = np.vstack(prototypes)

    train_sim = X_train_norm @ prototypes.T
    val_sim = X_val_norm @ prototypes.T
    test_sim = X_test_norm @ prototypes.T

    class_stats = {}
    class_rows = []
    for class_idx, c in enumerate(classes):
        own = train_sim[y_train == c, class_idx]
        class_stats[c] = {
            "mu": float(np.mean(own)),
            "sd": float(np.std(own) + 1e-12),
            "q05": float(np.quantile(own, 0.05)),
            "q50": float(np.quantile(own, 0.50)),
            "q95": float(np.quantile(own, 0.95)),
            "n_train_class": int((y_train == c).sum()),
        }
        class_rows.append({"label_id": int(c), **class_stats[c]})

    return classes, train_sim, val_sim, test_sim, class_stats, pd.DataFrame(class_rows)


def summarize_scores(values: np.ndarray, prefix: str):
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
    }


def analyze_split(dataset, embedding_name, split_path, X, y, label_lookup):
    split = np.load(split_path, allow_pickle=True)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]
    is_unknown = split["is_unknown_test"].astype(bool)
    holdout = str(split["holdout_cell_type"])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val = X[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    classes, train_sim, val_sim, test_sim, class_stats, class_df = build_prototype_state(
        X_train, y_train, X_val, X_test
    )

    best_sim, second_sim, margin = top2_margin(test_sim)
    best_idx = test_sim.argmax(axis=1)
    pred_label = classes[best_idx]

    raw_score = 1.0 - best_sim
    class_z = np.array([
        (class_stats[c]["mu"] - sim) / class_stats[c]["sd"]
        for c, sim in zip(pred_label, best_sim)
    ])

    class_df["analysis_type"] = "prototype_class_stats"
    class_df["dataset"] = dataset
    class_df["embedding"] = embedding_name
    class_df["split"] = split_path.name
    class_df["holdout_cell_type"] = holdout
    class_df["cell_type"] = class_df["label_id"].map(label_lookup)

    unknown_rows = []
    unknown_pred = pred_label[is_unknown]
    unknown_best_sim = best_sim[is_unknown]
    unknown_raw = raw_score[is_unknown]
    unknown_z = class_z[is_unknown]

    for c in classes:
        mask = unknown_pred == c
        if not np.any(mask):
            continue
        row = {
            "analysis_type": "unknown_nearest_class",
            "dataset": dataset,
            "embedding": embedding_name,
            "split": split_path.name,
            "holdout_cell_type": holdout,
            "pred_label_id": int(c),
            "pred_cell_type": label_lookup[int(c)],
            "n_unknown_assigned": int(mask.sum()),
            "frac_unknown_assigned": float(mask.mean()),
            "unknown_best_sim_mean": float(np.mean(unknown_best_sim[mask])),
            "unknown_raw_score_mean": float(np.mean(unknown_raw[mask])),
            "unknown_class_z_mean": float(np.mean(unknown_z[mask])),
            "pred_class_train_mu": class_stats[c]["mu"],
            "pred_class_train_sd": class_stats[c]["sd"],
            "pred_class_train_q05": class_stats[c]["q05"],
            "pred_class_train_q95": class_stats[c]["q95"],
            "pred_class_n_train": class_stats[c]["n_train_class"],
        }
        unknown_rows.append(row)

    score_rows = []
    for group_name, group_mask in [("known_test", ~is_unknown), ("unknown_test", is_unknown)]:
        row = {
            "analysis_type": "score_distribution",
            "dataset": dataset,
            "embedding": embedding_name,
            "split": split_path.name,
            "holdout_cell_type": holdout,
            "score_group": group_name,
            "n_cells": int(group_mask.sum()),
        }
        row.update(summarize_scores(best_sim[group_mask], "best_sim"))
        row.update(summarize_scores(raw_score[group_mask], "raw_score"))
        row.update(summarize_scores(class_z[group_mask], "class_z"))
        row.update(summarize_scores(margin[group_mask], "margin"))
        score_rows.append(row)

    overlap_row = {
        "analysis_type": "score_separation",
        "dataset": dataset,
        "embedding": embedding_name,
        "split": split_path.name,
        "holdout_cell_type": holdout,
        "raw_unknown_minus_known_mean": float(np.mean(raw_score[is_unknown]) - np.mean(raw_score[~is_unknown])),
        "class_z_unknown_minus_known_mean": float(np.mean(class_z[is_unknown]) - np.mean(class_z[~is_unknown])),
        "margin_unknown_minus_known_mean": float(np.mean(margin[is_unknown]) - np.mean(margin[~is_unknown])),
    }

    return [class_df, pd.DataFrame(unknown_rows), pd.DataFrame(score_rows), pd.DataFrame([overlap_row])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zheng68k")
    parser.add_argument("--embedding", type=Path, default=Path("data/embeddings/zheng68k_pca50.npy"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits/zheng68k"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/score_behavior_analysis.csv"))
    args = parser.parse_args()

    X, y, label_lookup = load_dataset(args.dataset, args.embedding)
    split_paths = sorted(args.splits_dir.glob("open_set_holdout_*_seed*.npz"))
    if not split_paths:
        raise FileNotFoundError(f"No open-set split files found in {args.splits_dir}")

    tables = []
    for split_path in split_paths:
        print(f"Analyzing prototype score behavior for {split_path}")
        tables.extend(analyze_split(args.dataset, args.embedding.name, split_path, X, y, label_lookup))

    output = pd.concat(tables, ignore_index=True, sort=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"Saved analysis to: {args.output}")

    unknown_nearest = output[output["analysis_type"] == "unknown_nearest_class"].copy()
    if not unknown_nearest.empty:
        unknown_nearest = unknown_nearest.sort_values(
            ["holdout_cell_type", "frac_unknown_assigned"],
            ascending=[True, False],
        )
        print("\nTop nearest known classes for held-out unknown cells:")
        print(
            unknown_nearest[
                [
                    "holdout_cell_type",
                    "pred_cell_type",
                    "n_unknown_assigned",
                    "frac_unknown_assigned",
                    "unknown_best_sim_mean",
                    "unknown_class_z_mean",
                    "pred_class_train_mu",
                    "pred_class_train_sd",
                ]
            ]
            .groupby("holdout_cell_type")
            .head(5)
            .to_string(index=False)
        )

    separation = output[output["analysis_type"] == "score_separation"].copy()
    print("\nMean score separation, unknown minus known:")
    print(
        separation[
            [
                "holdout_cell_type",
                "raw_unknown_minus_known_mean",
                "class_z_unknown_minus_known_mean",
                "margin_unknown_minus_known_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
