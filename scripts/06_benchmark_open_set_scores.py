from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, normalize

from experiment_utils import peak_memory_mb, runtime_context, timed_block, utc_now_iso


COVERAGE_TARGETS = (0.95, 0.90, 0.80)


def load_dataset(dataset: str, embedding_path: Path):
    emb = np.load(embedding_path)
    labels_df = pd.read_csv(Path("data/processed") / dataset / "labels.csv")
    y = labels_df["label_id"].astype(int).values

    if len(emb) != len(labels_df):
        raise ValueError(f"Embedding rows {len(emb)} != labels rows {len(labels_df)}")

    return emb, y


def top2_margin(values: np.ndarray):
    if values.shape[1] == 1:
        return values[:, 0], np.zeros(values.shape[0]), values[:, 0]
    top2 = np.partition(values, kth=-2, axis=1)[:, -2:]
    second = top2[:, 0]
    first = top2[:, 1]
    return first, second, first - second


def fpr_at_tpr(y_unknown: np.ndarray, unknown_score: np.ndarray, target_tpr: float = 0.95):
    if len(np.unique(y_unknown)) < 2:
        return np.nan

    fpr, tpr, _ = roc_curve(y_unknown, unknown_score)
    valid = np.where(tpr >= target_tpr)[0]
    if len(valid) == 0:
        return np.nan
    return float(np.min(fpr[valid]))


def threshold_metrics(y_true, y_pred, is_unknown, test_score, val_score):
    metrics = {}
    known_mask = ~is_unknown
    unknown_mask = is_unknown

    for coverage in COVERAGE_TARGETS:
        tag = f"val{int(coverage * 100)}"
        threshold = float(np.quantile(val_score, coverage))
        accepted = test_score <= threshold
        rejected = ~accepted

        known_accepted = known_mask & accepted
        if known_mask.sum() > 0:
            metrics[f"known_coverage_at_{tag}"] = float(known_accepted.sum() / known_mask.sum())
        else:
            metrics[f"known_coverage_at_{tag}"] = np.nan

        if unknown_mask.sum() > 0:
            metrics[f"unknown_recall_at_{tag}"] = float((unknown_mask & rejected).sum() / unknown_mask.sum())
        else:
            metrics[f"unknown_recall_at_{tag}"] = np.nan

        if known_accepted.sum() > 0:
            metrics[f"accepted_known_accuracy_at_{tag}"] = float(
                accuracy_score(y_true[known_accepted], y_pred[known_accepted])
            )
        else:
            metrics[f"accepted_known_accuracy_at_{tag}"] = np.nan

        metrics[f"rejection_rate_at_{tag}"] = float(rejected.mean())
        metrics[f"threshold_at_{tag}"] = threshold

    return metrics


def evaluate_score(y_true, y_pred, is_unknown, test_score, val_score):
    known_mask = ~is_unknown

    row = {
        "known_accuracy": float(accuracy_score(y_true[known_mask], y_pred[known_mask])),
        "unknown_auroc": float(roc_auc_score(is_unknown, test_score)),
        "unknown_auprc": float(average_precision_score(is_unknown, test_score)),
        "fpr_at_95_tpr": fpr_at_tpr(is_unknown, test_score, target_tpr=0.95),
    }
    row.update(threshold_metrics(y_true, y_pred, is_unknown, test_score, val_score))
    return row


def train_mlp_scores(X_train, y_train, X_val, X_test, seed):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
        verbose=False,
    )
    clf.fit(X_train_scaled, y_train)

    val_prob = clf.predict_proba(X_val_scaled)
    test_prob = clf.predict_proba(X_test_scaled)

    test_best, test_second, test_margin = top2_margin(test_prob)
    val_best, val_second, val_margin = top2_margin(val_prob)
    test_pred = clf.classes_[test_prob.argmax(axis=1)]

    eps = 1e-12
    return {
        "mlp_maxprob": {
            "y_pred": test_pred,
            "test_score": 1.0 - test_best,
            "val_score": 1.0 - val_best,
        },
        "mlp_entropy": {
            "y_pred": test_pred,
            "test_score": -np.sum(test_prob * np.log(test_prob + eps), axis=1) / np.log(test_prob.shape[1]),
            "val_score": -np.sum(val_prob * np.log(val_prob + eps), axis=1) / np.log(val_prob.shape[1]),
        },
        "mlp_margin": {
            "y_pred": test_pred,
            "test_score": -test_margin,
            "val_score": -val_margin,
        },
    }


def train_prototype_scores(X_train, y_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_norm = normalize(X_train_scaled)
    X_val_norm = normalize(X_val_scaled)
    X_test_norm = normalize(X_test_scaled)

    classes = np.unique(y_train)
    prototypes = []
    class_own_sims = {}

    for c in classes:
        proto = X_train_norm[y_train == c].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        prototypes.append(proto)

    prototypes = np.vstack(prototypes)
    train_sim = X_train_norm @ prototypes.T
    val_sim = X_val_norm @ prototypes.T
    test_sim = X_test_norm @ prototypes.T

    for class_idx, c in enumerate(classes):
        class_own_sims[c] = np.sort(train_sim[y_train == c, class_idx])

    test_best, test_second, test_margin = top2_margin(test_sim)
    val_best, val_second, val_margin = top2_margin(val_sim)

    test_best_idx = test_sim.argmax(axis=1)
    val_best_idx = val_sim.argmax(axis=1)
    test_pred = classes[test_best_idx]

    class_mu = {}
    class_sd = {}
    for class_idx, c in enumerate(classes):
        own = train_sim[y_train == c, class_idx]
        class_mu[c] = float(np.mean(own))
        class_sd[c] = float(np.std(own) + 1e-12)

    test_z = np.array([
        (class_mu[c] - sim) / class_sd[c]
        for c, sim in zip(test_pred, test_best)
    ])
    val_pred = classes[val_best_idx]
    val_z = np.array([
        (class_mu[c] - sim) / class_sd[c]
        for c, sim in zip(val_pred, val_best)
    ])

    def tail_score(pred_classes, best_sims):
        scores = []
        for c, sim in zip(pred_classes, best_sims):
            own = class_own_sims[c]
            percentile = np.searchsorted(own, sim, side="right") / len(own)
            scores.append(1.0 - percentile)
        return np.asarray(scores)

    return {
        "proto_1_minus_max_cosine": {
            "y_pred": test_pred,
            "test_score": 1.0 - test_best,
            "val_score": 1.0 - val_best,
        },
        "proto_margin": {
            "y_pred": test_pred,
            "test_score": -test_margin,
            "val_score": -val_margin,
        },
        "proto_class_z": {
            "y_pred": test_pred,
            "test_score": test_z,
            "val_score": val_z,
        },
        "proto_class_tail_quantile": {
            "y_pred": test_pred,
            "test_score": tail_score(test_pred, test_best),
            "val_score": tail_score(val_pred, val_best),
        },
    }


def train_mahalanobis_scores(X_train, y_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classes = np.unique(y_train)
    means = []
    variances = []
    for c in classes:
        X_c = X_train_scaled[y_train == c]
        means.append(X_c.mean(axis=0))
        variances.append(X_c.var(axis=0) + 1e-6)

    means = np.vstack(means)
    variances = np.vstack(variances)

    def distances(X):
        dists = []
        for mean, var in zip(means, variances):
            dists.append(np.mean(((X - mean) ** 2) / var, axis=1))
        return np.vstack(dists).T

    test_dist = distances(X_test_scaled)
    val_dist = distances(X_val_scaled)
    test_best_idx = test_dist.argmin(axis=1)

    return {
        "mahalanobis_diag_min": {
            "y_pred": classes[test_best_idx],
            "test_score": test_dist.min(axis=1),
            "val_score": val_dist.min(axis=1),
        }
    }


def train_knn_scores(X_train, y_train, X_val, X_test, n_neighbors):
    scaler = StandardScaler()
    X_train_scaled = normalize(scaler.fit_transform(X_train))
    X_val_scaled = normalize(scaler.transform(X_val))
    X_test_scaled = normalize(scaler.transform(X_test))

    n_neighbors = min(n_neighbors, len(X_train_scaled))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(X_train_scaled)

    test_dist, test_idx = nn.kneighbors(X_test_scaled)
    val_dist, _ = nn.kneighbors(X_val_scaled)

    y_pred = []
    for neighbors in test_idx:
        labels, counts = np.unique(y_train[neighbors], return_counts=True)
        y_pred.append(labels[counts.argmax()])

    return {
        f"knn_cosine_mean_distance_k{n_neighbors}": {
            "y_pred": np.asarray(y_pred),
            "test_score": test_dist.mean(axis=1),
            "val_score": val_dist.mean(axis=1),
        }
    }


def run_split(args, X, y, split_path):
    split = np.load(split_path, allow_pickle=True)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]
    is_unknown = split["is_unknown_test"].astype(bool)
    holdout = str(split["holdout_cell_type"]) if "holdout_cell_type" in split else "unknown"
    split_seed = int(split["seed"]) if "seed" in split else np.nan

    X_train, y_train = X[train_idx], y[train_idx]
    X_val = X[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    score_sets = {}
    with timed_block() as mlp_elapsed:
        score_sets.update(train_mlp_scores(X_train, y_train, X_val, X_test, args.seed))
    mlp_seconds = float(mlp_elapsed())

    with timed_block() as proto_elapsed:
        score_sets.update(train_prototype_scores(X_train, y_train, X_val, X_test))
    proto_seconds = float(proto_elapsed())

    with timed_block() as mahal_elapsed:
        score_sets.update(train_mahalanobis_scores(X_train, y_train, X_val, X_test))
    mahal_seconds = float(mahal_elapsed())

    with timed_block() as knn_elapsed:
        score_sets.update(train_knn_scores(X_train, y_train, X_val, X_test, args.knn_neighbors))
    knn_seconds = float(knn_elapsed())

    family_seconds = {
        "mlp": mlp_seconds,
        "proto": proto_seconds,
        "mahalanobis": mahal_seconds,
        "knn": knn_seconds,
    }

    rows = []
    for score_name, payload in score_sets.items():
        row = evaluate_score(
            y_true=y_test,
            y_pred=payload["y_pred"],
            is_unknown=is_unknown,
            test_score=payload["test_score"],
            val_score=payload["val_score"],
        )

        family = score_name.split("_", 1)[0]
        if score_name.startswith("proto_"):
            family = "prototype"

        row.update({
            "dataset": args.dataset,
            "embedding": args.embedding.name,
            "split": split_path.name,
            "holdout_cell_type": holdout,
            "score_method": score_name,
            "score_family": family,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_known_test": int((~is_unknown).sum()),
            "n_unknown_test": int(is_unknown.sum()),
            "fit_score_seconds": family_seconds.get(family if family != "prototype" else "proto", np.nan),
            "split_seed": split_seed,
            "model_seed": int(args.seed),
        })
        rows.append(row)

    return rows


def summarize_benchmark(output_df: pd.DataFrame):
    metrics = [
        "known_accuracy",
        "unknown_auroc",
        "unknown_auprc",
        "fpr_at_95_tpr",
        "known_coverage_at_val95",
        "unknown_recall_at_val95",
        "accepted_known_accuracy_at_val95",
        "known_coverage_at_val90",
        "unknown_recall_at_val90",
        "accepted_known_accuracy_at_val90",
        "known_coverage_at_val80",
        "unknown_recall_at_val80",
        "accepted_known_accuracy_at_val80",
    ]
    group_cols = ["dataset", "embedding", "holdout_cell_type", "score_family", "score_method"]
    summary = (
        output_df
        .groupby(group_cols, dropna=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    n_splits = (
        output_df
        .groupby(group_cols, dropna=False)["split_seed"]
        .nunique()
        .reset_index(name="n_split_seeds")
    )
    summary = summary.merge(n_splits, on=group_cols, how="left")

    sort_cols = ["holdout_cell_type", "unknown_auroc_mean", "unknown_auprc_mean"]
    return summary.sort_values(sort_cols, ascending=[True, False, False]).reset_index(drop=True)


def main():
    started_at = utc_now_iso()
    with timed_block() as elapsed:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="zheng68k")
        parser.add_argument("--embedding", type=Path, default=Path("data/embeddings/zheng68k_pca50.npy"))
        parser.add_argument("--splits_dir", type=Path, default=Path("data/splits/zheng68k"))
        parser.add_argument("--output", type=Path, default=Path("results/tables/open_set_score_benchmark.csv"))
        parser.add_argument(
            "--summary_output",
            type=Path,
            default=Path("results/tables/open_set_score_benchmark_summary.csv"),
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--knn_neighbors", type=int, default=10)
        args = parser.parse_args()

        X, y = load_dataset(args.dataset, args.embedding)
        split_paths = sorted(args.splits_dir.glob("open_set_holdout_*_seed*.npz"))
        if not split_paths:
            raise FileNotFoundError(f"No open-set split files found in {args.splits_dir}")

        rows = []
        for split_path in split_paths:
            print(f"Running score benchmark for {split_path}")
            rows.extend(run_split(args, X, y, split_path))

        finished_at = utc_now_iso()
        context = runtime_context()
        for row in rows:
            row.update({
                "runtime_seconds_total": float(elapsed()),
                "peak_memory_mb": peak_memory_mb(),
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
            })
            row.update(context)

        output_df = pd.DataFrame(rows)
        output_df = output_df.sort_values(["holdout_cell_type", "score_family", "score_method"]).reset_index(drop=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.output, index=False)
        summary_df = summarize_benchmark(output_df)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.summary_output, index=False)

        print(f"Saved score benchmark to: {args.output}")
        print(f"Saved score benchmark summary to: {args.summary_output}")
        display_cols = [
            "holdout_cell_type",
            "score_method",
            "split_seed",
            "known_accuracy",
            "unknown_auroc",
            "unknown_auprc",
            "unknown_recall_at_val95",
            "known_coverage_at_val95",
        ]
        print(output_df[display_cols].to_string(index=False))

        best_by_holdout = (
            summary_df.sort_values(["holdout_cell_type", "unknown_auroc_mean"], ascending=[True, False])
            .groupby("holdout_cell_type")
            .head(3)
        )
        print("\nTop 3 by mean AUROC per holdout:")
        print(
            best_by_holdout[
                [
                    "holdout_cell_type",
                    "score_method",
                    "unknown_auroc_mean",
                    "unknown_auroc_std",
                    "unknown_auprc_mean",
                    "n_split_seeds",
                ]
            ].to_string(index=False)
        )

        metadata = {
            "dataset": args.dataset,
            "embedding": str(args.embedding),
            "splits": [str(path) for path in split_paths],
            "output": str(args.output),
            "summary_output": str(args.summary_output),
            "n_rows": int(len(output_df)),
            "n_summary_rows": int(len(summary_df)),
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "runtime_seconds_total": float(elapsed()),
            "peak_memory_mb": peak_memory_mb(),
        }
        metadata.update(context)
        print("\nRun metadata:")
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
