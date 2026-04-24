from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)


def load_data(dataset, embedding_path):
    emb = np.load(embedding_path)
    labels_df = pd.read_csv(Path("data/processed") / dataset / "labels.csv")

    y = labels_df["label_id"].astype(int).values
    cell_types = labels_df["cell_type"].astype(str).values

    assert len(emb) == len(labels_df), (
        f"Embedding rows {len(emb)} != labels rows {len(labels_df)}"
    )

    return emb, y, cell_types, labels_df


def evaluate_closed_set(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def evaluate_open_set(y_true_known, y_pred_known, is_unknown, unknown_score):
    """
    is_unknown: True 表示 unknown cell
    unknown_score: 越大越像 unknown
    """
    results = {}

    if len(y_true_known) > 0:
        results["known_accuracy"] = float(accuracy_score(y_true_known, y_pred_known))
        results["known_macro_f1"] = float(
            f1_score(y_true_known, y_pred_known, average="macro")
        )
        results["known_weighted_f1"] = float(
            f1_score(y_true_known, y_pred_known, average="weighted")
        )

    if len(np.unique(is_unknown)) == 2:
        results["unknown_auroc"] = float(roc_auc_score(is_unknown, unknown_score))
        results["unknown_auprc"] = float(average_precision_score(is_unknown, unknown_score))
    else:
        results["unknown_auroc"] = None
        results["unknown_auprc"] = None

    return results


def train_mlp(X_train, y_train, seed):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

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
    return scaler, clf


def predict_mlp(scaler, clf, X):
    X_scaled = scaler.transform(X)
    pred = clf.predict(X_scaled)

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_scaled)
        max_prob = proba.max(axis=1)
        unknown_score = 1.0 - max_prob
    else:
        unknown_score = np.zeros(len(X))

    return pred, unknown_score


def train_prototype(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_norm = normalize(X_train_scaled)

    classes = np.unique(y_train)
    prototypes = []

    for c in classes:
        proto = X_train_norm[y_train == c].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        prototypes.append(proto)

    prototypes = np.vstack(prototypes)

    return scaler, classes, prototypes


def predict_prototype(scaler, classes, prototypes, X):
    X_scaled = scaler.transform(X)
    X_norm = normalize(X_scaled)

    sim = X_norm @ prototypes.T

    best_idx = sim.argmax(axis=1)
    best_sim = sim.max(axis=1)

    pred = classes[best_idx]

    # 越小的相似度越可能 unknown，所以 unknown_score = 1 - max similarity
    unknown_score = 1.0 - best_sim

    return pred, unknown_score


def run_closed_set(args, X, y):
    split = np.load(args.split, allow_pickle=True)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if args.classifier == "mlp":
        scaler, clf = train_mlp(X_train, y_train, seed=args.seed)
        y_pred, _ = predict_mlp(scaler, clf, X_test)

    elif args.classifier == "prototype":
        scaler, classes, prototypes = train_prototype(X_train, y_train)
        y_pred, _ = predict_prototype(scaler, classes, prototypes, X_test)

    else:
        raise ValueError(f"Unknown classifier: {args.classifier}")

    results = evaluate_closed_set(y_test, y_pred)
    return results


def run_open_set(args, X, y):
    split = np.load(args.split, allow_pickle=True)

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    is_unknown_test = split["is_unknown_test"].astype(bool)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if args.classifier == "mlp":
        scaler, clf = train_mlp(X_train, y_train, seed=args.seed)
        y_pred, unknown_score = predict_mlp(scaler, clf, X_test)

    elif args.classifier == "prototype":
        scaler, classes, prototypes = train_prototype(X_train, y_train)
        y_pred, unknown_score = predict_prototype(scaler, classes, prototypes, X_test)

    else:
        raise ValueError(f"Unknown classifier: {args.classifier}")

    known_mask = ~is_unknown_test

    results = evaluate_open_set(
        y_true_known=y_test[known_mask],
        y_pred_known=y_pred[known_mask],
        is_unknown=is_unknown_test,
        unknown_score=unknown_score,
    )

    if "holdout_cell_type" in split:
        holdout = str(split["holdout_cell_type"])
    else:
        holdout = "unknown"

    results["holdout_cell_type"] = holdout
    results["n_known_test"] = int(known_mask.sum())
    results["n_unknown_test"] = int(is_unknown_test.sum())

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zheng68k")
    parser.add_argument("--embedding", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["closed", "open"], required=True)
    parser.add_argument("--classifier", type=str, choices=["mlp", "prototype"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, cell_types, labels_df = load_data(args.dataset, args.embedding)

    print("Dataset:", args.dataset)
    print("Embedding:", args.embedding)
    print("Embedding shape:", X.shape)
    print("Split:", args.split)
    print("Mode:", args.mode)
    print("Classifier:", args.classifier)

    if args.mode == "closed":
        results = run_closed_set(args, X, y)
    else:
        results = run_open_set(args, X, y)

    results.update({
        "dataset": args.dataset,
        "embedding": Path(args.embedding).name,
        "split": Path(args.split).name,
        "mode": args.mode,
        "classifier": args.classifier,
        "seed": args.seed,
    })

    print("\nResults:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.dataset}_{args.mode}_{args.classifier}_{Path(args.split).stem}.csv"
    pd.DataFrame([results]).to_csv(out_path, index=False)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()