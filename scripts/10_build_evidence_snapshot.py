from pathlib import Path
import argparse
import json

import pandas as pd


METHODS = {
    "mlp_maxprob": {
        "method_name": "MLP max probability",
        "method_role": "closed-set confidence baseline",
    },
    "proto_1_minus_max_cosine": {
        "method_name": "Raw prototype distance",
        "method_role": "uncalibrated prototype baseline",
    },
    "proto_class_z": {
        "method_name": "CalibscAnno-v0 (class-z prototype)",
        "method_role": "main method",
    },
    "mahalanobis_diag_min": {
        "method_name": "Diagonal Mahalanobis",
        "method_role": "strong statistical-distance baseline",
    },
}

EMBEDDING_LABELS = {
    "pca": "PCA50",
    "scfoundation": "scFoundation",
}


MAIN_METRICS = [
    "known_accuracy_mean",
    "unknown_auroc_mean",
    "unknown_auprc_mean",
    "unknown_recall_at_val95_mean",
    "known_coverage_at_val95_mean",
    "accepted_known_accuracy_at_val95_mean",
]

SUMMARY_METRICS = [
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


def load_main_results(path: Path, embedding_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Result table not found: {path}")

    df = pd.read_csv(path)
    df["embedding_label"] = embedding_label
    return df


def summarize_benchmark_detail(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "embedding", "holdout_cell_type", "score_family", "score_method"]
    summary = (
        detail
        .groupby(group_cols, dropna=False)[SUMMARY_METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    n_splits = (
        detail
        .groupby(group_cols, dropna=False)["split_seed"]
        .nunique()
        .reset_index(name="n_split_seeds")
    )
    summary = summary.merge(n_splits, on=group_cols, how="left")
    return summary.sort_values(
        ["holdout_cell_type", "unknown_auroc_mean", "unknown_auprc_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    baseline_methods = {
        "mlp_maxprob": "mlp_maxprob",
        "raw_prototype": "proto_1_minus_max_cosine",
    }

    for baseline_name, baseline_method in baseline_methods.items():
        baseline = df[df["score_method"] == baseline_method].set_index("holdout_cell_type")
        for metric in [
            "unknown_auroc_mean",
            "unknown_auprc_mean",
            "unknown_recall_at_val95_mean",
        ]:
            delta_col = f"delta_{metric}_vs_{baseline_name}"
            df[delta_col] = df.apply(
                lambda row: row[metric] - baseline.loc[row["holdout_cell_type"], metric],
                axis=1,
            )
    return df


def add_average_rows(df: pd.DataFrame) -> pd.DataFrame:
    average_rows = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "n_split_seeds"]

    for score_method, method_df in df.groupby("score_method", sort=False):
        row = {
            "dataset": method_df["dataset"].iloc[0],
            "embedding": method_df["embedding"].iloc[0],
            "holdout_cell_type": "Average across holdouts",
            "score_family": method_df["score_family"].iloc[0],
            "score_method": score_method,
            "method_name": method_df["method_name"].iloc[0],
            "method_role": method_df["method_role"].iloc[0],
            "n_split_seeds": int(method_df["n_split_seeds"].min()),
        }
        for col in numeric_cols:
            row[col] = float(method_df[col].mean())
        average_rows.append(row)

    return pd.concat([df, pd.DataFrame(average_rows)], ignore_index=True, sort=False)


def build_main_results_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[summary["score_method"].isin(METHODS)].copy()
    if selected.empty:
        raise ValueError("No selected methods found in summary table")

    selected["method_name"] = selected["score_method"].map(
        {key: value["method_name"] for key, value in METHODS.items()}
    )
    selected["method_role"] = selected["score_method"].map(
        {key: value["method_role"] for key, value in METHODS.items()}
    )
    selected = add_deltas(selected)

    order = {method: index for index, method in enumerate(METHODS)}
    selected["method_order"] = selected["score_method"].map(order)
    selected = selected.sort_values(["holdout_cell_type", "method_order"]).reset_index(drop=True)
    selected = add_average_rows(selected)

    columns = [
        "dataset",
        "embedding",
        "holdout_cell_type",
        "method_name",
        "method_role",
        "score_method",
        "score_family",
        *[col for col in selected.columns if col.endswith("_mean") or col.endswith("_std")],
        "n_split_seeds",
        *[col for col in selected.columns if col.startswith("delta_unknown_")],
    ]
    columns = list(dict.fromkeys(col for col in columns if col in selected.columns))
    return selected[columns]


def build_seed_filtered_pca_results(args) -> pd.DataFrame | None:
    if not args.pca_benchmark_detail.exists():
        return None

    detail = pd.read_csv(args.pca_benchmark_detail)
    filtered = detail[detail["split_seed"].isin(args.pca_split_seeds)].copy()
    if filtered.empty:
        raise ValueError(
            f"No PCA benchmark rows found for split seeds: {args.pca_split_seeds}"
        )

    summary = summarize_benchmark_detail(filtered)
    args.pca_seed_filtered_summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.pca_seed_filtered_summary_output, index=False)

    main_results = build_main_results_from_summary(summary)
    args.pca_seed_filtered_main_output.parent.mkdir(parents=True, exist_ok=True)
    main_results.to_csv(args.pca_seed_filtered_main_output, index=False)
    print(f"Saved seed-filtered PCA summary to: {args.pca_seed_filtered_summary_output}")
    print(f"Saved seed-filtered PCA main results to: {args.pca_seed_filtered_main_output}")
    return main_results


def build_embedding_comparison(pca: pd.DataFrame, scfoundation: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([pca, scfoundation], ignore_index=True, sort=False)
    keep_cols = [
        "dataset",
        "embedding_label",
        "embedding",
        "holdout_cell_type",
        "method_name",
        "method_role",
        "score_method",
        "score_family",
        "n_split_seeds",
        *MAIN_METRICS,
    ]
    comparison = combined[[col for col in keep_cols if col in combined.columns]].copy()

    key_cols = ["holdout_cell_type", "score_method"]
    pca_reference = (
        comparison[comparison["embedding_label"] == EMBEDDING_LABELS["pca"]]
        .set_index(key_cols)
    )

    for metric in MAIN_METRICS:
        delta_col = f"delta_{metric}_vs_pca"
        comparison[delta_col] = pd.NA
        is_scfoundation = comparison["embedding_label"] == EMBEDDING_LABELS["scfoundation"]
        comparison.loc[is_scfoundation, delta_col] = comparison.loc[is_scfoundation].apply(
            lambda row: row[metric] - pca_reference.loc[
                (row["holdout_cell_type"], row["score_method"]), metric
            ],
            axis=1,
        )

    order = {
        EMBEDDING_LABELS["pca"]: 0,
        EMBEDDING_LABELS["scfoundation"]: 1,
    }
    comparison["embedding_order"] = comparison["embedding_label"].map(order)
    comparison = comparison.sort_values(
        ["holdout_cell_type", "score_method", "embedding_order"]
    ).drop(columns=["embedding_order"])
    return comparison


def build_claim_snapshot(comparison: pd.DataFrame) -> pd.DataFrame:
    avg = comparison[comparison["holdout_cell_type"] == "Average across holdouts"].copy()
    rows = []

    def get_metric(embedding_label: str, score_method: str, metric: str) -> float:
        values = avg[
            (avg["embedding_label"] == embedding_label)
            & (avg["score_method"] == score_method)
        ][metric]
        if values.empty:
            raise ValueError(f"Missing {embedding_label} / {score_method} / {metric}")
        return float(values.iloc[0])

    for embedding_label in [EMBEDDING_LABELS["pca"], EMBEDDING_LABELS["scfoundation"]]:
        mlp_auroc = get_metric(embedding_label, "mlp_maxprob", "unknown_auroc_mean")
        raw_auroc = get_metric(embedding_label, "proto_1_minus_max_cosine", "unknown_auroc_mean")
        class_z_auroc = get_metric(embedding_label, "proto_class_z", "unknown_auroc_mean")
        maha_auroc = get_metric(embedding_label, "mahalanobis_diag_min", "unknown_auroc_mean")
        rows.extend(
            [
                {
                    "embedding_label": embedding_label,
                    "claim": "Closed-set confidence is weak for unknown detection",
                    "metric": "average_unknown_auroc",
                    "reference_method": "MLP max probability",
                    "reference_value": mlp_auroc,
                    "candidate_method": "",
                    "candidate_value": pd.NA,
                    "delta": pd.NA,
                    "interpretation": "Softmax-style confidence is not sufficient as an open-set detector.",
                },
                {
                    "embedding_label": embedding_label,
                    "claim": "Prototype distance improves over closed-set confidence",
                    "metric": "average_unknown_auroc",
                    "reference_method": "MLP max probability",
                    "reference_value": mlp_auroc,
                    "candidate_method": "Raw prototype distance",
                    "candidate_value": raw_auroc,
                    "delta": raw_auroc - mlp_auroc,
                    "interpretation": "Distance to known class prototypes provides a stronger unknown score.",
                },
                {
                    "embedding_label": embedding_label,
                    "claim": "Class-conditional calibration improves prototype rejection on average",
                    "metric": "average_unknown_auroc",
                    "reference_method": "Raw prototype distance",
                    "reference_value": raw_auroc,
                    "candidate_method": "CalibscAnno-v0 class-z prototype",
                    "candidate_value": class_z_auroc,
                    "delta": class_z_auroc - raw_auroc,
                    "interpretation": "The benefit is embedding- and holdout-dependent, so this should be framed as a calibrated rejection component rather than the only final method.",
                },
                {
                    "embedding_label": embedding_label,
                    "claim": "Diagonal Mahalanobis is the strongest current rejection score",
                    "metric": "average_unknown_auroc",
                    "reference_method": "Raw prototype distance",
                    "reference_value": raw_auroc,
                    "candidate_method": "Diagonal Mahalanobis",
                    "candidate_value": maha_auroc,
                    "delta": maha_auroc - raw_auroc,
                    "interpretation": "This is the best current main-method candidate or core ablation target.",
                },
            ]
        )

    return pd.DataFrame(rows)


def build_metadata_table(metadata_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in metadata_paths:
        if not path.exists():
            raise FileNotFoundError(f"Metadata JSON not found: {path}")
        metadata = json.loads(path.read_text())
        input_shape = metadata.get("input_shape") or [None, None]
        rows.append(
            {
                "dataset": metadata.get("dataset"),
                "input_n_cells": input_shape[0],
                "input_n_genes": input_shape[1],
                "output_path": metadata.get("output_path"),
                "pool_type": metadata.get("pool_type"),
                "tgthighres": metadata.get("tgthighres"),
                "version": metadata.get("version"),
                "status": metadata.get("status"),
                "runtime_seconds": metadata.get("runtime_seconds"),
                "peak_gpu_memory_mb_nvidia_smi": metadata.get("peak_gpu_memory_mb_nvidia_smi"),
                "started_at_utc": metadata.get("started_at_utc"),
                "finished_at_utc": metadata.get("finished_at_utc"),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pca_main_results",
        type=Path,
        default=Path("results/tables/calibscanno_v0_main_results.csv"),
    )
    parser.add_argument(
        "--pca_benchmark_detail",
        type=Path,
        default=Path("results/tables/open_set_score_benchmark.csv"),
    )
    parser.add_argument(
        "--pca_split_seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--pca_seed_filtered_summary_output",
        type=Path,
        default=Path("results/tables/open_set_score_benchmark_pca_seed0-4_summary.csv"),
    )
    parser.add_argument(
        "--pca_seed_filtered_main_output",
        type=Path,
        default=Path("results/tables/calibscanno_v0_main_results_pca_seed0-4.csv"),
    )
    parser.add_argument(
        "--scfoundation_main_results",
        type=Path,
        default=Path("results/tables/calibscanno_v0_main_results_scfoundation.csv"),
    )
    parser.add_argument(
        "--comparison_output",
        type=Path,
        default=Path("results/tables/embedding_method_comparison.csv"),
    )
    parser.add_argument(
        "--claim_output",
        type=Path,
        default=Path("results/tables/paper_claim_snapshot.csv"),
    )
    parser.add_argument(
        "--metadata_json",
        type=Path,
        nargs="*",
        default=[],
        help="Optional scFoundation extraction metadata JSON files.",
    )
    parser.add_argument(
        "--metadata_output",
        type=Path,
        default=Path("results/tables/scfoundation_embedding_run_metadata.csv"),
    )
    args = parser.parse_args()

    pca = build_seed_filtered_pca_results(args)
    if pca is None:
        pca = pd.read_csv(args.pca_main_results)
        print(f"Using existing PCA main results: {args.pca_main_results}")
    pca["embedding_label"] = EMBEDDING_LABELS["pca"]

    scfoundation = load_main_results(
        args.scfoundation_main_results, EMBEDDING_LABELS["scfoundation"]
    )

    comparison = build_embedding_comparison(pca, scfoundation)
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.comparison_output, index=False)

    claim_snapshot = build_claim_snapshot(comparison)
    claim_snapshot.to_csv(args.claim_output, index=False)

    print(f"Saved embedding comparison to: {args.comparison_output}")
    print(f"Saved claim snapshot to: {args.claim_output}")
    display_cols = [
        "embedding_label",
        "holdout_cell_type",
        "method_name",
        "unknown_auroc_mean",
        "unknown_auprc_mean",
        "unknown_recall_at_val95_mean",
    ]
    print(
        comparison[
            comparison["holdout_cell_type"] == "Average across holdouts"
        ][display_cols].to_string(index=False)
    )

    if args.metadata_json:
        metadata_table = build_metadata_table(args.metadata_json)
        metadata_table.to_csv(args.metadata_output, index=False)
        print(f"Saved extraction metadata table to: {args.metadata_output}")


if __name__ == "__main__":
    main()
