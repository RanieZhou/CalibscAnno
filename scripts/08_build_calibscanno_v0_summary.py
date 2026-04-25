from pathlib import Path
import argparse

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


METRIC_COLUMNS = [
    "known_accuracy_mean",
    "known_accuracy_std",
    "unknown_auroc_mean",
    "unknown_auroc_std",
    "unknown_auprc_mean",
    "unknown_auprc_std",
    "known_coverage_at_val95_mean",
    "known_coverage_at_val95_std",
    "unknown_recall_at_val95_mean",
    "unknown_recall_at_val95_std",
    "accepted_known_accuracy_at_val95_mean",
    "accepted_known_accuracy_at_val95_std",
    "n_split_seeds",
]


def add_deltas(df: pd.DataFrame):
    baseline_methods = {
        "mlp_maxprob": "mlp_maxprob",
        "raw_prototype": "proto_1_minus_max_cosine",
    }

    for baseline_name, baseline_method in baseline_methods.items():
        baseline = (
            df[df["score_method"] == baseline_method]
            .set_index("holdout_cell_type")
        )

        for metric in ["unknown_auroc_mean", "unknown_auprc_mean", "unknown_recall_at_val95_mean"]:
            delta_col = f"delta_{metric}_vs_{baseline_name}"
            df[delta_col] = df.apply(
                lambda row: row[metric] - baseline.loc[row["holdout_cell_type"], metric],
                axis=1,
            )

    return df


def add_average_rows(df: pd.DataFrame):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/tables/open_set_score_benchmark_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/calibscanno_v0_main_results.csv"),
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.input)
    selected = summary[summary["score_method"].isin(METHODS)].copy()
    if selected.empty:
        raise ValueError(f"No selected methods found in {args.input}")

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

    delta_columns = [
        col for col in selected.columns
        if col.startswith("delta_unknown_")
    ]
    columns = [
        "dataset",
        "embedding",
        "holdout_cell_type",
        "method_name",
        "method_role",
        "score_method",
        "score_family",
        *METRIC_COLUMNS,
        *delta_columns,
    ]
    columns = [col for col in columns if col in selected.columns]

    output = selected[columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"Saved CalibscAnno-v0 summary to: {args.output}")
    display_cols = [
        "holdout_cell_type",
        "method_name",
        "unknown_auroc_mean",
        "unknown_auroc_std",
        "unknown_auprc_mean",
        "unknown_recall_at_val95_mean",
        "delta_unknown_auroc_mean_vs_raw_prototype",
        "delta_unknown_auroc_mean_vs_mlp_maxprob",
    ]
    print(output[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
