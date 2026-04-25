from pathlib import Path
import argparse

import pandas as pd


def is_baseline_result(path: Path) -> bool:
    name = path.name
    if name in {"summary_all_results.csv", "embedding_runs.csv"}:
        return False
    return name.endswith(".csv") and ("_closed_" in name or "_open_" in name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results/tables"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/summary_all_results.csv"),
    )
    args = parser.parse_args()

    result_files = sorted(
        path for path in args.results_dir.glob("*.csv") if is_baseline_result(path)
    )

    if not result_files:
        raise FileNotFoundError(f"No baseline result CSV files found in {args.results_dir}")

    rows = []
    for path in result_files:
        df = pd.read_csv(path)
        if len(df) != 1:
            raise ValueError(f"Expected exactly one row in {path}, found {len(df)}")
        row = df.iloc[0].to_dict()
        row["source_file"] = path.name
        rows.append(row)

    summary = pd.DataFrame(rows)

    preferred_columns = [
        "dataset",
        "embedding",
        "mode",
        "classifier",
        "split",
        "holdout_cell_type",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "known_accuracy",
        "known_macro_f1",
        "known_weighted_f1",
        "unknown_auroc",
        "unknown_auprc",
        "n_train",
        "n_test",
        "n_known_test",
        "n_unknown_test",
        "runtime_seconds",
        "fit_seconds",
        "predict_seconds",
        "peak_memory_mb",
        "seed",
        "source_file",
        "started_at_utc",
        "finished_at_utc",
        "python_version",
        "platform",
        "pid",
    ]

    ordered_columns = [col for col in preferred_columns if col in summary.columns]
    ordered_columns.extend(col for col in summary.columns if col not in ordered_columns)
    summary = summary[ordered_columns]

    sort_columns = [col for col in ["dataset", "embedding", "mode", "split", "classifier"] if col in summary.columns]
    summary = summary.sort_values(sort_columns).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    print(f"Loaded {len(result_files)} result files")
    print(f"Saved summary to: {args.output}")
    print(summary[["dataset", "mode", "classifier", "split"]].to_string(index=False))


if __name__ == "__main__":
    main()
