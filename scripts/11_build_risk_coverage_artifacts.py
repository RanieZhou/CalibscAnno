from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHODS = {
    "mlp_maxprob": "MLP maxprob",
    "proto_1_minus_max_cosine": "Raw prototype",
    "proto_class_z": "Class-z prototype",
    "mahalanobis_diag_min": "Diag. Mahalanobis",
}

METHOD_ORDER = list(METHODS)
COVERAGE_TAGS = ["val95", "val90", "val80"]
COVERAGE_LABELS = {
    "val95": "95%",
    "val90": "90%",
    "val80": "80%",
}
EMBEDDINGS = {
    "PCA50": "results/tables/open_set_score_benchmark_pca_seed0-4_summary.csv",
    "scFoundation": "results/tables/open_set_score_benchmark_scfoundation_summary.csv",
}

PALETTE = {
    "MLP maxprob": "#8A9199",
    "Raw prototype": "#B7A99A",
    "Class-z prototype": "#7F8F84",
    "Diag. Mahalanobis": "#4F6F63",
}


def load_summary(path: Path, embedding_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary table not found: {path}")
    df = pd.read_csv(path)
    df = df[df["score_method"].isin(METHODS)].copy()
    if df.empty:
        raise ValueError(f"No selected score methods in {path}")
    df["embedding_label"] = embedding_label
    df["method_label"] = df["score_method"].map(METHODS)
    return df


def add_average_rows(df: pd.DataFrame) -> pd.DataFrame:
    mean_cols = [col for col in df.columns if col.endswith("_mean")]
    std_cols = [col for col in df.columns if col.endswith("_std")]
    rows = []
    for (embedding_label, score_method), group in df.groupby(["embedding_label", "score_method"]):
        row = {
            "dataset": group["dataset"].iloc[0],
            "embedding": group["embedding"].iloc[0],
            "embedding_label": embedding_label,
            "holdout_cell_type": "Average across holdouts",
            "score_family": group["score_family"].iloc[0],
            "score_method": score_method,
            "method_label": group["method_label"].iloc[0],
            "n_split_seeds": int(group["n_split_seeds"].min()),
        }
        for col in mean_cols + std_cols:
            row[col] = float(group[col].mean())
        rows.append(row)
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True, sort=False)


def build_risk_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        for tag in COVERAGE_TAGS:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "embedding_label": row["embedding_label"],
                    "embedding": row["embedding"],
                    "holdout_cell_type": row["holdout_cell_type"],
                    "score_method": row["score_method"],
                    "method_label": row["method_label"],
                    "target_known_coverage": float(tag.replace("val", "")) / 100.0,
                    "target_known_coverage_label": COVERAGE_LABELS[tag],
                    "known_coverage_mean": row[f"known_coverage_at_{tag}_mean"],
                    "known_coverage_std": row[f"known_coverage_at_{tag}_std"],
                    "unknown_recall_mean": row[f"unknown_recall_at_{tag}_mean"],
                    "unknown_recall_std": row[f"unknown_recall_at_{tag}_std"],
                    "accepted_known_accuracy_mean": row[
                        f"accepted_known_accuracy_at_{tag}_mean"
                    ],
                    "accepted_known_accuracy_std": row[
                        f"accepted_known_accuracy_at_{tag}_std"
                    ],
                    "n_split_seeds": row["n_split_seeds"],
                }
            )

    risk = pd.DataFrame(rows)
    risk["method_order"] = risk["score_method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    risk = risk.sort_values(
        ["embedding_label", "holdout_cell_type", "method_order", "target_known_coverage"],
        ascending=[True, True, True, False],
    ).drop(columns=["method_order"])
    return risk


def apply_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.edgecolor": "#D8D1C7",
            "axes.labelcolor": "#4B5563",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "grid.color": "#E7E5E4",
            "grid.linewidth": 0.65,
            "grid.linestyle": "-",
            "grid.alpha": 0.65,
            "font.size": 8.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "xtick.color": "#6B7280",
            "ytick.color": "#6B7280",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_average_risk_coverage(risk: pd.DataFrame, png_path: Path, pdf_path: Path):
    apply_plot_style()
    avg = risk[risk["holdout_cell_type"] == "Average across holdouts"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    for ax, embedding_label in zip(axes, ["PCA50", "scFoundation"]):
        emb_df = avg[avg["embedding_label"] == embedding_label]
        for score_method in METHOD_ORDER:
            method_df = emb_df[emb_df["score_method"] == score_method].sort_values(
                "target_known_coverage"
            )
            label = METHODS[score_method]
            ax.plot(
                method_df["known_coverage_mean"],
                method_df["unknown_recall_mean"],
                marker="o",
                linewidth=2.0 if score_method == "mahalanobis_diag_min" else 1.45,
                markersize=4.6 if score_method == "mahalanobis_diag_min" else 4.0,
                color=PALETTE[label],
                label=label,
            )

        ax.set_title(embedding_label, fontsize=10, pad=8)
        ax.set_xlabel("Known coverage")
        ax.set_xlim(0.78, 0.98)
        ax.set_xticks([0.80, 0.90, 0.95])
        ax.set_xticklabels(["80%", "90%", "95%"])
        ax.set_ylim(0.0, 0.80)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Unknown recall")
    axes[1].legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    fig.suptitle(
        "Unknown rejection at validation-calibrated known coverage",
        y=1.02,
        fontsize=10.5,
    )
    fig.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pca_summary",
        type=Path,
        default=Path(EMBEDDINGS["PCA50"]),
    )
    parser.add_argument(
        "--scfoundation_summary",
        type=Path,
        default=Path(EMBEDDINGS["scFoundation"]),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/risk_coverage_summary.csv"),
    )
    parser.add_argument(
        "--figure_png",
        type=Path,
        default=Path("results/figures/risk_coverage_summary.png"),
    )
    parser.add_argument(
        "--figure_pdf",
        type=Path,
        default=Path("results/figures/risk_coverage_summary.pdf"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/figures/risk_coverage_summary_manifest.json"),
    )
    args = parser.parse_args()

    combined = pd.concat(
        [
            load_summary(args.pca_summary, "PCA50"),
            load_summary(args.scfoundation_summary, "scFoundation"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined = add_average_rows(combined)
    risk = build_risk_table(combined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    risk.to_csv(args.output, index=False)
    plot_average_risk_coverage(risk, args.figure_png, args.figure_pdf)

    manifest = {
        "source_tables": [str(args.pca_summary), str(args.scfoundation_summary)],
        "risk_table": str(args.output),
        "figure_png": str(args.figure_png),
        "figure_pdf": str(args.figure_pdf),
        "generating_script": "scripts/11_build_risk_coverage_artifacts.py",
        "surface_class": "paper_main",
        "main_claim": (
            "Diagonal Mahalanobis gives the strongest unknown recall at matched "
            "validation-calibrated known coverage, especially with scFoundation embeddings."
        ),
        "review_note": "Revised after visual inspection: reduced title size and expanded y-axis headroom to avoid clipping the strongest scFoundation curve.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Saved risk-coverage table to: {args.output}")
    print(f"Saved figure preview to: {args.figure_png}")
    print(f"Saved vector figure to: {args.figure_pdf}")
    print(f"Saved figure manifest to: {args.manifest}")
    print(
        risk[
            (risk["holdout_cell_type"] == "Average across holdouts")
            & (risk["target_known_coverage_label"] == "95%")
        ][
            [
                "embedding_label",
                "method_label",
                "known_coverage_mean",
                "unknown_recall_mean",
                "accepted_known_accuracy_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
