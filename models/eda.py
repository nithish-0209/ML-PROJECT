import json
import logging
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .model_utils import (
    detect_target_column,
    get_data_path,
    get_outputs_dir,
    get_project_root,
)

logger = logging.getLogger("SeverityPrediction.EDA")


def run_eda(data_path: Optional[Path] = None, output_dir: Optional[Path] = None) -> dict:
    """
    Perform comprehensive Exploratory Data Analysis:
      - Dataset dimension and schema analysis
      - Missing value and duplicate detection
      - Target distribution and class imbalance inspection
      - Numerical distributions and outlier detection
      - Categorical distributions
      - Correlation matrix heatmap
      - Bivariate feature-vs-target risk analysis
    All visual charts are saved to outputs/eda/.
    """
    csv_file = data_path or get_data_path()
    save_dir = output_dir or get_outputs_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_file}")

    print("=" * 70)
    print("      EXPLORATORY DATA ANALYSIS (EDA) — SEVERITY PREDICTION")
    print("=" * 70)

    df = pd.read_csv(csv_file)
    n_rows, n_cols = df.shape
    print(f"Loaded dataset: {csv_file}")
    print(f"Dimensions: {n_rows:,} rows, {n_cols} columns")

    # 1. Basic Metrics
    duplicates = int(df.duplicated().sum())
    missing_series = df.isnull().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing_series.items() if cnt > 0}
    total_missing = int(missing_series.sum())

    target_col = detect_target_column(df)
    target_counts = df[target_col].value_counts().to_dict()
    target_props = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()

    print(f"\n[1] Target Column: '{target_col}'")
    print(f"    Class Breakdown: {target_counts}")
    print(f"    Class Proportions (%): {target_props}")
    print(f"\n[2] Data Integrity:")
    print(f"    Duplicate rows: {duplicates}")
    print(f"    Columns with missing values: {missing_dict if missing_dict else 'None'}")
    print(f"    Total missing cells: {total_missing}")

    # Set visualization styling
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # --- Chart 1: Target Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette_colors = ["#22c55e", "#f59e0b", "#f97316", "#ef4444"]
    order = list(target_counts.keys())
    colors = palette_colors[:len(order)]

    sns.barplot(
        x=list(target_counts.keys()),
        y=list(target_counts.values()),
        palette=colors,
        ax=axes[0],
    )
    axes[0].set_title(f"Target Distribution: {target_col}", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Severity Level")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                         ha="center", va="bottom", fontsize=10, xytext=(0, 4), textcoords="offset points")

    axes[1].pie(
        list(target_counts.values()),
        labels=list(target_counts.keys()),
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops=dict(width=0.6, edgecolor="white"),
    )
    axes[1].set_title("Class Imbalance Ratio", fontsize=13, fontweight="bold")
    plt.savefig(save_dir / "01_target_distribution.png", dpi=200)
    plt.close()

    # --- Chart 2: Missing Values Analysis ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_cols = missing_pct[missing_pct > 0]
    if len(missing_cols) > 0:
        sns.barplot(x=missing_cols.index, y=missing_cols.values, color="#ef4444", ax=ax)
        ax.set_ylabel("Missing Percentage (%)")
        ax.set_title("Missing Values by Column", fontsize=13, fontweight="bold")
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                        ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    else:
        ax.text(0.5, 0.5, "No Missing Values in Dataset", ha="center", va="center", fontsize=14, color="#16a34a")
        ax.set_axis_off()
    plt.savefig(save_dir / "02_missing_values.png", dpi=200)
    plt.close()

    # --- Chart 3: Numerical Correlation Heatmap ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out ID columns
    num_cols = [c for c in num_cols if not c.lower().endswith("id") and not c.lower().endswith("index")]

    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 7))
        corr_matrix = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Numerical Features Correlation Matrix", fontsize=13, fontweight="bold")
        plt.savefig(save_dir / "03_correlation_matrix.png", dpi=200)
        plt.close()

    # --- Chart 4: Numerical Feature Distributions ---
    if len(num_cols) > 0:
        plot_cols = num_cols[:6]
        n_plots = len(plot_cols)
        fig, axes = plt.subplots(int(np.ceil(n_plots / 3)), min(3, n_plots), figsize=(15, 4 * np.ceil(n_plots / 3)))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(plot_cols):
            sns.histplot(df[col].dropna(), kde=True, color="#3b82f6", ax=axes[i])
            axes[i].set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
        for j in range(n_plots, len(axes)):
            fig.delaxes(axes[j])
        plt.savefig(save_dir / "04_numerical_distributions.png", dpi=200)
        plt.close()

    # --- Chart 5: Key Categorical Distributions ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target_col and not c.lower().endswith("id") and not c.lower().endswith("index")]

    if len(cat_cols) > 0:
        plot_cats = cat_cols[:4]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes = axes.flatten()
        for i, col in enumerate(plot_cats):
            top_counts = df[col].value_counts().head(6)
            sns.barplot(x=top_counts.index, y=top_counts.values, palette="Blues_r", ax=axes[i])
            axes[i].set_title(f"Categorical Feature: {col}", fontsize=11, fontweight="bold")
            axes[i].tick_params(axis="x", rotation=25)
        for j in range(len(plot_cats), len(axes)):
            fig.delaxes(axes[j])
        plt.savefig(save_dir / "05_categorical_distributions.png", dpi=200)
        plt.close()

    # --- Chart 6: Outlier Boxplots ---
    if len(num_cols) > 0:
        plot_outliers = num_cols[:4]
        fig, axes = plt.subplots(1, len(plot_outliers), figsize=(4 * len(plot_outliers), 5))
        if len(plot_outliers) == 1:
            axes = [axes]
        for i, col in enumerate(plot_outliers):
            sns.boxplot(y=df[col].dropna(), color="#a855f7", ax=axes[i])
            axes[i].set_title(f"Outliers: {col}", fontsize=11, fontweight="bold")
        plt.savefig(save_dir / "06_outlier_boxplots.png", dpi=200)
        plt.close()

    # --- Chart 7: Feature vs Target (Bivariate Risk) ---
    if "Vehicle_Speed" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.boxplot(x=target_col, y="Vehicle_Speed", data=df, palette="viridis", ax=axes[0])
        axes[0].set_title(f"Vehicle Speed vs {target_col}", fontsize=12, fontweight="bold")

        if "Driver_Alcohol" in df.columns:
            ct = pd.crosstab(df["Driver_Alcohol"], df[target_col], normalize="index") * 100
            ct.plot(kind="bar", stacked=True, colormap="plasma", ax=axes[1])
            axes[1].set_title("Driver Alcohol Involvement vs Severity (%)", fontsize=12, fontweight="bold")
            axes[1].set_ylabel("Percentage (%)")
            axes[1].tick_params(axis="x", rotation=0)
            axes[1].legend(title="Severity", bbox_to_anchor=(1.02, 1))

        plt.savefig(save_dir / "07_feature_vs_target.png", dpi=200)
        plt.close()

    summary_stats = {
        "dataset_path": str(csv_file),
        "total_rows": n_rows,
        "total_columns": n_cols,
        "duplicate_rows": duplicates,
        "missing_cells_total": total_missing,
        "target_column": target_col,
        "target_classes": list(target_counts.keys()),
        "class_counts": target_counts,
        "class_percentages": target_props,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
    }

    summary_json_path = save_dir / "eda_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=4)

    print(f"\n[3] EDA Visualizations generated successfully in: {save_dir}")
    print(f"    - 01_target_distribution.png")
    print(f"    - 02_missing_values.png")
    print(f"    - 03_correlation_matrix.png")
    print(f"    - 04_numerical_distributions.png")
    print(f"    - 05_categorical_distributions.png")
    print(f"    - 06_outlier_boxplots.png")
    print(f"    - 07_feature_vs_target.png")
    print(f"    - eda_summary.json")
    print("=" * 70)

    return summary_stats


if __name__ == "__main__":
    run_eda()
