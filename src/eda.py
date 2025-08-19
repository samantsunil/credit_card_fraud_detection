import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import ensure_dirs, EDA_DIR, load_data


def eda(df: pd.DataFrame, save_prefix: Optional[str] = None) -> None:
    ensure_dirs()

    # Basic info
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Target distribution (Class):\n", df["Class"].value_counts(normalize=False))
    print("\nMissing values per column:\n", df.isna().sum())

    # Save description
    desc = df.describe().T
    desc_path = os.path.join(EDA_DIR, f"{save_prefix or 'dataset'}_describe.csv")
    desc.to_csv(desc_path)
    print(f"Saved describe to {desc_path}")

    # Correlation heatmap (limit to numeric columns to avoid memory blowup)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, cbar=True)
    heatmap_path = os.path.join(EDA_DIR, f"{save_prefix or 'dataset'}_corr_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved correlation heatmap to {heatmap_path}")

    # Boxplots for a few features to visualize outliers (sample to avoid huge figure)
    sample_cols = [c for c in numeric_df.columns if c not in ["Class"]][:10]
    if sample_cols:
        melted = numeric_df[sample_cols].melt(var_name="feature", value_name="value")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=melted, x="feature", y="value")
        plt.xticks(rotation=45, ha="right")
        box_path = os.path.join(EDA_DIR, f"{save_prefix or 'dataset'}_boxplots.png")
        plt.tight_layout()
        plt.savefig(box_path, dpi=150)
        plt.close()
        print(f"Saved boxplots to {box_path}")

    # Class imbalance bar plot
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Class", data=df)
    plt.title("Class Distribution")
    class_dist_path = os.path.join(EDA_DIR, f"{save_prefix or 'dataset'}_class_distribution.png")
    plt.tight_layout()
    plt.savefig(class_dist_path, dpi=150)
    plt.close()
    print(f"Saved class distribution plot to {class_dist_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for Credit Card Fraud dataset")
    parser.add_argument("--data-path", required=True, help="Path to creditcard.csv")
    parser.add_argument("--prefix", default="dataset", help="Prefix for saved artifacts")
    args = parser.parse_args()

    df = load_data(args.data_path)
    eda(df, save_prefix=args.prefix)


if __name__ == "__main__":
    main()


