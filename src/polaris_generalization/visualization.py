"""Shared visualization utilities for the Polaris project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DEFAULT_DPI = 150

MODEL_COLORS = {"xgboost": "steelblue", "chemprop": "darkorange"}
MODEL_LABELS = {"xgboost": "XGBoost", "chemprop": "Chemprop (D-MPNN)"}


def set_style():
    """Set default plotting style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["figure.dpi"] = DEFAULT_DPI


def plot_model_comparison_bars(
    data_by_model: dict,
    endpoint_col: str,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Grouped bar chart comparing two or more models per endpoint.

    Parameters
    ----------
    data_by_model : {"xgboost": df, "chemprop": df} where each df has endpoint_col and metric_col
    endpoint_col : column name for x-axis grouping (e.g. "endpoint")
    metric_col : column name for bar height (e.g. "r2")
    ylabel, title : axis labels
    output_path : where to save the figure
    """
    model_names = list(data_by_model.keys())
    first_df = next(iter(data_by_model.values()))
    endpoints = sorted(first_df[endpoint_col].unique())
    n_ep = len(endpoints)
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(max(10, n_ep * 1.6), 5))
    x = np.arange(n_ep)
    width = 0.8 / n_models

    for i, model_name in enumerate(model_names):
        df = data_by_model[model_name]
        means, stds = [], []
        for ep in endpoints:
            ep_vals = df[df[endpoint_col] == ep][metric_col].dropna()
            means.append(ep_vals.mean() if len(ep_vals) else np.nan)
            stds.append(ep_vals.std() if len(ep_vals) > 1 else 0.0)

        offset = (i - n_models / 2 + 0.5) * width
        yerr = stds if any(s > 0 for s in stds) else None
        ax.bar(
            x + offset,
            means,
            width,
            yerr=yerr,
            label=MODEL_LABELS.get(model_name, model_name),
            color=MODEL_COLORS.get(model_name, f"C{i}"),
            edgecolor="white",
            alpha=0.85,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(endpoints, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
