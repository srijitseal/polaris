"""Shared visualization utilities for the Polaris project."""

import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_DPI = 150


def set_style():
    """Set default plotting style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["figure.dpi"] = DEFAULT_DPI
