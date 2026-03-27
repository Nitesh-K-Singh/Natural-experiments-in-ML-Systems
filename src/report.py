from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────


def _load_img(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return mpimg.imread(path)


def _get_fig_paths(run_dir):
    fig_dir = Path(run_dir) / "figures"
    return {
        "tau": fig_dir / "1_tau_distribution.jpg",
        "churn": fig_dir / "2_churn_distribution.jpg",
        "dist": fig_dir / "3_policy_distribution.jpg",
        "ecdf": fig_dir / "4_policy_ecdf.jpg",
    }


def _get_data_paths(run_dir):
    data_dir = Path(run_dir) / "data"
    return {
        "summary": data_dir / "3_summary_df.csv",
        "modal": data_dir / "4_df_modal_winner.csv",
    }


# ─────────────────────────────────────────────────────────────
# FIGURE 1: feature space (2x3 grid)
# ─────────────────────────────────────────────────────────────


def plot_feature_grid(run_map):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    for col_idx, (rho, run_dir) in enumerate(run_map.items()):
        paths = _get_fig_paths(run_dir)

        axes[0, col_idx].imshow(_load_img(paths["churn"]))
        axes[1, col_idx].imshow(_load_img(paths["tau"]))

        axes[0, col_idx].axis("off")
        axes[1, col_idx].axis("off")

    for col_idx, rho in enumerate(run_map.keys()):
        axes[0, col_idx].set_title(f"ρ = {rho}")

    axes[0, 0].set_ylabel("Churn Score")
    axes[1, 0].set_ylabel("Treatment Effect (τ)")

    fig.suptitle("Feature Space Comparison", fontsize=13)
    return fig


# ─────────────────────────────────────────────────────────────
# FIGURE 2: policy distribution (1x3)
# ─────────────────────────────────────────────────────────────


def plot_distribution_grid(run_map):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for col_idx, (rho, run_dir) in enumerate(run_map.items()):
        path = _get_fig_paths(run_dir)["dist"]
        axes[col_idx].imshow(_load_img(path))
        axes[col_idx].axis("off")
        axes[col_idx].set_title(f"ρ = {rho}")

    fig.suptitle("Policy Outcome Distribution", fontsize=13)
    return fig


# ─────────────────────────────────────────────────────────────
# FIGURE 3: ECDF (1x3)
# ─────────────────────────────────────────────────────────────


def plot_ecdf_grid(run_map):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for col_idx, (rho, run_dir) in enumerate(run_map.items()):
        path = _get_fig_paths(run_dir)["ecdf"]
        axes[col_idx].imshow(_load_img(path))
        axes[col_idx].axis("off")
        axes[col_idx].set_title(f"ρ = {rho}")

    fig.suptitle("ECDF of Policy Outcomes", fontsize=13)
    return fig


# ─────────────────────────────────────────────────────────────
# DATAFRAME 1: modal winner combined
# ─────────────────────────────────────────────────────────────


def get_modal_winner_df(run_map):
    dfs = []

    for rho, run_dir in run_map.items():
        df = pd.read_csv(_get_data_paths(run_dir)["modal"])
        df["rho"] = rho
        dfs.append(df)

    df = pd.concat(dfs)

    # pivot to wide format
    df_wide = df.pivot(index="policy", columns="rho", values="probability")

    return df_wide


# ─────────────────────────────────────────────────────────────
# DATAFRAME 2: summary combined
# ─────────────────────────────────────────────────────────────


def get_summary_df(run_map):
    dfs = []

    for rho, run_dir in run_map.items():
        df = pd.read_csv(_get_data_paths(run_dir)["summary"])
        df["rho"] = rho
        dfs.append(df)

    df = pd.concat(dfs)

    # create formatted string
    df["mean_ci"] = df.apply(
        lambda x: f"{x['mean']:.0f} [{x['ci_lower']:.0f}, {x['ci_upper']:.0f}]", axis=1
    )

    df_wide = df.pivot(index="policy", columns="rho", values="mean_ci")
    df_wide = df_wide[sorted(df_wide.columns)]

    return df_wide


# ─────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────


def generate_report(run_map):
    """
    run_map = {
        -1: Path(...),
         0: Path(...),
         1: Path(...)
    }
    """

    figs = {
        "feature_grid": plot_feature_grid(run_map),
        "distribution_grid": plot_distribution_grid(run_map),
        "ecdf_grid": plot_ecdf_grid(run_map),
    }

    dfs = {
        "modal_winner": get_modal_winner_df(run_map),
        "summary": get_summary_df(run_map),
    }

    return figs, dfs
