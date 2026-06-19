"""
Unified comparison table across all models and players.
Loads the three results CSVs and concatenates them into one summary.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results/full"

MODEL_ORDER = [
    "intercept_only",
    "binned_EB",
    "logistic_RBF",
    "gp_location",
    "gp_context",
]

MODEL_LABEL = {
    "intercept_only": "Intercept-only",
    "binned_EB":      "Binned EB",
    "logistic_RBF":   "Logistic RBF",
    "gp_location":    "GP location",
    "gp_context":     "GP context",
}

PLAYER_ORDER = ["Stephen Curry", "LeBron James", "Kevin Durant"]

DISPLAY_COLS = [
    "player", "model",
    "log_loss", "brier", "roc_auc", "ece",
    "ll_skill_%", "bs_skill_%",
    "reliability", "resolution",
]


def load_all() -> pd.DataFrame:
    frames = [
        pd.read_csv(f"{RESULTS_DIR}/baselines_results.csv"),
        pd.read_csv(f"{RESULTS_DIR}/gp_location_results.csv"),
        pd.read_csv(f"{RESULTS_DIR}/gp_context_results.csv"),
    ]
    df = pd.concat(frames, ignore_index=True)
    df["_model_rank"]  = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df["_player_rank"] = df["player"].map({p: i for i, p in enumerate(PLAYER_ORDER)})
    df = df.sort_values(["_player_rank", "_model_rank"]).drop(
        columns=["_model_rank", "_player_rank"]
    ).reset_index(drop=True)
    return df


def print_table(df: pd.DataFrame):
    print(f"\n{'='*60}\nUnified comparison — all models & players\n{'='*60}")
    for player in PLAYER_ORDER:
        sub = df[df["player"] == player][DISPLAY_COLS].copy()
        sub["model"] = sub["model"].map(MODEL_LABEL)
        print(f"\n  {player}")
        print(sub.drop(columns="player").to_string(index=False))


def plot_logloss(df: pd.DataFrame):
    """Grouped bar chart: log loss per player × model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    colors = ["#aec7e8", "#ffbb78", "#98df8a", "#1f77b4", "#2ca02c"]

    for ax, player in zip(axes, PLAYER_ORDER):
        sub = df[df["player"] == player].set_index("model").reindex(MODEL_ORDER)
        vals   = sub["log_loss"].values
        labels = [MODEL_LABEL[m] for m in MODEL_ORDER]
        bars   = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)

        # Annotate bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(player, fontsize=11)
        ax.set_ylabel("Log loss  (↓ better)" if ax == axes[0] else "")
        ax.set_ylim(min(vals) * 0.97, max(vals) * 1.02)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)

    fig.suptitle("Log loss by model and player  (lower = better)", fontsize=12)
    fig.tight_layout()
    out = f"{RESULTS_DIR}/unified_logloss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_skill_scores(df: pd.DataFrame):
    """Grouped bar chart: log-loss skill score (% over intercept-only) per player × model."""
    # skill score not defined for intercept_only itself — skip it
    models_to_plot = ["binned_EB", "logistic_RBF", "gp_location", "gp_context"]
    colors = ["#ffbb78", "#98df8a", "#1f77b4", "#2ca02c"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)

    for ax, player in zip(axes, PLAYER_ORDER):
        sub = df[(df["player"] == player) & (df["model"].isin(models_to_plot))].copy()
        sub = sub.set_index("model").reindex(models_to_plot)
        vals   = pd.to_numeric(sub["ll_skill_%"], errors="coerce").values
        labels = [MODEL_LABEL[m] for m in models_to_plot]
        bars   = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)

        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(player, fontsize=11)
        ax.set_ylabel("Log-loss skill score (%)" if ax == axes[0] else "")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    fig.suptitle("Log-loss skill score vs intercept-only  (↑ better)", fontsize=12)
    fig.tight_layout()
    out = f"{RESULTS_DIR}/unified_skill_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_ece(df: pd.DataFrame):
    """Bar chart: ECE per player × model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    colors = ["#aec7e8", "#ffbb78", "#98df8a", "#1f77b4", "#2ca02c"]

    for ax, player in zip(axes, PLAYER_ORDER):
        sub  = df[df["player"] == player].set_index("model").reindex(MODEL_ORDER)
        vals = sub["ece"].values
        labels = [MODEL_LABEL[m] for m in MODEL_ORDER]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(player, fontsize=11)
        ax.set_ylabel("ECE  (↓ better)" if ax == axes[0] else "")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)

    fig.suptitle("Expected Calibration Error by model and player  (lower = better)", fontsize=12)
    fig.tight_layout()
    out = f"{RESULTS_DIR}/unified_ece.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    df = load_all()

    # Save
    out_csv = f"{RESULTS_DIR}/unified_comparison.csv"
    df[DISPLAY_COLS].to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    print_table(df)
    plot_logloss(df)
    plot_skill_scores(df)
    plot_ece(df)


if __name__ == "__main__":
    main()
