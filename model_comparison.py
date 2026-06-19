"""
Model comparison: BIC, AIC, BIC Bayes factors, and (if per-sample
prediction CSVs are present) Wilcoxon signed-rank test.

Switches
--------
DATASET  : "full" (7-season deep-dive) or "broad" (30-player study)
TESTS    : which tests to run; "wilcoxon" is silently skipped if
           per-sample prediction CSVs don't exist in RESULTS_DIR
BF_PAIRS : (simpler_model, richer_model) pairs for the BIC Bayes factor

Adding a new player
-------------------
No code changes needed here. Just ensure their result CSVs exist in
RESULTS_DIR (run baselines.py, gp_location.py, gp_context.py first).
Their data CSV must also exist so that n_train can be read.

Per-sample Wilcoxon
-------------------
To enable the Wilcoxon test, add SAVE_PREDICTIONS = True to baselines.py /
gp_location.py / gp_context.py and re-run them. Each script will then write
results/{DATASET}/predictions_{player_slug}.csv with columns:
  model, y_true, y_prob
The Wilcoxon test here auto-detects those files.
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Switches ──────────────────────────────────────────────────────────────────
DATASET = "full"     # "full" | "broad"

TESTS = ["bic", "aic", "bic_bf", "wilcoxon"]

# Nested pairs for BIC Bayes factor — (simpler, richer).
BF_PAIRS = [
    ("intercept_only", "gp_location"),
    ("gp_location",    "gp_context"),
]
# ─────────────────────────────────────────────────────────────────────────────

# Free parameter counts per model.
# SVGPs: only kernel hyperparameters counted here.
# The ELBO's KL[q(u)||p(u)] term already penalises variational parameters,
# so including them in BIC/AIC would double-count that regularisation.
MODEL_PARAMS = {
    "intercept_only": 1,    # global FG%
    "binned_EB":      2,    # α, β of Beta prior (fitted by MLE)
    "logistic_RBF":   101,  # 100 RBF weights + 1 bias
    "gp_location":    4,    # ls_x, ls_y, output-scale, constant mean
    "gp_context":     12,   # 2 spatial ARD LS + σ²_sp + 7 context ARD LS + σ²_ct + mean
}

MODEL_ORDER = ["intercept_only", "binned_EB", "logistic_RBF", "gp_location", "gp_context"]

RESULTS_DIR = f"results/{DATASET}"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_metrics() -> pd.DataFrame:
    """Concatenate all aggregate results CSVs found in RESULTS_DIR.
    Tries the folder-prefixed filename first (e.g. broad_baselines_results.csv),
    then falls back to the unprefixed name."""
    frames = []
    for fname in ["baselines_results.csv", "gp_location_results.csv", "gp_context_results.csv"]:
        prefixed = f"{RESULTS_DIR}/{DATASET}_{fname}"
        unprefixed = f"{RESULTS_DIR}/{fname}"
        path = prefixed if os.path.exists(prefixed) else unprefixed
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No results CSVs found in {RESULTS_DIR}/")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["model"].isin(MODEL_PARAMS)].copy()
    df["_order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df = df.sort_values(["player", "_order"]).drop(columns="_order").reset_index(drop=True)
    return df


def get_split_sizes(players: list) -> tuple[dict, dict]:
    """Return ({player: n_train}, {player: n_test}) by reading the data CSVs."""
    suffix = "_shots.csv"
    n_train, n_test = {}, {}
    for player in players:
        slug = player.lower().replace(" ", "_")
        path = f"data/{slug}{suffix}"
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=["Split"])
            n_train[player] = int((df["Split"] == "train").sum())
            n_test[player]  = int((df["Split"] == "test").sum())
        else:
            print(f"  Warning: data file not found for {player}: {path}")
    return n_train, n_test


# ── BIC / AIC ─────────────────────────────────────────────────────────────────

def compute_ic(metrics: pd.DataFrame, n_train: dict) -> pd.DataFrame:
    """
    BIC = 2·n·train_log_loss + k·log(n)
    AIC = 2·n·train_log_loss + 2·k
    n and log_loss are training-set quantities (correct BIC/AIC definition).
    ΔBIC/ΔAIC measured from best model in each player group.

    ΔBIC interpretation (Kass & Raftery 1995):
      < 2   : models have similar support
      2–6   : positive evidence against the worse model
      6–10  : strong evidence
      > 10  : decisive evidence
    """
    rows = []
    for _, row in metrics.iterrows():
        player = row["player"]
        model  = row["model"]
        n = n_train.get(player)
        k = MODEL_PARAMS.get(model)
        if n is None or k is None:
            continue
        if "train_log_loss" not in row or pd.isna(row["train_log_loss"]):
            print(f"  Warning: train_log_loss missing for {player} / {model} — skipping")
            continue
        ll = row["train_log_loss"]
        total_ll = -n * ll
        rows.append({
            "player":          player,
            "model":           model,
            "k":               k,
            "n_train":         n,
            "train_log_loss":  ll,
            "BIC":             -2 * total_ll + k * np.log(n),
            "AIC":             -2 * total_ll + 2 * k,
        })

    df = pd.DataFrame(rows)
    for ic in ["BIC", "AIC"]:
        df[f"Δ{ic}"] = df.groupby("player")[ic].transform(lambda x: x - x.min())
    for col in ["BIC", "AIC", "ΔBIC", "ΔAIC"]:
        df[col] = df[col].round(2)
    return df


# ── BIC Bayes Factors ─────────────────────────────────────────────────────────

def compute_bic_bayes_factors(ic_df: pd.DataFrame) -> pd.DataFrame:
    """
    BIC Bayes factor approximation (Kass & Raftery 1995):
      ΔBIC = BIC_simple − BIC_rich
      BF   ≈ exp(ΔBIC / 2)

    BF > 1 favours the richer model (lower BIC).
    Interpretation (Kass & Raftery 1995 scale on log₁₀(BF)):
      BF 1–3   : weak evidence
      BF 3–20  : positive evidence
      BF 20–150: strong evidence
      BF > 150 : decisive evidence
    """
    rows = []
    for simple, rich in BF_PAIRS:
        for player, grp in ic_df.groupby("player"):
            s_row = grp[grp["model"] == simple]
            r_row = grp[grp["model"] == rich]
            if s_row.empty or r_row.empty:
                continue
            bic_s    = s_row["BIC"].iloc[0]
            bic_r    = r_row["BIC"].iloc[0]
            delta_bic = bic_s - bic_r          # positive → richer has lower BIC
            bf        = np.exp(delta_bic / 2)  # BF > 1 → richer model preferred
            rows.append({
                "player":       player,
                "simple_model": simple,
                "rich_model":   rich,
                "ΔBIC":         round(delta_bic, 2),
                "BF":           round(bf, 2),
                "favours":      rich if bf > 1 else simple,
            })

    return pd.DataFrame(rows)


# ── Wilcoxon Signed-Rank Test ─────────────────────────────────────────────────

def _per_sample_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    p = np.clip(y_prob, 1e-9, 1 - 1e-9)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def compute_wilcoxon(players: list) -> pd.DataFrame:
    """
    Two-sided Wilcoxon signed-rank test on per-sample log-losses.

    Reads all files matching RESULTS_DIR/predictions_{slug}_*.csv and
    concatenates them. Each file has columns: model, y_true, y_prob.
    These are written by baselines.py / gp_location.py / gp_context.py
    when SAVE_PREDICTIONS = True.

    Returns an empty DataFrame if no prediction CSVs are found.
    """
    rows = []
    any_found = False

    for player in players:
        slug  = player.lower().replace(" ", "_")
        files = glob.glob(f"{RESULTS_DIR}/predictions_{slug}_*.csv")
        if not files:
            continue
        any_found = True

        pred_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        pred_df["ll_i"] = _per_sample_logloss(
            pred_df["y_true"].values, pred_df["y_prob"].values
        )

        models = [m for m in MODEL_ORDER if m in pred_df["model"].unique()]
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                ll1 = pred_df[pred_df["model"] == m1]["ll_i"].values
                ll2 = pred_df[pred_df["model"] == m2]["ll_i"].values
                if len(ll1) != len(ll2) or len(ll1) == 0:
                    continue
                diff = ll1 - ll2
                if np.all(diff == 0):
                    continue
                stat, p_val = stats.wilcoxon(ll1, ll2, alternative="two-sided")
                winner = m1 if ll1.mean() < ll2.mean() else m2
                rows.append({
                    "player":    player,
                    "model_A":   m1,
                    "model_B":   m2,
                    "mean_ll_A": round(ll1.mean(), 4),
                    "mean_ll_B": round(ll2.mean(), 4),
                    "winner":    winner,
                    "W":         round(stat, 1),
                    "p_value":   round(float(p_val), 4),
                    "sig_0.05":  "yes" if p_val < 0.05 else "no",
                })

    if not any_found:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Printing ──────────────────────────────────────────────────────────────────

def print_ic_table(ic_df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("  BIC / AIC — complexity-penalised comparison  (training log-likelihood)")
    print("  GP k = kernel hyperparameters only (ELBO KL penalises variational params)")
    print("  ΔBIC: <2 similar support | 2–6 positive | 6–10 strong | >10 decisive")
    print(f"{'='*70}")
    for player in ic_df["player"].unique():
        sub = ic_df[ic_df["player"] == player].sort_values("BIC")
        print(f"\n  {player}  (n_train={sub['n_train'].iloc[0]:,})")
        cols = ["model", "k", "train_log_loss", "BIC", "ΔBIC", "AIC", "ΔAIC"]
        print(sub[cols].to_string(index=False))


def print_bf_table(bf_df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("  BIC Bayes Factors  (Kass & Raftery 1995 approximation)")
    print("  BF = exp(ΔBIC/2), ΔBIC = BIC_simple − BIC_rich")
    print("  BF > 1 favours richer model  |  1–3 weak | 3–20 positive | >20 strong")
    print(f"{'='*70}")
    cols = ["player", "simple_model", "rich_model", "ΔBIC", "BF", "favours"]
    print(bf_df[cols].to_string(index=False))


def print_wilcoxon_table(w_df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("  Wilcoxon Signed-Rank Test  (per-sample log-losses, two-sided)")
    print(f"{'='*70}")
    cols = ["player", "model_A", "model_B", "winner", "mean_ll_A", "mean_ll_B",
            "W", "p_value", "sig_0.05"]
    print(w_df[cols].to_string(index=False))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_delta_bic(ic_df: pd.DataFrame):
    """
    Horizontal bar chart of ΔBIC per player.
    Colour coding: green < 2, orange 2–10, red > 10.
    """
    players = ic_df["player"].unique()
    fig, axes = plt.subplots(1, len(players),
                             figsize=(5 * len(players), 4),
                             sharey=False)
    if len(players) == 1:
        axes = [axes]

    for ax, player in zip(axes, players):
        sub = ic_df[ic_df["player"] == player].sort_values("ΔBIC", ascending=False)
        colors = [
            "#2ca02c" if d < 2 else "#ff7f0e" if d < 10 else "#d62728"
            for d in sub["ΔBIC"]
        ]
        ax.barh(sub["model"], sub["ΔBIC"], color=colors, edgecolor="white")
        ax.axvline(2,  color="grey", linestyle="--", linewidth=1, alpha=0.8)
        ax.axvline(10, color="grey", linestyle=":",  linewidth=1, alpha=0.8)
        ax.set_xlabel("ΔBIC  (lower = better)")
        ax.set_title(player, fontsize=11)
        for val, patch in zip(sub["ΔBIC"], ax.patches):
            ax.text(val + 0.3, patch.get_y() + patch.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=8)

    fig.suptitle(
        "ΔBIC relative to best model per player\n"
        "green: similar support (<2)  |  orange: positive (2–10)  |  red: decisive (>10)",
        fontsize=10,
    )
    fig.tight_layout()
    out = f"{RESULTS_DIR}/model_comparison_bic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_delta_bic_broad(ic_df: pd.DataFrame):
    """
    Population-level BIC/AIC summary for many players.
    Left: box plots of ΔBIC per model (excluding winner).
    Right: fraction of players where each model wins (ΔBIC = 0).
    """
    models_plot = [m for m in MODEL_ORDER if m in ic_df["model"].unique()]
    data_bic  = [ic_df[ic_df["model"] == m]["ΔBIC"].values for m in models_plot]
    data_aic  = [ic_df[ic_df["model"] == m]["ΔAIC"].values for m in models_plot]
    win_frac  = [float((ic_df[ic_df["model"] == m]["ΔBIC"] == 0).mean()) for m in models_plot]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ΔBIC box plot
    axes[0].boxplot(data_bic, labels=models_plot, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#aec6e8"))
    axes[0].axhline(2,  color="orange", linestyle="--", linewidth=1, label="Δ=2")
    axes[0].axhline(10, color="red",    linestyle=":",  linewidth=1, label="Δ=10")
    axes[0].set_title("ΔBIC across 30 players")
    axes[0].set_ylabel("ΔBIC  (lower = better)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)

    # ΔAIC box plot
    axes[1].boxplot(data_aic, labels=models_plot, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#b5e8b0"))
    axes[1].set_title("ΔAIC across 30 players")
    axes[1].set_ylabel("ΔAIC  (lower = better)")
    axes[1].tick_params(axis="x", rotation=30)

    # Win fraction bar chart (ΔBIC = 0)
    colors = ["#2ca02c" if f > 0.5 else "#ff7f0e" if f > 0.2 else "#d62728"
              for f in win_frac]
    bars = axes[2].bar(models_plot, [f * 100 for f in win_frac], color=colors,
                       edgecolor="white")
    axes[2].set_title("BIC win rate (% of players where model is best)")
    axes[2].set_ylabel("% of players")
    axes[2].set_ylim(0, 105)
    axes[2].tick_params(axis="x", rotation=30)
    for bar, f in zip(bars, win_frac):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{f*100:.0f}%", ha="center", fontsize=9)

    fig.suptitle(
        f"Model comparison — BIC/AIC summary  (n={ic_df['player'].nunique()} players)",
        fontsize=11,
    )
    fig.tight_layout()
    out = f"{RESULTS_DIR}/model_comparison_bic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    metrics = load_metrics()
    players = sorted(metrics["player"].unique())
    print(f"  Players found: {players}")
    print(f"  Models found:  {sorted(metrics['model'].unique())}")

    n_train, _n_test = get_split_sizes(players)

    results = {}

    # ── BIC / AIC ─────────────────────────────────────────────────────────────
    if any(t in TESTS for t in ["bic", "aic"]):
        ic_df = compute_ic(metrics, n_train)
        results["ic"] = ic_df
        print_ic_table(ic_df)
        if len(players) > 5:
            plot_delta_bic_broad(ic_df)
        else:
            plot_delta_bic(ic_df)
        ic_df.to_csv(f"{RESULTS_DIR}/model_comparison_ic.csv", index=False)
        print(f"  Saved {RESULTS_DIR}/model_comparison_ic.csv")

    # ── BIC Bayes Factors ─────────────────────────────────────────────────────
    if "bic_bf" in TESTS and "ic" in results:
        bf_df = compute_bic_bayes_factors(results["ic"])
        if not bf_df.empty:
            results["bic_bf"] = bf_df
            print_bf_table(bf_df)
            bf_df.to_csv(f"{RESULTS_DIR}/model_comparison_bf.csv", index=False)
            print(f"  Saved {RESULTS_DIR}/model_comparison_bf.csv")

    # ── Wilcoxon ──────────────────────────────────────────────────────────────
    if "wilcoxon" in TESTS:
        w_df = compute_wilcoxon(players)
        if w_df.empty:
            print("\n  Wilcoxon: skipped — no per-sample prediction CSVs found.")
            print("  To enable: set SAVE_PREDICTIONS = True in baselines.py,")
            print("  gp_location.py, and gp_context.py, then re-run them.")
        else:
            results["wilcoxon"] = w_df
            print_wilcoxon_table(w_df)
            w_df.to_csv(f"{RESULTS_DIR}/model_comparison_wilcoxon.csv", index=False)
            print(f"  Saved {RESULTS_DIR}/model_comparison_wilcoxon.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
