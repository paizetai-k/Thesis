"""
H2 Limited Data Experiment — degradation curves on rookie seasons.

Design:
  - Players : Stephen Curry (2009-10), LeBron James (2003-04), Kevin Durant (2007-08)
  - Test set: shots from the last N_TEST_GAMES games of the rookie season (fixed)
  - Pool    : all remaining games
  - For each TRAIN_SIZE in TRAIN_SIZES, randomly sample that many games from
    the pool N_DRAWS times and evaluate all models.
  - Aggregate mean ± std across draws.

Models: intercept-only, binned EB, logistic RBF, GP location-only (ARD-2 Matérn-5/2 standard kernel)
Output: results/h2_limited/raw_results.csv
        results/h2_limited/aggregated_results.csv
        results/h2_limited/degradation_{player}.png
"""
import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from baselines import (intercept_only, binned_eb, logistic_rbf, full_metrics)
from gp_utils   import to_tensor, standardize, predict_probs
from kernels    import train_gp, build_kernel

torch.manual_seed(0)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH     = "data/NBA Shot Locations 1997 - 2020.csv"
RESULTS_DIR   = "results/h2_limited"
os.makedirs(RESULTS_DIR, exist_ok=True)

PLAYERS = [
    ("Stephen Curry", 2009),   # rookie season 2009-10
    ("LeBron James",  2003),   # rookie season 2003-04
    ("Kevin Durant",  2007),   # rookie season 2007-08
]

N_TEST_GAMES = 20
TRAIN_SIZES  = [10, 20, 30, 40, 50]
N_DRAWS      = 5

GP_N_ITER    = 300
GP_LR        = 0.01
# ─────────────────────────────────────────────────────────────────────────────


def derive_season_year(game_date_int) -> int:
    """YYYYMMDD int → NBA season start year (Oct–Sep convention)."""
    d     = int(game_date_int)
    year  = d // 10000
    month = (d % 10000) // 100
    return year if month >= 10 else year - 1


def load_rookie_season(raw: pd.DataFrame, player_name: str, rookie_year: int) -> pd.DataFrame:
    df = raw[raw["Player Name"] == player_name].copy()
    df = df[df["Season Type"] == "Regular Season"].copy()
    df["_season_year"] = df["Game Date"].apply(derive_season_year)
    df = df[df["_season_year"] == rookie_year].copy()
    df["is_3pt"] = (df["Shot Type"] == "3PT Field Goal").astype(int)
    df = df.sort_values("Game Date").reset_index(drop=True)
    return df


def split_test_pool(df: pd.DataFrame, n_test_games: int):
    """Return (pool_df, test_df, ordered_pool_game_ids)."""
    game_order = (df.drop_duplicates("Game ID")
                    .sort_values("Game Date")["Game ID"]
                    .tolist())
    test_game_ids = set(game_order[-n_test_games:])
    pool_game_ids = game_order[:-n_test_games]

    test_df = df[df["Game ID"].isin(test_game_ids)].copy()
    pool_df = df[df["Game ID"].isin(pool_game_ids)].copy()
    return pool_df, test_df, pool_game_ids


def sample_train(pool_df: pd.DataFrame, pool_game_ids: list, n_games: int, seed: int) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    chosen = rng.choice(pool_game_ids, size=n_games, replace=False)
    return pool_df[pool_df["Game ID"].isin(chosen)].copy()


def run_gp(train_df: pd.DataFrame, test_df: pd.DataFrame,
           y_true: np.ndarray, ref_ll: float, ref_bs: float, ref_acc: float,
           seed: int) -> dict:
    torch.manual_seed(seed)
    X_tr_raw, y_tr = to_tensor(train_df)
    X_te_raw, _    = to_tensor(test_df)
    X_tr, X_te, _, _ = standardize(X_tr_raw, X_te_raw)

    n_inducing = min(100, max(10, len(train_df) // 3))
    kernel = build_kernel("standard")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, likelihood = train_gp(
            X_tr, y_tr, kernel,
            learn_inducing=True,
            n_iter=GP_N_ITER,
            lr=GP_LR,
            n_inducing=n_inducing,
            print_every=0,
        )

    probs, _ = predict_probs(model, likelihood, X_te)
    return full_metrics(y_true, probs, "gp_location",
                        ref_logloss=ref_ll, ref_brier=ref_bs, ref_acc=ref_acc)


def main():
    print("Loading dataset...")
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  {len(raw):,} rows loaded.")

    all_rows = []

    for player_name, rookie_year in PLAYERS:
        slug = player_name.lower().replace(" ", "_")
        print(f"\n{'='*60}")
        print(f"  {player_name}  (rookie {rookie_year}-{str(rookie_year+1)[2:]})")
        print(f"{'='*60}")

        df = load_rookie_season(raw, player_name, rookie_year)
        pool_df, test_df, pool_game_ids = split_test_pool(df, N_TEST_GAMES)
        y_true = test_df["Shot Made Flag"].values

        print(f"  Pool : {len(pool_game_ids)} games  ({len(pool_df):,} shots)")
        print(f"  Test : {N_TEST_GAMES} games  ({len(test_df):,} shots)")

        for train_size in TRAIN_SIZES:
            if train_size > len(pool_game_ids):
                print(f"  [skip] train_size={train_size} exceeds pool ({len(pool_game_ids)} games)")
                continue

            print(f"\n  -- train_size = {train_size} games --")

            for draw in range(N_DRAWS):
                seed = draw * 1000 + train_size * 10
                train_df = sample_train(pool_df, pool_game_ids, train_size, seed)
                n_shots  = len(train_df)

                # ── Intercept-only (also provides skill-score reference) ────
                row_int, _ = intercept_only(train_df, test_df)
                ref_ll  = row_int["log_loss"]
                ref_bs  = row_int["brier"]
                ref_acc = row_int["accuracy"]

                # ── Binned EB ─────────────────────────────────────────────
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    row_eb, *_ = binned_eb(train_df, test_df, ref_ll, ref_bs, ref_acc)

                # ── Logistic RBF (adaptive centres) ───────────────────────
                n_centres = min(50, max(5, n_shots // 3))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    row_rbf, *_ = logistic_rbf(
                        train_df, test_df, ref_ll, ref_bs, ref_acc,
                        n_centres=n_centres,
                    )

                # ── GP location-only ──────────────────────────────────────
                row_gp = run_gp(train_df, test_df, y_true, ref_ll, ref_bs, ref_acc, seed)

                print(f"    draw {draw+1}/{N_DRAWS}  shots={n_shots:4d}"
                      f"  ll: int={ref_ll:.4f}"
                      f"  eb={row_eb['log_loss']:.4f}"
                      f"  rbf={row_rbf['log_loss']:.4f}"
                      f"  gp={row_gp['log_loss']:.4f}")

                for row in [row_int, row_eb, row_rbf, row_gp]:
                    all_rows.append({
                        "player":     player_name,
                        "train_size": train_size,
                        "draw":       draw,
                        "n_shots":    n_shots,
                        **row,
                    })

    # ── Save raw results ─────────────────────────────────────────────────────
    raw_df = pd.DataFrame(all_rows)
    raw_path = f"{RESULTS_DIR}/raw_results.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved raw results  → {raw_path}")

    # ── Aggregate across draws ────────────────────────────────────────────────
    agg = (raw_df
           .groupby(["player", "train_size", "model"])
           .agg(
               log_loss_mean=("log_loss",  "mean"),
               log_loss_std =("log_loss",  "std"),
               brier_mean   =("brier",     "mean"),
               brier_std    =("brier",     "std"),
               ece_mean     =("ece",       "mean"),
               ece_std      =("ece",       "std"),
               roc_auc_mean =("roc_auc",   "mean"),
               roc_auc_std  =("roc_auc",   "std"),
               n_shots_mean =("n_shots",   "mean"),
           )
           .reset_index())
    agg_path = f"{RESULTS_DIR}/aggregated_results.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Saved aggregated   → {agg_path}")

    plot_degradation_curves(agg)
    print("\nDone.")


# ── Plotting ─────────────────────────────────────────────────────────────────

MODEL_STYLE = {
    "intercept_only": dict(color="gray",       label="Intercept-only", ls="--"),
    "binned_EB":      dict(color="steelblue",   label="Binned EB"),
    "logistic_RBF":   dict(color="darkorange",  label="Logistic RBF"),
    "gp_location":    dict(color="seagreen",    label="GP location"),
}

METRICS = [
    ("log_loss", "Log Loss (↓ better)"),
    ("brier",    "Brier Score (↓ better)"),
    ("ece",      "ECE (↓ better)"),
]


def plot_degradation_curves(agg: pd.DataFrame):
    for player_name in agg["player"].unique():
        slug = player_name.lower().replace(" ", "_")
        pdf  = agg[agg["player"] == player_name]

        fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
        fig.suptitle(f"{player_name} — Limited-Data Degradation", fontsize=12, fontweight="bold")

        for ax, (metric, ylabel) in zip(axes, METRICS):
            for model, style in MODEL_STYLE.items():
                mdf = pdf[pdf["model"] == model].sort_values("train_size")
                if mdf.empty:
                    continue
                x  = mdf["train_size"].values
                mu = mdf[f"{metric}_mean"].values
                sd = mdf[f"{metric}_std"].values
                ax.plot(x, mu, marker="o", ms=4,
                        color=style["color"], ls=style.get("ls", "-"),
                        label=style["label"])
                ax.fill_between(x, mu - sd, mu + sd,
                                alpha=0.15, color=style["color"])

            ax.set_xlabel("Training games")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel.split(" ")[0])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"{RESULTS_DIR}/degradation_{slug}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot → {fname}")


if __name__ == "__main__":
    main()
