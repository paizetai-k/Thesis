"""
Spatial stratification: per-region metrics to test whether GP gains
concentrate in sparse court areas (evaluation criterion for H1 & H2).

Regions (geometric, from x / y / is_3pt):
  paint       : 2PT, |x| < 96, y < 190   — lane/paint area        (dense)
  mid_range   : 2PT, outside paint        — mid-range jumpers      (sparse)
  corner_3    : 3PT, |x| > 220            — corner threes          (sparse)
  above_break : 3PT, |x| <= 220           — above-the-break threes (moderate)

Models trained per player (5 total):
  intercept_only  — global FG% constant
  binned_eb       — hexagonal binned empirical Bayes
  logistic_rbf    — logistic regression on 100 RBF features
  gp_location     — SVGP, ARD-2 Matérn-5/2 kernel on (x, y)
  gp_context      — SVGP, additive spatial+context kernel (9D)

Output: results/full/spatial_stratification.csv
        results/full/spatial_stratification_*.png
"""
import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss

from baselines import intercept_only, binned_eb, logistic_rbf
from gp_utils import predict_probs
from gp_context import add_context_features, _to_tensor, _standardize, _build_additive_kernel
from kernels import build_kernel, train_gp

torch.manual_seed(42)

PLAYERS = [
    ("Stephen Curry", "data/stephen_curry_shots.csv"),
    ("LeBron James",  "data/lebron_james_shots.csv"),
    ("Kevin Durant",  "data/kevin_durant_shots.csv"),
]

RESULTS_DIR = "results/full"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_INDUCING = 200
N_ITER     = 500
LR         = 0.01

REGIONS = ["paint", "mid_range", "corner_3", "above_break"]
REGION_LABEL = {
    "paint":       "Paint\n(dense)",
    "mid_range":   "Mid-range\n(sparse)",
    "corner_3":    "Corner 3\n(sparse)",
    "above_break": "Above-break 3\n(moderate)",
}


def assign_region(df: pd.DataFrame) -> pd.Series:
    x   = df["X Location"].values
    y   = df["Y Location"].values
    is3 = df["is_3pt"].values.astype(bool)

    paint  = ~is3 & (np.abs(x) < 96) & (y < 190)
    corner = is3  & (np.abs(x) > 220)
    above  = is3  & ~corner
    mid    = ~is3 & ~paint

    region = pd.Series("", index=df.index)
    region[paint]  = "paint"
    region[mid]    = "mid_range"
    region[corner] = "corner_3"
    region[above]  = "above_break"
    return region


def safe_metrics(y_true: np.ndarray, probs: np.ndarray):
    """Return (log_loss, brier) or (None, None) if region too small / one class."""
    if len(y_true) < 5 or len(np.unique(y_true)) < 2:
        return None, None
    ll = log_loss(y_true, np.clip(probs, 1e-7, 1 - 1e-7))
    bs = brier_score_loss(y_true, probs)
    return round(ll, 4), round(bs, 4)


def main():
    all_rows = []

    for player_name, path in PLAYERS:
        print(f"\n{'='*60}\n  {player_name}\n{'='*60}")

        df = pd.read_csv(path)
        df = add_context_features(df)
        df["region"] = assign_region(df)
        train  = df[df["Split"] == "train"].copy()
        test   = df[df["Split"] == "test"].copy()
        y_true = test["Shot Made Flag"].values

        # Region composition
        print("  Region sizes (n_train | n_test):")
        for r in REGIONS:
            n_tr = (train["region"] == r).sum()
            n_te = (test["region"]  == r).sum()
            pct  = n_tr / len(train) * 100
            print(f"    {r:<14}: {n_tr:5,}  ({pct:.1f}%)  |  {n_te:4,}")

        # ── Baselines ──────────────────────────────────────────────────────────
        _, prob_io = intercept_only(train, test)
        ref_ll  = log_loss(y_true, np.clip(prob_io, 1e-7, 1 - 1e-7))
        ref_bs  = brier_score_loss(y_true, prob_io)
        ref_acc = ((prob_io >= 0.5) == y_true).mean()

        _, prob_eb, *_ = binned_eb(train, test, ref_ll, ref_bs, ref_acc)
        _, prob_lr, *_ = logistic_rbf(train, test, ref_ll, ref_bs, ref_acc)

        # ── GP location-only ───────────────────────────────────────────────────
        X_tr = torch.tensor(train[["X Location", "Y Location"]].values, dtype=torch.float32)
        X_te = torch.tensor(test[["X Location", "Y Location"]].values,  dtype=torch.float32)
        y_tr = torch.tensor(train["Shot Made Flag"].values,              dtype=torch.float32)
        mu  = X_tr.mean(0)
        std = X_tr.std(0).clamp(min=1e-6)
        X_tr_n = (X_tr - mu) / std
        X_te_n = (X_te - mu) / std

        kernel = build_kernel("standard")
        print("  Training GP location-only...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, lik = train_gp(
                X_tr_n, y_tr, kernel,
                n_iter=N_ITER, lr=LR, n_inducing=N_INDUCING, print_every=100,
            )
        prob_gp, _ = predict_probs(model, lik, X_te_n)

        # ── GP context (spatial + 7 context features) ──────────────────────────
        X_tr_ctx_raw, y_tr_ctx = _to_tensor(train)
        X_te_ctx_raw, _        = _to_tensor(test)
        X_tr_ctx, X_te_ctx, _, _ = _standardize(X_tr_ctx_raw, X_te_ctx_raw)

        ctx_kernel = _build_additive_kernel()
        print("  Training GP context...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx_model, ctx_lik = train_gp(
                X_tr_ctx, y_tr_ctx, ctx_kernel,
                learn_inducing=True,
                n_iter=N_ITER, lr=LR, n_inducing=N_INDUCING, print_every=100,
            )
        prob_ctx, _ = predict_probs(ctx_model, ctx_lik, X_te_ctx)

        model_probs = {
            "intercept_only": prob_io,
            "binned_eb":      prob_eb,
            "logistic_rbf":   prob_lr,
            "gp_location":    prob_gp,
            "gp_context":     prob_ctx,
        }

        # ── Per-region metrics ─────────────────────────────────────────────────
        for region in REGIONS:
            mask   = (test["region"] == region).values
            y_r    = y_true[mask]
            n_tr_r = int((train["region"] == region).sum())

            for model_name, probs in model_probs.items():
                ll, bs = safe_metrics(y_r, probs[mask])
                if ll is not None:
                    all_rows.append({
                        "player":   player_name,
                        "region":   region,
                        "model":    model_name,
                        "n_train":  n_tr_r,
                        "n_test":   int(mask.sum()),
                        "log_loss": ll,
                        "brier":    bs,
                    })

    # ── Save & print ───────────────────────────────────────────────────────────
    results = pd.DataFrame(all_rows)
    out_csv = f"{RESULTS_DIR}/spatial_stratification.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    MODEL_ORDER = ["intercept_only", "binned_eb", "logistic_rbf", "gp_location", "gp_context"]
    print(f"\n{'='*60}\nLog loss by region & model\n{'='*60}")
    for player_name, _ in PLAYERS:
        print(f"\n  {player_name}")
        df_p  = results[results["player"] == player_name]
        pivot = df_p.pivot_table(
            index="region", columns="model", values="log_loss", aggfunc="first"
        ).reindex(columns=MODEL_ORDER).reindex(REGIONS)
        n_col = df_p.groupby("region")["n_train"].first().reindex(REGIONS)
        pivot.insert(0, "n_train", n_col)
        print(pivot.round(4).to_string())

    # ── Plot ───────────────────────────────────────────────────────────────────
    _plot_advantage(results)
    _plot_logloss_heatmap(results)


def _plot_advantage(results: pd.DataFrame):
    """
    3-panel bar chart: GP log-loss reduction vs each baseline per region.
    Positive = GP is better; bars are grouped by baseline.
    """
    players   = [p for p, _ in PLAYERS]
    baselines = ["intercept_only", "binned_eb", "logistic_rbf"]
    bl_labels = ["Intercept-only", "Binned EB", "Logistic RBF"]
    colors    = ["#d62728", "#ff7f0e", "#1f77b4"]
    width     = 0.25
    x         = np.arange(len(REGIONS))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (ax, player_name) in enumerate(zip(axes, players)):
        df_p = results[results["player"] == player_name]

        for i, (bl, label, col) in enumerate(zip(baselines, bl_labels, colors)):
            advantages = []
            for region in REGIONS:
                gp = df_p[(df_p["region"] == region) & (df_p["model"] == "gp_location")]["log_loss"].values
                bv = df_p[(df_p["region"] == region) & (df_p["model"] == bl)]["log_loss"].values
                advantages.append(float(bv[0] - gp[0]) if len(gp) and len(bv) else float("nan"))
            ax.bar(x + (i - 1) * width, advantages, width, label=label, color=col, alpha=0.85)

        n_labels = []
        for region in REGIONS:
            row = df_p[(df_p["region"] == region) & (df_p["model"] == "gp_location")]
            n_labels.append(f"n={int(row['n_train'].iloc[0]):,}" if len(row) else "")

        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{REGION_LABEL[r]}\n{n}" for r, n in zip(REGIONS, n_labels)],
            fontsize=8,
        )
        ax.set_title(player_name, fontsize=11)
        if idx == 0:
            ax.set_ylabel("Log-loss reduction vs GP  (↑ = GP better)")
        ax.legend(fontsize=7)

    fig.suptitle(
        "GP location-only advantage over baselines by court region\n"
        "(positive bar = GP wins; n = training shots in region)",
        fontsize=11,
    )
    fig.tight_layout()
    out = f"{RESULTS_DIR}/spatial_stratification_advantage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_logloss_heatmap(results: pd.DataFrame):
    """
    Heatmap: log loss per (region × model) for each player side by side.
    Low = better (green); high = worse (red).
    """
    players     = [p for p, _ in PLAYERS]
    MODEL_ORDER = ["intercept_only", "binned_eb", "logistic_rbf", "gp_location", "gp_context"]
    MODEL_SHORT = ["Intercept", "Binned EB", "Logistic\nRBF", "GP\nlocation", "GP\ncontext"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, player_name in zip(axes, players):
        df_p = results[results["player"] == player_name]
        mat  = df_p.pivot_table(
            index="region", columns="model", values="log_loss", aggfunc="first"
        ).reindex(columns=MODEL_ORDER).reindex(REGIONS).values

        im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto",
                       vmin=np.nanmin(mat) * 0.99, vmax=np.nanmax(mat) * 1.01)

        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_SHORT, fontsize=8)
        ax.set_yticks(range(len(REGIONS)))
        ax.set_yticklabels([REGION_LABEL[r].replace("\n", " ") for r in REGIONS], fontsize=8)
        ax.set_title(player_name, fontsize=10)

        for i in range(len(REGIONS)):
            for j in range(len(MODEL_ORDER)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)

        plt.colorbar(im, ax=ax, label="Log loss" if ax == axes[-1] else "")

    fig.suptitle("Log loss by court region and model  (lower = better)", fontsize=11)
    fig.tight_layout()
    out = f"{RESULTS_DIR}/spatial_stratification_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
