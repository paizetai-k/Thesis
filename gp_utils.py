"""
Shared utilities for GP shot-prediction experiments.
Data loading, metrics, and plotting.  No GPyTorch model classes here.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import gpytorch
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------------------------
# Court dimensions
# ---------------------------------------------------------------------------
X_MIN, X_MAX = -250, 250
Y_MIN, Y_MAX = -50,  420

DISPLAY_COLS = [
    "model", "log_loss", "brier", "accuracy", "roc_auc", "ece",
    "ll_skill_%", "bs_skill_%", "acc_skill_%",
    "reliability", "resolution", "uncertainty",
]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(path):
    df    = pd.read_csv(path)
    train = df[df["Split"] == "train"].copy()
    test  = df[df["Split"] == "test"].copy()
    return train, test


def to_tensor(df, use_is3pt=False):
    cols = ["X Location", "Y Location"]
    if use_is3pt:
        cols.append("is_3pt")
    X = torch.tensor(df[cols].values, dtype=torch.float32)
    y = torch.tensor(df["Shot Made Flag"].values, dtype=torch.float32)
    return X, y


def standardize(X_tr, X_te):
    """Standardise the first two (location) columns; leave is_3pt unchanged."""
    mu  = X_tr[:, :2].mean(0)
    std = X_tr[:, :2].std(0).clamp(min=1e-6)
    X_tr_n, X_te_n = X_tr.clone(), X_te.clone()
    X_tr_n[:, :2] = (X_tr[:, :2] - mu) / std
    X_te_n[:, :2] = (X_te[:, :2] - mu) / std
    return X_tr_n, X_te_n, mu, std


# ---------------------------------------------------------------------------
# Prediction (thin wrapper — works with any GPyTorch model + likelihood)
# ---------------------------------------------------------------------------

def predict_probs(model, likelihood, X):
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X))
    return pred.mean.numpy(), pred.variance.numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return round(ece / n, 6)


def murphy_decomposition(y_true, y_prob, n_bins=10):
    n, p_bar = len(y_true), y_true.mean()
    bins    = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins[1:-1])
    rel = res = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        n_k = mask.sum()
        rel += n_k * (y_prob[mask].mean() - y_true[mask].mean()) ** 2
        res += n_k * (y_true[mask].mean() - p_bar) ** 2
    return {
        "reliability": round(rel / n, 6),
        "resolution":  round(res / n, 6),
        "uncertainty": round(p_bar * (1 - p_bar), 6),
    }


def full_metrics(y_true, y_prob, name, ref_logloss=None, ref_brier=None, ref_acc=None):
    y_true = np.array(y_true)
    y_prob = np.clip(np.array(y_prob), 1e-6, 1 - 1e-6)
    ll  = log_loss(y_true, y_prob)
    bs  = brier_score_loss(y_true, y_prob)
    acc = ((y_prob >= 0.5) == y_true).mean()
    auc = roc_auc_score(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    mur = murphy_decomposition(y_true, y_prob)
    ll_skill  = (ref_logloss - ll) / ref_logloss * 100 if ref_logloss else None
    bs_skill  = (ref_brier   - bs) / ref_brier   * 100 if ref_brier   else None
    acc_skill = (acc - ref_acc)    / ref_acc      * 100 if ref_acc     else None
    return {
        "model":       name,
        "log_loss":    round(ll,  4),
        "brier":       round(bs,  4),
        "accuracy":    round(acc, 4),
        "roc_auc":     round(auc, 4),
        "ece":         ece,
        "ll_skill_%":  round(ll_skill,  2) if ll_skill  is not None else "-",
        "bs_skill_%":  round(bs_skill,  2) if bs_skill  is not None else "-",
        "acc_skill_%": round(acc_skill, 2) if acc_skill is not None else "-",
        **mur,
    }


def intercept_ref(train_y_true, test_y_true):
    """Return (ref_logloss, ref_brier, ref_acc) from the intercept-only model."""
    ref_p    = np.mean(train_y_true)
    ref_prob = np.full(len(test_y_true), ref_p)
    return (
        log_loss(test_y_true, ref_prob),
        brier_score_loss(test_y_true, ref_prob),
        ((ref_prob >= 0.5) == test_y_true).mean(),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def draw_court(ax):
    lw, col = 1.5, "black"
    ax.add_patch(plt.Circle((0, 0), 7.5, color=col, fill=False, linewidth=lw))
    ax.plot([-30, 30], [-7.5, -7.5], color=col, linewidth=lw)
    ax.add_patch(patches.Rectangle((-80, -47.5), 160, 190,
                                   fill=False, color=col, linewidth=lw))
    ax.add_patch(patches.Rectangle((-60, -47.5), 120, 190,
                                   fill=False, color=col, linewidth=lw * 0.6,
                                   linestyle="--"))
    ax.add_patch(patches.Arc((0, 142.5), 120, 120,
                              theta1=0, theta2=180, color=col, linewidth=lw))
    ax.add_patch(patches.Arc((0, 0), 80, 80,
                              theta1=0, theta2=180, color=col, linewidth=lw))
    ax.plot([-220, -220], [-47.5, 92.5], color=col, linewidth=lw)
    ax.plot([ 220,  220], [-47.5, 92.5], color=col, linewidth=lw)
    ax.add_patch(patches.Arc((0, 0), 475, 475,
                              theta1=22, theta2=158, color=col, linewidth=lw))
    ax.plot([-250, 250], [422.5, 422.5], color=col, linewidth=lw)
    ax.plot([-250, 250], [-47.5, -47.5], color=col, linewidth=lw)
    ax.plot([-250, -250], [-47.5, 422.5], color=col, linewidth=lw)
    ax.plot([ 250,  250], [-47.5, 422.5], color=col, linewidth=lw)


def _grid_is3pt(xx_flat, yy_flat):
    """Approximate is_3pt for a visualisation grid using court geometry."""
    corner = np.abs(xx_flat) >= 220
    arc    = np.sqrt(xx_flat**2 + yy_flat**2) >= 237.5
    return (corner | arc).astype(np.float32)


def court_grid(model, likelihood, mu, std, use_is3pt=False, resolution=150):
    """Build a court-sized prediction grid and return (xx, yy, p_grid, std_grid)."""
    xs = np.linspace(X_MIN, X_MAX, resolution)
    ys = np.linspace(Y_MIN, Y_MAX, resolution)
    xx, yy  = np.meshgrid(xs, ys)
    xy_norm = (np.column_stack([xx.ravel(), yy.ravel()])
               - mu.numpy()) / std.numpy()

    if use_is3pt:
        is3pt = _grid_is3pt(xx.ravel(), yy.ravel()).reshape(-1, 1)
        grid  = torch.tensor(np.hstack([xy_norm, is3pt]), dtype=torch.float32)
    else:
        grid = torch.tensor(xy_norm, dtype=torch.float32)

    probs, variances = predict_probs(model, likelihood, grid)
    return xx, yy, probs.reshape(xx.shape), np.sqrt(variances).reshape(xx.shape)


def plot_surfaces(player_name, model, likelihood, mu, std,
                  tag, slug, results_dir, use_is3pt=False):
    """Save FG% surface and uncertainty surface for one player/model."""
    xx, yy, p_grid, std_grid = court_grid(model, likelihood, mu, std, use_is3pt)

    for data, cmap, label, suffix in [
        (p_grid,   "RdYlGn", "Predicted FG%", "surface"),
        (std_grid, "Blues",  "Posterior std",  "uncertainty"),
    ]:
        vkw = dict(vmin=0.2, vmax=0.8) if cmap == "RdYlGn" else {}
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(data, origin="lower",
                       extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                       cmap=cmap, aspect="auto", **vkw)
        plt.colorbar(im, ax=ax, label=label)
        draw_court(ax)
        ax.set_title(f"{player_name}\n{tag}")
        ax.set_xlim(-260, 260); ax.set_ylim(-55, 430)
        ax.set_aspect("equal"); ax.axis("off")
        fname = f"{results_dir}/{tag}_{suffix}_{slug}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"    Saved {fname}")


def plot_calibration(player_name, y_true, probs, tag, slug, results_dir):
    """Save a calibration curve plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    frac_pos, mean_pred = calibration_curve(
        y_true, np.clip(probs, 1e-6, 1 - 1e-6), n_bins=10, strategy="uniform"
    )
    ax.plot(mean_pred, frac_pos, marker="o", ms=4, label=tag)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{player_name} — {tag} calibration")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fname = f"{results_dir}/{tag}_calibration_{slug}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    Saved {fname}")
