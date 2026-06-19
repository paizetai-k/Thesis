"""
Baselines for NBA shot make/miss classification.
  4a — Intercept-only (global FG% as constant prediction)
  4b — Binned spatial estimator with empirical Bayes shrinkage (Beta-Binomial)

Evaluation:
  - Log-loss, Brier score, Accuracy
  - Skill scores (relative improvement over intercept-only)
  - ROC-AUC
  - Murphy decomposition of Brier (Reliability, Resolution, Uncertainty)
  - Histogram of predicted probabilities
  - Calibration curves

All models are trained and evaluated per player.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import os

# ── dataset selector ──────────────────────────────────────────────────────────
DATASET          = "full"   # "full" (7 seasons) or "broad" (30-player study)
SAVE_PREDICTIONS = True    # True → write per-sample predictions for Wilcoxon test
# ─────────────────────────────────────────────────────────────────────────────

PLAYERS = [
    ("Stephen Curry", "data/stephen_curry_shots.csv"),
    ("LeBron James",  "data/lebron_james_shots.csv"),
    ("Kevin Durant",  "data/kevin_durant_shots.csv"),
]

RESULTS_DIR = f"results/{DATASET}"
os.makedirs(RESULTS_DIR, exist_ok=True)

X_MIN, X_MAX = -250, 250
Y_MIN, Y_MAX = -50,  420
N_HEX = 25   # hex bins across the court width


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load(path):
    df = pd.read_csv(path)
    train = df[df["Split"] == "train"].copy()
    test  = df[df["Split"] == "test"].copy()
    return train, test


def murphy_decomposition(y_true, y_prob, n_bins=10):
    """
    Murphy (1973) decomposition:  Brier = Reliability - Resolution + Uncertainty

    Reliability  (lower is better): how far predicted probabilities deviate
                 from actual frequencies — calibration error.
    Resolution   (higher is better): how much predictions vary from the
                 climatological mean — discrimination power.
    Uncertainty  (fixed): inherent difficulty; = p_bar*(1-p_bar).
    """
    y_true = np.array(y_true)
    y_prob = np.clip(np.array(y_prob), 1e-9, 1 - 1e-9)
    n = len(y_true)
    p_bar = y_true.mean()

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins[1:-1])  # 0 … n_bins-1

    reliability = 0.0
    resolution  = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        n_k  = mask.sum()
        p_k  = y_prob[mask].mean()   # mean predicted prob in bin
        o_k  = y_true[mask].mean()   # observed frequency in bin
        reliability += n_k * (p_k - o_k) ** 2
        resolution  += n_k * (o_k - p_bar) ** 2

    reliability /= n
    resolution  /= n
    uncertainty  = p_bar * (1 - p_bar)

    return {
        "reliability": round(reliability, 6),
        "resolution":  round(resolution,  6),
        "uncertainty": round(uncertainty, 6),
    }


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc_k  = y_true[mask].mean()
        conf_k = y_prob[mask].mean()
        ece   += mask.sum() * abs(acc_k - conf_k)
    return round(ece / n, 6)


def full_metrics(y_true, y_prob, name, ref_logloss=None, ref_brier=None, ref_acc=None):
    y_true = np.array(y_true)
    y_prob = np.clip(np.array(y_prob), 1e-6, 1 - 1e-6)

    ll   = log_loss(y_true, y_prob)
    bs   = brier_score_loss(y_true, y_prob)
    acc  = ((y_prob >= 0.5) == y_true).mean()
    auc  = roc_auc_score(y_true, y_prob)
    ece  = expected_calibration_error(y_true, y_prob)
    mur  = murphy_decomposition(y_true, y_prob)

    # Skill scores (% improvement over intercept-only reference)
    ll_skill  = (ref_logloss - ll) / ref_logloss * 100 if ref_logloss else None
    bs_skill  = (ref_brier   - bs) / ref_brier   * 100 if ref_brier   else None
    acc_skill = (acc - ref_acc) / ref_acc         * 100 if ref_acc     else None

    return {
        "model":        name,
        "log_loss":     round(ll,  4),
        "brier":        round(bs,  4),
        "accuracy":     round(acc, 4),
        "roc_auc":      round(auc, 4),
        "ece":          ece,
        "ll_skill_%":   round(ll_skill,  2) if ll_skill  is not None else "-",
        "bs_skill_%":   round(bs_skill,  2) if bs_skill  is not None else "-",
        "acc_skill_%":  round(acc_skill, 2) if acc_skill is not None else "-",
        "reliability":  mur["reliability"],
        "resolution":   mur["resolution"],
        "uncertainty":  mur["uncertainty"],
    }


# ---------------------------------------------------------------------------
# 4a — intercept-only
# ---------------------------------------------------------------------------

def intercept_only(train, test):
    p = train["Shot Made Flag"].mean()
    y_prob = np.full(len(test), p)
    row = full_metrics(test["Shot Made Flag"], y_prob, "intercept_only")
    row["train_log_loss"] = round(log_loss(train["Shot Made Flag"], np.full(len(train), p)), 4)
    return row, y_prob


# ---------------------------------------------------------------------------
# 4b — binned spatial + empirical Bayes shrinkage
# ---------------------------------------------------------------------------

def _make_hex_grid(n_across=N_HEX):
    """Return (centers, circumradius) for a pointy-top hex grid covering the court."""
    r  = (X_MAX - X_MIN) / (n_across * np.sqrt(3))   # circumradius
    dx = np.sqrt(3) * r                                # col spacing
    dy = 1.5 * r                                       # row spacing

    centers = []
    row = 0
    y = Y_MIN - dy
    while y <= Y_MAX + dy:
        x_off = dx / 2 if row % 2 == 1 else 0.0
        x = X_MIN - dx + x_off
        while x <= X_MAX + dx:
            centers.append([x, y])
            x += dx
        y += dy
        row += 1
    return np.array(centers, dtype=np.float32), r


def _assign_hex(points, tree):
    """Return bin index (nearest hex center) for each point."""
    _, idx = tree.query(points)
    return idx


def _fit_beta_prior(k_arr, n_arr):
    from scipy.special import gammaln, betaln

    mask = n_arr > 0
    k, n = k_arr[mask], n_arr[mask]

    def neg_ll(params):
        a, b = np.exp(params)
        ll = (
            betaln(k + a, n - k + b)
            - betaln(a, b)
            + gammaln(n + 1)
            - gammaln(k + 1)
            - gammaln(n - k + 1)
        )
        return -ll.sum()

    mu    = k.sum() / n.sum()
    kappa = 20.0
    x0    = np.log([mu * kappa, (1 - mu) * kappa])
    res   = minimize(neg_ll, x0, method="Nelder-Mead",
                     options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5})

    if res.success or res.fun < neg_ll(x0):
        alpha, beta = np.exp(res.x)
    else:
        p_hat = k / n
        mu_p  = p_hat.mean()
        var_p = max(p_hat.var(), 1e-9)
        kap   = mu_p * (1 - mu_p) / var_p - 1
        alpha = mu_p * kap
        beta  = (1 - mu_p) * kap

    return max(alpha, 0.01), max(beta, 0.01)


def binned_eb(train, test, ref_logloss, ref_brier, ref_acc):
    x_tr    = train["X Location"].values
    y_tr    = train["Y Location"].values
    made_tr = train["Shot Made Flag"].values

    centers, hex_r = _make_hex_grid()
    tree = cKDTree(centers)

    bin_tr = _assign_hex(np.column_stack([x_tr, y_tr]), tree)
    n_bins = len(centers)
    k_arr  = np.zeros(n_bins)
    n_arr  = np.zeros(n_bins)
    np.add.at(k_arr, bin_tr, made_tr)
    np.add.at(n_arr, bin_tr, 1)

    alpha, beta = _fit_beta_prior(k_arr, n_arr)
    prior_mean  = alpha / (alpha + beta)
    print(f"    EB prior: α={alpha:.2f}  β={beta:.2f}  prior_mean={prior_mean:.3f}")

    p_arr = (k_arr + alpha) / (n_arr + alpha + beta)

    x_te, y_te = test["X Location"].values, test["Y Location"].values
    bin_te = _assign_hex(np.column_stack([x_te, y_te]), tree)
    y_prob = p_arr[bin_te]

    row = full_metrics(
        test["Shot Made Flag"], y_prob, "binned_EB",
        ref_logloss=ref_logloss, ref_brier=ref_brier, ref_acc=ref_acc,
    )
    y_prob_tr = np.clip(p_arr[bin_tr], 1e-6, 1 - 1e-6)
    row["train_log_loss"] = round(log_loss(made_tr, y_prob_tr), 4)
    return row, y_prob, centers, p_arr, hex_r, alpha, beta


# ---------------------------------------------------------------------------
# 4c — logistic regression with RBF features
# ---------------------------------------------------------------------------

def _rbf_features(X, centres, sigma):
    """Compute RBF activations: shape (n_samples, n_centres)."""
    # X: (n, 2), centres: (m, 2)
    diff = X[:, None, :] - centres[None, :, :]   # (n, m, 2)
    sq   = (diff ** 2).sum(axis=2)                # (n, m)
    return np.exp(-sq / (2 * sigma ** 2))


def logistic_rbf(train, test, ref_logloss, ref_brier, ref_acc, n_centres=100):
    from scipy.cluster.vq import kmeans2
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_tr = train[["X Location", "Y Location"]].values
    y_tr = train["Shot Made Flag"].values
    X_te = test[["X Location", "Y Location"]].values

    # Place centres via k-means on training locations (scipy avoids threadpoolctl issues)
    rng  = np.random.default_rng(42)
    init = X_tr[rng.choice(len(X_tr), n_centres, replace=False)]
    centres, _ = kmeans2(X_tr.astype(float), init, iter=20, minit="matrix")

    # σ = median nearest-neighbour distance between centres
    from scipy.spatial.distance import cdist
    D    = cdist(centres, centres)
    np.fill_diagonal(D, np.inf)
    sigma = np.median(D.min(axis=1))
    print(f"    RBF: {n_centres} centres, σ={sigma:.1f}")

    # Build feature matrices
    Phi_tr = _rbf_features(X_tr, centres, sigma)
    Phi_te = _rbf_features(X_te, centres, sigma)

    # Scale features (helps logistic regression converge)
    scaler = StandardScaler()
    Phi_tr = scaler.fit_transform(Phi_tr)
    Phi_te = scaler.transform(Phi_te)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(Phi_tr, y_tr)
    y_prob = clf.predict_proba(Phi_te)[:, 1]

    row = full_metrics(
        test["Shot Made Flag"], y_prob, "logistic_RBF",
        ref_logloss=ref_logloss, ref_brier=ref_brier, ref_acc=ref_acc,
    )
    y_prob_tr = clf.predict_proba(Phi_tr)[:, 1]
    row["train_log_loss"] = round(log_loss(y_tr, y_prob_tr), 4)
    return row, y_prob, centres, sigma, clf, scaler


def plot_rbf_surface(player_name, centres, sigma, clf, scaler, slug):
    """Smooth predicted FG% surface from the logistic RBF model."""
    xs = np.linspace(X_MIN, X_MAX, 200)
    ys = np.linspace(Y_MIN, Y_MAX, 200)
    xx, yy = np.meshgrid(xs, ys)
    grid   = np.column_stack([xx.ravel(), yy.ravel()])

    Phi_grid = _rbf_features(grid, centres, sigma)
    Phi_grid = scaler.transform(Phi_grid)
    p_grid   = clf.predict_proba(Phi_grid)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        p_grid, origin="lower",
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        vmin=0.2, vmax=0.8, cmap="RdYlGn", aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Predicted FG%")
    ax.set_title(f"{player_name}\nLogistic RBF surface")
    ax.set_xlabel("X Location"); ax.set_ylabel("Y Location")

    out = f"{RESULTS_DIR}/logistic_rbf_surface_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out}")


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_prob_histogram(player_name, probs_dict, slug):
    """Distribution of predicted probabilities per model."""
    models = list(probs_dict.items())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, (mname, y_prob) in zip(axes, models):
        ax.hist(y_prob, bins=40, range=(0, 1), color="steelblue",
                edgecolor="white", linewidth=0.4)
        ax.axvline(np.mean(y_prob), color="red", linestyle="--",
                   linewidth=1.2, label=f"mean={np.mean(y_prob):.3f}")
        ax.set_title(mname, fontsize=10)
        ax.set_xlabel("Predicted probability")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle(f"{player_name} — predicted probability distributions",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = f"{RESULTS_DIR}/prob_hist_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out}")


def plot_calibration(player_name, probs_dict, y_true, slug):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")

    for mname, y_prob in probs_dict.items():
        y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="uniform"
        )
        ax.plot(prob_pred, prob_true, marker="o", ms=4, label=mname)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{player_name} — calibration curve")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    out = f"{RESULTS_DIR}/calibration_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out}")


def _draw_court(ax):
    lw, col = 1.5, "black"
    ax.add_patch(plt.Circle((0, 0), 7.5, color=col, fill=False, linewidth=lw))
    ax.plot([-30, 30], [-7.5, -7.5], color=col, linewidth=lw)
    ax.add_patch(mpatches.Rectangle((-80, -47.5), 160, 190,
                                    fill=False, color=col, linewidth=lw))
    ax.add_patch(mpatches.Rectangle((-60, -47.5), 120, 190,
                                    fill=False, color=col, linewidth=lw * 0.6,
                                    linestyle="--"))
    ax.add_patch(mpatches.Arc((0, 142.5), 120, 120,
                               theta1=0, theta2=180, color=col, linewidth=lw))
    ax.add_patch(mpatches.Arc((0, 0), 80, 80,
                               theta1=0, theta2=180, color=col, linewidth=lw))
    ax.plot([-220, -220], [-47.5, 92.5], color=col, linewidth=lw)
    ax.plot([ 220,  220], [-47.5, 92.5], color=col, linewidth=lw)
    ax.add_patch(mpatches.Arc((0, 0), 475, 475,
                               theta1=22, theta2=158, color=col, linewidth=lw))
    ax.plot([-250, 250], [422.5, 422.5], color=col, linewidth=lw)
    ax.plot([-250, 250], [-47.5, -47.5], color=col, linewidth=lw)
    ax.plot([-250, -250], [-47.5, 422.5], color=col, linewidth=lw)
    ax.plot([ 250,  250], [-47.5, 422.5], color=col, linewidth=lw)


def plot_fg_surface(player_name, centers, p_arr, hex_r, alpha, beta, slug):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(6, 6))

    cmap  = plt.get_cmap("RdYlGn")
    norm  = mcolors.Normalize(vmin=0.2, vmax=0.8)
    hexes = [mpatches.RegularPolygon(
                 (cx, cy), numVertices=6, radius=hex_r * 1.01,
                 orientation=0)          # pointy-top (flat side vertical)
             for cx, cy in centers]
    col = PatchCollection(hexes, cmap=cmap, norm=norm, match_original=False)
    col.set_array(p_arr)
    ax.add_collection(col)
    plt.colorbar(col, ax=ax, label="Posterior FG%")

    _draw_court(ax)
    ax.set_title(f"{player_name}\nBinned EB surface (hex)  (α={alpha:.1f}, β={beta:.1f})")
    ax.set_xlim(-260, 260); ax.set_ylim(-55, 430)
    ax.set_aspect("equal"); ax.axis("off")

    out = f"{RESULTS_DIR}/binned_eb_surface_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out}")


def plot_murphy(player_name, rows, slug):
    """Bar chart of Murphy decomposition components per model."""
    models = [r["model"] for r in rows]
    rel    = [r["reliability"] for r in rows]
    res    = [r["resolution"]  for r in rows]
    unc    = [r["uncertainty"] for r in rows]

    x = np.arange(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, rel, w, label="Reliability (↓ better)", color="tomato")
    ax.bar(x,     res, w, label="Resolution  (↑ better)", color="steelblue")
    ax.bar(x + w, unc, w, label="Uncertainty (fixed)",    color="lightgrey")

    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Brier component")
    ax.set_title(f"{player_name} — Murphy decomposition")
    ax.legend(fontsize=8)
    plt.tight_layout()

    out = f"{RESULTS_DIR}/murphy_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

DISPLAY_COLS = [
    "model", "log_loss", "brier", "accuracy", "roc_auc", "ece",
    "ll_skill_%", "bs_skill_%", "acc_skill_%",
    "reliability", "resolution", "uncertainty",
]

def main():
    all_results = []

    for name, path in PLAYERS:
        slug = name.lower().replace(" ", "_")
        print(f"\n{'='*60}\n  {name}\n{'='*60}")

        train, test = load(path)
        y_true      = test["Shot Made Flag"].values
        probs_dict  = {}
        player_rows = []

        # 4a — intercept-only (reference for skill scores)
        row_4a, prob_4a = intercept_only(train, test)
        row_4a["player"] = name
        all_results.append(row_4a)
        player_rows.append(row_4a)
        probs_dict["intercept_only"] = prob_4a

        ref_ll  = row_4a["log_loss"]
        ref_bs  = row_4a["brier"]
        ref_acc = row_4a["accuracy"]

        # 4b — binned EB
        row_4b, prob_4b, centers, p_arr, hex_r, alpha, beta = binned_eb(
            train, test, ref_ll, ref_bs, ref_acc
        )
        row_4b["player"] = name
        all_results.append(row_4b)
        player_rows.append(row_4b)
        probs_dict["binned_EB"] = prob_4b

        # 4c — logistic RBF
        row_4c, prob_4c, centres, sigma, clf, scaler = logistic_rbf(
            train, test, ref_ll, ref_bs, ref_acc
        )
        row_4c["player"] = name
        all_results.append(row_4c)
        player_rows.append(row_4c)
        probs_dict["logistic_RBF"] = prob_4c

        # print per-player table
        pdf = pd.DataFrame(player_rows)[DISPLAY_COLS]
        print(pdf.to_string(index=False))

        # plots
        plot_prob_histogram(name, probs_dict, slug)
        plot_calibration(name, probs_dict, y_true, slug)
        plot_fg_surface(name, centers, p_arr, hex_r, alpha, beta, slug)
        plot_rbf_surface(name, centres, sigma, clf, scaler, slug)
        plot_murphy(name, player_rows, slug)

        if SAVE_PREDICTIONS:
            frames = [
                pd.DataFrame({"model": m, "y_true": y_true, "y_prob": p})
                for m, p in [
                    ("intercept_only", prob_4a),
                    ("binned_EB",      prob_4b),
                    ("logistic_RBF",   prob_4c),
                ]
            ]
            pred_path = f"{RESULTS_DIR}/predictions_{slug}_baselines.csv"
            pd.concat(frames, ignore_index=True).to_csv(pred_path, index=False)
            print(f"    Saved predictions → {pred_path}")

    # full summary
    results_df = pd.DataFrame(all_results)[["player", "train_log_loss"] + DISPLAY_COLS]
    print(f"\n{'='*60}\nFull results\n{'='*60}")
    print(results_df.to_string(index=False))
    out = f"{RESULTS_DIR}/baselines_results.csv"
    results_df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
