"""
H3 Context-Augmented GP — additive Matérn-5/2 kernel.

Features (9 total):
  Continuous (standardised): X Location, Y Location, opp_win_pct, time_remaining
  Binary     (unchanged)   : is_3pt, home_away
  Dummies    (unchanged)   : act_dunk, act_layup, act_hook_float  (jump = reference)

Kernel: k_spatial(x, y) + k_context(opp_win_pct, …, act_hook_float)
  k_spatial  = ScaleKernel( MaternKernel(nu=2.5, ard_num_dims=2) )
  k_context  = ScaleKernel( MaternKernel(nu=2.5, ard_num_dims=7) )

The additive structure exposes output-scales (σ²_spatial, σ²_context).
Note: the spatial fraction σ²_sp/(σ²_sp+σ²_ct) is sensitive to initialisation;
see diagnostic_random_restarts.py. Do not treat the ~58% default-init value as
a robust empirical finding — best-ELBO restarts converge near-pure-spatial (~0.97).

Comparison: metrics printed alongside gp_location results for direct H3 evaluation.
"""
import os
import warnings
import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, brier_score_loss
from gp_utils import (predict_probs, full_metrics, plot_calibration,
                      draw_court, DISPLAY_COLS, X_MIN, X_MAX, Y_MIN, Y_MAX)
from kernels import train_gp, init_inducing

torch.manual_seed(42)

# ── Dataset selector ──────────────────────────────────────────────────────────
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

N_INDUCING = 200
N_ITER     = 500
LR         = 0.01

FEAT_NAMES = [
    "x", "y", "opp_win_pct", "time_remaining",
    "is_3pt", "home_away", "act_dunk", "act_layup", "act_hook_float",
]
N_CONT = 4   # first 4 columns are continuous → standardised
N_FEAT = 9   # total feature dimensions


# ── Win% lookup (2012-13 through 2018-19) ────────────────────────────────────

WIN_PCT = {
    # 2012-13
    (2012,'ATL'):44/82,(2012,'BKN'):49/82,(2012,'BOS'):41/82,
    (2012,'CHA'):21/82,(2012,'CHI'):45/82,(2012,'CLE'):24/82,
    (2012,'DAL'):41/82,(2012,'DEN'):57/82,(2012,'DET'):29/82,
    (2012,'GSW'):47/82,(2012,'HOU'):45/82,(2012,'IND'):49/82,
    (2012,'LAC'):56/82,(2012,'LAL'):45/82,(2012,'MEM'):56/82,
    (2012,'MIA'):66/82,(2012,'MIL'):38/82,(2012,'MIN'):31/82,
    (2012,'NOH'):27/82,(2012,'NYK'):54/82,(2012,'OKC'):60/82,
    (2012,'ORL'):20/82,(2012,'PHI'):34/82,(2012,'PHX'):25/82,
    (2012,'POR'):33/82,(2012,'SAC'):28/82,(2012,'SAS'):58/82,
    (2012,'TOR'):34/82,(2012,'UTA'):43/82,(2012,'WAS'):29/82,
    # 2013-14
    (2013,'ATL'):38/82,(2013,'BKN'):44/82,(2013,'BOS'):25/82,
    (2013,'CHA'):43/82,(2013,'CHI'):48/82,(2013,'CLE'):33/82,
    (2013,'DAL'):49/82,(2013,'DEN'):36/82,(2013,'DET'):29/82,
    (2013,'GSW'):51/82,(2013,'HOU'):54/82,(2013,'IND'):56/82,
    (2013,'LAC'):57/82,(2013,'LAL'):27/82,(2013,'MEM'):50/82,
    (2013,'MIA'):54/82,(2013,'MIL'):15/82,(2013,'MIN'):40/82,
    (2013,'NOP'):34/82,(2013,'NYK'):37/82,(2013,'OKC'):59/82,
    (2013,'ORL'):23/82,(2013,'PHI'):19/82,(2013,'PHX'):48/82,
    (2013,'POR'):54/82,(2013,'SAC'):28/82,(2013,'SAS'):62/82,
    (2013,'TOR'):48/82,(2013,'UTA'):25/82,(2013,'WAS'):44/82,
    # 2014-15
    (2014,'ATL'):60/82,(2014,'BKN'):38/82,(2014,'BOS'):40/82,
    (2014,'CHA'):33/82,(2014,'CHI'):50/82,(2014,'CLE'):53/82,
    (2014,'DAL'):50/82,(2014,'DEN'):30/82,(2014,'DET'):32/82,
    (2014,'GSW'):67/82,(2014,'HOU'):56/82,(2014,'IND'):38/82,
    (2014,'LAC'):56/82,(2014,'LAL'):21/82,(2014,'MEM'):55/82,
    (2014,'MIA'):37/82,(2014,'MIL'):41/82,(2014,'MIN'):16/82,
    (2014,'NOP'):45/82,(2014,'NYK'):17/82,(2014,'OKC'):45/82,
    (2014,'ORL'):25/82,(2014,'PHI'):18/82,(2014,'PHX'):39/82,
    (2014,'POR'):51/82,(2014,'SAC'):29/82,(2014,'SAS'):55/82,
    (2014,'TOR'):49/82,(2014,'UTA'):38/82,(2014,'WAS'):46/82,
    # 2015-16
    (2015,'ATL'):48/82,(2015,'BKN'):21/82,(2015,'BOS'):48/82,
    (2015,'CHA'):48/82,(2015,'CHI'):42/82,(2015,'CLE'):57/82,
    (2015,'DAL'):42/82,(2015,'DEN'):33/82,(2015,'DET'):44/82,
    (2015,'GSW'):73/82,(2015,'HOU'):41/82,(2015,'IND'):45/82,
    (2015,'LAC'):53/82,(2015,'LAL'):17/82,(2015,'MEM'):42/82,
    (2015,'MIA'):48/82,(2015,'MIL'):33/82,(2015,'MIN'):29/82,
    (2015,'NOP'):30/82,(2015,'NYK'):32/82,(2015,'OKC'):55/82,
    (2015,'ORL'):35/82,(2015,'PHI'):10/82,(2015,'PHX'):23/82,
    (2015,'POR'):44/82,(2015,'SAC'):33/82,(2015,'SAS'):67/82,
    (2015,'TOR'):56/82,(2015,'UTA'):40/82,(2015,'WAS'):41/82,
    # 2016-17
    (2016,'ATL'):43/82,(2016,'BKN'):20/82,(2016,'BOS'):53/82,
    (2016,'CHA'):36/82,(2016,'CHI'):41/82,(2016,'CLE'):51/82,
    (2016,'DAL'):33/82,(2016,'DEN'):40/82,(2016,'DET'):37/82,
    (2016,'GSW'):67/82,(2016,'HOU'):55/82,(2016,'IND'):42/82,
    (2016,'LAC'):51/82,(2016,'LAL'):26/82,(2016,'MEM'):43/82,
    (2016,'MIA'):41/82,(2016,'MIL'):42/82,(2016,'MIN'):31/82,
    (2016,'NOP'):34/82,(2016,'NYK'):31/82,(2016,'OKC'):47/82,
    (2016,'ORL'):29/82,(2016,'PHI'):28/82,(2016,'PHX'):24/82,
    (2016,'POR'):41/82,(2016,'SAC'):32/82,(2016,'SAS'):61/82,
    (2016,'TOR'):51/82,(2016,'UTA'):51/82,(2016,'WAS'):49/82,
    # 2017-18
    (2017,'ATL'):24/82,(2017,'BKN'):28/82,(2017,'BOS'):55/82,
    (2017,'CHA'):36/82,(2017,'CHI'):27/82,(2017,'CLE'):50/82,
    (2017,'DAL'):24/82,(2017,'DEN'):46/82,(2017,'DET'):39/82,
    (2017,'GSW'):58/82,(2017,'HOU'):65/82,(2017,'IND'):48/82,
    (2017,'LAC'):42/82,(2017,'LAL'):35/82,(2017,'MEM'):22/82,
    (2017,'MIA'):44/82,(2017,'MIL'):44/82,(2017,'MIN'):47/82,
    (2017,'NOP'):48/82,(2017,'NYK'):29/82,(2017,'OKC'):48/82,
    (2017,'ORL'):25/82,(2017,'PHI'):52/82,(2017,'PHX'):21/82,
    (2017,'POR'):49/82,(2017,'SAC'):27/82,(2017,'SAS'):47/82,
    (2017,'TOR'):59/82,(2017,'UTA'):48/82,(2017,'WAS'):43/82,
    # 2018-19
    (2018,'ATL'):29/82,(2018,'BKN'):42/82,(2018,'BOS'):49/82,
    (2018,'CHA'):39/82,(2018,'CHI'):22/82,(2018,'CLE'):19/82,
    (2018,'DAL'):33/82,(2018,'DEN'):54/82,(2018,'DET'):41/82,
    (2018,'GSW'):57/82,(2018,'HOU'):53/82,(2018,'IND'):48/82,
    (2018,'LAC'):48/82,(2018,'LAL'):37/82,(2018,'MEM'):33/82,
    (2018,'MIA'):39/82,(2018,'MIL'):60/82,(2018,'MIN'):36/82,
    (2018,'NOP'):33/82,(2018,'NYK'):17/82,(2018,'OKC'):49/82,
    (2018,'ORL'):42/82,(2018,'PHI'):51/82,(2018,'PHX'):19/82,
    (2018,'POR'):53/82,(2018,'SAC'):39/82,(2018,'SAS'):48/82,
    (2018,'TOR'):58/82,(2018,'UTA'):50/82,(2018,'WAS'):32/82,
}

# Full team name → abbreviation (covers all 30 player-team combinations 2012-19)
TEAM_ABBREV = {
    "Atlanta Hawks":           "ATL",
    "Boston Celtics":          "BOS",
    "Brooklyn Nets":           "BKN",
    "Charlotte Bobcats":       "CHA",   # renamed to Hornets in 2014-15
    "Charlotte Hornets":       "CHA",
    "Chicago Bulls":           "CHI",
    "Cleveland Cavaliers":     "CLE",
    "Dallas Mavericks":        "DAL",
    "Denver Nuggets":          "DEN",
    "Detroit Pistons":         "DET",
    "Golden State Warriors":   "GSW",
    "Houston Rockets":         "HOU",
    "Indiana Pacers":          "IND",
    "LA Clippers":             "LAC",   # dataset uses both spellings
    "Los Angeles Clippers":    "LAC",
    "Los Angeles Lakers":      "LAL",
    "Memphis Grizzlies":       "MEM",
    "Miami Heat":              "MIA",
    "Milwaukee Bucks":         "MIL",
    "Minnesota Timberwolves":  "MIN",
    "New Orleans Hornets":     "NOH",   # became Pelicans in 2013-14
    "New Orleans Pelicans":    "NOP",
    "New York Knicks":         "NYK",
    "Oklahoma City Thunder":   "OKC",
    "Orlando Magic":           "ORL",
    "Philadelphia 76ers":      "PHI",
    "Phoenix Suns":            "PHX",
    "Portland Trail Blazers":  "POR",
    "Sacramento Kings":        "SAC",
    "San Antonio Spurs":       "SAS",
    "Toronto Raptors":         "TOR",
    "Utah Jazz":               "UTA",
    "Washington Wizards":      "WAS",
}

# ── Action type buckets ───────────────────────────────────────────────────────

DUNK = {
    'Dunk Shot','Driving Dunk Shot','Alley Oop Dunk Shot','Slam Dunk Shot',
    'Running Dunk Shot','Cutting Dunk Shot','Driving Slam Dunk Shot',
    'Putback Dunk Shot','Running Slam Dunk Shot','Reverse Dunk Shot',
    'Reverse Slam Dunk Shot','Driving Reverse Dunk Shot','Follow Up Dunk Shot',
    'Running Reverse Dunk Shot','Running Alley Oop Dunk Shot','Tip Dunk Shot',
    'Putback Slam Dunk Shot','Putback Reverse Dunk Shot',
}

LAYUP = {
    'Layup Shot','Driving Layup Shot','Driving Finger Roll Layup Shot',
    'Running Layup Shot','Reverse Layup Shot','Driving Reverse Layup Shot',
    'Cutting Layup Shot','Alley Oop Layup shot','Putback Layup Shot',
    'Running Reverse Layup Shot','Cutting Finger Roll Layup Shot',
    'Finger Roll Layup Shot','Running Finger Roll Layup Shot','Tip Layup Shot',
    'Tip Shot','Finger Roll Shot','Driving Finger Roll Shot',
    'Running Finger Roll Shot','Running Alley Oop Layup Shot',
    'Turnaround Finger Roll Shot',
}

HOOK_FLOAT = {
    'Hook Shot','Driving Hook Shot','Running Hook Shot','Jump Hook Shot',
    'Turnaround Hook Shot','Driving Bank Hook Shot','Hook Bank Shot',
    'Turnaround Bank Hook Shot','Running Bank Hook Shot','Jump Bank Hook Shot',
    'Floating Jump shot','Driving Floating Jump Shot',
    'Driving Floating Bank Jump Shot',
}


def _bucket(action: str) -> str:
    if action in DUNK:       return 'dunk'
    if action in LAYUP:      return 'layup'
    if action in HOOK_FLOAT: return 'hook_float'
    return 'jump'


def _derive_season_year(game_date_int) -> int:
    d     = int(game_date_int)
    year  = d // 10000
    month = (d % 10000) // 100
    return year if month >= 10 else year - 1


# ── Feature engineering ───────────────────────────────────────────────────────

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['_season_year']    = df['Game Date'].apply(_derive_season_year)
    df['_player_abbrev']  = df['Team Name'].map(TEAM_ABBREV)
    df['home_away']       = (df['Home Team'] == df['_player_abbrev']).astype(int)
    df['_opp_abbrev']     = df.apply(
        lambda r: r['Away Team'] if r['home_away'] == 1 else r['Home Team'], axis=1
    )
    df['opp_win_pct']     = df.apply(
        lambda r: WIN_PCT.get((r['_season_year'], r['_opp_abbrev']), 0.5), axis=1
    )
    df['time_remaining']  = (
        (4 - df['Period']).clip(lower=0) * 12
        + df['Minutes Remaining']
        + df['Seconds Remaining'] / 60
    ).clip(lower=0, upper=48)

    df['_action_cat']     = df['Action Type'].apply(_bucket)
    df['act_dunk']        = (df['_action_cat'] == 'dunk').astype(int)
    df['act_layup']       = (df['_action_cat'] == 'layup').astype(int)
    df['act_hook_float']  = (df['_action_cat'] == 'hook_float').astype(int)

    return df


CONT_COLS = ['X Location', 'Y Location', 'opp_win_pct', 'time_remaining']
BIN_COLS  = ['is_3pt', 'home_away', 'act_dunk', 'act_layup', 'act_hook_float']
ALL_COLS  = CONT_COLS + BIN_COLS   # length = N_FEAT = 9


def _to_tensor(df: pd.DataFrame):
    X = torch.tensor(df[ALL_COLS].values, dtype=torch.float32)
    y = torch.tensor(df['Shot Made Flag'].values, dtype=torch.float32)
    return X, y


def _standardize(X_tr: torch.Tensor, X_te: torch.Tensor):
    """Standardise continuous columns (0:N_CONT); leave binary columns unchanged."""
    mu  = X_tr[:, :N_CONT].mean(0)
    std = X_tr[:, :N_CONT].std(0).clamp(min=1e-6)
    X_tr_n, X_te_n = X_tr.clone(), X_te.clone()
    X_tr_n[:, :N_CONT] = (X_tr[:, :N_CONT] - mu) / std
    X_te_n[:, :N_CONT] = (X_te[:, :N_CONT] - mu) / std
    return X_tr_n, X_te_n, mu, std


SPATIAL_FEATS = FEAT_NAMES[:2]   # x, y
CONTEXT_FEATS = FEAT_NAMES[2:]   # opp_win_pct … act_hook_float  (7 dims)


def _build_additive_kernel() -> gpytorch.kernels.Kernel:
    k_spatial = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2,
                                      active_dims=(0, 1))
    )
    k_context = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=7,
                                      active_dims=(2, 3, 4, 5, 6, 7, 8))
    )
    return k_spatial + k_context


# ── Surface visualisation (context fixed at training mean) ────────────────────

def plot_context_surface(player_name: str, model, likelihood,
                         mu: torch.Tensor, std: torch.Tensor,
                         ctx_means: np.ndarray,
                         tag: str, slug: str, results_dir: str):
    """
    FG% surface and uncertainty surface with non-spatial features fixed at
    their training-set means.  ctx_means has shape (N_FEAT - 2,) covering
    [opp_win_pct, time_remaining, is_3pt, home_away, act_dunk, act_layup,
     act_hook_float] — all features after (x, y).
    """
    resolution = 150
    xs = np.linspace(X_MIN, X_MAX, resolution)
    ys = np.linspace(Y_MIN, Y_MAX, resolution)
    xx, yy = np.meshgrid(xs, ys)
    n_pts  = xx.size

    # Continuous (x, y) — standardise
    xy_raw  = np.column_stack([xx.ravel(), yy.ravel()])
    xy_norm = (xy_raw - mu[:2].numpy()) / std[:2].numpy()

    # Continuous context features — standardise with their own mu/std
    ctx_cont_raw  = np.tile(ctx_means[:2], (n_pts, 1))      # opp_win_pct, time_rem
    ctx_cont_norm = (ctx_cont_raw - mu[2:].numpy()) / std[2:].numpy()

    # Binary features — leave as-is (use means = expected proportions)
    ctx_bin = np.tile(ctx_means[2:], (n_pts, 1))            # is_3pt … act_hook_float

    grid_np = np.hstack([xy_norm, ctx_cont_norm, ctx_bin]).astype(np.float32)
    grid    = torch.tensor(grid_np)

    probs, variances = predict_probs(model, likelihood, grid)
    p_grid   = probs.reshape(xx.shape)
    std_grid = np.sqrt(variances).reshape(xx.shape)

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
        ax.set_title(f"{player_name}\n{tag}\n(context @ training mean)")
        ax.set_xlim(-260, 260); ax.set_ylim(-55, 430)
        ax.set_aspect("equal"); ax.axis("off")
        fname = f"{results_dir}/{tag}_{suffix}_{slug}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"    Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_results = []

    for player_name, path in PLAYERS:
        slug = player_name.lower().replace(" ", "_")
        print(f"\n{'='*60}\n  {player_name}\n{'='*60}")

        df    = pd.read_csv(path)
        df    = add_context_features(df)
        train = df[df["Split"] == "train"].copy()
        test  = df[df["Split"] == "test"].copy()
        y_true = test["Shot Made Flag"].values

        # Intercept-only reference
        ref_p   = train["Shot Made Flag"].mean()
        ref_ll  = log_loss(y_true, np.full(len(y_true), ref_p))
        ref_bs  = brier_score_loss(y_true, np.full(len(y_true), ref_p))
        ref_acc = ((np.full(len(y_true), ref_p) >= 0.5) == y_true).mean()

        # Tensors & standardisation
        X_tr_raw, y_tr = _to_tensor(train)
        X_te_raw, _    = _to_tensor(test)
        X_tr, X_te, mu_t, std_t = _standardize(X_tr_raw, X_te_raw)

        # Context feature means for surface visualisation (columns 2 onward)
        ctx_means = X_tr_raw[:, 2:].numpy().mean(axis=0)   # shape (7,)

        # Build additive kernel: k_spatial(x,y) + k_context(7 context dims)
        kernel = _build_additive_kernel()

        print(f"  Training on {len(train):,} shots  |  {N_FEAT} features  "
              f"|  {N_INDUCING} inducing pts")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, likelihood = train_gp(
                X_tr, y_tr, kernel,
                learn_inducing=True,
                n_iter=N_ITER, lr=LR, n_inducing=N_INDUCING,
                print_every=100, log_fn=None,
            )

        # Print additive kernel decomposition
        k_sp = model.covar_module.kernels[0]   # ScaleKernel (spatial)
        k_ct = model.covar_module.kernels[1]   # ScaleKernel (context)
        os_sp = k_sp.outputscale.item()
        os_ct = k_ct.outputscale.item()
        total = os_sp + os_ct
        ls_sp = k_sp.base_kernel.lengthscale.detach().squeeze().numpy()
        ls_ct = k_ct.base_kernel.lengthscale.detach().squeeze().numpy()
        print(f"  Variance shares:  k_spatial={os_sp:.4f} ({100*os_sp/total:.1f}%)   "
              f"k_context={os_ct:.4f} ({100*os_ct/total:.1f}%)")
        print("  Spatial length-scales:")
        for fname, lv in zip(SPATIAL_FEATS, ls_sp):
            print(f"    {fname:<18}: {lv:.4f}")
        print("  Context length-scales:")
        for fname, lv in zip(CONTEXT_FEATS, ls_ct):
            print(f"    {fname:<18}: {lv:.4f}")

        # Evaluate
        probs, _ = predict_probs(model, likelihood, X_te)
        train_probs, _ = predict_probs(model, likelihood, X_tr)
        row = full_metrics(y_true, probs, "gp_context",
                           ref_logloss=ref_ll, ref_brier=ref_bs, ref_acc=ref_acc)
        row["player"] = player_name
        row["train_log_loss"] = round(log_loss(train["Shot Made Flag"].values, train_probs), 4)
        all_results.append(row)
        print(pd.DataFrame([row])[DISPLAY_COLS].to_string(index=False))

        # Plots
        plot_context_surface(player_name, model, likelihood,
                             mu_t, std_t, ctx_means,
                             "gp_context", slug, RESULTS_DIR)
        plot_calibration(player_name, y_true, probs,
                         "gp_context", slug, RESULTS_DIR)

        if SAVE_PREDICTIONS:
            pred_path = f"{RESULTS_DIR}/predictions_{slug}_gp_context.csv"
            pd.DataFrame({"model": "gp_context", "y_true": y_true, "y_prob": probs}
                         ).to_csv(pred_path, index=False)
            print(f"    Saved predictions → {pred_path}")

    # Save results
    results_df = pd.DataFrame(all_results)[["player", "train_log_loss"] + DISPLAY_COLS]
    out = f"{RESULTS_DIR}/gp_context_results.csv"
    results_df.to_csv(out, index=False)
    print(f"\nSaved {out}")

    # Compare against location-only GP
    loc_path = f"{RESULTS_DIR}/gp_location_results.csv"
    if os.path.exists(loc_path):
        loc_df = pd.read_csv(loc_path)
        print(f"\n{'='*60}\nH3 comparison: gp_context vs gp_location\n{'='*60}")
        compare = pd.concat([loc_df, results_df]).sort_values(["player", "model"])
        print(compare[["player", "model", "log_loss", "brier", "ece",
                        "ll_skill_%", "bs_skill_%"]].to_string(index=False))


if __name__ == "__main__":
    main()
