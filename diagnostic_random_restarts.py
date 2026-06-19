"""
Diagnostic: random-restart sensitivity check for the context GP.

Runs 5 restarts for each of Stephen Curry, LeBron James, and Kevin Durant.
All three players use identical init pairs (same INIT_SEED), so any
differences in spatial_frac across players reflect the data, not the inits.

Every other setting matches gp_context.py exactly (full data, same
N_ITER/LR/N_INDUCING, same torch seed for the variational parameter init).

Run from the project root:
    python diagnostic_random_restarts.py
"""
import warnings
import numpy as np
import pandas as pd
import torch
import gpytorch

from kernels import train_gp
from gp_context import add_context_features, _to_tensor, _standardize

# ── Config ─────────────────────────────────────────────────────────────────────
PLAYERS = [
    ("Stephen Curry", "data/stephen_curry_shots.csv"),
    ("LeBron James",  "data/lebron_james_shots.csv"),
    ("Kevin Durant",  "data/kevin_durant_shots.csv"),
]
N_RESTARTS = 5
N_INDUCING = 200
N_ITER     = 500
LR         = 0.01
INIT_SEED  = 7     # seeds the Uniform draws; same for all three players
TORCH_SEED = 42    # matches gp_context.py — reset before every restart
# ───────────────────────────────────────────────────────────────────────────────


def _build_kernel(os_spatial: float, os_context: float) -> gpytorch.kernels.Kernel:
    k_sp = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2, active_dims=(0, 1))
    )
    k_ct = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=7,
                                      active_dims=(2, 3, 4, 5, 6, 7, 8))
    )
    k_sp.initialize(outputscale=os_spatial)
    k_ct.initialize(outputscale=os_context)
    return k_sp + k_ct


def _eval_elbo(model, likelihood, X_tr: torch.Tensor, y_tr: torch.Tensor) -> float:
    """Evaluate the ELBO at the converged parameters (no gradient tape)."""
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tr.size(0))
    with torch.no_grad():
        return mll(model(X_tr), y_tr).item()


def run_player(name: str, path: str,
               os_sp_inits: np.ndarray, os_ct_inits: np.ndarray) -> pd.DataFrame:
    """
    Run N_RESTARTS for one player and print the per-player results table.
    Returns a DataFrame with one row per restart.
    """
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")

    df    = pd.read_csv(path)
    df    = add_context_features(df)
    train = df[df["Split"] == "train"].copy()
    print(f"  Training shots: {len(train):,}")

    X_tr_raw, y_tr = _to_tensor(train)
    X_tr, _, _, _  = _standardize(X_tr_raw, X_tr_raw)

    rows = []
    for i, (osp_init, oct_init) in enumerate(zip(os_sp_inits, os_ct_inits)):
        print(f"\n  {'─'*55}")
        print(f"  Restart {i + 1}/{N_RESTARTS}  "
              f"σ²_sp_init={osp_init:.3f}  σ²_ct_init={oct_init:.3f}")
        print(f"  {'─'*55}")

        torch.manual_seed(TORCH_SEED)
        kernel = _build_kernel(float(osp_init), float(oct_init))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, likelihood = train_gp(
                X_tr, y_tr, kernel,
                learn_inducing=True,
                n_iter=N_ITER, lr=LR, n_inducing=N_INDUCING,
                print_every=100, log_fn=None,
            )

        k_sp       = model.covar_module.kernels[0]
        k_ct       = model.covar_module.kernels[1]
        os_sp_conv = k_sp.outputscale.item()
        os_ct_conv = k_ct.outputscale.item()
        sfrac      = os_sp_conv / (os_sp_conv + os_ct_conv)
        elbo       = _eval_elbo(model, likelihood, X_tr, y_tr)

        print(f"  Converged:  σ²_sp={os_sp_conv:.4f}  σ²_ct={os_ct_conv:.4f}  "
              f"spatial_frac={sfrac:.4f}  ELBO={elbo:.4f}")

        rows.append({
            "restart":      i + 1,
            "os_sp_init":   round(float(osp_init), 4),
            "os_ct_init":   round(float(oct_init), 4),
            "os_sp_conv":   round(os_sp_conv, 4),
            "os_ct_conv":   round(os_ct_conv, 4),
            "spatial_frac": round(sfrac, 4),
            "ELBO":         round(elbo, 4),
        })

    results = pd.DataFrame(rows)
    best    = results.loc[results["ELBO"].idxmax()]

    print(f"\n  Results — {name}")
    print(results.to_string(index=False))
    print(f"\n  Best restart: #{int(best['restart'])}  "
          f"ELBO={best['ELBO']:.4f}  "
          f"spatial_frac={best['spatial_frac']:.4f}  "
          f"(σ²_sp_init={best['os_sp_init']:.4f}, "
          f"σ²_ct_init={best['os_ct_init']:.4f})")

    return results


def main():
    # Generate init pairs once — reused identically for every player
    rng         = np.random.default_rng(INIT_SEED)
    os_sp_inits = rng.uniform(0.1, 3.0, size=N_RESTARTS)
    os_ct_inits = rng.uniform(0.1, 3.0, size=N_RESTARTS)

    print("Init pairs (same for all players):")
    for i, (osp, oct_) in enumerate(zip(os_sp_inits, os_ct_inits)):
        print(f"  Restart {i + 1}: σ²_sp_init={osp:.3f}  σ²_ct_init={oct_:.3f}")

    all_results  = {}
    summary_rows = []

    for name, path in PLAYERS:
        player_df = run_player(name, path, os_sp_inits, os_ct_inits)
        all_results[name] = player_df

        best = player_df.loc[player_df["ELBO"].idxmax()]
        summary_rows.append({
            "player":            name,
            "best_restart":      int(best["restart"]),
            "best_ELBO":         best["ELBO"],
            "best_spatial_frac": best["spatial_frac"],
            "os_sp_init":        best["os_sp_init"],
            "os_ct_init":        best["os_ct_init"],
            "os_sp_conv":        best["os_sp_conv"],
            "os_ct_conv":        best["os_ct_conv"],
        })

    # ── Cross-player summary ────────────────────────────────────────────────────
    summary = pd.DataFrame(summary_rows)

    print(f"\n{'='*65}")
    print("  Cross-player summary  (best-ELBO restart per player)")
    print(f"{'='*65}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
