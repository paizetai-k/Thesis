"""
GP classification — location-only model (ARD-2 Matérn-5/2 kernel).

  Kernel    : ScaleKernel(Matérn-5/2, ARD-2) on (x, y): independent ls_x, ls_y
              Focal players converge to ls_y > ls_x (ratios 1.34–1.64).
  Inference : Sparse variational GP (SVGP), Hensman et al. 2015
  Likelihood: BernoulliLikelihood (probit link)

Evaluated per player against intercept-only reference.
Metrics: log loss, Brier score, accuracy, ROC-AUC, ECE, Murphy decomposition.
"""
import os
import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.metrics import log_loss, brier_score_loss

from gp_utils import (load, to_tensor, standardize, predict_probs, full_metrics,
                       plot_surfaces, plot_calibration, DISPLAY_COLS)
from kernels import train_gp, hp_str, build_kernel

torch.manual_seed(42)

# ── dataset selector ──────────────────────────────────────────────────────────
DATASET          = "full"   # "full" (7 seasons) or "broad" (30-player study)
SAVE_PREDICTIONS = True     # True → write per-sample predictions for Wilcoxon test
# ─────────────────────────────────────────────────────────────────────────────

N_INDUCING = 200
N_ITER     = 500
LR         = 0.01

PLAYERS = [
    ("Stephen Curry", "data/stephen_curry_shots.csv"),
    ("LeBron James",  "data/lebron_james_shots.csv"),
    ("Kevin Durant",  "data/kevin_durant_shots.csv"),
]

RESULTS_DIR = f"results/{DATASET}"
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    all_results = []

    for name, path in PLAYERS:
        slug = name.lower().replace(" ", "_")
        print(f"\n{'='*60}\n  {name}\n{'='*60}")

        train, test     = load(path)
        X_tr_raw, y_tr  = to_tensor(train)
        X_te_raw, _     = to_tensor(test)
        X_tr, X_te, mu, std = standardize(X_tr_raw, X_te_raw)
        y_true = test["Shot Made Flag"].values

        ref_logloss = log_loss(y_true, np.full(len(y_true), train["Shot Made Flag"].mean()))
        ref_brier   = brier_score_loss(y_true, np.full(len(y_true), train["Shot Made Flag"].mean()))
        ref_acc     = ((np.full(len(y_true), train["Shot Made Flag"].mean()) >= 0.5) == y_true).mean()

        kernel  = build_kernel("standard")
        log_fn  = lambda m: hp_str(m, "standard")

        print(f"  Training on {len(train):,} points...")
        model, likelihood = train_gp(X_tr, y_tr, kernel, learn_inducing=True,
                                      n_iter=N_ITER, lr=LR, n_inducing=N_INDUCING,
                                      log_fn=log_fn)

        ls = model.covar_module.base_kernel.lengthscale[0]
        print(f"  Converged length-scales:  ls_x={ls[0].item():.4f}  ls_y={ls[1].item():.4f}")

        probs, _ = predict_probs(model, likelihood, X_te)
        train_probs, _ = predict_probs(model, likelihood, X_tr)
        row = full_metrics(y_true, probs, "gp_location",
                           ref_logloss=ref_logloss, ref_brier=ref_brier, ref_acc=ref_acc)
        row["player"] = name
        row["train_log_loss"] = round(log_loss(train["Shot Made Flag"].values, train_probs), 4)
        all_results.append(row)
        print(pd.DataFrame([row])[DISPLAY_COLS].to_string(index=False))

        plot_surfaces(name, model, likelihood, mu, std,
                      "gp_location", slug, RESULTS_DIR, use_is3pt=False)
        plot_calibration(name, y_true, probs, "gp_location", slug, RESULTS_DIR)

        if SAVE_PREDICTIONS:
            pred_path = f"{RESULTS_DIR}/predictions_{slug}_gp_location.csv"
            pd.DataFrame({"model": "gp_location", "y_true": y_true, "y_prob": probs}
                         ).to_csv(pred_path, index=False)
            print(f"    Saved predictions → {pred_path}")

    results_df = pd.DataFrame(all_results)[["player", "train_log_loss"] + DISPLAY_COLS]
    print(f"\n{'='*60}\nFull results\n{'='*60}")
    print(results_df.to_string(index=False))
    out = f"{RESULTS_DIR}/gp_location_results.csv"
    results_df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
