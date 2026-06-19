"""
Broad context study: gp_context on all 30 players, saving to results/broad/.
Run after broad_study.py so broad_results.csv already exists for comparison.

Requires: results/broad/broad_results.csv  (broad_study.py must have run first)
Produces: results/broad/broad_gp_context_results.csv
          results/broad/broad_gp_context_full_results.csv  (includes spatial_frac + ARD ls)
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import log_loss, brier_score_loss

from gp_context import (add_context_features, _to_tensor, _standardize,
                        _build_additive_kernel, FEAT_NAMES)
from gp_utils    import predict_probs, full_metrics, DISPLAY_COLS
from kernels     import train_gp

torch.manual_seed(42)

N_INDUCING  = 200
N_ITER      = 500
LR          = 0.01
RESULTS_DIR = "results/broad"
os.makedirs(RESULTS_DIR, exist_ok=True)

PLAYERS = [
    ("James Harden",           "data/james_harden_shots.csv"),
    ("Russell Westbrook",      "data/russell_westbrook_shots.csv"),
    ("Stephen Curry",          "data/stephen_curry_shots.csv"),
    ("Klay Thompson",          "data/klay_thompson_shots.csv"),
    ("Damian Lillard",         "data/damian_lillard_shots.csv"),
    ("LeBron James",           "data/lebron_james_shots.csv"),
    ("DeMar DeRozan",          "data/demar_derozan_shots.csv"),
    ("LaMarcus Aldridge",      "data/lamarcus_aldridge_shots.csv"),
    ("Kemba Walker",           "data/kemba_walker_shots.csv"),
    ("Paul George",            "data/paul_george_shots.csv"),
    ("Kyrie Irving",           "data/kyrie_irving_shots.csv"),
    ("Kevin Durant",           "data/kevin_durant_shots.csv"),
    ("Bradley Beal",           "data/bradley_beal_shots.csv"),
    ("Anthony Davis",          "data/anthony_davis_shots.csv"),
    ("Carmelo Anthony",        "data/carmelo_anthony_shots.csv"),
    ("Blake Griffin",          "data/blake_griffin_shots.csv"),
    ("John Wall",              "data/john_wall_shots.csv"),
    ("Dwyane Wade",            "data/dwyane_wade_shots.csv"),
    ("Kyle Lowry",             "data/kyle_lowry_shots.csv"),
    ("Chris Paul",             "data/chris_paul_shots.csv"),
    ("DeMarcus Cousins",       "data/demarcus_cousins_shots.csv"),
    ("Marc Gasol",             "data/marc_gasol_shots.csv"),
    ("Nikola Vucevic",         "data/nikola_vucevic_shots.csv"),
    ("Kawhi Leonard",          "data/kawhi_leonard_shots.csv"),
    ("Jimmy Butler",           "data/jimmy_butler_shots.csv"),
    ("CJ McCollum",            "data/cj_mccollum_shots.csv"),
    ("Serge Ibaka",            "data/serge_ibaka_shots.csv"),
    ("Tobias Harris",          "data/tobias_harris_shots.csv"),
    ("Goran Dragic",           "data/goran_dragic_shots.csv"),
    ("Thaddeus Young",         "data/thaddeus_young_shots.csv"),
]

CONTEXT_FEAT_NAMES = FEAT_NAMES[2:]  # opp_win_pct … act_hook_float (7 dims)


def run_player(name, path, idx, total):
    slug = name.lower().replace(" ", "_")
    print(f"\n[{idx}/{total}] {name}")
    t0 = time.time()

    df    = pd.read_csv(path)
    df    = add_context_features(df)
    train = df[df["Split"] == "train"].copy()
    test  = df[df["Split"] == "test"].copy()
    y_true = test["Shot Made Flag"].values

    ref_p   = train["Shot Made Flag"].mean()
    ref_ll  = log_loss(y_true, np.full(len(y_true), ref_p))
    ref_bs  = brier_score_loss(y_true, np.full(len(y_true), ref_p))
    ref_acc = ((np.full(len(y_true), ref_p) >= 0.5) == y_true).mean()

    X_tr_raw, y_tr = _to_tensor(train)
    X_te_raw, _    = _to_tensor(test)
    X_tr, X_te, _, _ = _standardize(X_tr_raw, X_te_raw)

    kernel = _build_additive_kernel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, likelihood = train_gp(
            X_tr, y_tr, kernel,
            learn_inducing=True,
            n_iter=N_ITER, lr=LR, n_inducing=N_INDUCING,
            print_every=0,
        )

    probs, _ = predict_probs(model, likelihood, X_te)
    train_probs, _ = predict_probs(model, likelihood, X_tr)
    row = full_metrics(y_true, probs, "gp_context",
                       ref_logloss=ref_ll, ref_brier=ref_bs, ref_acc=ref_acc)
    row["player"] = name
    row["train_log_loss"] = round(log_loss(train["Shot Made Flag"].values, train_probs), 4)

    # Variance decomposition: how much variance comes from spatial vs context
    k_sp = model.covar_module.kernels[0].outputscale.item()
    k_ct = model.covar_module.kernels[1].outputscale.item()
    row["spatial_frac"] = k_sp / (k_sp + k_ct)

    # ARD length-scales for each context feature
    ls_ctx = model.covar_module.kernels[1].base_kernel.lengthscale.detach().squeeze().tolist()
    if isinstance(ls_ctx, float):
        ls_ctx = [ls_ctx]
    for fname, ls in zip(CONTEXT_FEAT_NAMES, ls_ctx):
        row[f"ls_{fname}"] = ls

    elapsed = time.time() - t0
    print(f"  ctx={row['log_loss']:.4f} ({row['ll_skill_%']:+.2f}%)  "
          f"spat={row['spatial_frac']*100:.1f}%  [{elapsed:.0f}s]")

    # Print top-3 most informative context features (shortest length-scale)
    ls_sorted = sorted(zip(CONTEXT_FEAT_NAMES, ls_ctx), key=lambda x: x[1])
    top3 = ", ".join(f"{n}={v:.2f}" for n, v in ls_sorted[:3])
    print(f"  top-3 context feats (shortest ls): {top3}")

    return row


def cross_player_summary(ctx_df, broad_df):
    print(f"\n{'='*65}")
    print("  Cross-player summary — gp_context vs gp_location  (n=30)")
    print(f"{'='*65}")

    ctx_skills = pd.to_numeric(
        ctx_df.set_index("player")["ll_skill_%"], errors="coerce"
    )
    gp_skills = pd.to_numeric(
        broad_df[broad_df["model"] == "gp_location"]
        .set_index("player")["ll_skill_%"],
        errors="coerce",
    )

    print(f"\n  Log-loss skill score (% over intercept-only):")
    print(f"  {'Model':<16}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'#>0':>5}")
    for label, skills in [("gp_location", gp_skills), ("gp_context", ctx_skills)]:
        s = skills.dropna()
        print(f"  {label:<16}  {s.mean():>8.2f}  {s.std():>8.2f}  "
              f"{s.min():>8.2f}  {s.max():>8.2f}  {(s>0).sum():>5}/{len(s)}")

    # Per-player context gain
    common = ctx_skills.index.intersection(gp_skills.index)
    gain   = (ctx_skills[common] - gp_skills[common]).dropna()
    print(f"\n  Context gain over location (ctx_skill% − gp_skill%):")
    print(f"  Mean={gain.mean():.3f}  Std={gain.std():.3f}  "
          f"Min={gain.min():.3f}  Max={gain.max():.3f}  "
          f"#ctx>gp={(gain > 0).sum()}/{len(gain)}")

    # Sign test: is context > location for significantly more than half?
    n_pos  = int((gain > 0).sum())
    p_sign = stats.binomtest(n_pos, len(gain), p=0.5, alternative="greater").pvalue
    print(f"  Sign test ctx>gp: {n_pos}/{len(gain)}  (p={p_sign:.4f})")

    # Paired Wilcoxon: context skill vs location skill
    ctx_v = ctx_skills[common].dropna()
    gp_v  = gp_skills[common].dropna()
    idx   = ctx_v.index.intersection(gp_v.index)
    if len(idx) >= 5:
        stat, p = stats.wilcoxon(ctx_v[idx], gp_v[idx], alternative="two-sided")
        winner  = "ctx" if ctx_v[idx].mean() > gp_v[idx].mean() else "gp"
        print(f"  Wilcoxon ctx vs gp: winner={winner}  W={stat:.0f}  p={p:.4f}")

    # Variance decomposition summary
    sf = ctx_df["spatial_frac"].dropna()
    print(f"\n  Spatial fraction across players:")
    print(f"  Mean={sf.mean():.3f}  Std={sf.std():.3f}  "
          f"Min={sf.min():.3f}  Max={sf.max():.3f}")

    # Mean ARD length-scales across players
    ls_cols = [c for c in ctx_df.columns if c.startswith("ls_")]
    if ls_cols:
        print(f"\n  Mean ARD length-scales (lower = more informative):")
        ls_means = ctx_df[ls_cols].mean().sort_values()
        for col, val in ls_means.items():
            print(f"    {col[3:]:<22}  {val:.3f}")

    # Player ranking by context gain
    gain_sorted = gain.sort_values(ascending=False)
    print(f"\n  Player ranking by context gain:")
    print(f"  {'Rank':<5} {'Player':<25} {'Ctx%':>8} {'GP%':>8} {'Gain':>8}")
    for rank, (player, g) in enumerate(gain_sorted.items(), 1):
        cs = ctx_skills.get(player, float("nan"))
        gs = gp_skills.get(player, float("nan"))
        marker = " ◀" if player in {"Stephen Curry", "LeBron James", "Kevin Durant"} else ""
        print(f"  {rank:<5} {player:<25} {cs:>8.2f} {gs:>8.2f} {g:>8.3f}{marker}")


def main():
    print(f"Broad context study — {len(PLAYERS)} players — gp_context")
    print(f"Results → {RESULTS_DIR}/\n")

    all_rows = []
    total    = len(PLAYERS)

    for idx, (name, path) in enumerate(PLAYERS, 1):
        row = run_player(name, path, idx, total)
        all_rows.append(row)

    ctx_df = pd.DataFrame(all_rows)

    base_cols = ["player", "train_log_loss"] + DISPLAY_COLS
    ctx_df[base_cols].to_csv(f"{RESULTS_DIR}/broad_gp_context_results.csv", index=False)
    print(f"\nSaved {RESULTS_DIR}/broad_gp_context_results.csv")

    ctx_df.to_csv(f"{RESULTS_DIR}/broad_gp_context_full_results.csv", index=False)
    print(f"Saved {RESULTS_DIR}/broad_gp_context_full_results.csv")

    broad_path = f"{RESULTS_DIR}/broad_results.csv"
    if os.path.exists(broad_path):
        broad_df = pd.read_csv(broad_path)
        cross_player_summary(ctx_df, broad_df)
    else:
        print(f"\nWarning: {broad_path} not found — run broad_study.py first.")

    print("\nDone.")


if __name__ == "__main__":
    main()
