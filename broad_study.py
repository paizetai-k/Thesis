"""
Broad study: intercept_only, binned_EB, logistic_RBF, and gp_location
on all 30 players (top-30 by shot volume, 2012-13 through 2018-19).

Results saved to results/broad/ in the same CSV format as the deep-dive
scripts, so model_comparison.py works with DATASET="broad" unchanged.

Switches
--------
RUN_GP     : False → run only baselines (~2 min total, good for testing)
SAVE_PLOTS : True  → save GP surface + calibration plots per player
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
from scipy import stats

from sklearn.metrics import log_loss
from baselines import intercept_only, binned_eb, logistic_rbf
from gp_utils   import (load, to_tensor, standardize, predict_probs,
                         full_metrics, plot_surfaces, plot_calibration,
                         DISPLAY_COLS)
from kernels    import build_kernel, train_gp

torch.manual_seed(42)

# ── Switches ──────────────────────────────────────────────────────────────────
RUN_GP     = True    # False → baselines only (fast)
SAVE_PLOTS = False   # True  → save GP surface + calibration per player
# ─────────────────────────────────────────────────────────────────────────────

N_INDUCING = 200
N_ITER     = 500
LR         = 0.01

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


# ── Per-player run ────────────────────────────────────────────────────────────

def run_player(name, path, idx, total):
    slug = name.lower().replace(" ", "_")
    print(f"\n[{idx}/{total}] {name}")
    t0 = time.time()

    train, test = load(path)
    y_true = test["Shot Made Flag"].values
    rows = []

    # Intercept-only — also provides reference for skill scores
    row_int, _ = intercept_only(train, test)
    row_int["player"] = name
    rows.append(row_int)
    ref_ll  = row_int["log_loss"]
    ref_bs  = row_int["brier"]
    ref_acc = row_int["accuracy"]

    # Binned EB
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row_eb, _, *_ = binned_eb(train, test, ref_ll, ref_bs, ref_acc)
    row_eb["player"] = name
    rows.append(row_eb)

    # Logistic RBF
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row_rbf, _, *_ = logistic_rbf(train, test, ref_ll, ref_bs, ref_acc)
    row_rbf["player"] = name
    rows.append(row_rbf)

    baseline_str = (f"  int={ref_ll:.4f}  "
                    f"eb={row_eb['log_loss']:.4f} ({row_eb['ll_skill_%']:+.2f}%)  "
                    f"rbf={row_rbf['log_loss']:.4f} ({row_rbf['ll_skill_%']:+.2f}%)")

    gp_row  = None
    ls_pair = None
    if RUN_GP:
        X_tr_raw, y_tr = to_tensor(train)
        X_te_raw, _    = to_tensor(test)
        X_tr, X_te, mu, std = standardize(X_tr_raw, X_te_raw)

        kernel = build_kernel("standard")
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
        gp_row = full_metrics(y_true, probs, "gp_location",
                              ref_logloss=ref_ll, ref_brier=ref_bs, ref_acc=ref_acc)
        gp_row["player"] = name
        gp_row["train_log_loss"] = round(log_loss(train["Shot Made Flag"].values, train_probs), 4)
        rows.append(gp_row)

        ls     = model.covar_module.base_kernel.lengthscale[0]
        ls_x   = ls[0].item()
        ls_y   = ls[1].item()
        ls_pair = (ls_x, ls_y)

        print(baseline_str)
        print(f"  gp={gp_row['log_loss']:.4f} ({gp_row['ll_skill_%']:+.2f}%)  "
              f"ls_x={ls_x:.4f}  ls_y={ls_y:.4f}  ratio={ls_y/ls_x:.3f}  "
              f"[{time.time()-t0:.0f}s]")

        if SAVE_PLOTS:
            plot_surfaces(name, model, likelihood, mu, std,
                          "gp_location", slug, RESULTS_DIR, use_is3pt=False)
            plot_calibration(name, y_true, probs, "gp_location", slug, RESULTS_DIR)
    else:
        print(baseline_str + f"  [{time.time()-t0:.0f}s]")

    return rows, ls_pair


# ── Cross-player summary ──────────────────────────────────────────────────────

def cross_player_summary(df: pd.DataFrame):
    models = ["binned_EB", "logistic_RBF", "gp_location"]
    models = [m for m in models if m in df["model"].unique()]

    print(f"\n{'='*65}")
    print("  Cross-player summary  (n=30 players)")
    print(f"{'='*65}")

    # Mean ± std of log-loss skill score per model
    print("\n  Log-loss skill score (% over intercept-only):")
    print(f"  {'Model':<16}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'#>0':>5}")
    for m in models:
        sub = pd.to_numeric(
            df[df["model"] == m]["ll_skill_%"], errors="coerce"
        ).dropna()
        print(f"  {m:<16}  {sub.mean():>8.2f}  {sub.std():>8.2f}  "
              f"{sub.min():>8.2f}  {sub.max():>8.2f}  {(sub>0).sum():>5}/{len(sub)}")

    if "gp_location" not in df["model"].unique():
        return

    # Player ranking by GP skill score
    gp_skills = (df[df["model"] == "gp_location"]
                 .set_index("player")["ll_skill_%"]
                 .apply(pd.to_numeric, errors="coerce")
                 .sort_values(ascending=False))

    print("\n  Player ranking by GP location log-loss skill score:")
    print(f"  {'Rank':<5} {'Player':<25} {'GP skill%':>10}")
    for rank, (player, skill) in enumerate(gp_skills.items(), 1):
        marker = " ◀" if player in {"Stephen Curry", "LeBron James", "Kevin Durant"} else ""
        print(f"  {rank:<5} {player:<25} {skill:>10.2f}{marker}")

    # One-sample sign test: is GP skill > 0 for significantly more than half?
    skills = gp_skills.dropna()
    n_pos  = (skills > 0).sum()
    n_tot  = len(skills)
    # Binomial test: H0 = P(skill > 0) = 0.5
    p_binom = stats.binomtest(n_pos, n_tot, p=0.5, alternative="greater").pvalue
    print(f"\n  Sign test — GP skill > 0: {n_pos}/{n_tot} players  (p={p_binom:.4f})")

    # Across-player Wilcoxon: GP vs each baseline on skill scores
    print("\n  Across-player Wilcoxon (paired on log-loss skill scores):")
    for baseline in ["binned_EB", "logistic_RBF"]:
        if baseline not in df["model"].unique():
            continue
        gp_s  = df[df["model"] == "gp_location"].set_index("player")["ll_skill_%"]
        bl_s  = df[df["model"] == baseline].set_index("player")["ll_skill_%"]
        common = gp_s.index.intersection(bl_s.index)
        gp_v   = pd.to_numeric(gp_s[common], errors="coerce").dropna()
        bl_v   = pd.to_numeric(bl_s[common], errors="coerce").dropna()
        idx    = gp_v.index.intersection(bl_v.index)
        if len(idx) < 5:
            continue
        stat, p = stats.wilcoxon(gp_v[idx], bl_v[idx], alternative="two-sided")
        winner  = "GP" if gp_v[idx].mean() > bl_v[idx].mean() else baseline
        print(f"  GP vs {baseline:<16}  winner={winner:<16}  W={stat:.0f}  p={p:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mode = "baselines + GP location" if RUN_GP else "baselines only"
    print(f"Broad study — {len(PLAYERS)} players — {mode}")
    print(f"Results → {RESULTS_DIR}/\n")

    all_rows   = []
    ls_records = []
    total      = len(PLAYERS)

    for idx, (name, path) in enumerate(PLAYERS, 1):
        rows, ls_pair = run_player(name, path, idx, total)
        all_rows.extend(rows)
        if ls_pair is not None:
            ls_records.append({"player": name,
                                "ls_x":   round(ls_pair[0], 4),
                                "ls_y":   round(ls_pair[1], 4),
                                "ratio":  round(ls_pair[1] / ls_pair[0], 4)})

    df = pd.DataFrame(all_rows)

    # Save in same format as deep-dive scripts for model_comparison.py compatibility
    base_cols = ["player", "train_log_loss"] + DISPLAY_COLS
    bl_models = ["intercept_only", "binned_EB", "logistic_RBF"]

    bl_df = df[df["model"].isin(bl_models)][base_cols]
    bl_df.to_csv(f"{RESULTS_DIR}/broad_baselines_results.csv", index=False)
    print(f"\nSaved {RESULTS_DIR}/broad_baselines_results.csv")

    if RUN_GP and "gp_location" in df["model"].unique():
        gp_df = df[df["model"] == "gp_location"][base_cols]
        gp_df.to_csv(f"{RESULTS_DIR}/broad_gp_location_results.csv", index=False)
        print(f"Saved {RESULTS_DIR}/broad_gp_location_results.csv")

    # Full combined CSV
    df[base_cols].to_csv(f"{RESULTS_DIR}/broad_results.csv", index=False)
    print(f"Saved {RESULTS_DIR}/broad_results.csv")

    cross_player_summary(df)

    if ls_records:
        ls_df = (pd.DataFrame(ls_records)
                   .sort_values("ratio", ascending=False)
                   .reset_index(drop=True))
        ls_df.index += 1
        print(f"\n{'='*65}")
        print("  ARD-2 length-scales — all 30 players  (sorted by ls_y/ls_x)")
        print(f"{'='*65}")
        print(ls_df.to_string())
        print(f"\n  Mean ratio ls_y/ls_x : {ls_df['ratio'].mean():.3f}")
        print(f"  All ratios > 1       : {(ls_df['ratio'] > 1).all()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
