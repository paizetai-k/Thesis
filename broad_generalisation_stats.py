"""
Broad-study cross-player summary statistics (H1 and H3).

Computes aggregate sign tests and Wilcoxon signed-rank tests from the
already-saved per-player result CSVs, matching the call signatures used
in broad_study.py and broad_context.py:

  stats.wilcoxon(x, y, alternative="two-sided")   # no zero_method → 'wilcox'
  stats.binomtest(n_pos, n_tot, p=0.5, alternative="greater")

Saves a summary CSV to results/broad/broad_generalisation_stats.csv.
"""
import os
import pandas as pd
import numpy as np
from scipy import stats

BROAD_DIR   = "results/broad"
OUTPUT_CSV  = "results/broad/broad_generalisation_stats.csv"

FOCAL = {"Stephen Curry", "LeBron James", "Kevin Durant"}


# ── Load CSVs ─────────────────────────────────────────────────────────────────

broad_df = pd.read_csv(os.path.join(BROAD_DIR, "broad_results.csv"))
ctx_df   = pd.read_csv(os.path.join(BROAD_DIR, "broad_gp_context_full_results.csv"))

# ll_skill_% is stored as string "-" for intercept_only; coerce to numeric
for df in (broad_df, ctx_df):
    df["ll_skill_%"] = pd.to_numeric(df["ll_skill_%"], errors="coerce")

print(f"Loaded broad_results.csv        : {len(broad_df)} rows, "
      f"models={sorted(broad_df['model'].unique())}")
print(f"Loaded broad_gp_context_full... : {len(ctx_df)} rows, "
      f"models={sorted(ctx_df['model'].unique())}")
print(f"Players in broad study          : {broad_df['player'].nunique()}")


# ── H1: gp_location vs baselines ──────────────────────────────────────────────

print(f"\n{'='*65}")
print("  H1: Cross-player summary  (n=30 players)")
print(f"{'='*65}")

models_h1 = ["binned_EB", "logistic_RBF", "gp_location"]

print(f"\n  Log-loss skill score (% over intercept-only):")
print(f"  {'Model':<16}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'#>0':>5}")
for m in models_h1:
    sub = broad_df[broad_df["model"] == m]["ll_skill_%"].dropna()
    print(f"  {m:<16}  {sub.mean():>8.2f}  {sub.std():>8.2f}  "
          f"{sub.min():>8.2f}  {sub.max():>8.2f}  {(sub > 0).sum():>5}/{len(sub)}")

# Player ranking by GP skill score
gp_skills = (broad_df[broad_df["model"] == "gp_location"]
             .set_index("player")["ll_skill_%"]
             .sort_values(ascending=False))

print(f"\n  Player ranking by GP location log-loss skill score:")
print(f"  {'Rank':<5} {'Player':<25} {'GP skill%':>10}")
for rank, (player, skill) in enumerate(gp_skills.items(), 1):
    marker = " ◀" if player in FOCAL else ""
    print(f"  {rank:<5} {player:<25} {skill:>10.2f}{marker}")

# Sign test: GP skill > 0 for more than half?
skills = gp_skills.dropna()
n_pos  = int((skills > 0).sum())
n_tot  = len(skills)
p_binom_h1 = stats.binomtest(n_pos, n_tot, p=0.5, alternative="greater").pvalue
print(f"\n  Sign test — GP skill > 0: {n_pos}/{n_tot} players  (p={p_binom_h1:.6f})")

# Across-player Wilcoxon: GP vs each baseline on skill scores
print(f"\n  Across-player Wilcoxon (paired on log-loss skill scores):")
wilcoxon_h1 = {}
for baseline in ["binned_EB", "logistic_RBF"]:
    gp_s  = broad_df[broad_df["model"] == "gp_location"].set_index("player")["ll_skill_%"]
    bl_s  = broad_df[broad_df["model"] == baseline].set_index("player")["ll_skill_%"]
    common = gp_s.index.intersection(bl_s.index)
    gp_v  = gp_s[common].dropna()
    bl_v  = bl_s[common].dropna()
    idx   = gp_v.index.intersection(bl_v.index)
    stat, p = stats.wilcoxon(gp_v[idx], bl_v[idx], alternative="two-sided")
    winner  = "GP" if gp_v[idx].mean() > bl_v[idx].mean() else baseline
    wilcoxon_h1[baseline] = {"W": stat, "p": p, "n": len(idx), "winner": winner}
    print(f"  GP vs {baseline:<16}  winner={winner:<16}  W={stat:.0f}  p={p:.6f}")

# Counts where GP beats each model
print(f"\n  Counts (players where GP log-loss < baseline log-loss):")
gp_ll = broad_df[broad_df["model"] == "gp_location"].set_index("player")["log_loss"]
for baseline in ["intercept_only", "binned_EB", "logistic_RBF"]:
    bl_ll = broad_df[broad_df["model"] == baseline].set_index("player")["log_loss"]
    common = gp_ll.index.intersection(bl_ll.index)
    n_beats = (gp_ll[common] < bl_ll[common]).sum()
    print(f"  GP < {baseline:<16}: {n_beats}/{len(common)}")


# ── H3: gp_context vs gp_location ─────────────────────────────────────────────

print(f"\n{'='*65}")
print("  H3: Cross-player summary — gp_context vs gp_location  (n=30)")
print(f"{'='*65}")

ctx_skills = ctx_df.set_index("player")["ll_skill_%"]
gp_skills_h3 = (broad_df[broad_df["model"] == "gp_location"]
                .set_index("player")["ll_skill_%"])

print(f"\n  Log-loss skill score (% over intercept-only):")
print(f"  {'Model':<16}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'#>0':>5}")
for label, skills in [("gp_location", gp_skills_h3), ("gp_context", ctx_skills)]:
    s = skills.dropna()
    print(f"  {label:<16}  {s.mean():>8.2f}  {s.std():>8.2f}  "
          f"{s.min():>8.2f}  {s.max():>8.2f}  {(s > 0).sum():>5}/{len(s)}")

# Per-player context gain
common_h3 = ctx_skills.index.intersection(gp_skills_h3.index)
gain       = (ctx_skills[common_h3] - gp_skills_h3[common_h3]).dropna()
print(f"\n  Context gain over location (ctx_skill% − gp_skill%):")
print(f"  Mean={gain.mean():.3f}  Std={gain.std():.3f}  "
      f"Min={gain.min():.3f}  Max={gain.max():.3f}  "
      f"#ctx>gp={(gain > 0).sum()}/{len(gain)}")

# Sign test
n_pos_h3  = int((gain > 0).sum())
p_sign_h3 = stats.binomtest(n_pos_h3, len(gain), p=0.5, alternative="greater").pvalue
print(f"  Sign test ctx>gp: {n_pos_h3}/{len(gain)}  (p={p_sign_h3:.6f})")

# Paired Wilcoxon
ctx_v = ctx_skills[common_h3].dropna()
gp_v  = gp_skills_h3[common_h3].dropna()
idx_h3 = ctx_v.index.intersection(gp_v.index)
stat_h3, p_h3 = stats.wilcoxon(ctx_v[idx_h3], gp_v[idx_h3], alternative="two-sided")
winner_h3 = "ctx" if ctx_v[idx_h3].mean() > gp_v[idx_h3].mean() else "gp"
print(f"  Wilcoxon ctx vs gp: winner={winner_h3}  W={stat_h3:.0f}  p={p_h3:.6f}")

# Spatial fraction summary
sf = ctx_df["spatial_frac"].dropna()
print(f"\n  Spatial fraction across players:")
print(f"  Mean={sf.mean():.3f}  Std={sf.std():.3f}  "
      f"Min={sf.min():.3f}  Max={sf.max():.3f}")

# Mean ARD length-scales
ls_cols = [c for c in ctx_df.columns if c.startswith("ls_")]
if ls_cols:
    print(f"\n  Mean ARD length-scales (lower = more informative):")
    ls_means = ctx_df[ls_cols].mean().sort_values()
    for col, val in ls_means.items():
        print(f"    {col[3:]:<22}  {val:.4f}")

# Player ranking by context gain
gain_sorted = gain.sort_values(ascending=False)
print(f"\n  Player ranking by context gain (ctx_skill% − gp_skill%):")
print(f"  {'Rank':<5} {'Player':<25} {'Ctx%':>8} {'GP%':>8} {'Gain':>8}")
for rank, (player, g) in enumerate(gain_sorted.items(), 1):
    cs = ctx_skills.get(player, float("nan"))
    gs = gp_skills_h3.get(player, float("nan"))
    marker = " ◀" if player in FOCAL else ""
    print(f"  {rank:<5} {player:<25} {cs:>8.2f} {gs:>8.2f} {g:>8.3f}{marker}")


# ── Save summary CSV ───────────────────────────────────────────────────────────

rows = []

# H1 rows
for baseline in ["binned_EB", "logistic_RBF"]:
    w = wilcoxon_h1[baseline]
    rows.append({
        "comparison":  f"H1_GP_vs_{baseline}",
        "n_players":   w["n"],
        "winner":      w["winner"],
        "W_statistic": w["W"],
        "p_value":     w["p"],
        "test":        "wilcoxon_two_sided",
        "note":        "paired on ll_skill_%; zero_method=wilcox (scipy default)",
    })

gp_skill_stats = broad_df[broad_df["model"] == "gp_location"]["ll_skill_%"].dropna()
rows.append({
    "comparison":  "H1_GP_skill_gt0_sign_test",
    "n_players":   n_tot,
    "winner":      f"{n_pos}/{n_tot}",
    "W_statistic": float("nan"),
    "p_value":     p_binom_h1,
    "test":        "binomtest_greater",
    "note":        "H0: P(GP_skill>0)=0.5",
})
rows.append({
    "comparison":  "H1_GP_skill_mean",
    "n_players":   len(gp_skill_stats),
    "winner":      "gp_location",
    "W_statistic": float("nan"),
    "p_value":     float("nan"),
    "test":        "descriptive",
    "note":        f"mean={gp_skill_stats.mean():.4f} std={gp_skill_stats.std():.4f} "
                   f"min={gp_skill_stats.min():.4f} max={gp_skill_stats.max():.4f}",
})

# H3 rows
rows.append({
    "comparison":  "H3_ctx_vs_gp_sign_test",
    "n_players":   len(gain),
    "winner":      f"{n_pos_h3}/{len(gain)}",
    "W_statistic": float("nan"),
    "p_value":     p_sign_h3,
    "test":        "binomtest_greater",
    "note":        "H0: P(ctx_skill>gp_skill)=0.5",
})
rows.append({
    "comparison":  "H3_ctx_vs_gp_wilcoxon",
    "n_players":   len(idx_h3),
    "winner":      winner_h3,
    "W_statistic": stat_h3,
    "p_value":     p_h3,
    "test":        "wilcoxon_two_sided",
    "note":        "paired on ll_skill_%; zero_method=wilcox (scipy default)",
})
rows.append({
    "comparison":  "H3_ctx_gain_mean",
    "n_players":   len(gain),
    "winner":      "gp_context",
    "W_statistic": float("nan"),
    "p_value":     float("nan"),
    "test":        "descriptive",
    "note":        f"mean_gain={gain.mean():.4f} std={gain.std():.4f} "
                   f"min={gain.min():.4f} max={gain.max():.4f}",
})

out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}")
print("\nDone.")
