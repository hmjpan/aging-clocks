"""Step 7: Statistical validation of core claims.

Tests:
  A. Tissue-tissue correlation: Is cross-tissue drug structure non-random?
  B. Bootstrap stability: Are top drug rankings stable?
  C. Clock sensitivity: Does conclusion hold across parameters?
  D. Age permutation: Is the clock better than random age assignment?
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_DIR, TABLE_DIR, FIGURE_DIR, GTEX_DIR
from gct_io import read_gct, build_tissue_sample_table
from clock import train_tissue_clock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 1000
N_BOOT = 1000
N_AGE_PERM = 50
RNG_SEED = 42


def test_tissue_correlation():
    print("=" * 70)
    print("TEST A: Tissue-tissue correlation of drug effects")
    print("=" * 70, flush=True)
    matrix = pd.read_csv(TABLE_DIR / "drug_tissue_reversal_matrix.csv", index_col=0)
    obs_corr = matrix.corr()
    n_t = matrix.shape[1]
    triu = np.triu_indices(n_t, k=1)
    obs_off = obs_corr.values[triu]
    obs_mean = float(np.mean(np.abs(obs_off)))
    obs_nsig = int(np.sum(np.abs(obs_off) > 0.3))
    print("  Tissues: {}, pairs: {}".format(n_t, len(obs_off)), flush=True)
    print("  Observed mean |r|: {:.4f}, n_sig(|r|>0.3): {}".format(obs_mean, obs_nsig), flush=True)
    rng = np.random.RandomState(RNG_SEED)
    mv = matrix.values
    perm_mean = np.zeros(N_PERM)
    perm_nsig = np.zeros(N_PERM, dtype=int)
    print("  Running {} permutations...".format(N_PERM), flush=True)
    for i in range(N_PERM):
        if i % 200 == 0:
            print("    perm {}/{}".format(i, N_PERM), flush=True)
        pm = np.column_stack([rng.permutation(mv[:, j]) for j in range(n_t)])
        pc = np.corrcoef(pm, rowvar=False)
        po = pc[triu]
        perm_mean[i] = np.mean(np.abs(po))
        perm_nsig[i] = np.sum(np.abs(po) > 0.3)
    p_mean = float(np.mean(perm_mean >= obs_mean))
    p_nsig = float(np.mean(perm_nsig >= obs_nsig))
    null_m = float(np.mean(perm_mean))
    null_s = float(np.std(perm_mean))
    z = (obs_mean - null_m) / (null_s + 1e-10)
    sig = "SIGNIFICANT" if p_mean < 0.05 else "NOT SIG"
    print("  Null: {:.4f} +/- {:.4f}".format(null_m, null_s))
    print("  Z-score: {:.2f}".format(z))
    print("  P-value (mean |r|): {}".format(p_mean))
    print("  P-value (n_sig): {}".format(p_nsig))
    print("  Conclusion: {}".format(sig), flush=True)
    pd.DataFrame({"perm_mean_abs_corr": perm_mean, "perm_n_sig": perm_nsig}).to_csv(
        TABLE_DIR / "permutation_null_distribution.csv", index=False)
    obs_corr.to_csv(TABLE_DIR / "tissue_tissue_correlation.csv")
    return dict(obs_mean_abs_corr=obs_mean, null_mean_abs_corr=null_m,
                null_std_abs_corr=null_s, z_score=z, p_mean=p_mean, p_nsig=p_nsig,
                obs_n_significant=obs_nsig, perm_mean_abs_corrs=perm_mean,
                perm_n_significant=perm_nsig)


def test_bootstrap_stability():
    print("\n" + "=" * 70)
    print("TEST B: Bootstrap stability of drug rankings")
    print("=" * 70, flush=True)
    matrix = pd.read_csv(TABLE_DIR / "drug_tissue_reversal_matrix.csv", index_col=0)
    het_df = pd.read_csv(TABLE_DIR / "drug_heterogeneity_scores.csv")
    active = het_df[het_df["n_tissues_active"] >= 3].copy()
    orig_rank = active.sort_values("mean_reversal", ascending=False)
    orig_rank_dict = dict(zip(orig_rank["compound"], range(len(orig_rank))))
    rng = np.random.RandomState(RNG_SEED + 1)
    tissues = list(matrix.columns)
    n_t = len(tissues)
    top20_stab = np.zeros(N_BOOT)
    rank_corrs = np.zeros(N_BOOT)
    print("  Running {} boots...".format(N_BOOT), flush=True)
    for i in range(N_BOOT):
        if i % 200 == 0:
            print("    boot {}/{}".format(i, N_BOOT), flush=True)
        bt = rng.choice(tissues, size=n_t, replace=True)
        bm = matrix[bt]
        bmean = bm.mean(axis=1)
        bactive = bmean.loc[active["compound"].values]
        brank = bactive.rank(ascending=False)
        o20 = set(orig_rank["compound"].head(20).values)
        b20 = set(bactive.nlargest(20).index)
        top20_stab[i] = len(o20 & b20) / 20
        o_ranks = np.array([orig_rank_dict.get(c, len(orig_rank)) for c in bactive.index])
        b_ranks = brank.values
        if np.std(o_ranks) > 0 and np.std(b_ranks) > 0:
            r, _ = spearmanr(o_ranks, b_ranks)
            rank_corrs[i] = r
    ms = float(np.mean(top20_stab))
    mrc = float(np.mean(rank_corrs))
    cl, ch = np.percentile(top20_stab, [2.5, 97.5])
    rcl, rch = np.percentile(rank_corrs, [2.5, 97.5])
    stable = "STABLE" if mrc > 0.7 else "UNSTABLE"
    print("  Top-20 overlap: {:.3f} (CI: {:.3f}-{:.3f})".format(ms, cl, ch))
    print("  Rank corr: {:.3f} (CI: {:.3f}-{:.3f})".format(mrc, rcl, rch))
    print("  Conclusion: {}".format(stable), flush=True)
    pd.DataFrame({"top20_overlap": top20_stab, "rank_correlation": rank_corrs}).to_csv(
        TABLE_DIR / "bootstrap_stability.csv", index=False)
    return dict(mean_stability=ms, ci_low=cl, ci_high=ch, mean_rank_corr=mrc,
                rc_low=rcl, rc_high=rch, top20_stability=top20_stab,
                rank_correlations=rank_corrs)


def test_clock_sensitivity():
    print("\n" + "=" * 70)
    print("TEST C: Clock sensitivity analysis")
    print("=" * 70, flush=True)
    meta = build_tissue_sample_table()
    fpath = str(GTEX_DIR / "tpm_liver.gct.gz")
    expr = read_gct(fpath)
    param_sets = [
        ({"l1_ratio_grid": [0.2], "n_top_features_variance": 3000}, "l1=0.2,n=3000"),
        ({"l1_ratio_grid": [0.5], "n_top_features_variance": 3000}, "l1=0.5,n=3000"),
        ({"l1_ratio_grid": [0.8], "n_top_features_variance": 3000}, "l1=0.8,n=3000"),
        ({"l1_ratio_grid": [0.9], "n_top_features_variance": 3000}, "l1=0.9,n=3000"),
        ({"l1_ratio_grid": [0.5], "n_top_features_variance": 1000}, "l1=0.5,n=1000"),
        ({"l1_ratio_grid": [0.5], "n_top_features_variance": 2000}, "l1=0.5,n=2000"),
        ({"l1_ratio_grid": [0.5], "n_top_features_variance": 5000}, "l1=0.5,n=5000"),
    ]
    results = []
    for params, label in param_sets:
        t0 = time.time()
        result = train_tissue_clock("liver", expr, meta, params=params)
        elapsed = time.time() - t0
        if result:
            results.append({"params": label, "pearson_r": round(result.pearson_r, 4),
                            "spearman_r": round(result.spearman_r, 4),
                            "mae": round(result.mae, 3),
                            "n_features": result.n_features_used,
                            "best_alpha": round(result.best_alpha, 6),
                            "time_s": round(elapsed, 1)})
            print("  {}: r={:.4f}, MAE={:.3f}, n_feat={}".format(
                label, result.pearson_r, result.mae, result.n_features_used), flush=True)
    pd.DataFrame(results).to_csv(TABLE_DIR / "clock_sensitivity.csv", index=False)
    r_vals = [r["pearson_r"] for r in results]
    r_cv = np.std(r_vals) / (np.mean(r_vals) + 1e-10)
    labels = [label for _, label in param_sets]
    stable = "STABLE" if r_cv < 0.15 else "SENSITIVE"
    print("  r: mean={:.4f}, CV={:.3f}".format(np.mean(r_vals), r_cv))
    print("  Conclusion: {}".format(stable), flush=True)
    return dict(r_cv=r_cv, r_values=r_vals, labels=labels)


def test_age_permutation():
    print("\n" + "=" * 70)
    print("TEST D: Age permutation test (clock vs random)")
    print("=" * 70, flush=True)
    meta = build_tissue_sample_table()
    summary = pd.read_csv(TABLE_DIR / "clock_summary.csv")
    test_tissue = summary.iloc[0]["tissue"]
    obs_r = summary.iloc[0]["pearson_r"]
    print("  Testing: {} (obs r={:.4f})".format(test_tissue, obs_r), flush=True)
    fpath = str(GTEX_DIR / "tpm_{}.gct.gz".format(test_tissue))
    expr = read_gct(fpath)
    sub_meta = meta.set_index("SAMPID")
    sample_cols = [c for c in expr.columns if c != "gene_name" and c in sub_meta.index]
    actual_ages = sub_meta.loc[sample_cols, "AGE_NUM"].values.astype(float)
    valid = ~np.isnan(actual_ages)
    actual_ages = actual_ages[valid]
    rng = np.random.RandomState(RNG_SEED + 2)
    print("  Running {} age perms...".format(N_AGE_PERM), flush=True)
    perm_rs = np.zeros(N_AGE_PERM)
    for i in range(N_AGE_PERM):
        if i % 50 == 0:
            print("    perm {}/{}".format(i, N_AGE_PERM), flush=True)
        shuffled = rng.permutation(actual_ages)
        meta_perm = meta.copy()
        for idx, sid in enumerate(sample_cols):
            mask = meta_perm["SAMPID"] == sid
            if mask.any():
                meta_perm.loc[mask, "AGE_NUM"] = shuffled[idx]
        result = train_tissue_clock(test_tissue, expr, meta_perm)
        perm_rs[i] = result.pearson_r if result else 0.0
    p_val = float(np.mean(perm_rs >= obs_r))
    z = (obs_r - np.mean(perm_rs)) / (np.std(perm_rs) + 1e-10)
    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIG"
    print("  Observed r: {:.4f}".format(obs_r))
    print("  Permuted r: {:.4f} +/- {:.4f}".format(np.mean(perm_rs), np.std(perm_rs)))
    print("  Z-score: {:.2f}".format(z))
    print("  P-value: {}".format(p_val))
    print("  Conclusion: {}".format(sig), flush=True)
    pd.DataFrame({"perm_r": perm_rs}).to_csv(
        TABLE_DIR / "age_permutation_results.csv", index=False)
    return dict(obs_r=obs_r, perm_mean_r=float(np.mean(perm_rs)),
                perm_std_r=float(np.std(perm_rs)),                 z_score=z, p_value=p_val, perm_rs=perm_rs)


def plot_validation_figures(perm_res, boot_res, sens_res, age_res):
    print("\n" + "=" * 70)
    print("Generating validation figures...")
    print("=" * 70, flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    ax = axes[0, 0]
    ax.hist(perm_res["perm_mean_abs_corrs"], bins=40, color="lightgray",
            edgecolor="k", lw=0.3, alpha=0.7, label="Null (shuffled)")
    ax.axvline(perm_res["obs_mean_abs_corr"], color="red", lw=2,
               label="Observed (p={:.4f})".format(perm_res["p_mean"]))
    ax.axvline(perm_res["null_mean_abs_corr"], color="blue", ls="--", label="Null mean")
    ax.set_xlabel("Mean |correlation| across tissue pairs")
    ax.set_ylabel("Count")
    ax.set_title("A. Tissue Specificity vs Null (1000 perms)", fontweight="bold")
    ax.legend()
    ax = axes[0, 1]
    ax.hist(boot_res["rank_correlations"], bins=40, color="steelblue",
            edgecolor="k", lw=0.3, alpha=0.7)
    ax.axvline(boot_res["mean_rank_corr"], color="red", lw=2,
               label="Mean={:.3f}".format(boot_res["mean_rank_corr"]))
    ax.axvline(boot_res["rc_low"], color="orange", ls="--",
               label="95pct CI: {:.2f}-{:.2f}".format(boot_res["rc_low"], boot_res["rc_high"]))
    ax.axvline(boot_res["rc_high"], color="orange", ls="--")
    ax.set_xlabel("Spearman rank correlation")
    ax.set_ylabel("Count")
    ax.set_title("B. Drug Ranking Stability (1000 boots)", fontweight="bold")
    ax.legend()
    ax = axes[1, 0]
    labels = sens_res["labels"]
    r_vals = sens_res["r_values"]
    colors = ["#1f77b4" if "l1=" in l else "#ff7f0e" for l in labels]
    ax.barh(range(len(labels)), r_vals, color=colors, edgecolor="k", lw=0.3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Pearson r (liver clock)")
    ax.set_title("C. Clock Sensitivity to Parameters", fontweight="bold")
    ax.axvline(np.mean(r_vals), color="red", ls="--",
               label="Mean={:.3f}, CV={:.3f}".format(np.mean(r_vals), sens_res["r_cv"]))
    ax.legend()
    ax = axes[1, 1]
    ax.hist(age_res["perm_rs"], bins=40, color="lightgray",
            edgecolor="k", lw=0.3, alpha=0.7, label="Null (shuffled ages)")
    ax.axvline(age_res["obs_r"], color="red", lw=2,
               label="Observed (p={})".format(age_res["p_value"]))
    ax.axvline(age_res["perm_mean_r"], color="blue", ls="--", label="Null mean")
    ax.set_xlabel("Pearson r (age prediction)")
    ax.set_ylabel("Count")
    ax.set_title("D. Clock vs Random Age (200 perms)", fontweight="bold")
    ax.legend()
    plt.suptitle("Statistical Validation of Core Claims",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIGURE_DIR / "fig5_statistical_validation.png", dpi=200,
                bbox_inches="tight")
    plt.close()
    print("  fig5_statistical_validation.png")


def main():
    print("\n" + "=" * 70)
    print("STEP 7: STATISTICAL VALIDATION")
    print("=" * 70 + "\n", flush=True)
    perm_results = test_tissue_correlation()
    boot_results = test_bootstrap_stability()
    sens_results = test_clock_sensitivity()
    age_perm_results = test_age_permutation()
    plot_validation_figures(perm_results, boot_results, sens_results, age_perm_results)
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print("Test A - Tissue-tissue correlation:")
    print("  Observed mean |r|: {:.4f}".format(perm_results["obs_mean_abs_corr"]))
    print("  Null mean |r|: {:.4f} +/- {:.4f}".format(
        perm_results["null_mean_abs_corr"], perm_results["null_std_abs_corr"]))
    print("  Z-score: {:.2f}, P-value: {}".format(perm_results["z_score"], perm_results["p_mean"]))
    print()
    print("Test B - Bootstrap stability:")
    print("  Rank correlation: {:.3f} (CI: {:.3f}-{:.3f})".format(
        boot_results["mean_rank_corr"], boot_results["rc_low"], boot_results["rc_high"]))
    print()
    print("Test C - Clock sensitivity:")
    print("  CV across params: {:.3f}".format(sens_results["r_cv"]))
    print()
    print("Test D - Age permutation:")
    print("  Observed r: {:.4f}".format(age_perm_results["obs_r"]))
    print("  Permuted r: {:.4f} +/- {:.4f}".format(
        age_perm_results["perm_mean_r"], age_perm_results["perm_std_r"]))
    print("  Z-score: {:.2f}, P-value: {}".format(age_perm_results["z_score"], age_perm_results["p_value"]))
    print("\nStep 7 complete.")


if __name__ == "__main__":
    main()
