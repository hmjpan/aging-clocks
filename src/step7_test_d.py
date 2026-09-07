"""Quick Test D: age permutation (10 perms) + fig5 generation."""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import TABLE_DIR, FIGURE_DIR, GTEX_DIR
from gct_io import read_gct, build_tissue_sample_table
from clock import train_tissue_clock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = TABLE_DIR

# === Test D: 10 age permutations ===
print("TEST D: Age permutation (10 perms)", flush=True)
meta = build_tissue_sample_table()
summary = pd.read_csv(T / "clock_summary.csv")
test_tissue = summary.iloc[0]["tissue"]
obs_r = summary.iloc[0]["pearson_r"]
print("  Tissue: {} (obs r={:.4f})".format(test_tissue, obs_r), flush=True)

fpath = str(GTEX_DIR / "tpm_{}.gct.gz".format(test_tissue))
expr = read_gct(fpath)
sub_meta = meta.set_index("SAMPID")
sample_cols = [c for c in expr.columns if c != "gene_name" and c in sub_meta.index]
actual_ages = sub_meta.loc[sample_cols, "AGE_NUM"].values.astype(float)
valid = ~np.isnan(actual_ages)
actual_ages = actual_ages[valid]

rng = np.random.RandomState(44)
n_perm = 10
perm_rs = np.zeros(n_perm)
for i in range(n_perm):
    t0 = time.time()
    shuffled = rng.permutation(actual_ages)
    meta_perm = meta.copy()
    for idx, sid in enumerate(sample_cols):
        mask = meta_perm["SAMPID"] == sid
        if mask.any():
            meta_perm.loc[mask, "AGE_NUM"] = shuffled[idx]
    result = train_tissue_clock(test_tissue, expr, meta_perm)
    perm_rs[i] = result.pearson_r if result else 0.0
    print("  perm {}/{}: r={:.4f} ({:.0f}s)".format(i+1, n_perm, perm_rs[i], time.time()-t0), flush=True)

p_val = float(np.mean(perm_rs >= obs_r))
z = (obs_r - np.mean(perm_rs)) / (np.std(perm_rs) + 1e-10)
print("  RESULT: obs={:.4f}, perm={:.4f}+/-{:.4f}, z={:.2f}, p={}".format(
    obs_r, np.mean(perm_rs), np.std(perm_rs), z, p_val), flush=True)
pd.DataFrame({"perm_r": perm_rs}).to_csv(T / "age_permutation_results.csv", index=False)

# === Generate fig5 ===
print("\nGenerating fig5...", flush=True)
perm_null = pd.read_csv(T / "permutation_null_distribution.csv")
boot_res = pd.read_csv(T / "bootstrap_stability.csv")
sens_df = pd.read_csv(T / "clock_sensitivity.csv")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel A
ax = axes[0, 0]
ax.hist(perm_null["perm_mean_abs_corr"], bins=40, color="lightgray",
        edgecolor="k", lw=0.3, alpha=0.7, label="Null (shuffled)")
obs_mean = 0.0566
null_mean = perm_null["perm_mean_abs_corr"].mean()
ax.axvline(obs_mean, color="red", lw=2, label="Observed (p<0.001)")
ax.axvline(null_mean, color="blue", ls="--", label="Null mean")
ax.set_xlabel("Mean |correlation| across tissue pairs")
ax.set_ylabel("Count")
ax.set_title("A. Tissue Specificity vs Null (1000 perms)", fontweight="bold")
ax.legend()

# Panel B
ax = axes[0, 1]
ax.hist(boot_res["rank_correlation"], bins=40, color="steelblue",
        edgecolor="k", lw=0.3, alpha=0.7)
mrc = boot_res["rank_correlation"].mean()
rcl = boot_res["rank_correlation"].quantile(0.025)
rch = boot_res["rank_correlation"].quantile(0.975)
ax.axvline(mrc, color="red", lw=2, label="Mean={:.3f}".format(mrc))
ax.axvline(rcl, color="orange", ls="--", label="95pct CI: {:.2f}-{:.2f}".format(rcl, rch))
ax.axvline(rch, color="orange", ls="--")
ax.set_xlabel("Spearman rank correlation")
ax.set_ylabel("Count")
ax.set_title("B. Drug Ranking Stability (1000 boots)", fontweight="bold")
ax.legend()

# Panel C
ax = axes[1, 0]
labels_list = sens_df["params"].tolist()
r_vals = sens_df["pearson_r"].tolist()
colors = ["#1f77b4" if "l1=" in l else "#ff7f0e" for l in labels_list]
ax.barh(range(len(labels_list)), r_vals, color=colors, edgecolor="k", lw=0.3)
ax.set_yticks(range(len(labels_list)))
ax.set_yticklabels(labels_list, fontsize=8)
ax.set_xlabel("Pearson r (liver clock)")
ax.set_title("C. Clock Sensitivity to Parameters", fontweight="bold")
r_cv = sens_df["pearson_r"].std() / sens_df["pearson_r"].mean()
ax.axvline(np.mean(r_vals), color="red", ls="--",
           label="Mean={:.3f}, CV={:.3f}".format(np.mean(r_vals), r_cv))
ax.legend()

# Panel D
ax = axes[1, 1]
ax.hist(perm_rs, bins=10, color="lightgray",
        edgecolor="k", lw=0.3, alpha=0.7, label="Null (shuffled ages)")
ax.axvline(obs_r, color="red", lw=2, label="Observed (p={})".format(p_val))
ax.axvline(np.mean(perm_rs), color="blue", ls="--", label="Null mean")
ax.set_xlabel("Pearson r (age prediction)")
ax.set_ylabel("Count")
ax.set_title("D. Clock vs Random Age (10 perms)", fontweight="bold")
ax.legend()

plt.suptitle("Statistical Validation of Core Claims",
             fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIGURE_DIR / "fig5_statistical_validation.png", dpi=200,
            bbox_inches="tight")
plt.close()
print("  fig5_statistical_validation.png saved")
print("\nAll validation tests complete.")
