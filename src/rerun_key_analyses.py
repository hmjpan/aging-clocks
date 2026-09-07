"""Rerun key analyses for manuscript revision:
1. Age permutation x1000 (artery_aorta, fixed-alpha fast version)
2. Mixed-fraction permutation null test (|score|>1.0)
3. DepMap standardization fix + rerun aging scores
4. r>=0.5 (34 tissues) drug-ranking sensitivity
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from config import TABLE_DIR, CLOCK_DIR, GTEX_DIR, DEPMAP_DIR
from gct_io import read_gct, build_tissue_sample_table
from sklearn.linear_model import ElasticNet

rng = np.random.RandomState(42)

# ================================================================
# 1. Age permutation x1000 (fixed alpha, fast)
# ================================================================
def run_age_permutation_1000():
    print("=" * 70)
    print("RERUN 1: Age permutation x1000 (artery_aorta)")
    print("=" * 70, flush=True)

    tissue = "artery_aorta"
    meta = build_tissue_sample_table()
    fpath = str(GTEX_DIR / "tpm_{}.gct.gz".format(tissue))
    expr = read_gct(fpath)

    sample_cols = [c for c in expr.columns
                   if c != "gene_name" and c in set(meta["SAMPID"])]
    sub_meta = meta.set_index("SAMPID").loc[sample_cols]
    age = sub_meta["AGE_NUM"].values.astype(float)
    donors = sub_meta["SUBJID"].values
    valid = ~np.isnan(age)
    sample_cols = [s for s, v in zip(sample_cols, valid) if v]
    age = age[valid]
    donors = donors[valid]

    # Same preprocessing as training (fast version, fixed feature set)
    mat = expr.loc[:, sample_cols]
    present = (mat >= 1.0).mean(axis=1)
    mat = mat.loc[present[present >= 0.10].index]
    var = mat.var(axis=1)
    mat = mat.loc[var.nlargest(3000).index]
    X = np.log2(mat.values.astype(np.float64) + 1.0).T
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    y = age.copy()

    # Fixed alpha from original training (best_alpha for artery_aorta = 0.32)
    alpha = 0.32
    l1_ratio = 0.5

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    cv_splits = list(gkf.split(Xz, y, donors))

    def oof_pearson(y_shuffled):
        pred = np.full(len(y_shuffled), np.nan)
        for tr, te in cv_splits:
            m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
            m.fit(Xz[tr], y_shuffled[tr])
            pred[te] = m.predict(Xz[te])
        v = ~np.isnan(pred)
        if np.std(pred[v]) == 0 or np.std(y_shuffled[v]) == 0:
            return 0.0
        return pearsonr(pred[v], y_shuffled[v])[0]

    # Observed
    obs_r = oof_pearson(y)
    print("  Observed r: {:.4f}".format(obs_r), flush=True)

    n_perm = 1000
    perm_rs = np.zeros(n_perm)
    t0 = time.time()
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        perm_rs[i] = oof_pearson(y_shuf)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print("  {}/{} ({:.0f}s, est remaining {:.0f}s)".format(
                i + 1, n_perm, el, el / (i + 1) * (n_perm - i - 1)), flush=True)

    p_val = float(np.mean(perm_rs >= obs_r))
    z = (obs_r - perm_rs.mean()) / (perm_rs.std() + 1e-10)

    print("\n  RESULT:")
    print("  Observed r: {:.4f}".format(obs_r))
    print("  Permuted r: {:.4f} +/- {:.4f}".format(perm_rs.mean(), perm_rs.std()))
    print("  Z-score: {:.2f}".format(z))
    print("  Empirical p (1000 perms): {}".format(p_val))
    print("  Conservative p (n+1): {:.4f}".format((p_val * n_perm + 1) / (n_perm + 1)))

    pd.DataFrame({"perm_r": perm_rs}).to_csv(
        TABLE_DIR / "age_permutation_results_1000.csv", index=False)
    print("  Saved: age_permutation_results_1000.csv")
    return obs_r, perm_rs, p_val, z


# ================================================================
# 2. Mixed-fraction permutation null (|score|>1.0)
# ================================================================
def run_mixed_fraction_null():
    print("\n" + "=" * 70)
    print("RERUN 2: Mixed-fraction permutation null (94.2%)")
    print("=" * 70, flush=True)

    matrix = pd.read_csv(TABLE_DIR / "drug_tissue_reversal_matrix.csv", index_col=0)
    mv = matrix.values  # drugs x tissues
    n_drugs, n_tissues = mv.shape
    thresh = 1.0

    def mixed_fraction(mat):
        n_reju = (mat > thresh).sum(axis=1)
        n_pro = (mat < -thresh).sum(axis=1)
        return float(((n_reju > 0) & (n_pro > 0)).mean())

    obs = mixed_fraction(mv)
    print("  Observed mixed fraction (|score|>1.0): {:.4f} ({:.1f}%)".format(obs, obs * 100))

    n_perm = 1000
    null_frac = np.zeros(n_perm)
    for i in range(n_perm):
        pm = np.column_stack([rng.permutation(mv[:, j]) for j in range(n_tissues)])
        null_frac[i] = mixed_fraction(pm)
        if (i + 1) % 200 == 0:
            print("  perm {}/{}".format(i + 1, n_perm), flush=True)

    p = float(np.mean(null_frac >= obs))
    print("\n  Null mixed fraction: {:.4f} +/- {:.4f}".format(null_frac.mean(), null_frac.std()))
    print("  Null range: [{:.4f}, {:.4f}]".format(null_frac.min(), null_frac.max()))
    print("  Empirical p: {}".format(p))
    print("  CONCLUSION: observed {} vs null {:.3f} -> {}".format(
        obs, null_frac.mean(), "SIGNIFICANTLY ABOVE NULL" if p < 0.05 else "NOT ABOVE NULL"))

    pd.DataFrame({"null_mixed_fraction": null_frac}).to_csv(
        TABLE_DIR / "mixed_fraction_null.csv", index=False)
    print("  Saved: mixed_fraction_null.csv")
    return obs, null_frac, p


# ================================================================
# 3. DepMap standardization fix
# ================================================================
def run_depmap_standardized():
    print("\n" + "=" * 70)
    print("RERUN 3: DepMap aging scores with GTEx standardization")
    print("=" * 70, flush=True)

    meta = build_tissue_sample_table()
    models = pd.read_csv(DEPMAP_DIR / "Model.csv")
    lineage_map = {
        "lung": "lung", "breast": "breast_mammary_tissue",
        "skin": "skin_sun_exposed_lower_leg", "blood": "whole_blood",
        "colon": "colon_transverse", "liver": "liver", "brain": "brain_cortex",
        "prostate": "prostate", "pancreas": "pancreas", "kidney": "kidney_cortex",
        "ovary": "ovary", "stomach": "stomach", "thyroid": "thyroid",
        "uterus": "uterus", "bone": "muscle_skeletal",
    }
    cell_tissue = {}
    for _, row in models.iterrows():
        lin = str(row.get("OncotreeLineage", "")).lower()
        mid = row.get("ModelID", "")
        for key, tissue in lineage_map.items():
            if key in lin and tissue != "":
                cell_tissue[mid] = tissue
                break

    expr_df = pd.read_csv(DEPMAP_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
    expr_df.columns = [c.split(" (")[0] if " (" in c else c for c in expr_df.columns]

    # Gene map for Ensembl -> symbol
    fpath = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))[0]
    gtex_expr = read_gct(str(fpath))
    gene_map = dict(zip(gtex_expr.index, gtex_expr["gene_name"]))

    results = []
    for tissue in sorted(set(cell_tissue.values())):
        coef_path = CLOCK_DIR / "{}_coefficients.csv".format(tissue)
        if not coef_path.exists():
            continue
        coef = pd.read_csv(coef_path, index_col=0)
        coef.columns = ["weight"]
        if len(coef) < 10:
            continue

        # Recompute GTEx training mu/sd for these genes (same pipeline as training)
        # Load tissue GCT, filter, log2
        tfpath = str(GTEX_DIR / "tpm_{}.gct.gz".format(tissue))
        if not os.path.exists(tfpath):
            continue
        texpr = read_gct(tfpath)
        tsamples = [c for c in texpr.columns if c != "gene_name" and c in set(meta["SAMPID"])]
        if len(tsamples) < 10:
            continue
        tmat = texpr.loc[:, tsamples]
        present = (tmat >= 1.0).mean(axis=1)
        tmat = tmat.loc[present[present >= 0.10].index]
        X_t = np.log2(tmat.values.astype(np.float64) + 1.0)
        mu_g = X_t.mean(axis=1)
        sd_g = X_t.std(axis=1)
        sd_g[sd_g == 0] = 1.0
        gene_mu_sd = dict(zip(tmat.index, zip(mu_g, sd_g)))

        # Map clock genes to symbols, keep those with mu/sd
        mapped = {}
        for ensg, w in coef["weight"].items():
            sym = gene_map.get(ensg, "")
            if sym and sym in gene_mu_sd and sym in expr_df.columns:
                mapped[sym] = (w, gene_mu_sd[sym][0], gene_mu_sd[sym][1])

        if len(mapped) < 10:
            continue

        matching = [c for c, t in cell_tissue.items() if t == tissue and c in expr_df.index]
        if len(matching) < 5:
            continue

        # Standardized projection: (depmap_expr - mu)/sd * weight
        scores = np.zeros(len(matching))
        for j, cell in enumerate(matching):
            s = 0.0
            for sym, (w, mu, sd) in mapped.items():
                val = expr_df.loc[cell, sym]
                s += ((val - mu) / sd) * w
            scores[j] = s

        results.append({
            "tissue": tissue,
            "n_cells": len(matching),
            "n_genes_used": len(mapped),
            "aging_score_mean": round(float(np.mean(scores)), 4),
            "aging_score_std": round(float(np.std(scores)), 4),
        })
        print("  {}: {} cells, {} genes, score {:.2f}+/-{:.2f}".format(
            tissue, len(matching), len(mapped), np.mean(scores), np.std(scores)), flush=True)

    df = pd.DataFrame(results)
    df.to_csv(TABLE_DIR / "depmap_standardized_scores.csv", index=False)
    print("  Saved: depmap_standardized_scores.csv")
    return df


# ================================================================
# 4. r>=0.5 sensitivity (drug ranking)
# ================================================================
def run_highperf_sensitivity():
    print("\n" + "=" * 70)
    print("RERUN 4: r>=0.5 (34 tissues) drug-ranking sensitivity")
    print("=" * 70, flush=True)

    cs = pd.read_csv(TABLE_DIR / "clock_summary.csv")
    matrix = pd.read_csv(TABLE_DIR / "drug_tissue_reversal_matrix.csv", index_col=0)

    hi = cs[cs["pearson_r"] >= 0.5]["tissue"].tolist()
    print("  Tissues with r>=0.5: {} (of {})".format(len(hi), len(cs)))

    full_mean = matrix.mean(axis=1)
    hi_mean = matrix[hi].mean(axis=1)

    rho, p = spearmanr(full_mean, hi_mean)
    print("  Drug mean-reversal ranking: full-49 vs r>=0.5 subset")
    print("  Spearman rho: {:.4f} (p={:.2e})".format(rho, p))

    # Also compare heterogeneity (n_mixed) at threshold
    thresh = 1.0
    full_mixed = ((matrix > thresh).sum(axis=1) > 0) & ((matrix < -thresh).sum(axis=1) > 0)
    hi_mixed = ((matrix[hi] > thresh).sum(axis=1) > 0) & ((matrix[hi] < -thresh).sum(axis=1) > 0)
    print("  Mixed fraction full-49: {:.1f}%".format(full_mixed.mean() * 100))
    print("  Mixed fraction r>=0.5 subset: {:.1f}%".format(hi_mixed.mean() * 100))

    res = pd.DataFrame({
        "metric": ["drug_ranking_rho", "mixed_full", "mixed_r0.5"],
        "value": [rho, full_mixed.mean(), hi_mixed.mean()],
    })
    res.to_csv(TABLE_DIR / "highperf_sensitivity.csv", index=False)
    print("  Saved: highperf_sensitivity.csv")
    return rho, hi


if __name__ == "__main__":
    # Rerun 2 and 4 first (fast), then 3 (medium), then 1 (slow, x1000)
    print("### Fast analyses first ###")
    obs2, null2, p2 = run_mixed_fraction_null()
    rho4, hi4 = run_highperf_sensitivity()
    dep3 = run_depmap_standardized()
    print("\n### Slow analysis (age perm x1000) ###")
    obs1, perm1, p1, z1 = run_age_permutation_1000()
    print("\nDONE ALL RERUNS")