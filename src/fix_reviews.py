"""Fix 1+2: Re-analyze drug heterogeneity with significance filtering
and gene sharing with null model comparison."""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from config import TABLE_DIR, CLOCK_DIR, GTEX_DIR, FIGURE_DIR
from gct_io import read_gct

TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# FIX 1: Drug heterogeneity with significance filtering
# ================================================================
def fix_drug_heterogeneity():
    print("=" * 70)
    print("FIX 1: Drug heterogeneity with significance filtering")
    print("=" * 70, flush=True)

    matrix = pd.read_csv(TABLE_DIR / "drug_tissue_reversal_matrix.csv", index_col=0)
    all_scores = matrix.values.flatten()

    # Define significance tiers
    tiers = [
        ("|score| > 0 (all)", 0.0),
        ("|score| > 0.5", 0.5),
        ("|score| > 1.0", 1.0),
        ("|score| > 2.0", 2.0),
    ]

    print("\nScore distribution by threshold:")
    for name, thresh in tiers:
        n = (np.abs(all_scores) > thresh).sum()
        print("  {}: {}/{} ({:.1f}%)".format(name, n, len(all_scores), n/len(all_scores)*100))

    # Recompute heterogeneity at each threshold
    print("\nDrug heterogeneity at each threshold:")
    results = []
    for name, thresh in tiers:
        n_reju = (matrix > thresh).sum(axis=1)
        n_pro = (matrix < -thresh).sum(axis=1)
        n_nonsig = len(matrix.columns) - n_reju - n_pro

        # "Mixed" = has both significant reju AND significant pro
        is_mixed = ((n_reju > 0) & (n_pro > 0)).astype(int)
        # "Pure reju" = has reju but no pro
        is_pure_reju = ((n_reju > 0) & (n_pro == 0)).astype(int)
        # "Pure pro" = has pro but no reju
        is_pure_pro = ((n_pro > 0) & (n_reju == 0)).astype(int)
        # "Inactive" = neither
        is_inactive = ((n_reju == 0) & (n_pro == 0)).astype(int)

        n_mixed = is_mixed.sum()
        n_total = len(matrix)

        print("  {}: mixed={}/{} ({:.1f}%), pure_reju={}, pure_pro={}, inactive={}".format(
            name, n_mixed, n_total, n_mixed/n_total*100,
            is_pure_reju.sum(), is_pure_pro.sum(), is_inactive.sum()))

        results.append({
            "threshold": name,
            "n_mixed": n_mixed,
            "pct_mixed": round(n_mixed/n_total*100, 1),
            "n_pure_reju": int(is_pure_reju.sum()),
            "n_pure_pro": int(is_pure_pro.sum()),
            "n_inactive": int(is_inactive.sum()),
            "n_total": n_total,
        })

    tier_df = pd.DataFrame(results)
    tier_df.to_csv(TABLE_DIR / "heterogeneity_by_threshold.csv", index=False)

    # Use |score| > 1.0 as the recommended threshold for main text
    thresh = 1.0
    n_reju = (matrix > thresh).sum(axis=1)
    n_pro = (matrix < -thresh).sum(axis=1)
    is_mixed = ((n_reju > 0) & (n_pro > 0)).astype(int)
    n_nonsig = len(matrix.columns) - n_reju - n_pro

    het_v2 = pd.DataFrame({
        "compound": matrix.index,
        "n_rejuvenated_sig": n_reju.values,
        "n_pro_aging_sig": n_pro.values,
        "n_nonsignificant": n_nonsig.values,
        "is_mixed_sig": is_mixed.values,
        "mean_reversal": matrix.mean(axis=1).values,
        "mean_reversal_sig": matrix[abs(matrix) > thresh].mean(axis=1).fillna(0).values,
    })
    het_v2 = het_v2.sort_values("mean_reversal", ascending=False)
    het_v2.to_csv(TABLE_DIR / "drug_heterogeneity_sig_filtered.csv", index=False)

    print("\n  Recommended threshold: |score| > 1.0")
    print("  Mixed drugs: {}/{} ({:.1f}%)".format(
        is_mixed.sum(), len(matrix), is_mixed.sum()/len(matrix)*100))
    print("  (Original claim 99.7%, corrected: {:.1f}%)".format(
        is_mixed.sum()/len(matrix)*100))
    print("  Saved: drug_heterogeneity_sig_filtered.csv")

    return tier_df


# ================================================================
# FIX 2: Gene sharing with null model comparison
# ================================================================
def fix_gene_sharing():
    print("\n" + "=" * 70)
    print("FIX 2: Gene sharing with null model comparison")
    print("=" * 70, flush=True)

    wm = pd.read_csv(TABLE_DIR / "clock_weight_matrix.csv", index_col=0)
    if "gene_symbol" in wm.columns:
        wm = wm.drop(columns=["gene_symbol"])

    # Observed sharing
    gene_sharing = (wm != 0).sum(axis=1)
    obs_counts = gene_sharing.value_counts().sort_index()

    n_tissues = wm.shape[1]
    avg_genes = int((wm != 0).sum().mean())
    n_universe = len(wm)

    print("  Tissues: {}".format(n_tissues))
    print("  Avg genes per tissue: {}".format(avg_genes))
    print("  Universe size: {}".format(n_universe))
    print("  Observed genes shared >=50% (24): {}".format(
        (gene_sharing >= 24).sum()))

    # Null simulation: randomly select avg_genes from universe, n_tissues times
    rng = np.random.RandomState(42)
    n_sim = 1000
    null_shared_24 = np.zeros(n_sim)
    null_shared_10 = np.zeros(n_sim)
    null_max_sharing = np.zeros(n_sim)

    print("  Running {} null simulations...".format(n_sim), flush=True)
    for i in range(n_sim):
        all_genes_sim = []
        for _ in range(n_tissues):
            # Each tissue selects a slightly different number
            n_sel = rng.poisson(avg_genes)
            n_sel = min(n_sel, n_universe)
            genes = rng.choice(n_universe, n_sel, replace=False)
            all_genes_sim.extend(genes)
        counts = Counter(all_genes_sim)
        null_shared_24[i] = sum(1 for c in counts.values() if c >= 24)
        null_shared_10[i] = sum(1 for c in counts.values() if c >= 10)
        null_max_sharing[i] = max(counts.values()) if counts else 0

    print("\n  Null model results (random gene selection):")
    print("  Shared >=24: mean={:.1f}, max={:.0f}".format(
        null_shared_24.mean(), null_shared_24.max()))
    print("  Shared >=10: mean={:.1f}, max={:.0f}".format(
        null_shared_10.mean(), null_shared_10.max()))
    print("  Max sharing: mean={:.1f}, max={:.0f}".format(
        null_max_sharing.mean(), null_max_sharing.max()))

    obs_max = gene_sharing.max()
    obs_shared_10 = (gene_sharing >= 10).sum()
    p_24 = float(np.mean(null_shared_24 <= (gene_sharing >= 24).sum()))
    p_10 = float(np.mean(null_shared_10 <= obs_shared_10))

    print("\n  Comparison:")
    print("  Observed max sharing: {}".format(obs_max))
    print("  Null max sharing: {:.1f} +/- {:.1f}".format(
        null_max_sharing.mean(), null_max_sharing.std()))
    print("  P-value (>=24): {:.3f}".format(p_24))
    print("  P-value (>=10): {:.3f}".format(p_10))

    print("\n  CONCLUSION:")
    if p_24 > 0.05:
        print("  Zero shared genes is CONSISTENT with null model (P={:.3f})".format(p_24))
        print("  Cannot claim biological tissue-specificity from gene overlap alone")
        print("  -> Must change manuscript conclusion wording")
    else:
        print("  Zero shared genes is SIGNIFICANTLY different from null (P={:.3f})".format(p_24))

    # Save results
    null_df = pd.DataFrame({
        "null_shared_24": null_shared_24,
        "null_shared_10": null_shared_10,
        "null_max_sharing": null_max_sharing,
    })
    null_df.to_csv(TABLE_DIR / "gene_sharing_null_model.csv", index=False)

    comparison = pd.DataFrame({
        "metric": ["genes_shared_>=24", "genes_shared_>=10", "max_sharing"],
        "observed": [(gene_sharing >= 24).sum(), obs_shared_10, obs_max],
        "null_mean": [null_shared_24.mean(), null_shared_10.mean(), null_max_sharing.mean()],
        "null_std": [null_shared_24.std(), null_shared_10.std(), null_max_sharing.std()],
        "p_value": [p_24, p_10, float(np.mean(null_max_sharing <= obs_max))],
    })
    comparison.to_csv(TABLE_DIR / "gene_sharing_comparison.csv", index=False)

    print("  Saved: gene_sharing_null_model.csv, gene_sharing_comparison.csv")
    return comparison


# ================================================================
# FIX 5: Gene coverage analysis
# ================================================================
def fix_gene_coverage():
    print("\n" + "=" * 70)
    print("FIX 5: Gene coverage analysis")
    print("=" * 70, flush=True)

    import gzip
    lincs_genes = set()
    with gzip.open(r"H:\lunwencp\shuailao\data\lincs\GSE92742_Broad_LINCS_gene_info.txt.gz", "rt") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                lincs_genes.add(parts[1])

    wm = pd.read_csv(TABLE_DIR / "clock_weight_matrix.csv", index_col=0)
    gene_sym_col = None
    if "gene_symbol" in wm.columns:
        gene_sym_col = wm["gene_symbol"]
        wm = wm.drop(columns=["gene_symbol"])

    clock_genes = set(gene_sym_col.values) if gene_sym_col is not None else set(wm.index)
    clock_genes = {g for g in clock_genes if g and g != "nan"}

    overlap = clock_genes & lincs_genes
    excluded = clock_genes - lincs_genes

    print("  Total clock genes: {}".format(len(clock_genes)))
    print("  In LINCS universe: {} ({:.1f}%)".format(
        len(overlap), len(overlap)/len(clock_genes)*100))
    print("  Excluded from LINCS: {} ({:.1f}%)".format(
        len(excluded), len(excluded)/len(clock_genes)*100))

    # Per-tissue coverage
    cs = pd.read_csv(TABLE_DIR / "clock_summary.csv")
    coverage = []
    for _, row in cs.iterrows():
        tissue = row["tissue"]
        coef = pd.read_csv(CLOCK_DIR / "{}_coefficients.csv".format(tissue), index_col=0)
        tissue_syms = set(gene_sym_col.get(g, "") for g in coef.index) - {""} if gene_sym_col is not None else set()
        in_lincs = tissue_syms & lincs_genes
        pct = len(in_lincs)/len(tissue_syms)*100 if tissue_syms else 0
        coverage.append({
            "tissue": tissue,
            "n_clock_genes": len(tissue_syms),
            "n_in_lincs": len(in_lincs),
            "n_excluded": len(tissue_syms) - len(in_lincs),
            "pct_excluded": round(100 - pct, 1),
        })

    cov_df = pd.DataFrame(coverage)
    cov_df.to_csv(TABLE_DIR / "gene_coverage_analysis.csv", index=False)
    print("\n  Per-tissue coverage (top 10 worst):")
    print(cov_df.nlargest(10, "pct_excluded").to_string(index=False))
    print("\n  Mean excluded: {:.1f}%".format(cov_df["pct_excluded"].mean()))

    print("  Saved: gene_coverage_analysis.csv")
    return cov_df


if __name__ == "__main__":
    tier_df = fix_drug_heterogeneity()
    sharing_df = fix_gene_sharing()
    cov_df = fix_gene_coverage()

    print("\n" + "=" * 70)
    print("ALL FIXES COMPLETE")
    print("=" * 70)
    print("Fix 1: Drug heterogeneity re-analyzed with significance tiers")
    print("Fix 2: Gene sharing null model comparison done")
    print("Fix 5: Gene coverage analysis done")
    print("\nNext: Update manuscript text with corrected numbers")
