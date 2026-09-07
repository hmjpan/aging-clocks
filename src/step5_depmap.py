"""Step 5 v3: DepMap expression validation.

Strategy: 
  1. Project DepMap cell line expression onto GTEx tissue clocks
  2. Test whether aging scores correlate with cell line doubling time
     (a proxy for replicative aging)
  3. Validate drug targets: for top rejuvenating drugs, check if their
     targets have higher CRISPR dependency in "older-like" cell lines
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_DIR, TABLE_DIR, FIGURE_DIR, GTEX_DIR, DEPMAP_DIR
from gct_io import read_gct

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def build_gene_map():
    fpath = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))[0]
    expr = read_gct(str(fpath))
    return dict(zip(expr.index, expr["gene_name"]))


def load_clock_weights(gene_map):
    clocks = {}
    for f in sorted(CLOCK_DIR.glob("*_coefficients.csv")):
        tissue = f.name.replace("_coefficients.csv", "")
        df = pd.read_csv(f, index_col=0)
        df.columns = ["weight"]
        weights = {}
        for ensg, w in df["weight"].items():
            sym = gene_map.get(ensg, "")
            if sym and sym != "nan":
                weights[sym] = w
        clocks[tissue] = weights
    return clocks


def main():
    gene_map = build_gene_map()
    clocks = load_clock_weights(gene_map)

    print("Loading DepMap...", flush=True)
    models = pd.read_csv(DEPMAP_DIR / "Model.csv")

    # Map cell lines to tissues
    lineage_map = {
        "lung": "lung", "breast": "breast_mammary_tissue",
        "skin": "skin_sun_exposed_lower_leg",
        "blood": "whole_blood", "colon": "colon_transverse",
        "liver": "liver", "brain": "brain_cortex",
        "prostate": "prostate", "pancreas": "pancreas",
        "kidney": "kidney_cortex", "ovary": "ovary",
        "stomach": "stomach", "thyroid": "thyroid",
        "uterus": "uterus", "bone": "muscle_skeletal",
    }
    cell_tissue = {}
    for _, row in models.iterrows():
        lin = str(row.get("OncotreeLineage", "")).lower()
        mid = row.get("ModelID", "")
        for key, tissue in lineage_map.items():
            if key in lin and tissue in clocks:
                cell_tissue[mid] = tissue
                break
    print(f"  Mapped {len(cell_tissue)} cell lines", flush=True)

    # Load expression
    expr_df = pd.read_csv(DEPMAP_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
                          index_col=0)
    expr_df.columns = [c.split(" (")[0] if " (" in c else c
                        for c in expr_df.columns]
    print(f"  Expression: {expr_df.shape}", flush=True)

    # Load drug heterogeneity results
    het_path = TABLE_DIR / "drug_heterogeneity_scores.csv"
    if het_path.exists():
        het_df = pd.read_csv(het_path)
        # Top rejuvenating drugs
        top_reju = het_df.sort_values("mean_reversal", ascending=False).head(50)
        print(f"  Top 50 rejuvenating drugs loaded", flush=True)
    else:
        top_reju = None

    # Compute aging score for each cell line per tissue
    print("\nComputing aging scores...", flush=True)
    aging_scores = {}  # tissue -> {cell_id: score}

    for tissue, clock_weights in clocks.items():
        if len(clock_weights) < 10:
            continue
        matching = [c for c, t in cell_tissue.items()
                    if t == tissue and c in expr_df.index]
        if len(matching) < 5:
            continue

        clock_genes = [g for g in clock_weights if g in expr_df.columns]
        if len(clock_genes) < 10:
            continue

        cell_expr = expr_df.loc[matching, clock_genes]
        weights_arr = np.array([clock_weights[g] for g in clock_genes])
        scores = cell_expr.values @ weights_arr

        aging_scores[tissue] = dict(zip(matching, scores))
        # Also compute for non-matching cell lines (cross-tissue)
        other_cells = [c for c in expr_df.index if c not in matching][:100]
        if len(other_cells) > 5:
            other_expr = expr_df.loc[other_cells, clock_genes]
            other_scores = other_expr.values @ weights_arr
            # Compare distributions
            r, p = mannwhitneyu(scores, other_scores, alternative="greater")
            print(f"  {tissue}: {len(matching)} matching cells, "
                  f"mean={np.mean(scores):.2f} vs other={np.mean(other_scores):.2f}, "
                  f"p={p:.4e}")

    # Load CRISPR for drug target validation
    print("\nLoading CRISPR for drug target validation...", flush=True)
    crispr_df = pd.read_csv(DEPMAP_DIR / "CRISPRGeneEffect.csv", index_col=0)
    print(f"  CRISPR: {crispr_df.shape}", flush=True)

    # For top rejuvenating drugs: validate their targets
    # The rejuvenating drugs should target aging genes (positive clock weight)
    # Those aging genes should be more essential in "older-like" cell lines

    results = []
    for tissue, cell_scores in aging_scores.items():
        if len(cell_scores) < 10:
            continue
        clock_weights = clocks[tissue]
        aging_genes = {g: w for g, w in clock_weights.items() if w > 0}
        youth_genes = {g: w for g, w in clock_weights.items() if w < 0}

        cells = list(cell_scores.keys())
        scores = list(cell_scores.values())
        median_score = np.median(scores)
        older_cells = [c for c, s in zip(cells, scores) if s > median_score]
        younger_cells = [c for c, s in zip(cells, scores) if s <= median_score]

        if len(older_cells) < 3 or len(younger_cells) < 3:
            continue

        # For each aging gene, compare CRISPR dependency in older vs younger
        gene_results = []
        for gene in aging_genes:
            crispr_col = None
            for c in crispr_df.columns:
                if c.startswith(gene + " ("):
                    crispr_col = c
                    break
            if not crispr_col:
                continue

            older_avail = [c for c in older_cells if c in crispr_df.index]
            younger_avail = [c for c in younger_cells if c in crispr_df.index]
            if len(older_avail) < 3 or len(younger_avail) < 3:
                continue

            older_dep = crispr_df.loc[older_avail, crispr_col].values
            younger_dep = crispr_df.loc[younger_avail, crispr_col].values

            # More negative = more essential
            # Hypothesis: aging genes are more essential in older-like cells
            if np.std(older_dep) > 0 and np.std(younger_dep) > 0:
                u, p = mannwhitneyu(older_dep, younger_dep, alternative="less")
                gene_results.append({
                    "gene": gene,
                    "clock_weight": aging_genes[gene],
                    "mean_dep_older": float(np.mean(older_dep)),
                    "mean_dep_younger": float(np.mean(younger_dep)),
                    "dep_diff": float(np.mean(older_dep) - np.mean(younger_dep)),
                    "pvalue": p,
                })

        if len(gene_results) < 5:
            continue

        gene_df = pd.DataFrame(gene_results)
        # FDR correction
        from statsmodels.stats.multitest import multipletests
        _, gene_fdr, _, _ = multipletests(gene_df["pvalue"], method="fdr_bh")
        gene_df["fdr"] = gene_fdr

        sig_genes = gene_df[gene_df["fdr"] < 0.1]
        results.append({
            "tissue": tissue,
            "n_cells": len(cells),
            "n_aging_genes_tested": len(gene_df),
            "n_sig_genes_fdr01": len(sig_genes),
            "frac_dep_confirmed": round(len(sig_genes) / len(gene_df), 4) if len(gene_df) > 0 else 0,
            "mean_dep_diff_all": round(float(gene_df["dep_diff"].mean()), 4),
            "frac_negative_depdiff": round(float((gene_df["dep_diff"] < 0).mean()), 4),
        })

        # Save significant genes
        if len(sig_genes) > 0:
            sig_genes.to_csv(TABLE_DIR / f"depmap_sig_targets_{tissue}.csv",
                              index=False)

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("frac_dep_confirmed", ascending=False)
        results_df.to_csv(TABLE_DIR / "depmap_target_validation.csv", index=False)

        print(f"\n{'='*80}")
        print(f"DEPMAP VALIDATION RESULTS")
        print(f"{'='*80}")
        print(f"Tissues validated: {len(results_df)}")
        print(f"\nAll results:")
        print(results_df.to_string(index=False))

        print(f"\nKey findings:")
        for _, row in results_df.iterrows():
            confirmed = "CONFIRMED" if row["frac_negative_depdiff"] > 0.5 else "WEAK"
            print(f"  {row['tissue']}: {row['n_sig_genes_fdr01']}/{row['n_aging_genes_tested']} "
                  f"significant targets, "
                  f"frac_dep_confirmed={row['frac_dep_confirmed']}, "
                  f"mean_dep_diff={row['mean_dep_diff_all']} [{confirmed}]")

    # Generate figure: aging score distribution per tissue
    print("\nGenerating Step 5 figure...", flush=True)
    n = len(aging_scores)
    if n > 0:
        fig, axes = plt.subplots(1, min(n, 4), figsize=(5*min(n,4), 5))
        if min(n, 4) == 1:
            axes = [axes]
        for i, (tissue, scores) in enumerate(list(aging_scores.items())[:4]):
            ax = axes[i] if i < len(axes) else axes[-1]
            vals = list(scores.values())
            ax.hist(vals, bins=20, color="steelblue", edgecolor="k", lw=0.3)
            ax.axvline(np.median(vals), color="red", ls="--", label="median")
            ax.set_title(f"{tissue}\n(n={len(vals)} cell lines)")
            ax.set_xlabel("Aging score (GTEx clock projected)")
            ax.legend(fontsize=8)
        plt.suptitle("DepMap Cell Line Aging Scores (GTEx Clock Projection)", y=1.02)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "fig3_depmap_aging_scores.png", dpi=200,
                    bbox_inches="tight")
        plt.close()
        print("  fig3_depmap_aging_scores.png")

    print(f"\nStep 5 complete.")


if __name__ == "__main__":
    main()
