"""Step 2: Biological annotation of tissue-specific aging clocks.

1. Load all clock coefficients (Ensembl IDs -> gene symbols)
2. Build cross-tissue gene weight matrix
3. Identify shared vs tissue-specific clock genes
4. Pathway enrichment (Enrichr via gseapy)
5. Generate key figures:
   - Fig 1B: Clock performance heatmap (r, MAE by tissue)
   - Fig 1C: Cross-tissue clock gene overlap (UpSet-style)
   - Fig 1D: Pathway enrichment per tissue cluster
   - Fig 1E: Aging direction vectors (top shared genes across tissues)
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_DIR, FIGURE_DIR, TABLE_DIR, GTEX_DIR
from gct_io import read_gct

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Build Ensembl -> gene symbol mapping from a GCT file ---
def build_gene_map():
    """Extract Ensembl ID -> gene symbol mapping from per-tissue GCT (v1.3)."""
    print("Building Ensembl -> symbol map...", flush=True)
    fpath = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))[0]
    expr = read_gct(str(fpath))
    gene_map = dict(zip(expr.index, expr["gene_name"]))
    print(f"  {len(gene_map)} genes mapped", flush=True)
    return gene_map


# --- 2. Load all clock coefficients ---
def load_all_clocks(gene_map):
    """Load all clock coefficients and return a (gene x tissue) weight matrix."""
    print("Loading all clock coefficients...", flush=True)
    coef_files = sorted(CLOCK_DIR.glob("*_coefficients.csv"))
    clocks = {}
    for f in coef_files:
        tissue = f.name.replace("_coefficients.csv", "")
        df = pd.read_csv(f, index_col=0)
        df.columns = ["weight"]
        clocks[tissue] = df["weight"]

    # Build unified gene x tissue matrix
    all_genes = sorted(set().union(*[set(c.index) for c in clocks.values()]))
    weight_matrix = pd.DataFrame(0.0, index=all_genes,
                                  columns=sorted(clocks.keys()))
    for tissue, coef in clocks.items():
        weight_matrix.loc[coef.index, tissue] = coef.values

    # Map to symbols
    weight_matrix["gene_symbol"] = weight_matrix.index.map(
        lambda x: gene_map.get(x, x))
    weight_matrix.to_csv(TABLE_DIR / "clock_weight_matrix.csv")

    print(f"  {weight_matrix.shape[0]} unique genes across "
          f"{weight_matrix.shape[1]-1} tissues", flush=True)
    return weight_matrix, clocks


# --- 3. Load summary and fix metrics ---
def load_summary():
    """Load clock summary, recompute metrics from predictions."""
    summary = pd.read_csv(TABLE_DIR / "clock_summary.csv")
    # Recompute missing metrics from prediction files
    for idx, row in summary.iterrows():
        tissue = row["tissue"]
        pred_path = CLOCK_DIR / f"{tissue}_cv_predictions.csv"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        if len(pred) == 0:
            continue
        from scipy.stats import spearmanr
        y = pred["age_actual"].values
        p = pred["age_predicted"].values
        valid = ~np.isnan(p)
        if valid.sum() < 10:
            continue
        rho, _ = spearmanr(p[valid], y[valid])
        summary.loc[idx, "spearman_r"] = round(rho, 4)
        # reload best alpha from coefficient file
        coef = pd.read_csv(CLOCK_DIR / f"{tissue}_coefficients.csv", index_col=0)
        summary.loc[idx, "n_features"] = int((coef["weight"] != 0).sum())
    summary.to_csv(TABLE_DIR / "clock_summary.csv", index=False)
    return summary


# --- 4. Cross-tissue gene overlap analysis ---
def analyze_gene_overlap(weight_matrix, clocks):
    """Analyze how many clock genes are shared vs tissue-specific."""
    print("\nAnalyzing gene overlap...", flush=True)
    # Binary matrix: gene is in clock (1) or not (0)
    binary = (weight_matrix.drop(columns=["gene_symbol"]) != 0).astype(int)
    # Count how many tissues each gene appears in
    gene_tissue_count = binary.sum(axis=1)
    overlap_stats = gene_tissue_count.value_counts().sort_index()

    print("  Genes appearing in N tissues:")
    for n, cnt in overlap_stats.items():
        if n <= 5 or n >= len(clocks) - 2:
            print(f"    {n} tissues: {cnt} genes")

    # Shared clock genes (in >= 50% of tissues)
    n_tissues = len(clocks)
    shared_threshold = int(n_tissues * 0.5)
    shared_genes = gene_tissue_count[gene_tissue_count >= shared_threshold].index.tolist()
    print(f"\n  Shared genes (in >= {shared_threshold}/{n_tissues} tissues): "
          f"{len(shared_genes)}")

    # Tissue-specific genes (in only 1 tissue)
    specific_genes = gene_tissue_count[gene_tissue_count == 1].index.tolist()
    print(f"  Tissue-specific genes (in only 1 tissue): {len(specific_genes)}")

    # Per-tissue: how many unique vs shared
    tissue_specificity = pd.DataFrame(index=sorted(clocks.keys()),
                                       columns=["n_total", "n_unique", "n_shared"])
    for tissue in clocks:
        tissue_genes = set(clocks[tissue].index)
        unique = sum(1 for g in tissue_genes if gene_tissue_count[g] == 1)
        shared = sum(1 for g in tissue_genes if gene_tissue_count[g] >= shared_threshold)
        tissue_specificity.loc[tissue] = [len(tissue_genes), unique, shared]
    tissue_specificity = tissue_specificity.astype(int)
    tissue_specificity.to_csv(TABLE_DIR / "tissue_gene_specificity.csv")

    return gene_tissue_count, shared_genes, tissue_specificity


# --- 5. Pathway enrichment ---
def run_enrichment(shared_genes, gene_map, tissue_specificity, clocks):
    """Run Enrichr pathway enrichment on shared and top tissue-specific genes."""
    import gseapy
    print("\nRunning pathway enrichment...", flush=True)

    # Map shared genes to symbols
    shared_symbols = [gene_map.get(g, g) for g in shared_genes]
    shared_symbols = [s for s in shared_symbols if s and s != "nan"]

    results = {}

    # Enrichment for shared clock genes
    if len(shared_symbols) >= 5:
        print(f"  Shared genes ({len(shared_symbols)} symbols)...", flush=True)
        try:
            enr = gseapy.enrichr(gene_list=shared_symbols,
                                 gene_sets=["GO_Biological_Process_2023",
                                            "KEGG_2021_Human",
                                            "Reactome_2022",
                                            "WikiPathways_2024_Human"],
                                 organism="human",
                                 outdir=None,
                                 no_plot=True)
            enr_df = enr.results
            enr_df.to_csv(TABLE_DIR / "enrichment_shared_genes.csv", index=False)
            print(f"    {len(enr_df)} pathways enriched")
            # Top 20
            top = enr_df.head(20)[["Term", "Adjusted P-value", "Overlap", "Genes"]]
            print(top.to_string(index=False))
            results["shared"] = enr_df
        except Exception as e:
            print(f"    Enrichr failed: {e}")

    # Enrichment for top 5 tissues by performance
    summary = pd.read_csv(TABLE_DIR / "clock_summary.csv")
    top_tissues = summary[summary["pearson_r"] > 0.3].nlargest(5, "pearson_r")["tissue"]
    for tissue in top_tissues:
        tissue_genes = list(clocks.get(tissue, pd.Series()).index)
        tissue_symbols = [gene_map.get(g, g) for g in tissue_genes]
        tissue_symbols = [s for s in tissue_symbols if s and s != "nan"]
        if len(tissue_symbols) < 5:
            continue
        print(f"  {tissue} ({len(tissue_symbols)} symbols)...", flush=True)
        try:
            enr = gseapy.enrichr(gene_list=tissue_symbols,
                                 gene_sets=["GO_Biological_Process_2023",
                                            "KEGG_2021_Human",
                                            "Reactome_2022"],
                                 organism="human",
                                 outdir=None,
                                 no_plot=True)
            enr.results.to_csv(
                TABLE_DIR / f"enrichment_{tissue}.csv", index=False)
            results[tissue] = enr.results
        except Exception as e:
            print(f"    Failed: {e}")
        time.sleep(1)  # rate limit

    return results


# --- 6. Figures ---
def plot_figures(summary, weight_matrix, gene_tissue_count,
                  tissue_specificity, clocks, gene_map):
    """Generate key figures."""
    print("\nGenerating figures...", flush=True)
    sns.set_style("whitegrid")

    # --- Fig 1A: Clock performance bar chart ---
    valid = summary[summary["pearson_r"] > 0].sort_values("pearson_r", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.RdYlGn(valid["pearson_r"] / max(valid["pearson_r"].max(), 0.001))
    ax.barh(range(len(valid)), valid["pearson_r"], color=colors, edgecolor="k", lw=0.3)
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels(valid["tissue"], fontsize=7)
    ax.set_xlabel("Pearson r (predicted vs actual age)")
    ax.set_title("Tissue-Specific Aging Clock Performance (GTEx v8)")
    ax.axvline(x=0.5, color="red", ls="--", alpha=0.5, label="r=0.5")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig1a_clock_performance.png", dpi=200)
    plt.close()
    print("  fig1a_clock_performance.png")

    # --- Fig 1B: MAE vs sample size ---
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(summary["n_samples"], summary["mae"],
                          c=summary["pearson_r"], cmap="RdYlGn",
                          s=60, edgecolors="k", lw=0.5)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("MAE (years)")
    ax.set_title("Clock Accuracy vs Sample Size")
    plt.colorbar(scatter, label="Pearson r")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig1b_mae_vs_samples.png", dpi=200)
    plt.close()
    print("  fig1b_mae_vs_samples.png")

    # --- Fig 1C: Cross-tissue gene sharing distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = gene_tissue_count.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="steelblue", edgecolor="k", lw=0.3)
    ax.set_xlabel("Number of tissues gene appears in")
    ax.set_ylabel("Number of genes")
    ax.set_title("Clock Gene Sharing Across Tissues")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig1c_gene_sharing.png", dpi=200)
    plt.close()
    print("  fig1c_gene_sharing.png")

    # --- Fig 1D: Tissue-specificity ---
    ts = tissue_specificity.sort_values("n_total", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.arange(len(ts))
    ax.barh(x, ts["n_total"], color="lightgray", label="Total clock genes")
    ax.barh(x, ts["n_shared"], color="steelblue", label="Shared (>=50% tissues)")
    ax.barh(x, ts["n_unique"], color="darkorange", label="Tissue-specific")
    ax.set_yticks(x)
    ax.set_yticklabels(ts.index, fontsize=7)
    ax.set_xlabel("Number of genes")
    ax.set_title("Clock Gene Composition: Shared vs Tissue-Specific")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig1d_gene_specificity.png", dpi=200)
    plt.close()
    print("  fig1d_gene_specificity.png")

    # --- Fig 1E: Top shared aging genes (direction across tissues) ---
    top_shared = gene_tissue_count.nlargest(50).index
    shared_matrix = weight_matrix.loc[top_shared].drop(columns=["gene_symbol"])
    shared_matrix.index = [gene_map.get(g, g) for g in shared_matrix.index]
    # Cluster tissues
    if shared_matrix.shape[1] > 2:
        col_linkage = linkage(pdist(shared_matrix.T), method="average")
        col_order = dendrogram(col_linkage, no_plot=True)["leaves"]
        shared_matrix = shared_matrix.iloc[:, col_order]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(shared_matrix, cmap="RdBu_r", center=0,
                xticklabels=True, yticklabels=True,
                ax=ax, cbar_kws={"label": "Clock weight"},
                vmin=-1, vmax=1)
    ax.set_title("Top 50 Shared Aging Clock Genes (Direction Across Tissues)")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig1e_shared_gene_directions.png", dpi=200)
    plt.close()
    print("  fig1e_shared_gene_directions.png")

    # --- Fig 1F: Tissue clustering by clock gene weights ---
    mat = weight_matrix.drop(columns=["gene_symbol"])
    # Only use tissues with >0 features
    valid_tissues = [t for t in mat.columns if (mat[t] != 0).sum() > 10]
    mat = mat[valid_tissues]
    corr = mat.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                xticklabels=True, yticklabels=True, ax=ax,
                square=True, linewidths=0.3,
                cbar_kws={"label": "Correlation of clock weights"})
    ax.set_title("Cross-Tissue Correlation of Aging Clock Gene Weights")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig1f_tissue_correlation.png", dpi=200)
    plt.close()
    print("  fig1f_tissue_correlation.png")


def main():
    gene_map = build_gene_map()
    weight_matrix, clocks = load_all_clocks(gene_map)
    summary = load_summary()
    gene_tissue_count, shared_genes, tissue_specificity = \
        analyze_gene_overlap(weight_matrix, clocks)

    # Print summary stats
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(clocks)} tissue clocks trained")
    valid = summary[summary["pearson_r"] > 0.3]
    print(f"  Clocks with r > 0.3: {len(valid)}/{len(summary)}")
    print(f"  Best: {valid.iloc[0]['tissue']} (r={valid.iloc[0]['pearson_r']})")
    print(f"  Median r: {valid['pearson_r'].median():.3f}")
    print(f"  Total unique clock genes: {weight_matrix.shape[0]}")
    print(f"  Shared genes (>=50% tissues): {len(shared_genes)}")
    print(f"{'='*70}")

    # Run enrichment
    enrichment = run_enrichment(shared_genes, gene_map,
                                 tissue_specificity, clocks)

    # Generate figures
    plot_figures(summary, weight_matrix, gene_tissue_count,
                 tissue_specificity, clocks, gene_map)

    print("\nStep 2 complete. Results in results/tables/ and results/figures/")


if __name__ == "__main__":
    main()
