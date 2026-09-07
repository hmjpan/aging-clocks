"""Step 3-4: Drug x Tissue age-effect matrix via local GMT enrichment.

For each tissue clock:
  - aging_genes = positive weight genes (increase with age)
  - youth_genes = negative weight genes (decrease with age)

For each compound in LINCS L1000 GMT:
  - down_genes = genes DOWN-regulated by compound
  - up_genes   = genes UP-regulated by compound

Reversal score (Fisher exact -log10 p):
  rejuvenation = enrichment(aging_genes, down_genes)  - drug reverses aging
  pro_aging    = enrichment(aging_genes, up_genes)    - drug accelerates aging
  net_reversal = rejuvenation - pro_aging

Build compound x tissue matrix, compute heterogeneity.
"""
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_DIR, GTEX_DIR, TABLE_DIR, FIGURE_DIR, LINCS_DIR
from gct_io import read_gct

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --- 1. Load gene map ---
def build_gene_map():
    fpath = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))[0]
    expr = read_gct(str(fpath))
    return dict(zip(expr.index, expr["gene_name"]))


# --- 2. Load tissue clock genes ---
def load_tissue_clocks(gene_map):
    coef_files = sorted(CLOCK_DIR.glob("*_coefficients.csv"))
    clocks = {}
    for f in coef_files:
        tissue = f.name.replace("_coefficients.csv", "")
        df = pd.read_csv(f, index_col=0)
        df.columns = ["weight"]
        aging = df[df["weight"] > 0].index.tolist()
        youth = df[df["weight"] < 0].index.tolist()
        aging_syms = sorted(set(gene_map.get(g, "") for g in aging) - {""})
        youth_syms = sorted(set(gene_map.get(g, "") for g in youth) - {""})
        clocks[tissue] = {"aging": aging_syms, "youth": youth_syms,
                          "n": len(df)}
    return clocks


# --- 3. Parse GMT files ---
def parse_gmt(path):
    """Parse GMT: each line = gene_set_name\tdesc\tgene1\tgene2\t..."""
    gene_sets = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            # parts[1] is often empty (description placeholder)
            genes = set(g for g in parts[2:] if g and g != "")
            if genes:
                gene_sets[name] = genes
    return gene_sets


# --- 4. Fisher exact enrichment ---
def fisher_enrichment(query_genes, set_genes, background_size=12000):
    """Compute -log10(Fisher exact p) for overlap.

    Returns enrichment score (higher = more enriched).
    """
    query = set(query_genes)
    background = set()  # we use a fixed background size
    overlap = query & set_genes
    n_overlap = len(overlap)
    if n_overlap == 0:
        return 0.0, 0, 0.0

    # Contingency table
    n_query = len(query)
    n_set = len(set_genes)
    n_bg = background_size
    a = n_overlap
    b = n_query - n_overlap
    c = n_set - n_overlap
    d = n_bg - n_query - n_set + n_overlap
    if d < 0:
        d = 0
    _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
    score = -np.log10(pval) if pval > 0 else 300.0
    return score, n_overlap, pval


# --- 5. Build drug x tissue matrix ---
def build_matrix(clocks, gmt_down, gmt_up):
    """For each tissue x compound, compute net reversal score."""
    tissues = sorted(clocks.keys())
    compounds = sorted(set(gmt_down.keys()) | set(gmt_up.keys()))
    n_compounds = len(compounds)
    print(f"\n  {n_compounds} compounds x {len(tissues)} tissues", flush=True)
    print(f"  Total cells: {n_compounds * len(tissues):,}", flush=True)

    # Precompute: for each compound, extract down/up gene sets
    print("  Precomputing compound gene sets...", flush=True)
    comp_down = {}
    comp_up = {}
    comp_base = {}  # base compound name (without cell line)
    for comp in compounds:
        genes_down = gmt_down.get(comp, set())
        genes_up = gmt_up.get(comp, set())
        comp_down[comp] = genes_down
        comp_up[comp] = genes_up
        # Extract base compound name: "CPC001 HA1E 24H-hemado-10.0" -> "hemado"
        # Format: "<batch> <cell> <time>-<compound_name>-<dose>"
        name = comp
        if "-" in name:
            parts = name.split("-")
            if len(parts) >= 2:
                base = parts[-2] if len(parts) >= 2 else name
                comp_base[comp] = base

    # Build matrix
    matrix = pd.DataFrame(0.0, index=compounds, columns=tissues)

    print("  Computing enrichment scores...", flush=True)
    for i, tissue in enumerate(tissues):
        aging_genes = set(clocks[tissue]["aging"])
        youth_genes = set(clocks[tissue]["youth"])
        t0 = time.time()

        for comp in compounds:
            down_genes = comp_down[comp]
            up_genes = comp_up[comp]

            # Rejuvenation: compound DOWN-regulates aging genes
            rej_score, rej_n, _ = fisher_enrichment(aging_genes, down_genes)
            # Pro-aging: compound UP-regulates aging genes
            pro_score, pro_n, _ = fisher_enrichment(aging_genes, up_genes)
            # Also: compound UP-regulates youth genes (rejuvenating)
            rej2_score, _, _ = fisher_enrichment(youth_genes, up_genes)
            # compound DOWN-regulates youth genes (pro-aging)
            pro2_score, _, _ = fisher_enrichment(youth_genes, down_genes)

            # Net reversal = (rej + rej2) - (pro + pro2)
            net = rej_score + rej2_score - pro_score - pro2_score
            matrix.loc[comp, tissue] = net

        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(tissues)}] {tissue}: {elapsed:.1f}s", flush=True)

        # Save intermediate every 10 tissues
        if (i + 1) % 10 == 0:
            matrix.to_csv(TABLE_DIR / "drug_tissue_matrix_partial.csv")

    return matrix, comp_base


# --- 6. Analyze heterogeneity ---
def analyze_heterogeneity(matrix, comp_base):
    """Compute heterogeneity metrics and identify mixed compounds."""
    print("\nAnalyzing heterogeneity...", flush=True)

    # Only keep compounds with non-zero scores in >=5 tissues
    nonzero = (matrix.abs() > 0.01).sum(axis=1)
    valid = nonzero[nonzero >= 5].index
    m = matrix.loc[valid]

    # Collapse by base compound name (aggregate across cell lines)
    m["base_compound"] = [comp_base.get(c, c) for c in m.index]
    m_agg = m.groupby("base_compound").mean()

    # Heterogeneity metrics
    mean_reversal = m_agg.mean(axis=1)
    variance = m_agg.var(axis=1)
    mean_abs = m_agg.abs().mean(axis=1)
    cv = variance / (mean_abs ** 2 + 1e-10)

    n_rej = (m_agg > 0.5).sum(axis=1)
    n_pro = (m_agg < -0.5).sum(axis=1)
    is_mixed = ((n_rej > 0) & (n_pro > 0)).astype(int)

    het = pd.DataFrame({
        "compound": m_agg.index,
        "mean_reversal": mean_reversal.values,
        "variance": variance.values,
        "mean_abs_score": mean_abs.values,
        "heterogeneity_cv": cv.values,
        "n_rejuvenated": n_rej.values,
        "n_pro_aging": n_pro.values,
        "is_mixed": is_mixed.values,
        "n_tissues_active": (m_agg.abs() > 0.01).sum(axis=1).values,
    }).sort_values("heterogeneity_cv", ascending=False)

    het.to_csv(TABLE_DIR / "drug_heterogeneity.csv", index=False)
    m_agg.to_csv(TABLE_DIR / "drug_tissue_matrix_agg.csv")

    mixed = het[het["is_mixed"] == 1]
    print(f"\n  Total compounds: {len(m_agg)}")
    print(f"  Mixed (reju+pro): {len(mixed)}")
    print(f"\n  Top 30 heterogeneous:")
    print(mixed.head(30)[["compound", "mean_reversal", "heterogeneity_cv",
                            "n_rejuvenated", "n_pro_aging"]].to_string(index=False))

    return het, m_agg


def main():
    print("=== Step 3-4: Drug x Tissue Age-Effect Matrix ===\n", flush=True)

    gene_map = build_gene_map()
    clocks = load_tissue_clocks(gene_map)
    print(f"Loaded {len(clocks)} tissue clocks", flush=True)

    # Parse GMT files
    down_path = str(LINCS_DIR / "LINCS_L1000_Chem_Pert_down.txt")
    up_path = str(LINCS_DIR / "LINCS_L1000_Chem_Pert_up.txt")
    print("Parsing GMT files...", flush=True)
    gmt_down = parse_gmt(down_path)
    gmt_up = parse_gmt(up_path)
    print(f"  Down: {len(gmt_down)} gene sets", flush=True)
    print(f"  Up: {len(gmt_up)} gene sets", flush=True)

    # Build matrix
    matrix, comp_base = build_matrix(clocks, gmt_down, gmt_up)
    matrix.to_csv(TABLE_DIR / "drug_tissue_matrix_raw.csv")

    # Analyze
    het, m_agg = analyze_heterogeneity(matrix, comp_base)

    print(f"\n=== Step 3-4 Complete ===")
    print(f"  Raw matrix: {matrix.shape[0]} compounds x {matrix.shape[1]} tissues")
    print(f"  Aggregated: {m_agg.shape[0]} compounds x {m_agg.shape[1]} tissues")
    print(f"  Results in {TABLE_DIR}")


if __name__ == "__main__":
    main()
