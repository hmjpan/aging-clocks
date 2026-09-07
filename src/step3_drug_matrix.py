"""Step 3-4: Drug x Tissue age-effect matrix via local LINCS L1000 GMT.

For each tissue clock:
  1. Split clock genes into "aging genes" (positive weight, increase with age)
     and "youth genes" (negative weight, decrease with age)
  2. For each LINCS compound gene set:
     - aging_down: overlap with aging genes (drug REJUVENATES if enriched)
     - aging_up: overlap with aging genes (drug PRO-AGES if enriched)
  3. Compute hypergeometric enrichment p-value
  4. Age reversal score = -log10(p_aging_down) - (-log10(p_aging_up))
     Positive = rejuvenating, Negative = pro-aging

Build compound x tissue matrix, compute heterogeneity.
"""
import os
import sys
import time
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_DIR, TABLE_DIR, FIGURE_DIR, GTEX_DIR, LINCS_DIR
from gct_io import read_gct

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def build_gene_map():
    """Ensembl -> gene symbol from per-tissue GCT (v1.3)."""
    fpath = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))[0]
    expr = read_gct(str(fpath))
    return dict(zip(expr.index, expr["gene_name"]))


def load_tissue_clocks(gene_map):
    """Load all clock coefficients, split into aging/youth gene symbol sets."""
    print("Loading clock coefficients...", flush=True)
    coef_files = sorted(CLOCK_DIR.glob("*_coefficients.csv"))
    clocks = {}
    for f in coef_files:
        tissue = f.name.replace("_coefficients.csv", "")
        df = pd.read_csv(f, index_col=0)
        df.columns = ["weight"]
        aging_genes = df[df["weight"] > 0].index.tolist()
        youth_genes = df[df["weight"] < 0].index.tolist()
        aging_syms = sorted(set(gene_map.get(g, "") for g in aging_genes) - {""})
        youth_syms = sorted(set(gene_map.get(g, "") for g in youth_genes) - {""})
        clocks[tissue] = {
            "aging_genes": aging_syms,
            "youth_genes": youth_syms,
            "n_features": len(df),
        }
    print(f"  {len(clocks)} tissues loaded", flush=True)
    return clocks


def parse_gmt(path):
    """Parse a GMT file into {set_name: set(genes)}."""
    sets = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            # parts[1] is often empty (description placeholder)
            genes = set(g for g in parts[2:] if g and g != "")
            if len(genes) > 0:
                sets[name] = genes
    return sets


def extract_compound_name(set_name):
    """Extract compound name from LINCS gene set name.

    Format: 'CPC001 HA1E 24H-hemado-10.0'
    -> compound = 'hemado', cell = 'HA1E', time = '24H', dose = '10.0'
    """
    # Split on first '-' after the batch info
    parts = set_name.split("-")
    if len(parts) < 2:
        return set_name, set_name, ""
    # The part after first '-' contains compound name
    # Try to extract compound, dose
    rest = "-".join(parts[1:])
    # Dose is usually the last number
    dose_match = re.search(r"(\d+\.?\d*)$", rest)
    dose = dose_match.group(1) if dose_match else ""
    # Compound is everything before dose
    if dose:
        compound = rest[:dose_match.start()].rstrip("-")
    else:
        compound = rest
    # Cell line is in the part before '-'
    batch_cell = parts[0].split()
    cell = batch_cell[-1] if len(batch_cell) > 1 else ""
    return compound, cell, dose


def hypergeom_enrichment(query_genes, gene_universe_set, set_genes):
    """Compute -log10(p) for hypergeometric enrichment.

    Parameters:
    - query_genes: set of aging/youth gene symbols
    - gene_universe_set: set of all genes in the background universe
    - set_genes: set of genes in a LINCS compound gene set

    Returns: -log10(p-value), higher = more enriched

    NOTE: N (query size) is restricted to genes IN the universe,
    and M (population) is the universe size.
    """
    # Restrict query to universe
    query_in_universe = query_genes & gene_universe_set
    set_in_universe = set_genes & gene_universe_set
    overlap = len(query_in_universe & set_in_universe)
    if overlap == 0:
        return 0.0
    M = len(gene_universe_set)  # total population
    K = len(set_in_universe)     # successes in population
    N = len(query_in_universe)   # draws (query genes that are in universe)
    x = overlap                   # observed overlaps
    if N == 0 or K == 0:
        return 0.0
    # P(X >= x) = 1 - P(X < x) = sf(x-1)
    pval = hypergeom.sf(x - 1, M, K, N)
    if pval <= 0:
        return 300.0  # cap at extreme
    return -np.log10(pval)


def run_drug_analysis(clocks):
    """Compute drug x tissue age reversal matrix using local GMT."""
    print("\nLoading LINCS GMT files...", flush=True)
    down_sets = parse_gmt(str(LINCS_DIR / "LINCS_L1000_Chem_Pert_down.txt"))
    up_sets = parse_gmt(str(LINCS_DIR / "LINCS_L1000_Chem_Pert_up.txt"))
    print(f"  Down sets: {len(down_sets)}", flush=True)
    print(f"  Up sets: {len(up_sets)}", flush=True)

    # Build compound name -> list of (set_name, gene_set) mapping
    # Group by compound name to aggregate across cell lines/doses
    compound_down = defaultdict(list)
    compound_up = defaultdict(list)
    for name, genes in down_sets.items():
        comp, cell, dose = extract_compound_name(name)
        compound_down[comp].append((name, genes))
    for name, genes in up_sets.items():
        comp, cell, dose = extract_compound_name(name)
        compound_up[comp].append((name, genes))

    all_compounds = sorted(set(compound_down.keys()) | set(compound_up.keys()))
    print(f"  Unique compounds: {len(all_compounds)}", flush=True)

    # Define gene universe as union of all LINCS genes
    lincs_universe = set()
    for genes in down_sets.values():
        lincs_universe |= genes
    for genes in up_sets.values():
        lincs_universe |= genes
    N_universe = len(lincs_universe)
    print(f"  Gene universe: {N_universe}", flush=True)

    # Pre-compute clock gene sets restricted to universe
    tissue_aging = {}
    tissue_youth = {}
    for tissue, clock in clocks.items():
        aging = set(clock["aging_genes"]) & lincs_universe
        youth = set(clock["youth_genes"]) & lincs_universe
        tissue_aging[tissue] = aging
        tissue_youth[tissue] = youth
        if tissue == sorted(clocks.keys())[0]:
            print(f"  Example: {tissue} aging={len(set(clock['aging_genes']))} "
                  f"-> in universe: {len(aging)}", flush=True)

    tissues = sorted(clocks.keys())
    n_tissues = len(tissues)
    n_compounds = len(all_compounds)

    # Build matrix: compound x tissue -> reversal score
    print(f"\nComputing {n_compounds} x {n_tissues} reversal matrix...", flush=True)

    results = np.zeros((n_compounds, n_tissues), dtype=np.float32)

    for i, compound in enumerate(all_compounds):
        if i % 5000 == 0:
            print(f"  {i}/{n_compounds}...", flush=True)

        # Aggregate gene sets for this compound (union across cell lines)
        down_genes = set()
        for _, gs in compound_down.get(compound, []):
            down_genes |= gs
        up_genes = set()
        for _, gs in compound_up.get(compound, []):
            up_genes |= gs

        if len(down_genes) == 0 and len(up_genes) == 0:
            continue

        for j, tissue in enumerate(tissues):
            aging_set = tissue_aging[tissue]
            youth_set = tissue_youth[tissue]

            # Aging genes DOWN by drug = rejuvenating
            aging_down_score = 0.0
            if len(down_genes) > 0 and len(aging_set) > 0:
                aging_down_score = hypergeom_enrichment(
                    aging_set, lincs_universe, down_genes)

            # Aging genes UP by drug = pro-aging
            aging_up_score = 0.0
            if len(up_genes) > 0 and len(aging_set) > 0:
                aging_up_score = hypergeom_enrichment(
                    aging_set, lincs_universe, up_genes)

            # Youth genes UP by drug = rejuvenating
            youth_up_score = 0.0
            if len(up_genes) > 0 and len(youth_set) > 0:
                youth_up_score = hypergeom_enrichment(
                    youth_set, lincs_universe, up_genes)

            # Youth genes DOWN by drug = pro-aging
            youth_down_score = 0.0
            if len(down_genes) > 0 and len(youth_set) > 0:
                youth_down_score = hypergeom_enrichment(
                    youth_set, lincs_universe, down_genes)

            # Combined reversal score:
            # positive = rejuvenating, negative = pro-aging
            # Rejuvenating = aging genes down + youth genes up
            # Pro-aging = aging genes up + youth genes down
            reversal = (aging_down_score + youth_up_score) \
                     - (aging_up_score + youth_down_score)
            results[i, j] = reversal

    matrix = pd.DataFrame(results, index=all_compounds, columns=tissues)
    matrix.index.name = "compound"
    matrix.to_csv(TABLE_DIR / "drug_tissue_reversal_matrix.csv")
    print(f"\n  Matrix saved: {matrix.shape[0]} compounds x {matrix.shape[1]} tissues")

    return matrix


def analyze_heterogeneity(matrix):
    """Compute heterogeneity per compound and identify top candidates."""
    print("\nAnalyzing drug heterogeneity...", flush=True)

    # Only consider compounds with any non-zero score
    nonzero_count = (matrix.abs() > 0.5).sum(axis=1)  # threshold for meaningful signal
    valid_compounds = nonzero_count[nonzero_count >= 3].index
    matrix_valid = matrix.loc[valid_compounds]

    print(f"  Compounds with signal in >= 3 tissues: {len(matrix_valid)}")

    # Heterogeneity: std of reversal scores across tissues
    std_scores = matrix_valid.std(axis=1)
    mean_scores = matrix_valid.mean(axis=1)
    max_scores = matrix_valid.max(axis=1)
    min_scores = matrix_valid.min(axis=1)

    # Count rejuvenated vs pro-aged tissues per compound
    n_rejuvenated = (matrix_valid > 0.5).sum(axis=1)
    n_pro_aging = (matrix_valid < -0.5).sum(axis=1)

    # "Mixed" compounds: rejuvenating in some tissues, pro-aging in others
    is_mixed = ((n_rejuvenated > 0) & (n_pro_aging > 0)).astype(int)

    het_df = pd.DataFrame({
        "compound": matrix_valid.index,
        "mean_reversal": mean_scores.values,
        "std_reversal": std_scores.values,
        "max_reversal": max_scores.values,
        "min_reversal": min_scores.values,
        "range": (max_scores - min_scores).values,
        "n_rejuvenated": n_rejuvenated.values,
        "n_pro_aging": n_pro_aging.values,
        "is_mixed": is_mixed.values,
        "n_tissues_active": nonzero_count[valid_compounds].values,
    })

    # Sort by heterogeneity (range) for mixed compounds
    het_df = het_df.sort_values("range", ascending=False)
    het_df.to_csv(TABLE_DIR / "drug_heterogeneity_scores.csv", index=False)

    mixed = het_df[het_df["is_mixed"] == 1]
    print(f"\n  Total active compounds: {len(matrix_valid)}")
    print(f"  Mixed (rejuvenating + pro-aging): {len(mixed)}")

    print(f"\n  Top 30 most heterogeneous compounds:")
    print(het_df.head(30)[["compound", "mean_reversal", "range",
                            "n_rejuvenated", "n_pro_aging"]].to_string(index=False))

    print(f"\n  Top 30 most rejuvenating (mean across tissues):")
    top_reju = het_df.sort_values("mean_reversal", ascending=False).head(30)
    print(top_reju[["compound", "mean_reversal", "n_rejuvenated",
                      "n_pro_aging"]].to_string(index=False))

    print(f"\n  Top 30 most pro-aging:")
    top_pro = het_df.sort_values("mean_reversal").head(30)
    print(top_pro[["compound", "mean_reversal", "n_rejuvenated",
                    "n_pro_aging"]].to_string(index=False))

    return het_df, matrix_valid


def main():
    gene_map = build_gene_map()
    clocks = load_tissue_clocks(gene_map)

    # Check if matrix already exists
    matrix_path = TABLE_DIR / "drug_tissue_reversal_matrix.csv"
    if matrix_path.exists():
        print("Loading existing reversal matrix...", flush=True)
        matrix = pd.read_csv(matrix_path, index_col=0)
    else:
        matrix = run_drug_analysis(clocks)

    # Analyze
    het_df, matrix_valid = analyze_heterogeneity(matrix)

    # Save a focused table: top 100 mixed compounds
    mixed = het_df[het_df["is_mixed"] == 1].head(100)
    if len(mixed) > 0:
        mixed_matrix = matrix_valid.loc[mixed["compound"].values]
        mixed_matrix.to_csv(TABLE_DIR / "top100_mixed_compounds_matrix.csv")

    print(f"\nStep 3-4 complete.")
    print(f"  Matrix: {matrix_valid.shape[0]} compounds x {matrix_valid.shape[1]} tissues")
    print(f"  Results in {TABLE_DIR}")


if __name__ == "__main__":
    main()
