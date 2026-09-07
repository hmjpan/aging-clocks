# -*- coding: utf-8 -*-
"""Sensitivity analyses for manuscript revision.

Analysis 1: Exclude weak clocks (Pearson r < 0.3) and recompute drug
            heterogeneity statistics.

Analysis 2: Concordance between hypergeometric -log10(p) and a Jaccard-
            weighted enrichment score.

Analysis 3: Per drug-tissue pair effect size (z-score relative to a
            within-tissue permutation null) and BH-FDR correction across
            all drug x tissue pairs.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr, norm

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_DIR, TABLE_DIR, LINCS_DIR, GTEX_DIR
from gct_io import read_gct

TABLE_DIR.mkdir(parents=True, exist_ok=True)

WEAK_TISSUES = [
    "kidney_cortex",
    "cells_ebv-transformed_lymphocytes",
    "brain_spinal_cord_cervical_c-1",
]


def build_gene_map():
    fpath = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))[0]
    expr = read_gct(str(fpath))
    return dict(zip(expr.index, expr["gene_name"]))


def load_clocks(gene_map):
    clocks = {}
    for f in sorted(CLOCK_DIR.glob("*_coefficients.csv")):
        tissue = f.name.replace("_coefficients.csv", "")
        df = pd.read_csv(f, index_col=0)
        df.columns = ["weight"]
        aging = sorted(set(gene_map.get(g, "") for g in df[df["weight"] > 0].index) - {""})
        youth = sorted(set(gene_map.get(g, "") for g in df[df["weight"] < 0].index) - {""})
        clocks[tissue] = {"aging": aging, "youth": youth, "weights": df}
    return clocks


def parse_gmt(path):
    sets = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            sets[parts[0]] = set(g for g in parts[2:] if g)
    return sets


def extract_compound(set_name):
    parts = set_name.split("-")
    if len(parts) < 2:
        return set_name
    rest = "-".join(parts[1:])
    import re
    m = re.search(r"(\d+\.?\d*)$", rest)
    if m:
        return rest[:m.start()].rstrip("-")
    return rest


def load_lincs_compounds():
    print("Loading LINCS GMT...", flush=True)
    down_sets = parse_gmt(str(LINCS_DIR / "LINCS_L1000_Chem_Pert_down.txt"))
    up_sets = parse_gmt(str(LINCS_DIR / "LINCS_L1000_Chem_Pert_up.txt"))
    compound_down = defaultdict(set)
    compound_up = defaultdict(set)
    for name, genes in down_sets.items():
        compound_down[extract_compound(name)] |= genes
    for name, genes in up_sets.items():
        compound_up[extract_compound(name)] |= genes
    all_compounds = sorted(set(compound_down) | set(compound_up))
    universe = set()
    for gs in down_sets.values():
        universe |= gs
    for gs in up_sets.values():
        universe |= gs
    print(f"  {len(all_compounds)} compounds, universe={len(universe)}", flush=True)
    return all_compounds, compound_down, compound_up, universe


def hypergeom_neglog10(query, universe, drug_set):
    q = query & universe
    s = drug_set & universe
    overlap = len(q & s)
    if overlap == 0:
        return 0.0
    M = len(universe); K = len(s); N = len(q)
    p = hypergeom.sf(overlap - 1, M, K, N)
    return 0.0 if p <= 0 else -np.log10(p)


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def compute_matrix(clocks, compounds, cdown, cup, universe, score="hypergeom"):
    tissues = sorted(clocks.keys())
    n_c, n_t = len(compounds), len(tissues)
    mat = np.zeros((n_c, n_t), dtype=np.float32)
    tissue_aging = {t: set(clocks[t]["aging"]) & universe for t in tissues}
    tissue_youth = {t: set(clocks[t]["youth"]) & universe for t in tissues}
    for i, comp in enumerate(compounds):
        dg = cdown.get(comp, set()); ug = cup.get(comp, set())
        if not dg and not ug:
            continue
        for j, t in enumerate(tissues):
            ag = tissue_aging[t]; yg = tissue_youth[t]
            if score == "hypergeom":
                r = (hypergeom_neglog10(ag, universe, dg) +
                     hypergeom_neglog10(yg, universe, ug)) \
                    - (hypergeom_neglog10(ag, universe, ug) +
                       hypergeom_neglog10(yg, universe, dg))
            elif score == "jaccard":
                r = (jaccard(ag, dg) + jaccard(yg, ug)) \
                    - (jaccard(ag, ug) + jaccard(yg, dg))
            else:
                raise ValueError(score)
            mat[i, j] = r
    return pd.DataFrame(mat, index=compounds, columns=tissues)


def analysis1_weak_tissues(main_matrix):
    print("\n=== Analysis 1: Exclude weak clocks (r<0.3) ===", flush=True)
    keep = [t for t in main_matrix.columns if t not in WEAK_TISSUES]
    sub = main_matrix[keep]
    n_rej = (sub > 0.5).sum(axis=1)
    n_pro = (sub < -0.5).sum(axis=1)
    is_mixed = ((n_rej > 0) & (n_pro > 0)).astype(int)
    out = pd.DataFrame({
        "compound": sub.index,
        "mean_reversal_46tissue": sub.mean(axis=1).values,
        "range_46tissue": (sub.max(axis=1) - sub.min(axis=1)).values,
        "n_rejuvenated_46tissue": n_rej.values,
        "n_pro_aging_46tissue": n_pro.values,
        "is_mixed_46tissue": is_mixed.values,
    })
    out.to_csv(TABLE_DIR / "sensitivity_weak_tissues.csv", index=False)

    main_het = pd.read_csv(TABLE_DIR / "drug_heterogeneity_scores.csv")
    merged = out.merge(main_het[["compound", "mean_reversal", "range",
                                  "n_rejuvenated", "n_pro_aging", "is_mixed"]],
                       on="compound", how="inner")
    rho_mean, _ = spearmanr(merged["mean_reversal_46tissue"], merged["mean_reversal"])
    rho_range, _ = spearmanr(merged["range_46tissue"], merged["range"])
    rho_rej, _ = spearmanr(merged["n_rejuvenated_46tissue"], merged["n_rejuvenated"])

    mixed_46 = int((out["is_mixed_46tissue"] == 1).sum())
    mixed_49 = int((main_het["is_mixed"] == 1).sum())
    n_total = len(out)

    res = {
        "n_tissues_kept": len(keep),
        "n_tissues_excluded": len(WEAK_TISSUES),
        "excluded_tissues": WEAK_TISSUES,
        "excluded_pearson_r": [0.065, 0.070, 0.242],
        "n_compounds": n_total,
        "n_mixed_49tissue": mixed_49,
        "n_mixed_46tissue": mixed_46,
        "pct_mixed_49tissue": float(mixed_49 / n_total * 100),
        "pct_mixed_46tissue": float(mixed_46 / n_total * 100),
        "spearman_rho_mean_reversal": float(rho_mean),
        "spearman_rho_range": float(rho_range),
        "spearman_rho_n_rejuvenated": float(rho_rej),
    }
    print(f"  Kept {len(keep)} tissues, excluded {len(WEAK_TISSUES)}")
    print(f"  Mixed drugs: {mixed_49}/{n_total} ({res['pct_mixed_49tissue']:.1f}%) [49-tissue] vs "
          f"{mixed_46}/{n_total} ({res['pct_mixed_46tissue']:.1f}%) [46-tissue]")
    print(f"  Spearman rho (mean reversal): {rho_mean:.3f}")
    print(f"  Spearman rho (range): {rho_range:.3f}")
    return res


def analysis2_jaccard_concordance(clocks, compounds, cdown, cup, universe, main_matrix):
    print("\n=== Analysis 2: Jaccard vs hypergeometric concordance ===", flush=True)
    jac = compute_matrix(clocks, compounds, cdown, cup, universe, score="jaccard")
    jac.to_csv(TABLE_DIR / "sensitivity_jaccard_matrix.csv")

    tissues = sorted(set(main_matrix.columns) & set(jac.columns))
    rows = []
    for t in tissues:
        rho, p = spearmanr(main_matrix[t].values, jac[t].values)
        rows.append({"tissue": t, "spearman_rho": rho, "p_value": p})
    per_tissue = pd.DataFrame(rows)
    per_tissue.to_csv(TABLE_DIR / "sensitivity_jaccard_concordance.csv", index=False)

    main_mean = main_matrix.mean(axis=1)
    jac_mean = jac.mean(axis=1)
    common = main_mean.index.intersection(jac_mean.index)
    rho_global, _ = spearmanr(main_mean.loc[common], jac_mean.loc[common])

    top_main = set(main_mean.sort_values(ascending=False).head(50).index)
    top_jac = set(jac_mean.sort_values(ascending=False).head(50).index)
    overlap_top50 = len(top_main & top_jac)

    res = {
        "n_tissues": len(tissues),
        "per_tissue_rho_median": float(per_tissue["spearman_rho"].median()),
        "per_tissue_rho_q25": float(per_tissue["spearman_rho"].quantile(0.25)),
        "per_tissue_rho_q75": float(per_tissue["spearman_rho"].quantile(0.75)),
        "per_tissue_rho_min": float(per_tissue["spearman_rho"].min()),
        "per_tissue_rho_max": float(per_tissue["spearman_rho"].max()),
        "global_mean_reversal_rho": float(rho_global),
        "top50_overlap_count": int(overlap_top50),
        "top50_overlap_pct": float(overlap_top50 / 50 * 100),
    }
    print(f"  Per-tissue Spearman rho: median={res['per_tissue_rho_median']:.3f} "
          f"[IQR {res['per_tissue_rho_q25']:.3f}-{res['per_tissue_rho_q75']:.3f}]")
    print(f"  Global mean-reversal rho: {rho_global:.3f}")
    print(f"  Top-50 most rejuvenating overlap: {overlap_top50}/50")
    return res


def analysis3_effectsize_fdr(main_matrix):
    print("\n=== Analysis 3: Effect size (z) and BH-FDR ===", flush=True)
    tissues = list(main_matrix.columns)
    z = np.zeros_like(main_matrix.values, dtype=np.float64)
    for j, t in enumerate(tissues):
        col = main_matrix[t].values
        rng = np.random.default_rng(42 + j)
        null_samples = np.empty((1000, len(col)))
        for i in range(1000):
            null_samples[i] = rng.permutation(col)
        null_mean = null_samples.mean(axis=0)
        null_std = null_samples.std(axis=0)
        null_std[null_std == 0] = 1.0
        z[:, j] = (col - null_mean) / null_std

    p = 2 * norm.sf(np.abs(z))
    flat_p = p.ravel()
    n = flat_p.size
    order = np.argsort(flat_p)
    ranked = flat_p[order]
    fdr = ranked * n / (np.arange(1, n + 1))
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    fdr = np.clip(fdr, 0, 1)
    fdr_full = np.empty_like(flat_p)
    fdr_full[order] = fdr

    fdr_mat = fdr_full.reshape(z.shape)
    sig_fdr005 = int((fdr_mat < 0.05).sum())
    sig_fdr010 = int((fdr_mat < 0.10).sum())
    sig_fdr020 = int((fdr_mat < 0.20).sum())
    n_pairs = int(z.size)

    flat_z = z.ravel()
    n_drugs, n_tis = z.shape
    flat_tissue = np.repeat(np.arange(n_tis), n_drugs)
    flat_drug = np.tile(np.arange(n_drugs), n_tis)
    sample_idx = np.argsort(flat_p)[:200]
    sample_df = pd.DataFrame({
        "compound": [main_matrix.index[flat_drug[k]] for k in sample_idx],
        "tissue": [tissues[flat_tissue[k]] for k in sample_idx],
        "reversal_score": [main_matrix.values[flat_drug[k], flat_tissue[k]] for k in sample_idx],
        "z_score": [flat_z[k] for k in sample_idx],
        "p_value": [flat_p[k] for k in sample_idx],
        "fdr_bh": [fdr_full[k] for k in sample_idx],
    })
    sample_df.to_csv(TABLE_DIR / "sensitivity_effectsize_fdr.csv", index=False)

    res = {
        "n_drug_tissue_pairs": n_pairs,
        "n_sig_fdr_0.05": sig_fdr005,
        "n_sig_fdr_0.10": sig_fdr010,
        "n_sig_fdr_0.20": sig_fdr020,
        "pct_sig_fdr_0.05": float(sig_fdr005 / n_pairs * 100),
        "pct_sig_fdr_0.10": float(sig_fdr010 / n_pairs * 100),
        "pct_sig_fdr_0.20": float(sig_fdr020 / n_pairs * 100),
        "z_abs_median": float(np.median(np.abs(z))),
        "z_abs_p95": float(np.percentile(np.abs(z), 95)),
    }
    print(f"  Total pairs: {n_pairs}")
    print(f"  FDR<0.05: {sig_fdr005} ({res['pct_sig_fdr_0.05']:.2f}%)")
    print(f"  FDR<0.10: {sig_fdr010} ({res['pct_sig_fdr_0.10']:.2f}%)")
    print(f"  FDR<0.20: {sig_fdr020} ({res['pct_sig_fdr_0.20']:.2f}%)")
    print(f"  |z| median: {res['z_abs_median']:.3f}, p95: {res['z_abs_p95']:.3f}")
    return res


def main():
    print("Loading main reversal matrix...", flush=True)
    main_matrix = pd.read_csv(TABLE_DIR / "drug_tissue_reversal_matrix.csv", index_col=0)
    print(f"  Shape: {main_matrix.shape}")

    summary = {}

    res1 = analysis1_weak_tissues(main_matrix)
    summary["analysis1_weak_tissues"] = res1

    gene_map = build_gene_map()
    clocks = load_clocks(gene_map)
    compounds, cdown, cup, universe = load_lincs_compounds()

    res2 = analysis2_jaccard_concordance(clocks, compounds, cdown, cup, universe, main_matrix)
    summary["analysis2_jaccard_concordance"] = res2

    res3 = analysis3_effectsize_fdr(main_matrix)
    summary["analysis3_effectsize_fdr"] = res3

    with open(TABLE_DIR / "sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll sensitivity analyses complete. Summary: {TABLE_DIR / 'sensitivity_summary.json'}")


if __name__ == "__main__":
    main()
