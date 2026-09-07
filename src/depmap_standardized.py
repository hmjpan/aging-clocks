"""DepMap standardized aging scores - fixed version."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_DIR, GTEX_DIR, DEPMAP_DIR, TABLE_DIR
from gct_io import read_gct, build_tissue_sample_table

meta = build_tissue_sample_table()
models = pd.read_csv(DEPMAP_DIR / "Model.csv")
expr_df = pd.read_csv(DEPMAP_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
expr_df.columns = [c.split(" (")[0] if " (" in c else c for c in expr_df.columns]

lineage_map = {
    "lung": "lung", "breast": "breast_mammary_tissue",
    "skin": "skin_sun_exposed_lower_leg", "blood": "whole_blood",
    "colon": "colon_transverse", "liver": "liver", "brain": "brain_cortex",
    "prostate": "prostate", "pancreas": "pancreas", "kidney": "kidney_cortex",
    "ovary": "ovary", "stomach": "stomach", "thyroid": "thyroid",
    "uterus": "uterus", "bone": "muscle_skeletal",
}
clock_tissues = {f.replace("_coefficients.csv", "") for f in os.listdir(CLOCK_DIR)
                 if f.endswith("_coefficients.csv")}
cell_tissue = {}
for _, row in models.iterrows():
    lin = str(row.get("OncotreeLineage", "")).lower()
    mid = row.get("ModelID", "")
    for key, tissue in lineage_map.items():
        if key in lin and tissue in clock_tissues:
            cell_tissue[mid] = tissue
            break

# gene map (Ensembl -> symbol) from any GCT
fpath = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))[0]
gtex_expr = read_gct(str(fpath))
gene_map = dict(zip(gtex_expr.index, gtex_expr["gene_name"]))

print("Gene map size:", len(gene_map))
print("Cell-tissue mapping:", len(cell_tissue), "cells across",
      len(set(cell_tissue.values())), "tissues")

results = []
for tissue in sorted(set(cell_tissue.values())):
    coef_path = CLOCK_DIR / "{}_coefficients.csv".format(tissue)
    if not coef_path.exists():
        continue
    coef = pd.read_csv(coef_path, index_col=0)
    coef.columns = ["weight"]
    if len(coef) < 10:
        continue

    # GTEx training mu/sd for the genes in this clock
    # Use the tissue's own GCT: filter + log2 same as training
    tfpath = GTEX_DIR / "tpm_{}.gct.gz".format(tissue)
    if not tfpath.exists():
        print("  SKIP {}: no GCT".format(tissue))
        continue
    texpr = read_gct(str(tfpath))
    tsamples = [c for c in texpr.columns if c != "gene_name" and c in set(meta["SAMPID"])]
    if len(tsamples) < 10:
        print("  SKIP {}: too few samples".format(tissue))
        continue
    tmat = texpr.loc[:, tsamples]
    present = (tmat >= 1.0).mean(axis=1)
    tmat = tmat.loc[present[present >= 0.10].index]
    X_t = np.log2(tmat.values.astype(np.float64) + 1.0)
    mu_g = X_t.mean(axis=1)
    sd_g = X_t.std(axis=1)
    sd_g[sd_g == 0] = 1.0
    gene_mu_sd = dict(zip(tmat.index, zip(mu_g, sd_g)))

    # Map clock gene Ensembl -> (symbol, weight), require mu/sd by Ensembl and symbol in DepMap
    mapped = {}
    for ensg, w in coef["weight"].items():
        sym = gene_map.get(ensg, "")
        if sym and ensg in gene_mu_sd and sym in expr_df.columns:
            mapped[sym] = (w, gene_mu_sd[ensg][0], gene_mu_sd[ensg][1])

    if len(mapped) < 10:
        print("  SKIP {}: only {} genes mappable".format(tissue, len(mapped)))
        continue

    matching = [c for c, t in cell_tissue.items() if t == tissue and c in expr_df.index]
    if len(matching) < 5:
        print("  SKIP {}: only {} cells".format(tissue, len(matching)))
        continue

    # Standardized projection: sum((depmap - mu)/sd * weight)
    syms = list(mapped.keys())
    W = np.array([mapped[s][0] for s in syms])
    MU = np.array([mapped[s][1] for s in syms])
    SD = np.array([mapped[s][2] for s in syms])
    sub = expr_df.loc[matching, syms].values  # cells x genes
    std_expr = (sub - MU) / SD
    scores = std_expr @ W

    results.append({
        "tissue": tissue,
        "n_cells": len(matching),
        "n_genes_used": len(mapped),
        "aging_score_mean": round(float(np.mean(scores)), 4),
        "aging_score_std": round(float(np.std(scores)), 4),
        "n_genes_in_depmap": len(mapped),
    })
    print("  {}: {} cells, {} genes, score {:.2f}+/-{:.2f}".format(
        tissue, len(matching), len(mapped), np.mean(scores), np.std(scores)))

df = pd.DataFrame(results)
df.to_csv(TABLE_DIR / "depmap_standardized_scores.csv", index=False)
print("\nSaved:", len(df), "tissues to depmap_standardized_scores.csv")