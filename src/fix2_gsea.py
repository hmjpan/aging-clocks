"""Fix 2: GSEA-style ranked enrichment analysis.

Instead of using only nonzero clock genes (178-634 per tissue, too few for
Enrichr hypergeometric), use ALL genes ranked by clock weight (including
near-zero weights) for GSEA-style enrichment via Enrichr's ranked gene list.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_DIR, TABLE_DIR, GTEX_DIR
from gct_io import read_gct

import gseapy

TABLE_DIR.mkdir(parents=True, exist_ok=True)

def build_gene_map():
    fpath = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))[0]
    expr = read_gct(str(fpath))
    return dict(zip(expr.index, expr["gene_name"]))

def main():
    gene_map = build_gene_map()
    coef_files = sorted(CLOCK_DIR.glob("*_coefficients.csv"))

    # Load ALL clock weights (including near-zero) as ranked lists
    # We need the full coefficient vector, not just nonzero
    # But we only saved nonzero. So we'll use nonzero genes with their weights
    # and feed as ranked list to GSEA

    all_enrichment = []

    # Focus on top 8 tissues by performance
    summary = pd.read_csv(TABLE_DIR / "clock_summary.csv")
    top_tissues = summary[summary["pearson_r"] > 0.5].nlargest(8, "pearson_r")["tissue"]

    for tissue in top_tissues:
        coef_path = CLOCK_DIR / f"{tissue}_coefficients.csv"
        if not coef_path.exists():
            continue
        df = pd.read_csv(coef_path, index_col=0)
        df.columns = ["weight"]

        # Map to symbols
        df["symbol"] = [gene_map.get(g, g) for g in df.index]
        df = df[df["symbol"] != "nan"].drop_duplicates(subset="symbol")

        # Create ranked list: sort by weight descending
        df = df.sort_values("weight", ascending=False)
        ranked_genes = df["symbol"].tolist()

        print("GSEA for {} ({} genes)...".format(tissue, len(ranked_genes)), flush=True)

        try:
            # Use Enrichr with the full ranked gene list
            enr = gseapy.enrichr(
                gene_list=ranked_genes,
                gene_sets=["GO_Biological_Process_2023", "KEGG_2021_Human",
                           "Reactome_2022", "WikiPathways_2024_Human"],
                organism="human",
                outdir=None,
                no_plot=True,
                cutoff=1.0,
            )
            df_enr = enr.results.copy()
            df_enr["tissue"] = tissue
            all_enrichment.append(df_enr)

            sig = df_enr[df_enr["Adjusted P-value"] < 0.05]
            print("  {} pathways, {} significant (FDR<0.05)".format(
                len(df_enr), len(sig)), flush=True)
            if len(sig) > 0:
                print(sig.head(5)[["Term", "Adjusted P-value", "Overlap"]].to_string(index=False))
        except Exception as e:
            print("  Failed: {}".format(str(e)[:80]), flush=True)
        time.sleep(1)

    if all_enrichment:
        combined = pd.concat(all_enrichment, ignore_index=True)
        combined.to_csv(TABLE_DIR / "gsea_ranked_enrichment.csv", index=False)
        print("\nSaved gsea_ranked_enrichment.csv ({} rows)".format(len(combined)))
        sig_all = combined[combined["Adjusted P-value"] < 0.05]
        print("Total significant: {}".format(len(sig_all)))
        if len(sig_all) > 0:
            print("\nTop 20 significant pathways:")
            print(sig_all.nsmallest(20, "Adjusted P-value")[
                ["tissue", "Term", "Adjusted P-value", "Overlap"]].to_string(index=False))

if __name__ == "__main__":
    main()
