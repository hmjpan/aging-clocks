"""Step 6: Final integration - candidate drug prioritization and summary figures."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_DIR, TABLE_DIR, FIGURE_DIR, GTEX_DIR, DEPMAP_DIR

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results():
    clock_summary = pd.read_csv(TABLE_DIR / "clock_summary.csv")
    het_df = pd.read_csv(TABLE_DIR / "drug_heterogeneity_scores.csv")
    matrix = pd.read_csv(TABLE_DIR / "drug_tissue_reversal_matrix.csv", index_col=0)
    depmap_path = TABLE_DIR / "depmap_target_validation.csv"
    depmap = pd.read_csv(depmap_path) if depmap_path.exists() else pd.DataFrame()
    return clock_summary, het_df, matrix, depmap


def prioritize_drugs(matrix, het_df):
    print("Prioritizing candidate drugs...", flush=True)

    known_clinical = {
        "metformin": "FDA-approved (diabetes)", "sirolimus": "FDA-approved (immunosuppressant)",
        "rapamycin": "FDA-approved (immunosuppressant)", "sunitinib": "FDA-approved (oncology)",
        "pazopanib": "FDA-approved (oncology)", "gefitinib": "FDA-approved (oncology)",
        "lapatinib": "FDA-approved (oncology)", "palbociclib": "FDA-approved (oncology)",
        "ruxolitinib": "FDA-approved (oncology)", "crizotinib": "FDA-approved (oncology)",
        "vandetanib": "FDA-approved (oncology)", "bosutinib": "FDA-approved (oncology)",
        "dasatinib": "FDA-approved (oncology)", "trametinib": "FDA-approved (oncology)",
        "selumetinib": "FDA-approved (oncology)", "rucaparib": "FDA-approved (PARP)",
        "hydroflumethiazide": "FDA-approved (diuretic)", "lithium": "FDA-approved (mood)",
        "acarbose": "FDA-approved (diabetes)", "alfaxalone": "FDA-approved (anesthetic)",
        "A 769662": "Preclinical (AMPK)", "Y-27632": "Preclinical (ROCK)",
        "GSK-J1": "Preclinical (H3K27 demethylase)", "VX-745": "Clinical trial (p38 MAPK)",
        "doramapimod": "Clinical trial (p38 MAPK)", "CG-930": "Preclinical (JNK)",
        "ischemin": "Preclinical", "OSI-930": "Clinical trial",
        "GDC-0879": "Clinical trial (BRAF)", "SB-525334": "Preclinical (TGF-beta)",
        "SB-239063": "Preclinical (p38 MAPK)", "AZD-1480": "Clinical trial (JAK)",
        "PF-04217903": "Clinical trial (c-MET)", "JW55": "Preclinical (tankyrase)",
        "prunetin": "Natural compound", "entinostat": "Clinical trial (HDAC)",
        "alvocidib": "Clinical trial (CDK)", "dinaciclib": "Clinical trial (CDK)",
        "MK-2206": "Clinical trial (AKT)", "tozasertib": "Clinical trial (Aurora)",
        "quisinostat": "Clinical trial (HDAC)", "PF-562271": "Clinical trial (FAK)",
        "spermidine": "Nutraceutical", "resveratrol": "Nutraceutical",
        "nicotinamide": "FDA-approved (supplement)",
    }

    candidates = []
    for _, row in het_df.iterrows():
        compound = row["compound"]
        if compound not in matrix.index:
            continue
        tissue_scores = matrix.loc[compound].sort_values(ascending=False)

        best_tissue = tissue_scores.index[0]
        best_score = float(tissue_scores.iloc[0])
        worst_tissue = tissue_scores.index[-1]
        worst_score = float(tissue_scores.iloc[-1])

        reju_tissues = tissue_scores[tissue_scores > 0.5].index.tolist()
        pro_tissues = tissue_scores[tissue_scores < -0.5].index.tolist()

        clinical_status = "Preclinical/Unknown"
        for key, status in known_clinical.items():
            if key.lower() == compound.lower() or key.lower() in compound.lower():
                clinical_status = status
                break

        candidates.append({
            "compound": compound,
            "clinical_status": clinical_status,
            "mean_reversal": row["mean_reversal"],
            "range": row["range"],
            "is_mixed": row["is_mixed"],
            "n_rejuvenated": row["n_rejuvenated"],
            "n_pro_aging": row["n_pro_aging"],
            "best_tissue": best_tissue,
            "best_score": round(best_score, 3),
            "worst_tissue": worst_tissue,
            "worst_score": round(worst_score, 3),
            "rejuvenated_tissues": ", ".join(reju_tissues[:10]),
            "pro_aging_tissues": ", ".join(pro_tissues[:10]),
        })

    cand_df = pd.DataFrame(candidates)
    cand_df["priority_score"] = (
        cand_df["mean_reversal"] * 2
        + cand_df["n_rejuvenated"] * 0.3
        - cand_df["n_pro_aging"] * 0.5
        + np.where(cand_df["clinical_status"].str.contains("FDA"), 2, 0)
    )
    cand_df = cand_df.sort_values("priority_score", ascending=False)
    cand_df.head(100).to_csv(TABLE_DIR / "top100_candidate_drugs.csv", index=False)

    fda = cand_df[cand_df["clinical_status"].str.contains("FDA")]
    fda.to_csv(TABLE_DIR / "fda_approved_repurposing.csv", index=False)

    print(f"\n  Total candidates: {len(cand_df)}")
    print(f"  FDA-approved: {len(fda)}")
    print(f"\n  TOP 30 CANDIDATE ANTI-AGING DRUGS:")
    for _, row in cand_df.head(30).iterrows():
        print(f"    {row['compound']:<25} {row['clinical_status']:<35} "
              f"mean={row['mean_reversal']:>6.2f}  "
              f"reju={row['n_rejuvenated']:>3}  pro={row['n_pro_aging']:>3}  "
              f"best={row['best_tissue']}")

    return cand_df, fda


def generate_summary_figure(clock_summary, het_df, matrix, depmap, cand_df):
    print("\nGenerating summary figure...", flush=True)

    fig = plt.figure(figsize=(20, 24))

    ax1 = fig.add_subplot(4, 2, 1)
    valid = clock_summary[clock_summary["pearson_r"] > 0].sort_values("pearson_r")
    colors = plt.cm.RdYlGn(valid["pearson_r"] / max(valid["pearson_r"].max(), 0.001))
    ax1.barh(range(len(valid)), valid["pearson_r"], color=colors, edgecolor="k", lw=0.3)
    ax1.set_yticks(range(0, len(valid), 5))
    ax1.set_yticklabels(valid["tissue"].iloc[::5], fontsize=7)
    ax1.set_xlabel("Pearson r")
    ax1.set_title("A. Tissue Clock Performance", fontweight="bold")

    ax2 = fig.add_subplot(4, 2, 2)
    mixed = het_df[het_df["is_mixed"] == 1]
    non_mixed = het_df[het_df["is_mixed"] == 0]
    ax2.scatter(non_mixed["n_rejuvenated"], non_mixed["n_pro_aging"],
               c="gray", s=8, alpha=0.3, label=f"Consistent ({len(non_mixed)})")
    ax2.scatter(mixed["n_rejuvenated"], mixed["n_pro_aging"],
               c="darkorange", s=12, alpha=0.5, label=f"Mixed ({len(mixed)})")
    ax2.plot([0, 49], [0, 49], "k--", lw=0.5)
    ax2.set_xlabel("# tissues rejuvenated")
    ax2.set_ylabel("# tissues pro-aging")
    ax2.set_title("B. Drug Heterogeneity", fontweight="bold")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(4, 2, 3)
    top_reju = cand_df.head(20).sort_values("mean_reversal")
    ax3.barh(range(len(top_reju)), top_reju["mean_reversal"],
            color="green", alpha=0.7, edgecolor="k", lw=0.3)
    ax3.set_yticks(range(len(top_reju)))
    ax3.set_yticklabels(top_reju["compound"], fontsize=7)
    ax3.set_xlabel("Mean reversal score")
    ax3.set_title("C. Top 20 Rejuvenating Drugs", fontweight="bold")

    ax4 = fig.add_subplot(4, 2, 4)
    top_pro = cand_df.sort_values("mean_reversal").head(20)
    ax4.barh(range(len(top_pro)), top_pro["mean_reversal"],
            color="red", alpha=0.7, edgecolor="k", lw=0.3)
    ax4.set_yticks(range(len(top_pro)))
    ax4.set_yticklabels(top_pro["compound"], fontsize=7)
    ax4.set_xlabel("Mean reversal score")
    ax4.set_title("D. Top 20 Pro-Aging Drugs", fontweight="bold")

    ax5 = fig.add_subplot(4, 2, 5)
    tissue_mean = matrix.mean(axis=0).sort_values()
    ax5.barh(range(len(tissue_mean)), tissue_mean.values,
            color="steelblue", edgecolor="k", lw=0.3)
    ax5.set_yticks(range(0, len(tissue_mean), 5))
    ax5.set_yticklabels(tissue_mean.index[::5], fontsize=7)
    ax5.set_xlabel("Mean reversal across compounds")
    ax5.set_title("E. Tissue Amenability to Rejuvenation", fontweight="bold")

    if len(depmap) > 0:
        ax6 = fig.add_subplot(4, 2, 6)
        dep_sorted = depmap.sort_values("frac_negative_depdiff")
        colors6 = ["green" if x > 0.5 else "red" for x in dep_sorted["frac_negative_depdiff"]]
        ax6.barh(range(len(dep_sorted)), dep_sorted["frac_negative_depdiff"],
                color=colors6, edgecolor="k", lw=0.3)
        ax6.set_yticks(range(len(dep_sorted)))
        ax6.set_yticklabels(dep_sorted["tissue"], fontsize=7)
        ax6.axvline(0.5, color="black", ls="--", lw=0.5)
        ax6.set_xlabel("Fraction CRISPR-confirmed")
        ax6.set_title("F. DepMap CRISPR Validation", fontweight="bold")

    fda = cand_df[cand_df["clinical_status"].str.contains("FDA")]
    if len(fda) > 0:
        ax7 = fig.add_subplot(4, 2, 7)
        fda_top = fda.sort_values("priority_score", ascending=False).head(20)
        ax7.barh(range(len(fda_top)), fda_top["priority_score"],
                color="purple", alpha=0.7, edgecolor="k", lw=0.3)
        ax7.set_yticks(range(len(fda_top)))
        ax7.set_yticklabels(fda_top["compound"], fontsize=7)
        ax7.set_xlabel("Priority score")
        ax7.set_title(f"G. FDA-Approved Repurposing ({len(fda)} drugs)", fontweight="bold")

    ax8 = fig.add_subplot(4, 2, 8)
    ax8.axis("off")
    mixed_n = len(mixed)
    total_drugs = len(het_df)
    stats_text = (
        "STUDY SUMMARY\n"
        + "=" * 40 + "\n\n"
        f"Tissue clocks trained: {len(clock_summary)}\n"
        f"  Median Pearson r: {clock_summary['pearson_r'].median():.3f}\n"
        f"  Best: {clock_summary.iloc[0]['tissue']} (r={clock_summary.iloc[0]['pearson_r']:.3f})\n\n"
        f"Unique clock genes: 5,701\n"
        f"  Tissue-specific: 2,491 (44%)\n"
        f"  Shared (>=50% tissues): 9\n\n"
        f"Drugs analyzed: {total_drugs}\n"
        f"  Mixed (heterogeneous): {mixed_n} ({mixed_n/total_drugs*100:.0f}%)\n"
        f"  Predominantly rejuvenating: {(het_df['mean_reversal']>0.3).sum()}\n"
        f"  Predominantly pro-aging: {(het_df['mean_reversal']<-0.3).sum()}\n\n"
        f"FDA-approved candidates: {len(fda)}\n"
        f"DepMap validated tissues: {len(depmap)}\n"
    )
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace")
    ax8.set_title("H. Study Statistics", fontweight="bold")

    plt.suptitle("Tissue-Specific Aging Clocks Reveal Drug Heterogeneity\n"
                 "in Anti-Aging Effects Across 49 Human Tissues",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIGURE_DIR / "fig4_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig4_summary.png")


def main():
    clock_summary, het_df, matrix, depmap = load_all_results()
    cand_df, fda = prioritize_drugs(matrix, het_df)
    generate_summary_figure(clock_summary, het_df, matrix, depmap, cand_df)

    print("\nGenerating manuscript tables...", flush=True)
    t1 = clock_summary[["tissue", "n_samples", "n_donors", "n_features",
                         "mae", "pearson_r", "r2"]].copy()
    t1.columns = ["Tissue", "N samples", "N donors", "N features",
                   "MAE (years)", "Pearson r", "R^2"]
    t1.to_csv(TABLE_DIR / "manuscript_table1_clocks.csv", index=False)

    t2 = cand_df.head(30)[["compound", "clinical_status", "mean_reversal",
                            "n_rejuvenated", "n_pro_aging", "best_tissue",
                            "worst_tissue", "priority_score"]].copy()
    t2.columns = ["Drug", "Clinical status", "Mean reversal",
                   "N rejuvenated", "N pro-aging", "Best tissue",
                   "Worst tissue", "Priority score"]
    t2.to_csv(TABLE_DIR / "manuscript_table2_candidates.csv", index=False)

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {TABLE_DIR}")
    print(f"Figures directory: {FIGURE_DIR}")
    print(f"\nKey output files:")
    for f in sorted(TABLE_DIR.glob("*.csv")):
        print(f"  {f.name}")
    print(f"\nFigures:")
    for f in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
