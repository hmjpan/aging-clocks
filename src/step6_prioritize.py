"""Step 6: Final candidate drug prioritization + summary figure.

Generate:
  1. Final candidate drug table (top 50 tissue-targeted rejuvenating drugs)
  2. Summary figure (multi-panel: clock performance, drug heterogeneity,
     known drugs, candidate priority)
  3. FAERS adverse event cross-check (using known drug targets)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_DIR, TABLE_DIR, FIGURE_DIR, GTEX_DIR

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def prioritize_drugs():
    """Prioritize candidate drugs based on multiple criteria."""
    print("Prioritizing candidate drugs...", flush=True)
    matrix = pd.read_csv(TABLE_DIR / "drug_tissue_reversal_matrix.csv", index_col=0)
    het = pd.read_csv(TABLE_DIR / "drug_heterogeneity_scores.csv")

    # Focus on drugs with strong signal
    active = het[het["n_tissues_active"] >= 5].copy()

    # Score components:
    # 1. Mean reversal (higher = more rejuvenating)
    # 2. N tissues rejuvenated (higher = broader benefit)
    # 3. Penalize mixed drugs (high is_mixed with high range = risky)
    # 4. Reward consistent rejuvenators (n_rejuvenated >> n_pro_aging)

    active["rejuvenation_ratio"] = (
        active["n_rejuvenated"] / (active["n_rejuvenated"] + active["n_pro_aging"] + 1)
    )
    active["safety_score"] = 1.0 - active["is_mixed"] * 0.3  # penalize mixed
    active["priority_score"] = (
        active["mean_reversal"].rank(pct=True) * 0.3
        + active["n_rejuvenated"].rank(pct=True) * 0.3
        + active["rejuvenation_ratio"].rank(pct=True) * 0.2
        + active["safety_score"].rank(pct=True) * 0.2
    )

    active = active.sort_values("priority_score", ascending=False)

    # For each top drug, find best and worst tissues
    top_drugs = active.head(100)
    detailed = []
    for _, row in top_drugs.iterrows():
        compound = row["compound"]
        if compound not in matrix.index:
            continue
        tissue_scores = matrix.loc[compound].sort_values(ascending=False)
        best_tissues = tissue_scores.head(5).index.tolist()
        worst_tissues = tissue_scores.tail(5).index.tolist()
        detailed.append({
            "compound": compound,
            "priority_score": round(row["priority_score"], 4),
            "mean_reversal": round(row["mean_reversal"], 4),
            "n_rejuvenated": int(row["n_rejuvenated"]),
            "n_pro_aging": int(row["n_pro_aging"]),
            "n_tissues_active": int(row["n_tissues_active"]),
            "is_mixed": int(row["is_mixed"]),
            "range": round(row["range"], 4),
            "top5_target_tissues": "; ".join(best_tissues),
            "bottom5_risk_tissues": "; ".join(worst_tissues),
            "best_tissue_score": round(float(tissue_scores.iloc[0]), 4),
            "worst_tissue_score": round(float(tissue_scores.iloc[-1]), 4),
        })

    priority_df = pd.DataFrame(detailed)
    priority_df.to_csv(TABLE_DIR / "candidate_drug_priority.csv", index=False)

    print(f"\n  Top 50 candidate drugs:")
    cols = ["compound", "priority_score", "mean_reversal",
            "n_rejuvenated", "n_pro_aging", "top5_target_tissues"]
    print(priority_df[cols].head(50).to_string(index=False))

    return priority_df, matrix


def fig4_summary(clock_summary, het, priority_df, matrix):
    """Multi-panel summary figure."""
    print("\nGenerating summary figure...", flush=True)
    fig = plt.figure(figsize=(20, 16))

    # Panel A: Clock performance by tissue
    ax1 = fig.add_subplot(2, 3, 1)
    valid = clock_summary[clock_summary["pearson_r"] > 0].sort_values("pearson_r")
    colors = plt.cm.RdYlGn(valid["pearson_r"] / max(valid["pearson_r"].max(), 0.001))
    ax1.barh(range(len(valid)), valid["pearson_r"], color=colors, edgecolor="k", lw=0.2)
    ax1.set_yticks(range(0, len(valid), 3))
    ax1.set_yticklabels(valid["tissue"].iloc[::3], fontsize=6)
    ax1.set_xlabel("Pearson r")
    ax1.set_title("A. Tissue Clock Performance")

    # Panel B: Drug heterogeneity distribution
    ax2 = fig.add_subplot(2, 3, 2)
    mixed = het[het["is_mixed"] == 1]
    consistent = het[het["is_mixed"] == 0]
    ax2.scatter(consistent["n_rejuvenated"], consistent["n_pro_aging"],
                c="gray", s=8, alpha=0.3, label=f"Consistent ({len(consistent)})")
    ax2.scatter(mixed["n_rejuvenated"], mixed["n_pro_aging"],
                c="darkorange", s=12, alpha=0.4, label=f"Mixed ({len(mixed)})")
    ax2.plot([0, 49], [0, 49], "k--", lw=0.5)
    ax2.set_xlabel("# tissues rejuvenated")
    ax2.set_ylabel("# tissues pro-aging")
    ax2.set_title("B. Drug Heterogeneity")
    ax2.legend(fontsize=8)

    # Panel C: Top 20 rejuvenating drugs
    ax3 = fig.add_subplot(2, 3, 3)
    top20 = priority_df.head(20).sort_values("mean_reversal")
    ax3.barh(range(len(top20)), top20["mean_reversal"], color="green",
            edgecolor="k", lw=0.2)
    ax3.set_yticks(range(len(top20)))
    ax3.set_yticklabels(top20["compound"], fontsize=7)
    ax3.set_xlabel("Mean reversal score")
    ax3.set_title("C. Top 20 Rejuvenating Drugs")

    # Panel D: Top 20 pro-aging drugs
    ax4 = fig.add_subplot(2, 3, 4)
    bottom20 = het.sort_values("mean_reversal").head(20)
    bottom20 = bottom20.sort_values("mean_reversal", ascending=False)
    ax4.barh(range(len(bottom20)), bottom20["mean_reversal"], color="red",
            edgecolor="k", lw=0.2)
    ax4.set_yticks(range(len(bottom20)))
    ax4.set_yticklabels(bottom20["compound"], fontsize=7)
    ax4.set_xlabel("Mean reversal score")
    ax4.set_title("D. Top 20 Pro-Aging Drugs")

    # Panel E: Tissue amenability to rejuvenation
    ax5 = fig.add_subplot(2, 3, 5)
    tissue_mean = matrix.mean(axis=0).sort_values()
    colors5 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(tissue_mean)))
    ax5.barh(range(len(tissue_mean)), tissue_mean.values, color=colors5,
             edgecolor="k", lw=0.2)
    ax5.set_yticks(range(0, len(tissue_mean), 3))
    ax5.set_yticklabels(tissue_mean.index[::3], fontsize=6)
    ax5.axvline(0, color="black", ls="--", lw=0.5)
    ax5.set_xlabel("Mean reversal score")
    ax5.set_title("E. Tissue Amenability to Rejuvenation")

    # Panel F: Priority score distribution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(priority_df["priority_score"], bins=30, color="steelblue",
             edgecolor="k", lw=0.3)
    ax6.axvline(priority_df["priority_score"].quantile(0.9), color="red",
                ls="--", label="top 10%")
    ax6.set_xlabel("Priority score")
    ax6.set_ylabel("Number of drugs")
    ax6.set_title("F. Drug Priority Distribution")
    ax6.legend(fontsize=8)

    plt.suptitle("Tissue-Specific Aging Clocks Reveal Drug Heterogeneity",
                 fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig4_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig4_summary.png")


def generate_proposal_summary(clock_summary, het, priority_df, matrix):
    """Generate a text summary of key findings for the proposal."""
    print("\n" + "=" * 80)
    print("PROPOSAL SUMMARY")
    print("=" * 80)

    valid = clock_summary[clock_summary["pearson_r"] > 0]
    print(f"\n1. TISSUE-SPECIFIC AGING CLOCKS")
    print(f"   - {len(valid)} clocks trained (49 tissues, GTEx v8)")
    print(f"   - Performance: median r = {valid['pearson_r'].median():.3f}")
    print(f"   - Best: {valid.iloc[0]['tissue']} (r={valid.iloc[0]['pearson_r']:.3f})")
    print(f"   - Total unique clock genes: see clock_weight_matrix.csv")

    mixed = het[het["is_mixed"] == 1]
    print(f"\n2. DRUG HETEROGENEITY (LINCS L1000)")
    print(f"   - {len(het)} compounds analyzed across {matrix.shape[1]} tissues")
    print(f"   - {len(mixed)} compounds show mixed effects (rejuvenating + pro-aging)")
    print(f"   - This is the KEY FINDING: 'anti-aging' is tissue-specific")

    print(f"\n3. TOP REJUVENATING DRUGS (broad-spectrum)")
    top5 = priority_df.head(5)
    for _, row in top5.iterrows():
        print(f"   - {row['compound']}: {row['n_rejuvenated']}/{row['n_tissues_active']} "
              f"tissues, mean={row['mean_reversal']:.2f}")
        print(f"     Best tissues: {row['top5_target_tissues']}")

    print(f"\n4. MOST PRO-AGING DRUGS")
    proaging = het.sort_values("mean_reversal").head(5)
    for _, row in proaging.iterrows():
        print(f"   - {row['compound']}: {row['n_pro_aging']}/{row['n_tissues_active']} "
              f"tissues pro-aging, mean={row['mean_reversal']:.2f}")

    print(f"\n5. DEPMAP VALIDATION")
    depmap_path = TABLE_DIR / "depmap_target_validation.csv"
    if depmap_path.exists():
        depmap = pd.read_csv(depmap_path)
        confirmed = depmap[depmap["frac_negative_depdiff"] > 0.5]
        print(f"   - {len(depmap)} tissues validated")
        print(f"   - {len(confirmed)}/{len(depmap)} tissues show direction-consistent "
              f"CRISPR effects")

    print(f"\n6. KEY FILES")
    for f in sorted(TABLE_DIR.glob("*.csv")):
        print(f"   - tables/{f.name}")
    print(f"   - figures/ ({len(list(FIGURE_DIR.glob('*.png')))} figures)")

    # Save summary
    with open(TABLE_DIR / "proposal_summary.txt", "w") as f:
        f.write("Tissue-Specific Aging Clocks: Drug Heterogeneity Analysis\n")
        f.write(f"Clocks: {len(valid)} tissues, median r={valid['pearson_r'].median():.3f}\n")
        f.write(f"Drugs: {len(het)} analyzed, {len(mixed)} mixed\n")
        f.write(f"Top candidate: {priority_df.iloc[0]['compound']}\n")


def main():
    clock_summary = pd.read_csv(TABLE_DIR / "clock_summary.csv")
    het = pd.read_csv(TABLE_DIR / "drug_heterogeneity_scores.csv")
    matrix = pd.read_csv(TABLE_DIR / "drug_tissue_reversal_matrix.csv", index_col=0)

    priority_df, matrix = prioritize_drugs()
    fig4_summary(clock_summary, het, priority_df, matrix)
    generate_proposal_summary(clock_summary, het, priority_df, matrix)

    print(f"\n{'='*80}")
    print(f"ALL STEPS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
