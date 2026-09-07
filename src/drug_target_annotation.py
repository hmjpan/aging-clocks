"""Drug target pathway annotation for top drugs and FDA-approved drugs."""
import os, sys, re
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import TABLE_DIR

TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Known drug target annotations (manually curated from DrugBank/LINCS metadata)
DRUG_TARGETS = {
    # Kinase inhibitors
    "sunitinib": {"targets": "VEGFR2, PDGFR, FLT3, c-Kit", "pathway": "Receptor tyrosine kinase / VEGF signaling", "class": "Multi-kinase inhibitor"},
    "pazopanib": {"targets": "VEGFR1-3, PDGFR, c-Kit", "pathway": "Angiogenesis / VEGF signaling", "class": "Multi-kinase inhibitor"},
    "gefitinib": {"targets": "EGFR", "pathway": "EGFR/ERBB signaling", "class": "EGFR inhibitor"},
    "lapatinib": {"targets": "EGFR, HER2", "pathway": "EGFR/ERBB signaling", "class": "Dual HER inhibitor"},
    "palbociclib": {"targets": "CDK4, CDK6", "pathway": "Cell cycle / CDK signaling", "class": "CDK4/6 inhibitor"},
    "ruxolitinib": {"targets": "JAK1, JAK2", "pathway": "JAK-STAT signaling", "class": "JAK inhibitor"},
    "crizotinib": {"targets": "ALK, MET, ROS1", "pathway": "ALK/MET signaling", "class": "ALK/MET inhibitor"},
    "vandetanib": {"targets": "VEGFR, EGFR, RET", "pathway": "VEGF/EGFR signaling", "class": "Multi-kinase inhibitor"},
    "bosutinib": {"targets": "Bcr-Abl, Src", "pathway": "BCR-ABL signaling", "class": "TKI"},
    "dasatinib": {"targets": "Bcr-Abl, Src family", "pathway": "BCR-ABL/Src signaling", "class": "TKI"},
    "trametinib": {"targets": "MEK1, MEK2", "pathway": "MAPK/ERK signaling", "class": "MEK inhibitor"},
    "selumetinib": {"targets": "MEK1, MEK2", "pathway": "MAPK/ERK signaling", "class": "MEK inhibitor"},
    "rucaparib": {"targets": "PARP1, PARP2", "pathway": "DNA damage repair / PARP", "class": "PARP inhibitor"},
    "foretinib": {"targets": "MET, VEGFR, RON", "pathway": "HGF/MET signaling", "class": "MET inhibitor"},
    "tozasertib": {"targets": "Aurora A/B/C", "pathway": "Mitotic / Aurora kinase", "class": "Aurora inhibitor"},
    "entfinib": {"targets": "HER2, EGFR", "pathway": "ERBB signaling", "class": "HER inhibitor"},
    # Preclinical
    "PI-103": {"targets": "PI3K, mTOR, DNA-PK", "pathway": "PI3K/AKT/mTOR signaling", "class": "PI3K/mTOR dual inhibitor"},
    "TGX-221": {"targets": "PI3K beta", "pathway": "PI3K/AKT signaling", "class": "PI3K beta inhibitor"},
    "NU-7441": {"targets": "DNA-PK", "pathway": "DNA damage repair", "class": "DNA-PK inhibitor"},
    "A 769662": {"targets": "AMPK", "pathway": "AMPK / energy sensing", "class": "AMPK activator"},
    "Y-27632": {"targets": "ROCK1, ROCK2", "pathway": "RhoA/ROCK signaling", "class": "ROCK inhibitor"},
    "VX-745": {"targets": "p38 MAPK", "pathway": "p38 MAPK / inflammation", "class": "p38 inhibitor"},
    "doramapimod": {"targets": "p38 MAPK", "pathway": "p38 MAPK / inflammation", "class": "p38 inhibitor"},
    "SB-239063": {"targets": "p38 MAPK", "pathway": "p38 MAPK / inflammation", "class": "p38 inhibitor"},
    "SB-525334": {"targets": "TGFBR1 (ALK5)", "pathway": "TGF-beta signaling", "class": "TGF-beta inhibitor"},
    "GSK-J1": {"targets": "JMJD3 (KDM6B)", "pathway": "Histone demethylation / epigenetic", "class": "H3K27 demethylase inhibitor"},
    "PFI-1": {"targets": "BRD4, BRD2", "pathway": "BET / chromatin reader", "class": "BET inhibitor"},
    "I-BET": {"targets": "BRD2, BRD3, BRD4", "pathway": "BET / chromatin reader", "class": "BET inhibitor"},
    "CG-930": {"targets": "JNK1, JNK2, JNK3", "pathway": "JNK / stress signaling", "class": "JNK inhibitor"},
    "OSI-930": {"targets": "c-Kit, VEGFR", "pathway": "VEGF/c-Kit signaling", "class": "Multi-kinase inhibitor"},
    "GDC-0879": {"targets": "BRAF", "pathway": "MAPK/RAF signaling", "class": "BRAF inhibitor"},
    "AZD-1480": {"targets": "JAK1, JAK2", "pathway": "JAK-STAT signaling", "class": "JAK inhibitor"},
    "PF-04217903": {"targets": "c-MET", "pathway": "HGF/MET signaling", "class": "MET inhibitor"},
    "JW55": {"targets": "TNKS1, TNKS2", "pathway": "Wnt/beta-catenin signaling", "class": "Tankyrase inhibitor"},
    "MK-2206": {"targets": "AKT1, AKT2, AKT3", "pathway": "PI3K/AKT signaling", "class": "AKT inhibitor"},
    "alvocidib": {"targets": "CDK1, CDK2, CDK4, CDK7, CDK9", "pathway": "Cell cycle / CDK signaling", "class": "Pan-CDK inhibitor"},
    "dinaciclib": {"targets": "CDK1, CDK2, CDK5, CDK9", "pathway": "Cell cycle / CDK signaling", "class": "CDK inhibitor"},
    "HG-9-91-01": {"targets": "GSK3, CLK, DYRK", "pathway": "Multiple kinase / glycogen synthase", "class": "Multi-kinase inhibitor"},
    "XMD-1499": {"targets": "ULK1, ULK2", "pathway": "Autophagy initiation", "class": "ULK inhibitor"},
    # Others
    "metformin": {"targets": "AMPK (indirect), mTORC1", "pathway": "AMPK / mTOR / energy sensing", "class": "Biguanide"},
    "rapamycin": {"targets": "mTORC1", "pathway": "mTOR signaling", "class": "mTOR inhibitor"},
    "sirolimus": {"targets": "mTORC1", "pathway": "mTOR signaling", "class": "mTOR inhibitor"},
    "resveratrol": {"targets": "SIRT1 (activator), AMPK", "pathway": "Sirtuin / AMPK signaling", "class": "Polyphenol"},
    "hydroflumethiazide": {"targets": "NCC (SLC12A3)", "pathway": "Sodium-chloride cotransporter", "class": "Thiazide diuretic"},
    "lithium": {"targets": "GSK3beta, IMPase", "pathway": "GSK3 / Wnt signaling", "class": "Mood stabilizer"},
    "acarbose": {"targets": "Alpha-glucosidase", "pathway": "Carbohydrate metabolism", "class": "Anti-diabetic"},
    "spermidine": {"targets": "eIF5A, autophagy", "pathway": "Autophagy / polyamine", "class": "Polyamine"},
    "nicotinamide": {"targets": "NAD+ precursor, PARP, SIRT1", "pathway": "NAD+ metabolism", "class": "Vitamin B3"},
    "quercetin": {"targets": "Senescence markers, PI3K", "pathway": "Flavonoid / senolytic", "class": "Flavonoid"},
    "fisetin": {"targets": "Senescence markers", "pathway": "Flavonoid / senolytic", "class": "Flavonoid"},
    "dasatinib": {"targets": "Bcr-Abl, Src", "pathway": "BCR-ABL/Src signaling", "class": "TKI / senolytic"},
    "pravastatin sodium": {"targets": "HMG-CoA reductase", "pathway": "Cholesterol biosynthesis", "class": "Statin"},
    "ondansetron hydrochloride": {"targets": "5-HT3 receptor", "pathway": "Serotonergic signaling", "class": "5-HT3 antagonist"},
}


def annotate_drugs():
    """Annotate top drugs with target and pathway information."""
    het = pd.read_csv(TABLE_DIR / "drug_heterogeneity_scores.csv")
    fda = pd.read_csv(TABLE_DIR / "fda_approved_repurposing.csv")
    matrix = pd.read_csv(TABLE_DIR / "drug_tissue_reversal_matrix.csv", index_col=0)

    # Get top 50 rejuvenating and pro-aging
    top_reju = het.sort_values("mean_reversal", ascending=False).head(50)
    top_pro = het.sort_values("mean_reversal").head(50)

    # Annotate
    annotated = []
    for _, row in het.iterrows():
        compound = row["compound"]
        # Try to match
        target_info = None
        for key, info in DRUG_TARGETS.items():
            if key.lower() == compound.lower() or key.lower() in compound.lower():
                target_info = info
                break

        # Get best and worst tissues
        if compound in matrix.index:
            scores = matrix.loc[compound].sort_values(ascending=False)
            best_tissue = scores.index[0]
            best_score = scores.iloc[0]
            worst_tissue = scores.index[-1]
            worst_score = scores.iloc[-1]
        else:
            best_tissue = worst_tissue = ""
            best_score = worst_score = 0

        is_fda = compound in fda["compound"].values

        annotated.append({
            "compound": compound,
            "targets": target_info["targets"] if target_info else "Unknown",
            "pathway": target_info["pathway"] if target_info else "Unknown",
            "drug_class": target_info["class"] if target_info else "Unknown",
            "mean_reversal": row["mean_reversal"],
            "n_rejuvenated": row["n_rejuvenated"],
            "n_pro_aging": row["n_pro_aging"],
            "is_mixed": row["is_mixed"],
            "best_tissue": best_tissue,
            "best_score": round(best_score, 3),
            "worst_tissue": worst_tissue,
            "worst_score": round(worst_score, 3),
            "fda_approved": is_fda,
        })

    df = pd.DataFrame(annotated)
    df = df.sort_values("mean_reversal", ascending=False)
    df.to_csv(TABLE_DIR / "drug_target_pathway_annotation.csv", index=False)

    # Summary by pathway
    known = df[df["pathway"] != "Unknown"]
    print("Annotated {}/{} drugs with known targets".format(len(known), len(df)))
    print()

    # Pathway enrichment among rejuvenating vs pro-aging
    reju = known[known["mean_reversal"] > 0.3]
    pro = known[known["mean_reversal"] < -0.3]

    print("=== Rejuvenating drugs (mean > 0.3): {} known ===".format(len(reju)))
    pathway_counts = Counter(reju["pathway"].values)
    for p, c in pathway_counts.most_common(15):
        print("  {}: {}".format(p, c))

    print()
    print("=== Pro-aging drugs (mean < -0.3): {} known ===".format(len(pro)))
    pathway_counts = Counter(pro["pathway"].values)
    for p, c in pathway_counts.most_common(15):
        print("  {}: {}".format(p, c))

    print()
    print("=== Heterogeneous drugs (mixed, known targets) ===")
    mixed_known = known[known["is_mixed"] == 1].sort_values(
        "mean_reversal", key=abs, ascending=False).head(30)
    print(mixed_known[["compound", "targets", "pathway", "mean_reversal",
                        "n_rejuvenated", "n_pro_aging"]].to_string(index=False))

    # Save pathway summary table
    pathway_summary = []
    for pathway in set(known["pathway"]):
        sub = known[known["pathway"] == pathway]
        pathway_summary.append({
            "pathway": pathway,
            "n_drugs": len(sub),
            "mean_reversal": round(sub["mean_reversal"].mean(), 3),
            "n_rejuvenating": (sub["mean_reversal"] > 0).sum(),
            "n_pro_aging": (sub["mean_reversal"] < 0).sum(),
            "n_mixed": sub["is_mixed"].sum(),
        })
    ps_df = pd.DataFrame(pathway_summary).sort_values("n_drugs", ascending=False)
    ps_df.to_csv(TABLE_DIR / "pathway_level_summary.csv", index=False)

    print()
    print("=== Pathway-level summary ===")
    print(ps_df.head(20).to_string(index=False))

    return df


if __name__ == "__main__":
    annotate_drugs()
