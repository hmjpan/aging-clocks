"""Parse LINCS L1000 metadata to understand data structure."""
import gzip
import os
import sys

import pandas as pd

LINCS_DIR = r"H:\lunwencp\shuailao\data\lincs"


def read_gz(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return pd.read_csv(f, sep="\t", low_memory=False)


print("=== Gene Info (landmark genes) ===")
lm = read_gz(os.path.join(LINCS_DIR, "GSE92742_Broad_LINCS_gene_info_delta_landmark.txt.gz"))
print(f"Landmark genes: {len(lm)}")
print(f"Cols: {list(lm.columns)}")
print(lm.head(3).to_string())
print()

print("=== Gene Info (all genes) ===")
gi = read_gz(os.path.join(LINCS_DIR, "GSE92742_Broad_LINCS_gene_info.txt.gz"))
print(f"All genes: {len(gi)}")
print(f"Cols: {list(gi.columns)}")
print(gi["pr_gene_id"].dtype, gi["pr_gene_symbol"].dtype)
# Check landmark vs inferred
print(f"Gene types: {gi['pr_is_lm'].value_counts().to_dict()}")
print()

print("=== Pert Info (compounds) ===")
pi = read_gz(os.path.join(LINCS_DIR, "GSE92742_Broad_LINCS_pert_info.txt.gz"))
print(f"Perturbagens: {len(pi)}")
print(f"Cols: {list(pi.columns)}")
print(f"Pert types: {pi['pert_type'].value_counts().head(10).to_dict()}")
# Count compounds
chem = pi[pi["pert_type"] == "trt_cp"]
print(f"Chemical perturbagens: {len(chem)}")
print(chem[["pert_id", "pert_iname", "pert_type"]].head(5).to_string())
print()

print("=== Sig Info (signatures) ===")
si = read_gz(os.path.join(LINCS_DIR, "GSE92742_Broad_LINCS_sig_info.txt.gz"))
print(f"Signatures: {len(si)}")
print(f"Cols: {list(si.columns)}")
print(si.head(3).to_string())
# Count by pert type
print(f"Sig pert types: {si['pert_type'].value_counts().head(10).to_dict()}")
# Count cell lines
print(f"Cell lines: {si['cell_id'].value_counts().head(15).to_dict()}")
print()

print("=== Cell Info ===")
ci = read_gz(os.path.join(LINCS_DIR, "GSE92742_Broad_LINCS_cell_info.txt.gz"))
print(f"Cell lines: {len(ci)}")
print(f"Cols: {list(ci.columns)}")
print(ci[["cell_id", "cell_type", "primary_site"]].head(20).to_string())
