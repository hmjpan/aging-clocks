"""Verify all downloaded files."""
import gzip
import os

import pandas as pd

LINCS = r"H:\lunwencp\shuailao\data\lincs"
DEPMAP = r"H:\lunwencp\shuailao\data\depmap"

print("=" * 60)
print("1. LINCS sig_info.txt.gz")
print("=" * 60)
path = os.path.join(LINCS, "GSE92742_Broad_LINCS_sig_info.txt.gz")
try:
    with gzip.open(path, "rt") as f:
        header = f.readline().strip().split("\t")
        first = f.readline().strip().split("\t")
        n = 1
        for _ in f:
            n += 1
    print(f"  OK - {n} rows, {len(header)} cols")
    print(f"  Cols: {header}")
    print(f"  First: {first[:6]}")
except Exception as e:
    print(f"  CORRUPT: {e}")

print()
print("=" * 60)
print("2. LINCS_L1000_Chem_Pert_down.txt (GMT format)")
print("=" * 60)
path = os.path.join(LINCS, "LINCS_L1000_Chem_Pert_down.txt")
with open(path, "r") as f:
    lines = []
    for i, line in enumerate(f):
        lines.append(line)
        if i >= 4:
            break
    n_total = sum(1 for _ in open(path))
print(f"  {n_total} gene sets (lines)")
for i, line in enumerate(lines[:3]):
    parts = line.strip().split("\t")
    print(f"  Line {i+1}: set_name='{parts[0][:60]}' | n_genes={len(parts)-1} | first_genes={parts[1:5]}")

print()
print("=" * 60)
print("3. LINCS_L1000_Chem_Pert_up.txt (GMT format)")
print("=" * 60)
path = os.path.join(LINCS, "LINCS_L1000_Chem_Pert_up.txt")
with open(path, "r") as f:
    first = f.readline().strip().split("\t")
    n_total = sum(1 for _ in open(path))
print(f"  {n_total} gene sets (lines)")
parts = first
print(f"  Line 1: set_name='{parts[0][:60]}' | n_genes={len(parts)-1} | first_genes={parts[1:5]}")

print()
print("=" * 60)
print("4. DepMap OmicsExpression")
print("=" * 60)
path = os.path.join(DEPMAP, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
df = pd.read_csv(path, index_col=0, nrows=5)
print(f"  Shape (first 5 rows): {df.shape}")
print(f"  Index sample: {list(df.index[:3])}")
print(f"  Col sample: {list(df.columns[:5])}")

print()
print("=" * 60)
print("5. DepMap CRISPRGeneEffect")
print("=" * 60)
path = os.path.join(DEPMAP, "CRISPRGeneEffect.csv")
df = pd.read_csv(path, index_col=0, nrows=5)
print(f"  Shape (first 5 rows): {df.shape}")
print(f"  Index sample: {list(df.index[:3])}")
print(f"  Col sample: {list(df.columns[:5])}")

print()
print("=" * 60)
print("6. DepMap Model.csv")
print("=" * 60)
path = os.path.join(DEPMAP, "Model.csv")
df = pd.read_csv(path, nrows=5)
print(f"  Shape (first 5 rows): {df.shape}")
print(f"  Cols: {list(df.columns[:15])}")

print()
print("ALL CHECKS DONE")
