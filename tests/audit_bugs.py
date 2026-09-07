"""Audit: verify critical bugs found in code review."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import GTEX_DIR, CLOCK_DIR
from gct_io import read_gct

print("=" * 70)
print("BUG 1: Z-score normalization on full data (data leakage)")
print("=" * 70)
print("""
clock.py line 111-114:
  mu = X.mean(axis=0)      # computed on ALL data (train+test)
  sd = X.std(axis=0)       # computed on ALL data (train+test)
  Xz = (X - mu) / sd       # then used for CV

PROBLEM: Test fold statistics leak into normalization.
FIX: Must compute mu/sd inside each CV fold on training data only.
""")

print("=" * 70)
print("BUG 2: Feature selection on full data (data leakage)")
print("=" * 70)
print("""
clock.py line 98-99:
  expr_sub = _filter_genes(expr_sub, ...)    # on ALL data
  expr_sub = _select_variance(expr_sub, ...)  # on ALL data

PROBLEM: Test fold gene expression used to select features.
FIX: Must do feature selection inside each CV fold.
""")

print("=" * 70)
print("BUG 3: OOF predictions use ElasticNetCV(cv=3) not GroupKFold")
print("=" * 70)
print("""
clock.py line 141-148:
  model = ElasticNetCV(cv=3, ...)  # 3-fold, NOT grouped by donor
  model.fit(Xz[tr_idx], y[tr_idx])  # donor leakage within training

PROBLEM: Same donor's samples can be in both train and test
         within the 3-fold inner CV.
FIX: Use ElasticNet (fixed alpha) or GroupKFold for inner CV.
""")

print("=" * 70)
print("BUG 4: Hypergeometric universe mismatch")
print("=" * 70)
# Check overlap between clock genes and LINCS universe
gene_map = {}
fpath = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))[0]
expr = read_gct(str(fpath))
gene_map = dict(zip(expr.index, expr["gene_name"]))

# Load artery_aorta clock genes
coef = pd.read_csv(CLOCK_DIR / "artery_aorta_coefficients.csv", index_col=0)
clock_syms = set(gene_map.get(g, "") for g in coef.index) - {""}
print(f"Clock genes (symbols): {len(clock_syms)}")

# Parse LINCS GMT to get universe
down_sets = {}
with open(r"H:\lunwencp\shuailao\data\lincs\LINCS_L1000_Chem_Pert_down.txt") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            genes = set(g for g in parts[2:] if g)
            down_sets[parts[0]] = genes

up_sets = {}
with open(r"H:\lunwencp\shuailao\data\lincs\LINCS_L1000_Chem_Pert_up.txt") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            genes = set(g for g in parts[2:] if g)
            up_sets[parts[0]] = genes

lincs_universe = set()
for genes in down_sets.values():
    lincs_universe |= genes
for genes in up_sets.values():
    lincs_universe |= genes

print(f"LINCS universe: {len(lincs_universe)}")
overlap = clock_syms & lincs_universe
print(f"Clock genes IN LINCS universe: {len(overlap)}")
print(f"Clock genes NOT in LINCS universe: {len(clock_syms) - len(overlap)}")
print(f"""
PROBLEM: N (query size) in hypergeom = len(aging_set) = all clock genes
         But many clock genes are NOT in the LINCS universe.
         Hypergeom assumes all N draws are from the universe.
         Should use N = len(aging_set & lincs_universe).
""")

print("=" * 70)
print("BUG 5: Compound name extraction error")
print("=" * 70)
import re
test_names = [
    "CPC001 HA1E 24H-hemado-10.0",
    "CPC001 HA1E 24H-BRD-A00100033-10.0",  # BRD- prefix
    "CPC001 HA1E 24H-5-hydroxytryptophan-10.0",  # multiple hyphens
    "CPC001 MCF7 24H-n-benzylnaltrindole hydrochloride-10.0",
]
for name in test_names:
    parts = name.split("-")
    rest = "-".join(parts[1:])
    dose_match = re.search(r"(\d+\.?\d*)$", rest)
    dose = dose_match.group(1) if dose_match else ""
    compound = rest[:dose_match.start()].rstrip("-") if dose else rest
    print(f"  '{name}'")
    print(f"    -> compound='{compound}', dose='{dose}'")
    if "BRD" in name:
        print(f"    ERROR: 'BRD-A00100033' split into 'BRD' + 'A00100033'")
print()

print("=" * 70)
print("BUG 6: train_all_clocks.py spearman_r = 0")
print("=" * 70)
summary = pd.read_csv(r"H:\lunwencp\shuailao\results\tables\clock_summary.csv")
zero_spearman = (summary["spearman_r"] == 0.0).sum()
print(f"Tissues with spearman_r = 0: {zero_spearman}/{len(summary)}")
print("These are from the 'skip' path which doesn't compute spearman_r")
