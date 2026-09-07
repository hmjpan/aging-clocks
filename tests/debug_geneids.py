"""Debug v2: test with per-tissue file."""
import gzip
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_DIR, GTEX_DIR
from gct_io import read_gct

# Check median TPM version
with gzip.open(str(GTEX_DIR / "gene_median_tpm.gct.gz"), "rt") as f:
    line1 = f.readline().strip()
    line2 = f.readline().strip().split("\t")
    line3 = f.readline().rstrip().split("\t")
    first_data = f.readline().rstrip().split("\t")
print(f"Median TPM: version={line1}, dims={line2[:2]}")
print(f"  Header[:5]: {line3[:5]}")
print(f"  Data[:5]: {first_data[:5]}")
print()

# Check per-tissue file
fpath = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))[0]
with gzip.open(str(fpath), "rt") as f:
    line1 = f.readline().strip()
    line2 = f.readline().strip().split("\t")
    line3 = f.readline().rstrip().split("\t")
    first_data = f.readline().rstrip().split("\t")
print(f"Per-tissue ({fpath.name}): version={line1}, dims={line2[:2]}")
print(f"  Header[:5]: {line3[:5]}")
print(f"  Data[:5]: {first_data[:5]}")
print()

# Load gene map from per-tissue file
expr = read_gct(str(fpath))
gene_map = dict(zip(expr.index, expr["gene_name"]))
print(f"Per-tissue gene map: {len(gene_map)} genes")
print(f"  Index sample: {expr.index[:3].tolist()}")
print(f"  gene_name sample: {expr['gene_name'][:3].tolist()}")

# Test with coefficient file
coef = pd.read_csv(CLOCK_DIR / "artery_aorta_coefficients.csv", index_col=0)
overlap = set(coef.index) & set(gene_map.keys())
print(f"\n  Coef IDs: {len(coef.index)}")
print(f"  Overlap: {len(overlap)}")
if overlap:
    for ensg in list(overlap)[:3]:
        print(f"    {ensg} -> {gene_map[ensg]}")
