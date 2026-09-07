"""Verify all GTEx tissue TPM files: gzip integrity + dimensions + sample overlap."""
import gzip
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import GTEX_DIR
from gct_io import build_tissue_sample_table

meta = build_tissue_sample_table()
meta_sampids = set(meta["SAMPID"])

files = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))
print(f"Found {len(files)} tissue files\n")
print(f"{'#':>3} {'Tissue':<50} {'Genes':>6} {'Samples':>8} {'Overlap':>7} {'Status':>6}")
print("-" * 85)

results = []
ok = 0
fail = 0
for i, f in enumerate(files, 1):
    tissue = f.name.replace("tpm_", "").replace(".gct.gz", "")
    try:
        with gzip.open(f, "rt") as fh:
            ver = fh.readline().strip()
            dims = fh.readline().strip().split("\t")
            n_genes, n_samples = int(dims[0]), int(dims[1])
            header = fh.readline().strip().split("\t")
            sample_ids = header[3:]
            overlap = len(set(sample_ids) & meta_sampids)
            status = "OK"
            ok += 1
        results.append((tissue, n_genes, n_samples, overlap))
    except Exception as e:
        status = "FAIL"
        n_genes = n_samples = overlap = 0
        results.append((tissue, 0, 0, 0))
        fail += 1
    print(f"{i:>3} {tissue:<50} {n_genes:>6} {n_samples:>8} {overlap:>7} {status:>6}")

print(f"\n=== Summary: {ok} OK, {fail} FAIL ===")
print(f"Total samples across tissues: {sum(r[3] for r in results)}")

# Check which tissues meet min_samples threshold (80)
min_n = 80
eligible = [r for r in results if r[3] >= min_n]
print(f"\nTissues with >= {min_n} overlapping samples: {len(eligible)}/{len(results)}")
for t, g, ns, ov in sorted(eligible, key=lambda x: x[3], reverse=True):
    print(f"  {t:<50} {ov:>5} samples")
