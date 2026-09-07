"""Quick test: train clock on liver tissue to validate the pipeline."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from gct_io import read_gct, build_tissue_sample_table
from clock import train_tissue_clock

meta = build_tissue_sample_table()
print(f"Metadata: {len(meta)} samples, {meta['SUBJID'].nunique()} donors")

tissue = "liver"
fpath = f"data/gtex/tpm_{tissue}.gct.gz"
print(f"\nParsing {tissue} GCT...")
t0 = time.time()
expr = read_gct(fpath)
print(f"  Parsed in {time.time()-t0:.1f}s, shape={expr.shape}")

print(f"\nTraining clock for {tissue}...")
t0 = time.time()
result = train_tissue_clock(tissue, expr, meta)
if result:
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  N_samples={result.n_samples}, N_donors={result.n_donors}")
    print(f"  N_features={result.n_features_used}")
    print(f"  MAE={result.mae:.3f}, median_AE={result.median_ae:.3f}")
    print(f"  Pearson r={result.pearson_r:.4f}, r2={result.r2:.4f}")
    print(f"  Best alpha={result.best_alpha:.6f}, l1_ratio={result.best_l1_ratio}")
    print(f"  Top 10 genes:")
    for gene, w in result.coefficients.head(10).items():
        print(f"    {gene}: {w:.4f}")
else:
    print("  Skipped (insufficient data)")
