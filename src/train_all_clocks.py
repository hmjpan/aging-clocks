"""Train tissue-specific aging clocks for all eligible GTEx tissues.

This is Step 1 of the pipeline:
  1. Load GTEx metadata (ages, donor IDs)
  2. For each tissue (>=80 samples, >=4 age brackets):
     a. Parse GCT TPM file
     b. Filter low-expression genes, select top-variance genes
     c. Train elastic net with donor-grouped K-fold CV
     d. Save: coefficients, CV predictions, metrics
  3. Write summary table (all tissues)

Usage:
    python src/train_all_clocks.py
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import GTEX_DIR, CLOCK_DIR, TABLE_DIR, CLOCK_PARAMS
from gct_io import read_gct, build_tissue_sample_table
from clock import train_tissue_clock


def find_tissue_files():
    """Return list of (tissue_slug, filepath) for all downloaded GCT files."""
    files = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))
    out = []
    for f in files:
        slug = f.name.replace("tpm_", "").replace(".gct.gz", "")
        out.append((slug, str(f)))
    return out


def save_clock_result(result, out_dir):
    """Persist a ClockResult to disk."""
    tissue_slug = result.tissue
    # coefficients
    result.coefficients.to_csv(
        out_dir / f"{tissue_slug}_coefficients.csv", header=["weight"]
    )
    # CV predictions
    result.cv_predictions.to_csv(
        out_dir / f"{tissue_slug}_cv_predictions.csv", index=False
    )
    # bracket stats
    result.age_bracket_stats.to_csv(
        out_dir / f"{tissue_slug}_bracket_stats.csv", index=False
    )


def result_to_summary_row(r):
    return {
        "tissue": r.tissue,
        "n_samples": r.n_samples,
        "n_donors": r.n_donors,
        "n_features": r.n_features_used,
        "mae": round(r.mae, 3),
        "median_ae": round(r.median_ae, 3),
        "pearson_r": round(r.pearson_r, 4),
        "spearman_r": round(r.spearman_r, 4),
        "r2": round(r.r2, 4),
        "best_alpha": round(r.best_alpha, 6),
        "best_l1_ratio": r.best_l1_ratio,
    }


def main():
    CLOCK_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata once
    print("Loading GTEx metadata...", flush=True)
    meta = build_tissue_sample_table()
    print(f"  {len(meta)} samples, {meta['SUBJID'].nunique()} donors\n", flush=True)

    tissue_files = find_tissue_files()
    print(f"Found {len(tissue_files)} tissue files\n", flush=True)
    print(f"{'#':>3} {'Tissue':<50} {'N':>5} {'Genes':>5} {'MAE':>6} "
          f"{'r':>6} {'r2':>6} {'Time':>7}")
    print("-" * 90, flush=True)

    summary_rows = []
    trained = 0
    skipped = 0
    failed = 0

    for i, (tissue_slug, fpath) in enumerate(tissue_files, 1):
        t0 = time.time()
        try:
            # Check if already done (skip if results exist)
            coef_path = CLOCK_DIR / f"{tissue_slug}_coefficients.csv"
            if coef_path.exists():
                # Load existing results
                pred = pd.read_csv(CLOCK_DIR / f"{tissue_slug}_cv_predictions.csv")
                coef = pd.read_csv(coef_path, index_col=0)["weight"]
                from scipy.stats import spearmanr
                rho_val, _ = spearmanr(pred["age_predicted"], pred["age_actual"])
                row = {
                    "tissue": tissue_slug,
                    "n_samples": len(pred),
                    "n_donors": pred["SUBJID"].nunique(),
                    "n_features": int((coef != 0).sum()),
                    "mae": round(float(np.mean(np.abs(
                        pred["age_actual"] - pred["age_predicted"]))), 3),
                    "median_ae": round(float(np.median(np.abs(
                        pred["age_actual"] - pred["age_predicted"]))), 3),
                    "pearson_r": round(float(np.corrcoef(
                        pred["age_predicted"], pred["age_actual"])[0, 1]), 4),
                    "spearman_r": round(float(rho_val), 4),
                    "r2": round(1.0 - np.sum(
                        (pred["age_actual"] - pred["age_predicted"])**2) /
                        np.sum((pred["age_actual"] -
                                pred["age_actual"].mean())**2), 4),
                    "best_alpha": 0.0,
                    "best_l1_ratio": 0.0,
                }
                summary_rows.append(row)
                elapsed = time.time() - t0
                print(f"{i:>3} {tissue_slug:<50} {row['n_samples']:>5} "
                      f"{row['n_features']:>5} {row['mae']:>6} "
                      f"{row['pearson_r']:>6} {row['r2']:>6} "
                      f"{elapsed:>6.1f}s [skip]", flush=True)
                skipped += 1
                continue

            # Parse GCT
            expr = read_gct(fpath)

            # Train clock
            result = train_tissue_clock(tissue_slug, expr, meta)
            if result is None:
                print(f"{i:>3} {tissue_slug:<50}  --- skipped (insufficient data)",
                      flush=True)
                skipped += 1
                continue

            save_clock_result(result, CLOCK_DIR)
            row = result_to_summary_row(result)
            summary_rows.append(row)
            elapsed = time.time() - t0
            print(f"{i:>3} {tissue_slug:<50} {row['n_samples']:>5} "
                  f"{row['n_features']:>5} {row['mae']:>6} "
                  f"{row['pearson_r']:>6} {row['r2']:>6} "
                  f"{elapsed:>6.1f}s", flush=True)
            trained += 1

        except Exception as e:
            elapsed = time.time() - t0
            print(f"{i:>3} {tissue_slug:<50}  FAIL: {str(e)[:60]}  "
                  f"({elapsed:.1f}s)", flush=True)
            traceback.print_exc()
            failed += 1

    # Write summary table
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("pearson_r", ascending=False)
    summary.to_csv(TABLE_DIR / "clock_summary.csv", index=False)

    print(f"\n{'='*90}")
    print(f"Trained: {trained}, Skipped: {skipped}, Failed: {failed}")
    print(f"\nTop 10 clocks by Pearson r:")
    print(summary.head(10).to_string(index=False))
    print(f"\nSummary saved to {TABLE_DIR / 'clock_summary.csv'}")


if __name__ == "__main__":
    main()
