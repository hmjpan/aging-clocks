# Tissue-Specific Aging Clocks Reveal Drug Effect Heterogeneity Across 49 Human Tissues

## Overview

This repository contains all analysis code to reproduce the results in:

> Zhang D (2026) Tissue-specific aging clocks reveal drug effect heterogeneity across 49 human tissues.

The pipeline constructs 49 tissue-specific transcriptomic aging clocks from GTEx v8, projects 3,926 LINCS L1000 compounds onto each clock, and performs exploratory cross-dataset validation using DepMap CRISPR data. All data used are publicly available without application.

## Data requirements

All datasets are publicly available (no application required):

| Dataset | Source | Size |
|---|---|---|
| GTEx v8 TPM matrices | https://storage.googleapis.com/adult-gtex/ | ~2 GB |
| LINCS L1000 GMT (up/down) | https://maayanlab.cloud/Enrichr/ | ~30 MB |
| DepMap CRISPR + Omics | https://depmap.org/portal/ | ~1 GB |

Place data under `data/gtex/`, `data/lincs/`, `data/depmap/` respectively.

## Pipeline

```
Step 1: Download data
  ├── download_tissues_api.py    # GTEx per-tissue TPM (resume-capable)
  ├── download_lincs_meta.py     # LINCS L1000 metadata
  └── download_lincs_gmt.py      # LINCS L1000 gene sets (up/down)

Step 2: Train tissue-specific clocks
  ├── clock.py                   # Core elastic-net clock (donor-grouped CV, per-fold normalization)
  ├── train_all_clocks.py        # Train all 49 clocks
  └── run_clocks.py              # Individual clock runner

Step 3: Drug x tissue matrix
  └── step3_drug_matrix.py       # Hypergeometric enrichment matrix (3,926 x 49)

Step 4: Exploratory validation
  ├── depmap_standardized.py     # DepMap scores using GTEx training mean/SD standardization (CORRECTED)
  └── step5_depmap.py            # Original DepMap analysis (for reference only)

Step 5: Statistical validation
  ├── rerun_key_analyses.py      # PRIMARY: age permutation x1000, mixed-fraction null,
  │                              #   DepMap standardization, high-performance tissue sensitivity
  ├── interaction_test.py        # Drug x tissue interaction via tissue clock gene-set randomization
  ├── step7_fixes.py             # Significance tiers, gene-sharing null, gene coverage
  ├── sensitivity_analyses.py    # Weak-tissue exclusion, Jaccard concordance, FDR calibration
  └── fix_reviews.py             # Additional reviewer-response analyses

Step 6: Prioritization
  ├── step6_prioritize.py        # Candidate drug prioritization
  ├── step6_final.py             # Final integration and summary figure
  └── drug_target_annotation.py  # Drug target/pathway annotation
```

## Usage (reproduce current manuscript results)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
python src/download_tissues_api.py
python src/download_lincs_meta.py
python src/download_lincs_gmt.py

# 3. Train clocks
python src/train_all_clocks.py

# 4. Build drug matrix
python src/step3_drug_matrix.py

# 5. Primary statistical validation (as reported in the manuscript)
python src/rerun_key_analyses.py   # age permutation x1000, mixed-fraction null, DepMap standardization
python src/interaction_test.py     # drug x tissue interaction test

# 6. Additional robustness checks
python src/sensitivity_analyses.py
python src/step7_fixes.py

# 7. Prioritize candidates
python src/step6_prioritize.py
```

**IMPORTANT:** `rerun_key_analyses.py` is the entry point that reproduces the statistical analyses as reported in the current manuscript (age permutation with 1,000 permutations; conservative p = 0.001). The older `step7_validation.py` and `step7_test_d.py` scripts (10 permutations) are retained for reference only and should NOT be used to reproduce the manuscript's reported p-values.

## Age permutation note

The age permutation test reported in the manuscript uses 1,000 permutations with **fixed model hyperparameters** (alpha selected during original training) for computational tractability. This is a fast re-run version of the full training pipeline; see the Methods section of the manuscript for details.

## Key results

| Metric | Value |
|---|---|
| Tissue clocks trained | 49 |
| Median Pearson r | 0.543 |
| Best clock (artery aorta) | r = 0.838 |
| Drugs projected | 3,926 |
| Unique clock genes | 10,253 |
| Mixed drug-tissue pairs (|score| > 1.0) | 94.2% (below random null 97.5%; see Methods) |
| Strong bidirectional drugs (|score| > 2, >=3+3 tissues) | 111 (2.8%; p < 0.001 vs null) |
| Age permutation (1,000 perms) | Z = 12.7, conservative p = 0.001 |
| Permutation Z (cross-tissue correlation) | 160.7 |
| Bootstrap Spearman rho | 0.771 |
| FDR < 0.05 drug-tissue pairs | 0.79% |

## License

MIT License

## Contact

Dongdong Zhang - mcmojiepan@gmail.com