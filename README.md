# Tissue-Specific Aging Clocks Map Heterogeneous Aging-Modulatory Drug-Score Patterns Across 49 Human Tissue and Cell-Line Categories

## Overview

This repository contains all analysis code to reproduce the results in:

> Zhang D (2026) Tissue-specific aging clocks map heterogeneous aging-modulatory drug-score patterns across 49 human tissue and cell-line categories.

The pipeline constructs 49 transcriptomic aging clocks (47 tissue types and 2 cell-line categories) from GTEx v8 (948 unique donors contributing samples to the clocks; 17,329 samples), projects 3,926 LINCS L1000 compound-name entries onto each clock, and performs an external DepMap CRISPR concordance analysis after GTEx-based standardization. All data used are publicly available without application.

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

Step 2: Train category-specific clocks
  ├── clock.py                   # Core elastic-net clock (donor-grouped CV, per-fold normalization)
  ├── train_all_clocks.py        # Train all 49 clocks
  └── run_clocks.py              # Individual clock runner

Step 3: Drug x category matrix
  └── step3_drug_matrix.py       # Hypergeometric enrichment matrix (3,926 compound-name entries x 49)

Step 4: Exploratory validation
  ├── depmap_standardized.py     # DepMap scores using GTEx training mean/SD standardization (CORRECTED)
  └── step5_depmap.py            # Original DepMap analysis (for reference only)

Step 5: Statistical validation
  ├── rerun_key_analyses.py      # PRIMARY: age permutation x1000, mixed-fraction null,
  │                              #   DepMap standardization, high-performance category sensitivity
  ├── interaction_test.py        # Drug x category interaction via clock gene-set randomization
  ├── step7_fixes.py             # Significance tiers, gene-sharing null, gene coverage
  ├── sensitivity_analyses.py    # Weak-clock exclusion, Jaccard concordance, FDR calibration
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
python src/interaction_test.py     # drug x category interaction test

# 6. Additional robustness checks
python src/sensitivity_analyses.py
python src/step7_fixes.py

# 7. Prioritize candidates
python src/step6_prioritize.py
```

**IMPORTANT:** `rerun_key_analyses.py` is the entry point that reproduces the statistical analyses as reported in the current manuscript (age permutation with 1,000 permutations; conservative p = 0.001). The older `step7_validation.py` and `step7_test_d.py` scripts (10 permutations) are retained for reference only and should NOT be used to reproduce the manuscript's reported p-values.

## Age permutation note

The age permutation test reported in the manuscript uses 1,000 permutations with **fixed model hyperparameters** (alpha = 0.32, l1_ratio = 0.5; artery aorta) for computational tractability. This is a fast re-run version of the full training pipeline; the observed CV r = 0.855 with MAE = 5.2 years, R2 = 0.729, and n_features = 309 for artery aorta are self-consistent and reported in the manuscript. See the Methods section for details.

## Key results

| Metric | Value |
|---|---|
| Category clocks trained | 49 (47 tissue types + 2 cell-line categories) |
| Donors contributing to clocks | 948 |
| Median Pearson r | 0.543 |
| Best clock (artery aorta) | r = 0.855, MAE = 5.2 years, R2 = 0.729 |
| Compound-name entries projected | 3,926 |
| Unique clock genes | 10,253 |
| Mixed score directions (\|score\| > 1.0) | 94.2% (descriptive; below random null 97.5%; see Methods) |
| Strong bidirectional (\|score\| > 2, >=3+3 categories) | 111 (2.8%; p < 0.001 vs null) |
| Age permutation (1,000 perms) | Z = 12.7, conservative p = 0.001 |
| Permutation Z (cross-category correlation) | 160.7 |
| Bootstrap Spearman rho | 0.771 |
| 34-clock (r >= 0.5) ranking concordance | rho = 0.897 |
| Jaccard vs hypergeometric concordance | median per-category rho = 0.961 |
| FDR < 0.05 drug-category pairs | 0.79% |

## Terminology note

GTEx v8 includes 47 tissue sites and 2 cell-line categories (cultured fibroblasts, EBV-transformed lymphocytes); "49 categories" throughout refers to these. "3,926 compound-name entries" refers to distinct parsed LINCS perturbagen names; after case-normalization and synonym deduplication (e.g., sirolimus = rapamycin; MK-2206/mk-2206), the number of unique molecular entities is approximately 3,919.

## License

MIT License

## Contact

Dongdong Zhang - mcmojiepan@gmail.com