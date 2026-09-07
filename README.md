# Tissue-specific aging clocks reveal pervasive heterogeneity in drug anti-aging effects across 49 human tissues

## Overview

This repository contains all analysis code to reproduce the results in:

> Zhang D (2026) Tissue-specific aging clocks reveal pervasive heterogeneity in drug anti-aging effects across 49 human tissues. *Biogerontology* (submitted)

The pipeline constructs 49 tissue-specific transcriptomic aging clocks from GTEx v8, projects 3,926 LINCS L1000 compounds onto each clock, and performs cross-dataset validation using DepMap CRISPR data.

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
  ├── download_tissues.py        # GTEx per-tissue TPM
  ├── download_lincs_gmt.py      # LINCS L1000 gene sets
  └── parse_lincs_meta.py        # LINCS metadata

Step 2: Train tissue-specific clocks
  ├── clock.py                   # Core elastic-net clock (donor-grouped CV)
  ├── train_all_clocks.py        # Train all 49 clocks
  └── run_clocks.py              # Batch runner

Step 3: Drug × tissue matrix
  ├── step3_drug_matrix.py       # Hypergeometric enrichment matrix
  ├── step3_local.py             # Local LINCS GMT version
  └── drug_target_annotation.py  # Drug target/pathway annotation

Step 4: Validation
  ├── step5_depmap.py            # DepMap CRISPR validation
  ├── step7_validation.py        # Permutation, bootstrap, sensitivity, age-perm
  ├── step7_fixes.py             # Additional robustness checks
  ├── fix2_gsea.py               # GSEA pathway enrichment
  └── fix_enrichment_gender_fig5.py  # Gender-stratified analysis

Step 5: Sensitivity analyses (new)
  ├── sensitivity_analyses.py    # Weak-tissue exclusion, Jaccard concordance, FDR
  └── fix_reviews.py             # Reviewer response analyses

Step 6: Prioritization
  ├── step6_prioritize.py        # Candidate drug prioritization
  └── step6_final.py             # Final integration
```

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
python src/download_tissues.py
python src/download_lincs_gmt.py

# 3. Train clocks
python src/train_all_clocks.py

# 4. Build drug matrix
python src/step3_drug_matrix.py

# 5. Validate
python src/step5_depmap.py
python src/step7_validation.py
python src/sensitivity_analyses.py

# 6. Prioritize candidates
python src/step6_prioritize.py
```

## Key results

| Metric | Value |
|---|---|
| Tissue clocks trained | 49 |
| Median Pearson r | 0.543 |
| Best clock (artery aorta) | r = 0.838 |
| Drugs projected | 3,919 |
| Mixed-effect drugs (|score| > 1.0) | 94.2% |
| FDR < 0.05 drug-tissue pairs | 0.79% |
| Permutation Z (heterogeneity) | 160.7 |
| Bootstrap Spearman ρ | 0.771 |

## License

MIT License

## Contact

Dongdong Zhang — mcmojiepan@gmail.com
