"""Configuration: paths, data sources, clock hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
GTEX_DIR = DATA / "gtex"
LINCS_DIR = DATA / "lincs"
DEPMAP_DIR = DATA / "depmap"

RESULTS = ROOT / "results"
CLOCK_DIR = RESULTS / "clocks"
FIGURE_DIR = RESULTS / "figures"
TABLE_DIR = RESULTS / "tables"
for d in (CLOCK_DIR, FIGURE_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

GCS_BASE = "https://storage.googleapis.com/adult-gtex"
GTEX_TPM_TISSUE_PREFIX = f"{GCS_BASE}/bulk-gex/v8/rna-seq/tpms-by-tissue"
GTEX_SAMPLE_ATTRS_URL = f"{GCS_BASE}/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
GTEX_SUBJECT_PHENO_URL = f"{GCS_BASE}/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"

AGE_BRACKET_MID = {
    "20-29": 25.0,
    "30-39": 35.0,
    "40-49": 45.0,
    "50-59": 55.0,
    "60-69": 65.0,
    "70-79": 75.0,
}

CLOCK_PARAMS = {
    "alpha": 1.0,
    "l1_ratio_grid": [0.5],
    "cv_folds": 5,
    "min_samples": 80,
    "min_age_brackets": 4,
    "n_top_features_variance": 3000,
    "n_alphas": 30,
    "random_state": 42,
    "log_transform": True,
    "min_expression": 1.0,
}
