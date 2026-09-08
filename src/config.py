"""Project configuration: paths, data sources, clock hyperparameters."""
from pathlib import Path

# Root
ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA = ROOT / "data"
GTEX_DIR = DATA / "gtex"
LINCS_DIR = DATA / "lincs"
DEPMAP_DIR = DATA / "depmap"

# Results
RESULTS = ROOT / "results"
CLOCK_DIR = RESULTS / "clocks"
FIGURE_DIR = RESULTS / "figures"
TABLE_DIR = RESULTS / "tables"
for d in (CLOCK_DIR, FIGURE_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# GTEx GCS base (open-access, no application required)
GCS_BASE = "https://storage.googleapis.com/adult-gtex"
GTEX_TPM_TISSUE_PREFIX = f"{GCS_BASE}/bulk-gex/v8/rna-seq/tpms-by-tissue"
GTEX_SAMPLE_ATTRS_URL = f"{GCS_BASE}/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
GTEX_SUBJECT_PHENO_URL = f"{GCS_BASE}/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"

# Age bracket -> numeric midpoint mapping (GTEx privacy-binned ages)
AGE_BRACKET_MID = {
    "20-29": 25.0,
    "30-39": 35.0,
    "40-49": 45.0,
    "50-59": 55.0,
    "60-69": 65.0,
    "70-79": 75.0,
}

# Clock training hyperparameters
CLOCK_PARAMS = {
    "alpha": 1.0,            # overall L1/L2 penalty weight (tuned by CV)
    "l1_ratio_grid": [0.5],  # single l1_ratio for speed
    "cv_folds": 5,            # grouped K-fold by donor (no donor leakage)
    "min_samples": 80,        # skip tissues with fewer samples
    "min_age_brackets": 4,    # require >=4 of 6 age brackets represented
    "n_top_features_variance": 3000,  # pre-filter by variance before elastic net
    "n_alphas": 30,           # alpha grid size
    "random_state": 42,
    "log_transform": True,    # log2(TPM+1)
    "min_expression": 1.0,    # mean TPM >= 1 in >=10% samples to keep gene
}

# Tissue name -> GCS filename slug mapping (derived from SMTSD lowercased + _ )
# These are the tissues we will build clocks for.
