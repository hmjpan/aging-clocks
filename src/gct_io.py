"""GCT file parser and GTEx metadata loader.

GCT v1.3 format:
  Line 1: #1.3
  Line 2: <n_genes>\t<n_samples>\t0\t0
  Line 3: id\tName\tDescription\t<sample1>\t<sample2>...
  Data:   <ensg>\t<gene_name>\t<desc>\t<tpm1>\t<tpm2>...
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import AGE_BRACKET_MID, GTEX_DIR, GTEX_SAMPLE_ATTRS_URL, GTEX_SUBJECT_PHENO_URL
except ImportError:
    from config import AGE_BRACKET_MID, GTEX_DIR, GTEX_SAMPLE_ATTRS_URL, GTEX_SUBJECT_PHENO_URL


def read_gct(path: str | Path) -> pd.DataFrame:
    """Parse a GCT v1.3 file (optionally gzipped).

    Returns a DataFrame indexed by gene_id (Ensembl) with a 'gene_name'
    column and one column per sample (float64 TPM).
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt") as f:
        version = f.readline().strip()
        if not version.startswith("#1."):
            raise ValueError(f"Unexpected GCT version line: {version}")
        is_v13 = version == "#1.3"
        dims = f.readline().strip().split("\t")
        n_genes, n_samples = int(dims[0]), int(dims[1])
        header = f.readline().rstrip("\n").split("\t")

        if is_v13:
            # v1.3: id, Name, Description, sample1, ...
            # parts[1] = Ensembl ID, parts[2] = gene symbol
            col_names = header[3:]
            id_idx, name_idx, data_start = 1, 2, 3
        else:
            # v1.2: Name(Ensembl ID), Description(gene symbol), sample1, ...
            col_names = header[2:]
            id_idx, name_idx, data_start = 0, 1, 2

        gene_ids = []
        gene_names = []
        rows = []
        for _ in range(n_genes):
            parts = f.readline().rstrip("\n").split("\t")
            gene_ids.append(parts[id_idx])
            gene_names.append(parts[name_idx])
            rows.append([float(x) for x in parts[data_start:data_start + n_samples]])

    mat = np.array(rows, dtype=np.float64)
    df = pd.DataFrame(mat, index=gene_ids, columns=col_names[:n_samples])
    df.insert(0, "gene_name", gene_names)
    df.index.name = "gene_id"
    return df


def load_sample_attributes() -> pd.DataFrame:
    """Load GTEx sample attributes, extract SUBJID, return cleaned frame."""
    path = GTEX_DIR / "sample_attributes.txt"
    if not path.exists():
        raise FileNotFoundError(f"Sample attributes not found at {path}. Run download.")
    df = pd.read_csv(path, sep="\t", dtype=str)
    # SUBJID = first two hyphen-separated tokens of SAMPID (e.g. GTEX-1117F)
    df["SUBJID"] = df["SAMPID"].str.split("-").str[:2].str.join("-")
    return df


def load_subject_phenotypes() -> pd.DataFrame:
    """Load subject phenotypes and map age brackets to numeric midpoints."""
    path = GTEX_DIR / "subject_phenotypes.txt"
    if not path.exists():
        raise FileNotFoundError(f"Subject phenotypes not found at {path}. Run download.")
    df = pd.read_csv(path, sep="\t")
    df["AGE_NUM"] = df["AGE"].map(AGE_BRACKET_MID)
    df["SEX_LABEL"] = df["SEX"].map({1: "M", 2: "F"})
    return df


def build_tissue_sample_table() -> pd.DataFrame:
    """Merge sample attributes with subject phenotypes.

    Returns one row per sample with columns:
      SAMPID, SUBJID, SMTS (broad tissue), SMTSD (detailed tissue),
      AGE (bracket), AGE_NUM, SEX, SEX_LABEL, DTHHRDY
    """
    sa = load_sample_attributes()
    sp = load_subject_phenotypes()
    merged = sa.merge(sp, on="SUBJID", how="inner", suffixes=("_sa", "_sp"))
    keep = ["SAMPID", "SUBJID", "SMTS", "SMTSD",
            "AGE", "AGE_NUM", "SEX", "SEX_LABEL", "DTHHRDY"]
    # AGE might be duplicated; handle gracefully
    for c in ("AGE", "DTHHRDY"):
        if f"{c}_sa" in merged.columns and f"{c}_sp" in merged.columns:
            merged[c] = merged[f"{c}_sp"]
            merged = merged.drop(columns=[f"{c}_sa", f"{c}_sp"])
    return merged[[c for c in keep if c in merged.columns]].copy()


def sample_id_to_tissue_file(tissue_sd: str) -> str:
    """Convert an SMTSD detailed-tissue name to its GCS filename slug.

    e.g. 'Whole Blood' -> 'whole_blood'
         'Brain - Frontal Cortex (BA9)' -> 'brain_frontal_cortex_ba9'
    """
    slug = tissue_sd.lower()
    # remove parenthetical content but keep the text inside
    slug = slug.replace("(", "").replace(")", "")
    # replace non-alphanumeric with underscore
    chars = []
    for ch in slug:
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars)
    # collapse multiple underscores
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return slug
