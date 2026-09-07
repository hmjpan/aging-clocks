"""Load and clean GTEx v8 metadata: subject phenotypes + sample attributes.

Links samples to donor age via SUBJID (first two hyphen-delimited tokens of SAMPID).
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from .config import GTEX_DIR, GTEX_SAMPLE_ATTRS_URL, GTEX_SUBJECT_PHENO_URL, AGE_BRACKET_MID


def load_subject_phenotypes(path: Path | None = None) -> pd.DataFrame:
    """Load GTEx subject phenotypes (SUBJID, SEX, AGE, DTHHRDY)."""
    path = path or (GTEX_DIR / "subject_phenotypes.txt")
    df = pd.read_csv(path, sep="\t")
    df["SEX"] = df["SEX"].map({1: "M", 2: "F"})
    # Map age bracket string -> numeric midpoint
    df["AGE_NUM"] = df["AGE"].map(AGE_BRACKET_MID)
    # Age bracket as ordered categorical for stratification
    df["AGE_BRACKET"] = pd.Categorical(
        df["AGE"], categories=list(AGE_BRACKET_MID.keys()), ordered=True
    )
    return df


def load_sample_attributes(path: Path | None = None) -> pd.DataFrame:
    """Load GTEx sample attributes, extract SUBJID from SAMPID."""
    path = path or (GTEX_DIR / "sample_attributes.txt")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # SUBJID = first two tokens of SAMPID (e.g. GTEX-1117F-0006-SM-5NQBE -> GTEX-1117F)
    df["SUBJID"] = df["SAMPID"].str.split("-").str[:2].str.join("-")
    return df


def build_sample_age_table(
    sa: pd.DataFrame | None = None, sp: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Join sample attributes with subject phenotypes to get per-sample age.

    Returns DataFrame with columns: SAMPID, SUBJID, SMTS, SMTSD, AGE_NUM, SEX.
    """
    sa = sa if sa is not None else load_sample_attributes()
    sp = sp if sp is not None else load_subject_phenotypes()
    merged = sa.merge(sp[["SUBJID", "SEX", "AGE_NUM", "AGE_BRACKET"]], on="SUBJID", how="left")
    keep = ["SAMPID", "SUBJID", "SMTS", "SMTSD", "SEX", "AGE_NUM", "AGE_BRACKET"]
    return merged[keep].dropna(subset=["AGE_NUM"])


def load_tissue_sample_map(
    sa: pd.DataFrame | None = None, sp: pd.DataFrame | None = None
) -> dict[str, list[str]]:
    """Return {detailed_tissue: [SAMPID, ...]} for all tissues."""
    table = build_sample_age_table(sa, sp)
    return table.groupby("SMTSD")["SAMPID"].apply(list).to_dict()
