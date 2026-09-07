"""Robust GCT (Gene Cluster Text) file parser for GTEx per-tissue TPM files.

GCT v1.3 format:
  Line 1: #1.3
  Line 2: <n_rows>\t<n_cols>\t<n_row_annotations>\t<n_col_annotations>
  Line 3: id\tName\tDescription\t<sample_id>...
  Line 4+: <gene_id>\t<gene_name>\t<description>\t<values>...
"""
from __future__ import annotations
import gzip
import io
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def read_gct(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a .gct or .gct.gz file.

    Returns
    -------
    expr : pd.DataFrame
        Genes x Samples expression matrix (float), index=gene_id (Ensembl),
        columns=sample_id.
    row_meta : pd.DataFrame
        Gene metadata (Name, Description) indexed by gene_id.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        version = f.readline().rstrip("\n")
        if not version.startswith("#"):
            raise ValueError(f"Unexpected GCT version line: {version!r}")
        dims = f.readline().rstrip("\n").split("\t")
        n_rows, n_cols = int(dims[0]), int(dims[1])
        header = f.readline().rstrip("\n").split("\t")
        # header: [id, Name, Description, sample1, sample2, ...]
        sample_ids = header[3:3 + n_cols]
        # Read data in chunks for memory efficiency
        rows = []
        row_meta_list = []
        gene_ids = []
        for _ in range(n_rows):
            parts = f.readline().rstrip("\n").split("\t")
            gene_id = parts[0]
            name = parts[1]
            desc = parts[2]
            vals = parts[3:3 + n_cols]
            gene_ids.append(gene_id)
            row_meta_list.append((gene_id, name, desc))
            rows.append(vals)
    expr = pd.DataFrame(rows, index=gene_ids, columns=sample_ids, dtype="float64")
    row_meta = pd.DataFrame(row_meta_list, columns=["gene_id", "Name", "Description"]).set_index("gene_id")
    return expr, row_meta


def read_gct_chunked(path: str | Path, gene_filter=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse GCT reading values directly via pandas after skipping header.

    Faster path that lets pandas handle the numeric parsing.
    If *gene_filter* is provided (set of Ensembl IDs), only those rows are kept.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        f.readline()  # version
        dims = f.readline().rstrip("\n").split("\t")
        n_rows, n_cols = int(dims[0]), int(dims[1])
        header_line = f.readline()  # keep raw for column names
        # Read the rest as a TSV
        df = pd.read_csv(io.StringIO(f.read()), sep="\t", header=None,
                         names=header_line.rstrip("\n").split("\t"),
                         nrows=n_rows, dtype={0: str, 1: str, 2: str})
    df = df.set_index(df.columns[0])
    sample_ids = df.columns[3:3 + n_cols]
    row_meta = df.iloc[:, :2].copy()
    row_meta.columns = ["Name", "Description"]
    expr = df.iloc[:, 3:3 + n_cols].astype("float64")
    expr.columns = sample_ids
    if gene_filter is not None:
        expr = expr.loc[expr.index.isin(gene_filter)]
        row_meta = row_meta.loc[row_meta.index.isin(gene_filter)]
    return expr, row_meta
