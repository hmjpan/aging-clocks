"""Batch download all GTEx per-tissue TPM files via GCS JSON API media endpoint.

The JSON API endpoint is more resilient than the direct storage URL against
connection resets.  Each file is retried up to 20 times with exponential backoff.
Already-complete files are skipped.

Usage:
    python src/download_tissues_api.py
"""
import gzip
import os
import sys
import time
import urllib.parse

import requests

sys.path.insert(0, os.path.dirname(__file__))
from config import GTEX_DIR

GCS_MEDIA = "https://storage.googleapis.com/storage/v1/b/adult-gtex/o"


def list_tissue_files():
    """List all per-tissue TPM GCT files with sizes."""
    url = (
        "https://storage.googleapis.com/storage/v1/b/adult-gtex/o"
        "?prefix=bulk-gex/v8/rna-seq/tpms-by-tissue/&maxResults=200"
        "&fields=items(name,size)"
    )
    r = requests.get(url, timeout=30)
    items = r.json().get("items", [])
    out = []
    for it in items:
        nm = it["name"]
        if not nm.endswith(".gct.gz"):
            continue
        slug = nm.split("/")[-1].replace("gene_tpm_2017-06-05_v8_", "").replace(".gct.gz", "")
        out.append((slug, nm, int(it.get("size", 0))))
    out.sort(key=lambda x: x[2])
    return out


def download_api(gcs_name, dst, expected, max_retries=25, chunk=256 * 1024):
    """Download via GCS JSON API media endpoint with retries + resume."""
    encoded = urllib.parse.quote(gcs_name, safe="")
    url = f"{GCS_MEDIA}/{encoded}?alt=media"

    for attempt in range(1, max_retries + 1):
        existing = os.path.getsize(dst) if os.path.exists(dst) else 0
        if expected and existing >= expected:
            break
        headers = {}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
        try:
            s = requests.Session()
            r = s.get(url, stream=True, headers=headers, timeout=(30, 90))
            if r.status_code not in (200, 206):
                s.close()
                time.sleep(min(5 * attempt, 60))
                continue
            mode = "ab" if existing > 0 and r.status_code == 206 else "wb"
            if mode == "wb":
                existing = 0
            with open(dst, mode) as f:
                for c in r.iter_content(chunk_size=chunk):
                    if c:
                        f.write(c)
            cur = os.path.getsize(dst)
            s.close()
            if expected and cur >= expected:
                break
        except Exception:
            time.sleep(min(5 * attempt, 60))

    actual = os.path.getsize(dst) if os.path.exists(dst) else 0
    if expected and actual != expected:
        return False
    try:
        with gzip.open(dst, "rt") as f:
            f.read(8192)
        return True
    except Exception:
        try:
            os.remove(dst)
        except OSError:
            pass
        return False


if __name__ == "__main__":
    GTEX_DIR.mkdir(parents=True, exist_ok=True)
    tissues = list_tissue_files()
    print(f"Found {len(tissues)} tissue TPM files\n")
    ok = 0
    skip = 0
    fail = 0
    for i, (slug, gcs_name, size) in enumerate(tissues, 1):
        dst = str(GTEX_DIR / f"tpm_{slug}.gct.gz")
        existing = os.path.getsize(dst) if os.path.exists(dst) else 0
        # quick gzip check for existing
        if existing == size:
            try:
                with gzip.open(dst, "rt") as f:
                    f.read(4096)
                print(f"[{i}/{len(tissues)}] SKIP {slug} ({size/1e6:.1f} MB)")
                skip += 1
                continue
            except Exception:
                pass
        print(f"[{i}/{len(tissues)}] GET  {slug} ({size/1e6:.1f} MB)")
        if download_api(gcs_name, dst, expected=size):
            print(f"        OK   {slug}")
            ok += 1
        else:
            print(f"        FAIL {slug}")
            fail += 1
    print(f"\nDone: {ok} new, {skip} skipped, {fail} failed")
