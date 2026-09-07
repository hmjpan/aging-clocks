"""Robust chunked downloader with resume support and gzip verification.

Usage:
    python src/download_tissues.py
"""
import gzip
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
from config import GTEX_DIR, GCS_BASE, GTEX_TPM_TISSUE_PREFIX

GCS_META = f"{GCS_BASE}/storage/v1/b/adult-gtex/o"


def get_gcs_size(name: str) -> int:
    """Get the size of a GCS object via the JSON API (with retry)."""
    url = f"https://storage.googleapis.com/storage/v1/b/adult-gtex/o?prefix={name}&maxResults=1"
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            items = r.json().get("items", [])
            for it in items:
                if it["name"] == name:
                    return int(it["size"])
            return 0
        except Exception:
            time.sleep(3 * (attempt + 1))
    return 0


def download_resume(url: str, path: str, expected: int = 0,
                    max_retries: int = 15, chunk: int = 256 * 1024) -> bool:
    """Download with HTTP Range resume. Returns True if file is complete & valid."""
    for attempt in range(1, max_retries + 1):
        existing = os.path.getsize(path) if os.path.exists(path) else 0
        if expected and existing >= expected:
            break
        headers = {}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
        try:
            r = requests.get(url, stream=True, headers=headers, timeout=60)
            r.raise_for_status()
            mode = "ab" if existing > 0 and r.status_code == 206 else "wb"
            if mode == "wb":
                existing = 0
            with open(path, mode) as f:
                for c in r.iter_content(chunk_size=chunk):
                    if c:
                        f.write(c)
            cur = os.path.getsize(path)
            print(f"  attempt {attempt}: {cur/1e6:.1f} MB / {expected/1e6:.1f} MB")
        except Exception as e:
            print(f"  attempt {attempt} error: {str(e)[:80]}")
            time.sleep(3 * attempt)

    actual = os.path.getsize(path) if os.path.exists(path) else 0
    if expected and actual != expected:
        print(f"  SIZE MISMATCH: {actual} != {expected}")
        return False
    # verify gzip
    try:
        with gzip.open(path, "rt") as f:
            f.read(1024 * 64)
        return True
    except Exception as e:
        print(f"  GZIP CORRUPT: {e}")
        try:
            os.remove(path)
        except OSError:
            pass
        return False


def list_tissue_files() -> list[tuple[str, str, int]]:
    """List all per-tissue TPM GCT files: (tissue_slug, gcs_name, size)."""
    r = requests.get(
        "https://storage.googleapis.com/storage/v1/b/adult-gtex/o"
        "?prefix=bulk-gex/v8/rna-seq/tpms-by-tissue/&maxResults=200"
        "&fields=items(name,size)",
        timeout=30,
    )
    items = r.json().get("items", [])
    out = []
    for it in items:
        nm = it["name"]
        if not nm.endswith(".gct.gz"):
            continue
        slug = nm.split("/")[-1].replace("gene_tpm_2017-06-05_v8_", "").replace(".gct.gz", "")
        out.append((slug, nm, int(it.get("size", 0))))
    out.sort(key=lambda x: x[2])  # smallest first
    return out


if __name__ == "__main__":
    GTEX_DIR.mkdir(parents=True, exist_ok=True)
    tissues = list_tissue_files()
    print(f"Found {len(tissues)} tissue TPM files")
    ok = 0
    fail = 0
    for slug, gcs_name, size in tissues:
        url = f"https://storage.googleapis.com/adult-gtex/{gcs_name}"
        dst = GTEX_DIR / f"tpm_{slug}.gct.gz"
        existing = dst.stat().st_size if dst.exists() else 0
        if existing == size:
            # quick gzip check
            try:
                with gzip.open(dst, "rt") as f:
                    f.read(4096)
                print(f"[OK]  {slug} ({size/1e6:.1f} MB) already complete")
                ok += 1
                continue
            except Exception:
                dst.unlink()
        print(f"[GET] {slug} ({size/1e6:.1f} MB)")
        if download_resume(url, str(dst), expected=size):
            print(f"[OK]  {slug}")
            ok += 1
        else:
            print(f"[FAIL] {slug}")
            fail += 1
    print(f"\nDone: {ok} ok, {fail} failed")
