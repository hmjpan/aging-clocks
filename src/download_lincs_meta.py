"""Download LINCS L1000 metadata files from GEO."""
import os
import sys
import time

import requests

DST = r"H:\lunwencp\shuailao\data\lincs"
os.makedirs(DST, exist_ok=True)

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl"

FILES = [
    "GSE92742_Broad_LINCS_gene_info.txt.gz",
    "GSE92742_Broad_LINCS_gene_info_delta_landmark.txt.gz",
    "GSE92742_Broad_LINCS_sig_info.txt.gz",
    "GSE92742_Broad_LINCS_pert_info.txt.gz",
    "GSE92742_Broad_LINCS_sig_metrics.txt.gz",
    "GSE92742_Broad_LINCS_cell_info.txt.gz",
    "GSE92742_Broad_LINCS_pert_metrics.txt.gz",
]

for fname in FILES:
    url = f"{BASE}/{fname}"
    dst = os.path.join(DST, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 100:
        print(f"SKIP {fname} ({os.path.getsize(dst)/1e6:.1f} MB)")
        continue
    for attempt in range(5):
        try:
            r = requests.get(url, stream=True, timeout=(30, 120))
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(256 * 1024):
                    if chunk:
                        f.write(chunk)
            sz = os.path.getsize(dst) / 1e6
            print(f"OK   {fname} ({sz:.1f} MB)")
            break
        except Exception as e:
            print(f"  attempt {attempt}: {str(e)[:80]}")
            time.sleep(5 * (attempt + 1))

# Check Level 5 file size
l5_url = f"{BASE}/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"
try:
    r = requests.head(l5_url, timeout=30, allow_redirects=True)
    cl = r.headers.get("content-length", "?")
    print(f"\nLevel5 GCTX size: {int(cl)/1e9:.2f} GB" if cl != "?" else "\nLevel5: size unknown")
except Exception as e:
    print(f"\nLevel5 head failed: {e}")
