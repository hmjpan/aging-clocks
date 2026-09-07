"""Download remaining LINCS metadata + check L5 size."""
import os
import time
import requests

DST = r"H:\lunwencp\shuailao\data\lincs"
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl"

# Check L5 size
l5_url = f"{BASE}/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"
try:
    r = requests.head(l5_url, timeout=30, allow_redirects=True)
    cl = r.headers.get("content-length", "0")
    sz_gb = int(cl) / 1e9
    print(f"Level5 GCTX: {sz_gb:.2f} GB")
except Exception as e:
    print(f"L5 size check failed: {e}")

# Download remaining metadata
REMAINING = [
    "GSE92742_Broad_LINCS_pert_info.txt.gz",
    "GSE92742_Broad_LINCS_sig_metrics.txt.gz",
    "GSE92742_Broad_LINCS_cell_info.txt.gz",
    "GSE92742_Broad_LINCS_pert_metrics.txt.gz",
]

for fname in REMAINING:
    dst = os.path.join(DST, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 100:
        print(f"SKIP {fname} ({os.path.getsize(dst)/1e6:.1f} MB)")
        continue
    url = f"{BASE}/{fname}"
    for attempt in range(5):
        try:
            r = requests.get(url, stream=True, timeout=(30, 120))
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(256 * 1024):
                    if chunk:
                        f.write(chunk)
            print(f"OK   {fname} ({os.path.getsize(dst)/1e6:.1f} MB)")
            break
        except Exception as e:
            print(f"  retry {attempt}: {str(e)[:60]}")
            time.sleep(3 * (attempt + 1))
