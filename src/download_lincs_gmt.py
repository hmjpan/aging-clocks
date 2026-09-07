"""Download LINCS L1000 gene set libraries from Enrichr for local computation."""
import os
import sys
import time

import requests

DST = r"H:\lunwencp\shuailao\data\lincs"
os.makedirs(DST, exist_ok=True)

LIBS = [
    "LINCS_L1000_Chem_Pert_down",
    "LINCS_L1000_Chem_Pert_up",
]

for lib in LIBS:
    url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={lib}"
    dst = os.path.join(DST, f"{lib}.gmt")
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        print(f"SKIP {lib} ({os.path.getsize(dst)/1e6:.1f} MB)")
        continue
    for attempt in range(5):
        try:
            print(f"Downloading {lib}...")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(dst, "w") as f:
                f.write(r.text)
            sz = os.path.getsize(dst) / 1e6
            n_lines = r.text.count("\n")
            print(f"  OK {sz:.1f} MB, {n_lines} gene sets")
            break
        except Exception as e:
            print(f"  attempt {attempt}: {str(e)[:80]}")
            time.sleep(10 * (attempt + 1))
