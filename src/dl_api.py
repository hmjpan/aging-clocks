"""Download tissue via GCS JSON API media endpoint (different from direct URL)."""
import sys
import time
import urllib.parse

import requests

BUCKET = "adult-gtex"
GCS_MEDIA = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"

tissue = sys.argv[1] if len(sys.argv) > 1 else "uterus"
object_name = f"bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_{tissue}.gct.gz"
dst = f"H:/lunwencp/shuailao/data/gtex/tpm_{tissue}.gct.gz"

encoded = urllib.parse.quote(object_name, safe="")
url = f"{GCS_MEDIA}/{encoded}?alt=media"

for attempt in range(1, 13):
    try:
        s = requests.Session()
        r = s.get(url, stream=True, timeout=(30, 90))
        print(f"attempt {attempt}: status={r.status_code}")
        if r.status_code == 200:
            total = 0
            with open(dst, "wb") as f:
                for chunk in r.iter_content(256 * 1024):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
            print(f"downloaded {total / 1e6:.1f} MB")
            s.close()
            break
        s.close()
    except Exception as e:
        print(f"attempt {attempt}: {str(e)[:100]}")
    time.sleep(5 * attempt)
