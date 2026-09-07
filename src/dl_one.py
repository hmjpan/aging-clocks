"""Download a single tissue file with a fresh session."""
import requests
import sys

tissue = sys.argv[1] if len(sys.argv) > 1 else "brain_amygdala"
url = f"https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_{tissue}.gct.gz"
dst = f"H:/lunwencp/shuailao/data/gtex/tpm_{tissue}.gct.gz"

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0"})
try:
    r = s.get(url, stream=True, timeout=(30, 120))
    cl = r.headers.get("content-length", "?")
    print(f"status={r.status_code} content-length={cl}")
    total = 0
    with open(dst, "wb") as f:
        for chunk in r.iter_content(256 * 1024):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    print(f"downloaded {total / 1e6:.1f} MB")
except Exception as e:
    print(f"FAIL: {str(e)[:120]}")
finally:
    s.close()
