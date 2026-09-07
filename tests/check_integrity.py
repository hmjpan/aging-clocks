"""Check all GTEx tissue files for gzip integrity."""
import gzip
import glob
import os

dl_dir = r"H:\lunwencp\shuailao\data\gtex"
files = sorted(glob.glob(os.path.join(dl_dir, "tpm_*.gct.gz")))
print(f"Total files: {len(files)}")
bad = []
for f in files:
    name = os.path.basename(f)
    size_mb = round(os.path.getsize(f) / 1e6, 1)
    try:
        with gzip.open(f, "rt") as fh:
            fh.readline()  # version
            fh.readline()  # dims
            fh.readline()  # header
            fh.read(8192)  # some data
        print(f"  OK   {size_mb:>7.1f} MB  {name}")
    except Exception as e:
        print(f"  BAD  {size_mb:>7.1f} MB  {name}  ({str(e)[:50]})")
        bad.append(name)
print(f"\nCorrupt: {len(bad)}")
for b in bad:
    print(f"  {b}")
