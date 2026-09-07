import gzip, os, glob

dl_dir = r"H:\lunwencp\shuailao\data\gtex"
files = sorted(glob.glob(os.path.join(dl_dir, "tpm_*.gct.gz")))
bad = []
for f in files:
    try:
        with gzip.open(f, "rt") as fh:
            fh.readline()
            fh.readline()
            fh.readline()
            fh.read(8192)
    except Exception as e:
        bad.append((os.path.basename(f), str(e)[:60]))

print("Total files:", len(files))
print("Corrupt:", len(bad))
for name, err in bad:
    print("  " + name + ": " + err)
