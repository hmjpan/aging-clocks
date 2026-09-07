"""Check GEO FTP for LINCS L1000 files."""
import requests
from xml.etree import ElementTree

url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
r = requests.get(url, timeout=30)
print(f"Status: {r.status_code}")
# Parse HTML listing
import re
files = re.findall(r'href="([^"]+)"', r.text)
files = [f for f in files if f not in ("../", "./") and not f.startswith("?")]
print(f"Files found: {len(files)}")
for f in files:
    print(f"  {f}")
