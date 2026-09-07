"""Check data source accessibility for LINCS L1000."""
import requests

urls = {
    "geo_ftp_dir": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/",
    "geo_lincs_l5": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_L1000_Level5_COMPZ.MODZ_n473647.gct.gz",
    "geo_lincs_l5b": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level5_COMPZ_n118050x12328.gct.gz",
    "clue_api": "https://api.clue.io/api/perts",
    "enrichr_lincs_down": "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=LINCS_L1000_Chem_Pert_down",
    "enrichr_lincs_up": "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=LINCS_L1000_Chem_Pert_up",
}
for k, u in urls.items():
    try:
        r = requests.head(u, timeout=20, allow_redirects=True)
        cl = r.headers.get("content-length", "?")
        print(f"{k}: {r.status_code} len={cl}")
    except Exception as e:
        print(f"{k}: FAIL {str(e)[:80]}")
