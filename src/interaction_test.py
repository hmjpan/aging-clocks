"""Drug x tissue interaction test via tissue clock gene-set randomization.

Null: for each tissue, randomly draw aging/youth gene sets from the LINCS
universe keeping the SAME gene-set sizes as observed, recompute the full
drug x tissue matrix. This preserves:
  - each drug's LINCS up/down signature and its size
  - each tissue clock gene-set size
  - the gene universe structure
and destroys only the biological link between tissue aging programs and
drug perturbation signatures.

Test statistic: drug-wise cross-tissue dispersion of DOUBLE-CENTERED
residuals R_dt = S_dt - mean_d - mean_t + mean, summarized as
mean(drug-wise SD), median(drug-wise IQR), P95(drug-wise robust range).
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, iqr

sys.path.insert(0, os.path.dirname(__file__))
from config import TABLE_DIR, CLOCK_DIR, GTEX_DIR, LINCS_DIR

rng = np.random.RandomState(42)


def parse_gmt(path):
    sets = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            sets[parts[0]] = set(g for g in parts[2:] if g)
    return sets


def extract_compound_name(set_name):
    parts = set_name.split("-")
    if len(parts) < 2:
        return set_name
    rest = "-".join(parts[1:])
    import re
    dose_match = re.search(r"(\d+\.?\d*)$", rest)
    dose = dose_match.group(1) if dose_match else ""
    if dose:
        return rest[:dose_match.start()].rstrip("-")
    return rest


def build_universe_and_drugs():
    """Universe (all LINCS genes) + per-drug up/down sets restricted to universe."""
    down_sets = parse_gmt(str(LINCS_DIR / "LINCS_L1000_Chem_Pert_down.txt"))
    up_sets = parse_gmt(str(LINCS_DIR / "LINCS_L1000_Chem_Pert_up.txt"))

    universe = set()
    for gs in down_sets.values():
        universe |= gs
    for gs in up_sets.values():
        universe |= gs
    universe = sorted(universe)
    U_idx = {g: i for i, g in enumerate(universe)}
    M = len(universe)

    from collections import defaultdict
    comp_down = defaultdict(set)
    comp_up = defaultdict(set)
    for name, genes in down_sets.items():
        comp_down[extract_compound_name(name)] |= genes
    for name, genes in up_sets.items():
        comp_up[extract_compound_name(name)] |= genes

    all_compounds = sorted(set(comp_down) | set(comp_up))
    n_drugs = len(all_compounds)

    # bool matrices: drugs x universe
    D_down = np.zeros((n_drugs, M), dtype=bool)
    D_up = np.zeros((n_drugs, M), dtype=bool)
    for i, c in enumerate(all_compounds):
        for g in comp_down[c]:
            if g in U_idx:
                D_down[i, U_idx[g]] = True
        for g in comp_up[c]:
            if g in U_idx:
                D_up[i, U_idx[g]] = True

    return universe, all_compounds, D_down, D_up, M


def load_tissue_gene_sets(universe):
    """Per-tissue aging/youth gene symbol sets restricted to universe."""
    U_set = set(universe)
    fpath = sorted(GTEX_DIR.glob("tpm_*.gct.gz"))[0]
    from gct_io import read_gct
    expr = read_gct(str(fpath))
    gene_map = dict(zip(expr.index, expr["gene_name"]))

    tissue_sets = {}
    for f in sorted(CLOCK_DIR.glob("*_coefficients.csv")):
        tissue = f.name.replace("_coefficients.csv", "")
        coef = pd.read_csv(f, index_col=0)
        coef.columns = ["weight"]
        aging = set()
        youth = set()
        for ensg, w in coef["weight"].items():
            sym = gene_map.get(ensg, "")
            if not sym or sym == "nan":
                continue
            if w > 0 and sym in U_set:
                aging.add(sym)
            elif w < 0 and sym in U_set:
                youth.add(sym)
        if len(aging) >= 5 and len(youth) >= 5:
            tissue_sets[tissue] = (aging, youth)
    return tissue_sets


def compute_scores(D_down, D_up, aging_idx, youth_idx, M, K_down, K_up):
    """Full matrix scores (3926 x n_tissues) for given aging/youth gene idx."""
    n_a = len(aging_idx)
    n_y = len(youth_idx)
    # aging-down term: -log10 P(X >= overlap) hypergeom
    x_ad = D_down[:, aging_idx].sum(axis=1)
    p_ad = np.where(x_ad > 0, -np.log10(hypergeom.sf(x_ad - 1, M, K_down, n_a)), 0.0)
    x_au = D_up[:, aging_idx].sum(axis=1)
    p_au = np.where(x_au > 0, -np.log10(hypergeom.sf(x_au - 1, M, K_up, n_a)), 0.0)
    x_yu = D_up[:, youth_idx].sum(axis=1)
    p_yu = np.where(x_yu > 0, -np.log10(hypergeom.sf(x_yu - 1, M, K_up, n_y)), 0.0)
    x_yd = D_down[:, youth_idx].sum(axis=1)
    p_yd = np.where(x_yd > 0, -np.log10(hypergeom.sf(x_yd - 1, M, K_down, n_y)), 0.0)
    return (p_ad + p_yu) - (p_au + p_yd)


def main():
    print("Building universe + drug signature matrices...", flush=True)
    universe, compounds, D_down, D_up, M = build_universe_and_drugs()
    print("  Universe: {} genes, {} compounds".format(M, len(compounds)), flush=True)

    K_down = D_down.sum(axis=1)
    K_up = D_up.sum(axis=1)

    tissue_sets = load_tissue_gene_sets(universe)
    tissues = sorted(tissue_sets.keys())
    print("  Tissues with aging/youth sets: {}".format(len(tissues)), flush=True)

    U_idx = {g: i for i, g in enumerate(universe)}

    # Observed matrix (recompute exactly as original pipeline)
    print("Computing observed matrix...", flush=True)
    n_drugs = len(compounds)
    obs_mat = np.zeros((n_drugs, len(tissues)))
    for j, t in enumerate(tissues):
        aging, youth = tissue_sets[t]
        aging_idx = [U_idx[g] for g in aging]
        youth_idx = [U_idx[g] for g in youth]
        obs_mat[:, j] = compute_scores(D_down, D_up, aging_idx, youth_idx, M, K_down, K_up)

    # Double-centered residuals + dispersion stats
    def dispersion_stats(mat):
        """Return (mean_drug_SD, median_drug_IQR, p95_drug_range) on double-centered residuals."""
        S = mat - mat.mean(axis=1, keepdims=True) - mat.mean(axis=0, keepdims=True) + mat.mean()
        sd_d = S.std(axis=1)
        iqr_d = np.array([iqr(S[i]) for i in range(S.shape[0])])
        rng_d = np.ptp(S, axis=1)
        return sd_d.mean(), np.median(iqr_d), np.percentile(rng_d, 95)

    obs_mean_sd, obs_med_iqr, obs_p95 = dispersion_stats(obs_mat)
    print("  Observed dispersion (double-centered residuals):")
    print("    mean(drug SD)   = {:.4f}".format(obs_mean_sd))
    print("    median(drug IQR)= {:.4f}".format(obs_med_iqr))
    print("    P95(drug range) = {:.4f}".format(obs_p95), flush=True)

    # Also mixed fraction at |score|>1 for reference
    n_reju = (obs_mat > 1.0).sum(axis=1)
    n_pro = (obs_mat < -1.0).sum(axis=1)
    obs_mixed = ((n_reju > 0) & (n_pro > 0)).mean()
    print("  Observed mixed fraction (|score|>1): {:.4f}".format(obs_mixed), flush=True)

    # Null: randomize tissue clock gene sets (keep sizes), recompute matrix
    n_perm = 500
    null_mean_sd = np.zeros(n_perm)
    null_med_iqr = np.zeros(n_perm)
    null_p95 = np.zeros(n_perm)
    null_mixed = np.zeros(n_perm)

    sizes = {t: (len(tissue_sets[t][0]), len(tissue_sets[t][1])) for t in tissues}
    all_idx = np.arange(M)

    print("Running {} gene-set randomization permutations...".format(n_perm), flush=True)
    t0 = time.time()
    for i in range(n_perm):
        perm_mat = np.zeros((n_drugs, len(tissues)))
        for j, t in enumerate(tissues):
            n_a, n_y = sizes[t]
            # draw aging from universe, youth from remainder
            aging_idx = rng.choice(all_idx, n_a, replace=False)
            rest = np.setdiff1d(all_idx, aging_idx, assume_unique=True)
            youth_idx = rng.choice(rest, n_y, replace=False)
            perm_mat[:, j] = compute_scores(D_down, D_up, aging_idx, youth_idx, M, K_down, K_up)
        null_mean_sd[i], null_med_iqr[i], null_p95[i] = dispersion_stats(perm_mat)
        nr = (perm_mat > 1.0).sum(axis=1)
        nn = (perm_mat < -1.0).sum(axis=1)
        null_mixed[i] = ((nr > 0) & (nn > 0)).mean()
        if (i + 1) % 100 == 0:
            print("  {}/{} ({:.0f}s)".format(i + 1, n_perm, time.time() - t0), flush=True)

    print("\n" + "=" * 72)
    print("RESULTS: drug x tissue interaction (gene-set randomization null)")
    print("=" * 72)
    p_sd = np.mean(null_mean_sd >= obs_mean_sd)
    p_iqr = np.mean(null_med_iqr >= obs_med_iqr)
    p_rng = np.mean(null_p95 >= obs_p95)
    p_mix = np.mean(null_mixed >= obs_mixed)
    print("  Statistic              Observed      Null mean +/- SD        p")
    print("  mean(drug SD)          {:.4f}      {:.4f} +/- {:.4f}     {}".format(
        obs_mean_sd, null_mean_sd.mean(), null_mean_sd.std(), p_sd))
    print("  median(drug IQR)       {:.4f}      {:.4f} +/- {:.4f}     {}".format(
        obs_med_iqr, null_med_iqr.mean(), null_med_iqr.std(), p_iqr))
    print("  P95(drug range)        {:.4f}      {:.4f} +/- {:.4f}     {}".format(
        obs_p95, null_p95.mean(), null_p95.std(), p_rng))
    print("  mixed frac (|s|>1)     {:.4f}      {:.4f} +/- {:.4f}     {}".format(
        obs_mixed, null_mixed.mean(), null_mixed.std(), p_mix))

    print("\nCONCLUSION:")
    sig = [s for s in (p_sd, p_iqr, p_rng) if s < 0.05]
    if len(sig) == 3:
        print("  Observed cross-tissue dispersion is SIGNIFICANTLY HIGHER than "
              "the gene-set-randomization null on all 3 statistics.")
        print("  -> drug x tissue interaction exists beyond random clock composition.")
    elif len(sig) > 0:
        print("  Partially significant ({}/3).".format(len(sig)))
    else:
        print("  Observed dispersion does NOT exceed gene-set-randomization null.")
        print("  -> dispersion is consistent with random tissue clock composition.")

    # Save
    pd.DataFrame({
        "stat": ["mean_drug_SD", "median_drug_IQR", "p95_drug_range", "mixed_frac"],
        "observed": [obs_mean_sd, obs_med_iqr, obs_p95, obs_mixed],
        "null_mean": [null_mean_sd.mean(), null_med_iqr.mean(), null_p95.mean(), null_mixed.mean()],
        "null_sd": [null_mean_sd.std(), null_med_iqr.std(), null_p95.std(), null_mixed.std()],
        "p_value": [p_sd, p_iqr, p_rng, p_mix],
    }).to_csv(TABLE_DIR / "interaction_test_results.csv", index=False)
    print("\nSaved: interaction_test_results.csv")


if __name__ == "__main__":
    main()