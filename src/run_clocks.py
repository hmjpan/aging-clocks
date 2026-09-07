"""Run aging clock training for all eligible GTEx tissues.

For each tissue (>=80 samples, >=4 age brackets):
  1. Read per-tissue GCT (log2(TPM+1))
  2. Filter low-expression genes, select top-variance genes
  3. Train ElasticNetCV with donor-grouped K-fold CV
  4. Save: coefficients, z-score params, CV predictions, metrics

Outputs (per tissue):
  results/clocks/<tissue>_coef.csv      — gene_id, gene_name, weight (nonzero)
  results/clocks/<tissue>_params.npz    — gene_mean, gene_std, intercept, alpha, l1_ratio
  results/clocks/<tissue>_predictions.csv — SAMPID, age, predicted, residual

Outputs (global):
  results/tables/clock_summary.csv      — one row per tissue with metrics
  results/tables/clock_permutation.csv  — permutation test null distribution

Usage:
    python src/run_clocks.py [--n-perm 1000] [--tissue whole_blood]
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(__file__))
from config import CLOCK_PARAMS, CLOCK_DIR, GTEX_DIR, TABLE_DIR
from gct_io import build_tissue_sample_table, read_gct

warnings.filterwarnings("ignore")

CLOCK_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare(tissue_slug, meta, params):
    """Read GCT, filter, transform, return X (samples×genes), y, groups, gene_ids."""
    path = GTEX_DIR / f"tpm_{tissue_slug}.gct.gz"
    if not path.exists():
        return None

    expr = read_gct(path)
    sample_cols = [c for c in expr.columns if c != "gene_name" and c in meta.index]
    if len(sample_cols) < params["min_samples"]:
        return None

    sub_meta = meta.loc[sample_cols]
    age = sub_meta["AGE_NUM"].values.astype(float)
    donors = sub_meta["SUBJID"].values
    brackets = sub_meta["AGE"].values
    valid = ~np.isnan(age)
    sample_cols = [s for s, v in zip(sample_cols, valid) if v]
    age = age[valid]
    donors = donors[valid]
    brackets = brackets[valid]

    if len(pd.Series(brackets).unique()) < params["min_age_brackets"]:
        return None

    # gene filtering
    mat = expr.loc[:, sample_cols]
    present = (mat >= params["min_expression"]).mean(axis=1)
    keep = present[present >= 0.10].index
    mat = mat.loc[keep]

    # variance filter
    var = mat.var(axis=1)
    n_top = min(params["n_top_features_variance"], len(var))
    keep = var.nlargest(n_top).index
    mat = mat.loc[keep]

    gene_ids = mat.index.tolist()
    gene_names = expr.loc[gene_ids, "gene_name"].values

    # log2 transform
    X = np.log2(mat.values.astype(np.float64) + 1.0).T  # samples × genes
    y = age.copy()

    return {
        "X": X, "y": y, "groups": donors, "gene_ids": gene_ids,
        "gene_names": gene_names, "n_samples": len(y),
        "n_donors": len(np.unique(donors)), "brackets": brackets,
        "sample_ids": sample_cols,
    }


def train_clock(data, params, do_perm=False, n_perm=1000):
    """Train elastic net clock with grouped CV. Returns results dict."""
    X, y, groups = data["X"], data["y"], data["groups"]
    gene_ids = data["gene_ids"]
    rng = np.random.RandomState(params["random_state"])

    # z-score
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    n_splits = min(params["cv_folds"], data["n_donors"])
    gkf = GroupKFold(n_splits=n_splits)

    # ElasticNetCV for hyperparameter selection
    enet_cv = ElasticNetCV(
        l1_ratio=params["l1_ratio_grid"],
        alphas=None,
        cv=gkf,
        max_iter=10000,
        n_alphas=100,
        random_state=params["random_state"],
        n_jobs=-1,
    )
    enet_cv.fit(Xz, y, groups=groups)
    best_alpha = enet_cv.alpha_
    best_l1 = enet_cv.l1_ratio_

    # OOF predictions via refit with best params
    oof_pred = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(Xz, y, groups):
        m = ElasticNet(
            alpha=best_alpha, l1_ratio=best_l1,
            max_iter=10000, random_state=params["random_state"],
        )
        m.fit(Xz[tr_idx], y[tr_idx])
        oof_pred[te_idx] = m.predict(Xz[te_idx])

    # final model (fit on all data)
    final_model = ElasticNet(
        alpha=best_alpha, l1_ratio=best_l1,
        max_iter=10000, random_state=params["random_state"],
    )
    final_model.fit(Xz, y)

    # metrics
    mae = float(np.mean(np.abs(oof_pred - y)))
    median_ae = float(np.median(np.abs(oof_pred - y)))
    r, p_r = pearsonr(oof_pred, y)
    rho, p_rho = spearmanr(oof_pred, y)
    ss_res = np.sum((y - oof_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # permutation test
    perm_mae_null = None
    perm_p = None
    if do_perm and n_perm > 0:
        perm_maes = []
        for _ in range(n_perm):
            y_perm = y[rng.permutation(len(y))]
            enet_p = ElasticNet(
                alpha=best_alpha, l1_ratio=best_l1,
                max_iter=10000, random_state=params["random_state"],
            )
            preds = np.full(len(y), np.nan)
            for tr_idx, te_idx in gkf.split(Xz, y_perm, groups):
                enet_p.fit(Xz[tr_idx], y_perm[tr_idx])
                preds[te_idx] = enet_p.predict(Xz[te_idx])
            perm_maes.append(np.mean(np.abs(preds - y_perm)))
        perm_mae_null = np.array(perm_maes)
        perm_p = float(np.mean(perm_mae_null <= mae))

    # coefficients
    coef = pd.Series(final_model.coef_, index=gene_ids)
    coef_nonzero = coef[coef != 0].sort_values(key=lambda x: x.abs(), ascending=False)

    return {
        "mae": mae, "median_ae": median_ae,
        "pearson_r": float(r), "pearson_p": float(p_r),
        "spearman_r": float(rho), "spearman_p": float(p_rho),
        "r2": r2,
        "best_alpha": best_alpha, "best_l1_ratio": best_l1,
        "n_features": int((final_model.coef_ != 0).sum()),
        "intercept": float(final_model.intercept_),
        "coef": coef_nonzero,
        "gene_ids": gene_ids,
        "mu": mu, "sd": sd,
        "oof_pred": oof_pred,
        "perm_mae_null": perm_mae_null,
        "perm_p": perm_p,
    }


def save_results(tissue, data, res):
    """Save per-tissue results."""
    # coefficients
    coef_df = pd.DataFrame({
        "gene_id": res["coef"].index,
        "weight": res["coef"].values,
    })
    coef_df["gene_name"] = [data["gene_names"][data["gene_ids"].index(gid)]
                           for gid in coef_df["gene_id"]]
    coef_df.to_csv(CLOCK_DIR / f"{tissue}_coef.csv", index=False)

    # params (for projection in Step 3)
    np.savez(
        CLOCK_DIR / f"{tissue}_params.npz",
        gene_ids=np.array(res["gene_ids"], dtype=object),
        gene_names=np.array(data["gene_names"], dtype=object),
        mu=res["mu"], sd=res["sd"],
        intercept=res["intercept"],
        alpha=res["best_alpha"], l1_ratio=res["best_l1_ratio"],
        coef=final_coef_full(res),
    )

    # predictions
    pred_df = pd.DataFrame({
        "SAMPID": data["sample_ids"],
        "SUBJID": data["groups"],
        "AGE_bracket": data["brackets"],
        "age_actual": data["y"],
        "age_predicted": res["oof_pred"],
        "age_residual": data["y"] - res["oof_pred"],
    })
    pred_df.to_csv(CLOCK_DIR / f"{tissue}_predictions.csv", index=False)


def final_coef_full(res):
    """Return full coefficient vector (including zeros)."""
    full = np.zeros(len(res["gene_ids"]))
    for i, gid in enumerate(res["gene_ids"]):
        if gid in res["coef"].index:
            full[i] = res["coef"][gid]
    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", default=None, help="Single tissue slug (for testing)")
    parser.add_argument("--n-perm", type=int, default=1000, help="Permutation test iterations")
    args = parser.parse_args()

    params = CLOCK_PARAMS.copy()
    meta = build_tissue_sample_table().set_index("SAMPID")

    # find eligible tissues
    if args.tissue:
        tissues = [args.tissue]
    else:
        tissues = sorted(f.name.replace("tpm_", "").replace(".gct.gz", "")
                         for f in GTEX_DIR.glob("tpm_*.gct.gz"))

    print(f"Processing {len(tissues)} tissues\n")
    summary_rows = []
    perm_rows = []

    for i, tissue in enumerate(tissues, 1):
        t0 = time.time()
        print(f"[{i}/{len(tissues)}] {tissue} ...", end=" ", flush=True)

        data = load_and_prepare(tissue, meta, params)
        if data is None:
            print("SKIP (insufficient samples/age brackets)")
            continue

        res = train_clock(data, params, do_perm=True, n_perm=args.n_perm)
        save_results(tissue, data, res)

        elapsed = time.time() - t0
        print(f"n={data['n_samples']} donors={data['n_donors']} "
              f"feat={res['n_features']} MAE={res['mae']:.2f} "
              f"r={res['pearson_r']:.3f} R2={res['r2']:.3f} "
              f"perm_p={res['perm_p']:.4f} ({elapsed:.0f}s)")

        summary_rows.append({
            "tissue": tissue,
            "n_samples": data["n_samples"],
            "n_donors": data["n_donors"],
            "n_features": res["n_features"],
            "mae": res["mae"],
            "median_ae": res["median_ae"],
            "pearson_r": res["pearson_r"],
            "pearson_p": res["pearson_p"],
            "spearman_r": res["spearman_r"],
            "spearman_p": res["spearman_p"],
            "r2": res["r2"],
            "best_alpha": res["best_alpha"],
            "best_l1_ratio": res["best_l1_ratio"],
            "perm_p": res["perm_p"],
        })

        if res["perm_mae_null"] is not None:
            for v in res["perm_mae_null"]:
                perm_rows.append({"tissue": tissue, "null_mae": v})

    # save summary
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "clock_summary.csv", index=False)
    print(f"\nSummary saved: {TABLE_DIR / 'clock_summary.csv'}")

    if perm_rows:
        perm_df = pd.DataFrame(perm_rows)
        perm_df.to_csv(TABLE_DIR / "clock_permutation.csv", index=False)
        print(f"Permutation null saved: {TABLE_DIR / 'clock_permutation.csv'}")

    # print top/bottom performers
    if len(summary) > 0:
        print("\n=== Top 10 clocks by Pearson r ===")
        print(summary.nlargest(10, "pearson_r")[
            ["tissue", "n_samples", "n_features", "mae", "pearson_r", "r2", "perm_p"]
        ].to_string(index=False))
        print("\n=== Bottom 10 clocks ===")
        print(summary.nsmallest(10, "pearson_r")[
            ["tissue", "n_samples", "n_features", "mae", "pearson_r", "r2", "perm_p"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
