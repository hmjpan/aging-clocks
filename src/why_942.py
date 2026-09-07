import numpy as np
import pandas as pd

T = r"H:\lunwencp\shuailao\results\tables"
matrix = pd.read_csv(T + r"\drug_tissue_reversal_matrix.csv", index_col=0)
mv = matrix.values
n_drugs, n_tissues = mv.shape
rng = np.random.RandomState(42)

print("=" * 70)
print("WHY 94.2%?  - decomposition")
print("=" * 70)

# 1. Per-tissue marginal: fraction of |score|>1 positive / negative
thresh = 1.0
pos_frac = (mv > thresh).mean(axis=0)   # per tissue
neg_frac = (mv < -thresh).mean(axis=0)
print("\n1. Per-tissue marginals (|score|>1.0):")
print("   mean P(+1) across tissues: {:.3f}".format(pos_frac.mean()))
print("   mean P(-1) across tissues: {:.3f}".format(neg_frac.mean()))
print("   range P(+1): [{:.3f}, {:.3f}]".format(pos_frac.min(), pos_frac.max()))
print("   range P(-1): [{:.3f}, {:.3f}]".format(neg_frac.min(), neg_frac.max()))

# 2. Independence assumption: P(drug has >=1 pos) = 1-(1-p)^49
p_pos_ind = 1 - np.prod(1 - pos_frac)
p_neg_ind = 1 - np.prod(1 - neg_frac)
p_mixed_ind = p_pos_ind * p_neg_ind  # approx if independent (approx, slight overcount)
print("\n2. If tissues were INDEPENDENT (null-like):")
print("   P(>=1 positive tissue) = 1 - prod(1-p_t) = {:.4f}".format(p_pos_ind))
print("   P(>=1 negative tissue) = {:.4f}".format(p_neg_ind))
print("   P(mixed) approx = {:.4f} ({:.1f}%)".format(p_mixed_ind, p_mixed_ind * 100))

# 3. Observed
n_pos = (mv > thresh).sum(axis=1)
n_neg = (mv < -thresh).sum(axis=1)
mixed_obs = ((n_pos > 0) & (n_neg > 0)).mean()
print("\n3. OBSERVED (real matrix, tissues correlated):")
print("   P(drug has >=1 pos): {:.4f}".format((n_pos > 0).mean()))
print("   P(drug has >=1 neg): {:.4f}".format((n_neg > 0).mean()))
print("   P(mixed) observed: {:.4f} ({:.1f}%)".format(mixed_obs, mixed_obs * 100))

# 4. Why observed < independence? -> cross-tissue correlation
tissue_corr = np.corrcoef(mv.T)  # tissues x tissues
off = tissue_corr[np.triu_indices(n_tissues, 1)]
print("\n4. Cross-tissue correlation of drug scores:")
print("   mean off-diagonal r = {:.4f} (POSITIVE = tissues agree)")
print("   -> positive correlation means same drug tends to score"
      "\n      same sign across tissues -> FEWER mixed drugs than independent case")

# 5. Null simulation: within-tissue shuffle = break correlation, keep marginals
null_mixed = []
for i in range(1000):
    pm = np.column_stack([rng.permutation(mv[:, j]) for j in range(n_tissues)])
    nr = (pm > thresh).sum(axis=1)
    nn = (pm < -thresh).sum(axis=1)
    null_mixed.append(((nr > 0) & (nn > 0)).mean())
null_mixed = np.array(null_mixed)
print("\n5. NULL (shuffle within each tissue, 1000x):")
print("   mixed fraction: {:.4f} +/- {:.4f}".format(null_mixed.mean(), null_mixed.std()))
print("   -> higher than observed {} because shuffling destroys the"
      "\n      positive cross-tissue correlation".format(mixed_obs))

# 6. Where does the 'signal' actually live? -> more stringent definitions
print("\n6. More stringent definitions (observed vs null):")
for th, k in [(2.0, 1), (2.0, 3), (2.0, 5)]:
    obs = ((mv > th).sum(axis=1) >= k) & ((mv < -th).sum(axis=1) >= k)
    obs_f = obs.mean()
    nulls = []
    for i in range(500):
        pm = np.column_stack([rng.permutation(mv[:, j]) for j in range(n_tissues)])
        nulls.append((((pm > th).sum(axis=1) >= k) & ((pm < -th).sum(axis=1) >= k)).mean())
    nulls = np.array(nulls)
    p = np.mean(nulls >= obs_f)
    print("   thr={} k>={}: obs={:.4f} null={:.4f}+/-{:.4f} p={}".format(
        th, k, obs_f, nulls.mean(), nulls.std(), p))

# 7. What actually drives the >2.0 signal: drugs with truly extreme opposite scores
print("\n7. Top drugs by |score|>2.0 mixed pattern:")
sig = ((mv > 2.0).sum(axis=1) >= 3) & ((mv < -2.0).sum(axis=1) >= 3)
if sig.sum() > 0:
    for c in matrix.index[sig][:15]:
        row = matrix.loc[c]
        print("   {:<25} n>2={:>2} n<-2={:>2} range={:>6.1f}".format(
            c, (row > 2.0).sum(), (row < -2.0).sum(), row.max() - row.min()))
print("   total drugs with thr=2.0 k>=3 mixed: {}".format(sig.sum()))