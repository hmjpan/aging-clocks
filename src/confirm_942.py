import numpy as np
import pandas as pd

T = r"H:\lunwencp\shuailao\results\tables"
matrix = pd.read_csv(T + r"\drug_tissue_reversal_matrix.csv", index_col=0)
mv = matrix.values
rng = np.random.RandomState(42)

print("=" * 72)
print("94.2% 数字链条的完整演变与核查")
print("=" * 72)

print("\n【1】不同阈值下的混合比例（同一矩阵 3926x49）")
for th in [0.0, 0.5, 1.0, 2.0]:
    n_reju = (mv > th).sum(axis=1)
    n_pro = (mv < -th).sum(axis=1)
    mixed = ((n_reju > 0) & (n_pro > 0)).sum()
    print("  |score|>{}: mixed = {}/3926 = {:.1f}%".format(th, mixed, mixed / 3926 * 100))

print("\n【2】99.7% 是从哪来的（历史遗留）")
h = pd.read_csv(T + r"\drug_heterogeneity_scores.csv")
# 这个文件是 3919 行（>=3组织有|score|>0.5信号），is_mixed 列用的是 >0.5
print("  drug_heterogeneity_scores.csv 行数: {}".format(len(h)))
print("  is_mixed=1 数: {}".format(h["is_mixed"].sum()))
print("  {}/{} = {:.1f}%  <- 这就是99.7%的来历（阈值|score|>0.5，分母3919）".format(
    h["is_mixed"].sum(), len(h), h["is_mixed"].sum() / len(h) * 100))

print("\n【3】94.2% 是怎么来的")
print("  阈值 |score|>1.0, 分母改为完整矩阵 3926")
print("  mixed = {} / 3926 = {:.2f}%".format(
    ((mv > 1.0).sum(axis=1) > 0).sum() & 0, 0))  # placeholder

# 正确计算
n_reju = (mv > 1.0).sum(axis=1)
n_pro = (mv < -1.0).sum(axis=1)
mixed_1 = ((n_reju > 0) & (n_pro > 0)).sum()
print("  mixed(|score|>1.0) = {} / 3926 = {:.2f}%".format(mixed_1, mixed_1 / 3926 * 100))

print("\n【4】这个 94.2% 到底证不证明异质性？(null 检验)")
obs = mixed_1 / 3926
nulls = []
for i in range(1000):
    pm = np.column_stack([rng.permutation(mv[:, j]) for j in range(mv.shape[1])])
    nr = (pm > 1.0).sum(axis=1)
    nn = (pm < -1.0).sum(axis=1)
    nulls.append(((nr > 0) & (nn > 0)).mean())
nulls = np.array(nulls)
print("  观测混合比例: {:.4f}".format(obs))
print("  随机null混合比例: {:.4f} +/- {:.4f}".format(nulls.mean(), nulls.std()))
print("  结论: 观测{}随机null -> 94.2% 不能证明异质性，应排除该表述".format(
    "显著高于" if obs > np.percentile(nulls, 95) else "不高于"))

print("\n【5】真正显著的异质性定义")
for th, k in [(2.0, 3), (2.0, 5)]:
    obs2 = ((mv > th).sum(axis=1) >= k) & ((mv < -th).sum(axis=1) >= k)
    obs2f = obs2.mean()
    null2s = []
    for i in range(500):
        pm = np.column_stack([rng.permutation(mv[:, j]) for j in range(mv.shape[1])])
        null2s.append((((pm > th).sum(axis=1) >= k) & ((pm < -th).sum(axis=1) >= k)).mean())
    null2s = np.array(null2s)
    p = np.mean(null2s >= obs2f)
    print("  |score|>{} 且正负各>={}组织: 观测={:.3f} null={:.4f} p={}".format(
        th, k, obs2f, null2s.mean(), p))