import numpy as np
import random

# 请修改为您的 .npz 文件路径
npz_path = 'result/GCN/dataset/dataset_7_1.npz'

# 加载数据
data = np.load(npz_path, allow_pickle=True)
windows = data['windows']
N = len(windows)
print(f"数据集共 {N} 个窗口，每个窗口形状 {windows.shape[1:]}")

# 由于N可能很大，我们随机抽取部分窗口对计算距离（避免全量计算）
max_samples = 2000  # 最多计算2000个窗口对（可调）
if N * (N-1) / 2 > max_samples:
    print(f"窗口总数较大，随机抽样 {max_samples} 个窗口对进行统计")
    # 随机抽样索引对
    pairs = set()
    while len(pairs) < max_samples:
        i = random.randrange(N)
        j = random.randrange(N)
        if i != j:
            pairs.add((i, j) if i < j else (j, i))
    pairs = list(pairs)
else:
    # 全量计算
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

dists = []
for i, j in pairs:
    w_i = windows[i]
    w_j = windows[j]
    # 计算逐帧逐关键点欧氏距离，然后取平均
    diff = np.linalg.norm(w_i - w_j, axis=-1)  # (T, V)
    dist = np.mean(diff)
    dists.append(dist)

dists = np.array(dists)
print("\n=== 窗口间平均帧间关键点距离统计 ===")
print(f"样本数（窗口对）：{len(dists)}")
print(f"最小值：{dists.min():.6f}")
print(f"最大值：{dists.max():.6f}")
print(f"均值：{dists.mean():.6f}")
print(f"中位数：{np.median(dists):.6f}")
print(f"25%分位数：{np.percentile(dists, 25):.6f}")
print(f"10%分位数：{np.percentile(dists, 10):.6f}")
print(f"5%分位数：{np.percentile(dists, 5):.6f}")

# 建议
print("\n建议的 diversity_threshold 参考值：")
print(f" - 若希望过滤最相似的10%窗口对：{np.percentile(dists, 10):.4f}")
print(f" - 若希望过滤最相似的5%窗口对：{np.percentile(dists, 5):.4f}")
print(f" - 中位数的一半：{np.median(dists)/2:.4f}")