# dist_feature.py
# 仿照 dist.py，使用 Feature.py 的 26 维特征 + L2 欧氏距离计算窗口间距离并推荐 threshold
import numpy as np
import random
import os
import sys

# 确保能导入 Feature（如果 Feature.py 不在当前目录，改成你的实际路径）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Feature import Feature


def extract_window_features(windows, idx, window_size=7):
    """
    提取第 idx 个窗口的逐帧特征。
    返回: (T, 26)，每帧一行，26 维 = 角度16 + 中心4 + 朝向2 + 协调4
    """
    win = windows[idx]  # (T, V, 2)
    feats = np.zeros((window_size, 26), dtype=np.float32)
    for t in range(window_size):
        try:
            feats[t] = Feature(win[t].tolist()).get_all_features()
        except Exception as e:
            feats[t] = 0.0
    return feats


def window_distance(feat_i, feat_j, agg='mean'):
    """
    按 L2 欧氏距离计算两个窗口的特征距离（与 ST_filter_fea.py 一致）：
    1. 同一帧特征差 -> (T, 26)
    2. 对 26 维求 L2 范数 -> 该帧特征距离标量 (T,)
    3. 对 T 帧取平均(默认)或累加 -> 窗口间距离标量
    """
    diff = feat_i - feat_j                      # (T, 26)
    frame_dist = np.linalg.norm(diff, axis=-1)  # (T,) 每帧 L2 距离
    if agg == 'sum':
        return float(frame_dist.sum())          # 累加
    return float(frame_dist.mean())             # 取平均（默认）


def recommend_threshold(npz_path, sample_pairs=5000, seed=42, agg='mean'):
    """
    分析数据集窗口间 Feature 特征 L2 距离分布，推荐 diversity_threshold 值。

    参数:
        npz_path: .npz 数据集文件路径
        sample_pairs: 随机采样的窗口对数量
        seed: 随机种子
        agg: 'mean' 对 T 帧距离取平均；'sum' 累加
    """
    random.seed(seed)
    np.random.seed(seed)

    # 加载数据
    data = np.load(npz_path, allow_pickle=True)
    windows = data['windows']
    N = len(windows)
    T = windows.shape[1]
    print(f"数据集共 {N} 个窗口，每个窗口形状: {windows.shape[1:]}")
    print(f"特征定义: 26 维 = 角度16 + 四肢中心4 + 身体朝向2 + 对侧协调4")
    print(f"距离定义: 逐帧 26 维特征差 -> L2 欧氏距离 -> {T} 帧{'平均' if agg=='mean' else '累加'}")

    if N <= 1:
        print("数据集中窗口数量不足，无法计算距离分布。")
        return None

    # 采样窗口对（与 dist.py 相同逻辑）
    total_pairs = N * (N - 1) // 2
    if total_pairs <= sample_pairs:
        print(f"总窗口对数 {total_pairs}，进行全量计算...")
        pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    else:
        print(f"总窗口对数 {total_pairs}，随机采样 {sample_pairs} 对...")
        pairs = set()
        while len(pairs) < sample_pairs:
            i = random.randrange(N)
            j = random.randrange(N)
            if i != j:
                pairs.add((i, j) if i < j else (j, i))
        pairs = list(pairs)

    # 预计算所有被采样窗口的逐帧特征（避免同一窗口在多对里重复提取）
    unique_idx = sorted(set([i for i, j in pairs] + [j for i, j in pairs]))
    print(f"采样窗口对 {len(pairs)} 对，涉及 {len(unique_idx)} 个唯一窗口，预计算特征...")
    feat_cache = {}
    for idx in unique_idx:
        feat_cache[idx] = extract_window_features(windows, idx, window_size=T)

    # 计算距离
    dists = []
    for i, j in pairs:
        d = window_distance(feat_cache[i], feat_cache[j], agg=agg)
        dists.append(d)
    dists = np.array(dists)

    # 统计信息
    print("\n" + "=" * 50)
    print("窗口间 Feature 特征 L2 距离统计（逐帧 L2 -> 帧平均）")
    print("=" * 50)
    print(f"样本数（窗口对）: {len(dists)}")
    print(f"最小值: {dists.min():.6f}")
    print(f"最大值: {dists.max():.6f}")
    print(f"均值:   {dists.mean():.6f}")
    print(f"标准差: {dists.std():.6f}")
    print(f"中位数: {np.median(dists):.6f}")
    print(f"25%分位数: {np.percentile(dists, 25):.6f}")
    print(f"10%分位数: {np.percentile(dists, 10):.6f}")
    print(f"5%分位数:  {np.percentile(dists, 5):.6f}")
    print(f"1%分位数:  {np.percentile(dists, 1):.6f}")

    # 直方图分布简要提示
    print("\n距离分布区间:")
    percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    for p in percentiles:
        print(f"  {p}% 分位数: {np.percentile(dists, p):.6f}")

    # 推荐阈值
    print("\n" + "=" * 50)
    print("推荐 diversity_threshold 值")
    print("=" * 50)

    # 推荐1: 5%分位数（过滤最相似的5%窗口对）
    rec1 = np.percentile(dists, 5)
    print(f"推荐值1（5%分位数，严格）: {rec1:.4f}")
    print(f"  -> 含义：约5%的窗口对会被视为相似而被过滤")

    # 推荐2: 10%分位数（较宽松）
    rec2 = np.percentile(dists, 10)
    print(f"推荐值2（10%分位数，适中）: {rec2:.4f}")
    print(f"  -> 含义：约10%的窗口对会被视为相似而被过滤")

    # 推荐3: 25%分位数（更宽松）
    rec3 = np.percentile(dists, 25)
    print(f"推荐值3（25%分位数，宽松）: {rec3:.4f}")
    print(f"  -> 含义：约25%的窗口对会被视为相似而被过滤")

    # 推荐4: 中位数的一半
    rec4 = np.median(dists) / 2
    print(f"推荐值4（中位数的一半）: {rec4:.4f}")

    # 最佳推荐（基于数据分布自动选择）
    best_rec = rec2
    best_rec = max(best_rec, 0.01)  # L2 距离下限
    print(f"\n★ 综合推荐值: {best_rec:.4f}")
    print(f"   建议先将 diversity_threshold={best_rec:.4f} 设为初始值")
    print(f"   预期过滤约 10% 的窗口对（最相似的负样本）")
    print(f"   训练时若过滤比例过高（负样本太少），可降低阈值；")
    print(f"   若发现负样本仍太相似，可适当提高。")

    # 额外建议
    print("\n" + "-" * 50)
    print("使用建议:")
    print(f"  1. L2 距离量级与 L1 平均不同：L2 ≈ L1 × √26 ≈ L1 × 5.1")
    print(f"     当前 L2 均值约 {dists.mean():.3f}，阈值应在 {rec1:.2f}~{rec3:.2f} 之间")
    print(f"  2. 原来的 threshold=0.13（L1 时代）在 L2 下几乎不过滤，必须换新值")
    print(f"  3. 想更严格过滤，取 25% 分位数；想更宽松，取 5% 分位数")
    print(f"  4. 如果想关闭过滤，将 diversity_threshold=0")
    print(f"  5. agg='sum'（帧距离累加）时阈值约为 'mean' 的 {T} 倍，需按对应模式取值")
    print("-" * 50)

    return best_rec


if __name__ == '__main__':
    # 请修改为您的 .npz 文件路径
    npz_path = 'result/GCN/dataset/dataset_7_1.npz'

    # 分析并推荐阈值（默认帧距离取平均）
    recommended = recommend_threshold(
        npz_path,
        sample_pairs=5000,  # 采样对数，可根据需要调整
        seed=42,
        agg='mean'          # 可选 'mean'（取平均）或 'sum'（累加）
    )