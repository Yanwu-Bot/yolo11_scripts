import numpy as np
import random

def recommend_threshold(npz_path, sample_pairs=5000, seed=42):
    """
    分析数据集窗口间距离分布，推荐 diversity_threshold 值。
    
    参数:
        npz_path: .npz 数据集文件路径
        sample_pairs: 随机采样的窗口对数量（大数据集时使用）
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 加载数据
    data = np.load(npz_path, allow_pickle=True)
    windows = data['windows']
    N = len(windows)
    
    print(f"数据集共 {N} 个窗口，每个窗口形状: {windows.shape[1:]}")
    
    if N <= 1:
        print("数据集中窗口数量不足，无法计算距离分布。")
        return None
    
    # 计算距离
    dists = []
    
    # 如果窗口总数不多，可以全量计算；否则随机采样
    total_pairs = N * (N - 1) // 2
    
    if total_pairs <= sample_pairs:
        # 全量计算
        print(f"总窗口对数 {total_pairs}，进行全量计算...")
        for i in range(N):
            w_i = windows[i]
            for j in range(i+1, N):
                w_j = windows[j]
                diff = np.linalg.norm(w_i - w_j, axis=-1)  # (T, V)
                dist = float(np.mean(diff))
                dists.append(dist)
    else:
        # 随机采样
        print(f"总窗口对数 {total_pairs}，随机采样 {sample_pairs} 对...")
        pairs = set()
        while len(pairs) < sample_pairs:
            i = random.randrange(N)
            j = random.randrange(N)
            if i != j:
                pairs.add((i, j) if i < j else (j, i))
        pairs = list(pairs)
        
        for i, j in pairs:
            w_i = windows[i]
            w_j = windows[j]
            diff = np.linalg.norm(w_i - w_j, axis=-1)
            dist = float(np.mean(diff))
            dists.append(dist)
    
    dists = np.array(dists)
    
    # 统计信息
    print("\n" + "=" * 50)
    print("窗口间平均帧间关键点距离统计")
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
    
    # 推荐3: 中位数的一半
    rec3 = np.median(dists) / 2
    print(f"推荐值3（中位数的一半，宽松）: {rec3:.4f}")
    
    # 推荐4: 均值的0.3倍
    rec4 = dists.mean() * 0.3
    print(f"推荐值4（均值的0.3倍，中间值）: {rec4:.4f}")
    
    # 最佳推荐（基于数据分布自动选择）
    # 选择10%分位数和中位数的一半中较小的那个，但不要太小
    best_rec = min(rec2, rec3)
    best_rec = max(best_rec, 0.01)  # 确保至少为0.01
    print(f"\n★ 综合推荐值: {best_rec:.4f}")
    print(f"   建议先将 diversity_threshold={best_rec:.4f} 设为初始值")
    print(f"   训练时若出现大量'多样性采样警告'，可适当降低；")
    print(f"   若发现负样本仍太相似，可适当提高。")
    
    # 额外建议
    print("\n" + "-" * 50)
    print("使用建议:")
    print("  1. 如果训练速度很快，建议从推荐值开始")
    print("  2. 如果出现大量'多样性采样警告'，降低阈值（如减半）")
    print("  3. 如果希望更严格过滤，可提高阈值（如取25%分位数）")
    print("  4. 如果想关闭过滤，将 diversity_threshold=0")
    print("-" * 50)
    
    return best_rec


if __name__ == '__main__':
    # 请修改为您的 .npz 文件路径
    npz_path = 'result/GCN/dataset/dataset_7_1.npz'
    
    # 分析并推荐阈值
    recommended = recommend_threshold(
        npz_path,
        sample_pairs=5000,  # 采样对数，可根据需要调整
        seed=42
    )