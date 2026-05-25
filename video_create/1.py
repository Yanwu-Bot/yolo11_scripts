import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成二维数据点（两个簇，模拟相似数据）
np.random.seed(42)
X, _ = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=42)

# 普通哈希：随机划分空间（用随机斜线分割）
def random_hash(point):
    # 使用一个随机直线将空间分为两部分（0或1）
    np.random.seed(hash(tuple(point)) % 2**32)  # 伪随机种子
    a = np.random.randn(2)
    b = np.random.randn()
    return int(np.dot(a, point) + b > 0)

# LSH：基于欧氏距离的局部敏感哈希（使用随机超平面投影）
def lsh_hash(point, planes):
    # 用多组随机超平面投影，每个投影给出一个bit
    return tuple(np.dot(planes, point) > 0)

# 生成随机超平面（用于LSH）
num_planes = 2  # 投影维度，产生2^2=4个桶
planes = np.random.randn(num_planes, 2)

# 计算每个点的哈希值
random_buckets = [random_hash(p) for p in X]
lsh_buckets = [lsh_hash(p, planes) for p in X]

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, buckets, title in zip(axes, [random_buckets, lsh_buckets], 
                               ['Ordinary Hash (Random Split)', 'LSH (Random Hyperplane Projection)']):
    # 绘制点，按桶着色
    unique_buckets = list(set(buckets))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_buckets)))
    for i, p in enumerate(X):
        bucket = buckets[i]
        idx = unique_buckets.index(bucket)
        ax.scatter(p[0], p[1], c=[colors[idx]], s=40, edgecolors='k', linewidth=0.5, zorder=3)
    
    # 如果是LSH，绘制超平面（分割线）
    if 'LSH' in title:
        x_range = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
        for plane in planes:
            # 直线方程: plane[0]*x + plane[1]*y = 0  => y = -plane[0]/plane[1] * x
            if plane[1] != 0:
                y_vals = -plane[0]/plane[1] * x_range
                ax.plot(x_range, y_vals, 'gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title('LSH: Similar points fall into same bucket (same color)\nBuckets: {}'.format(len(unique_buckets)), fontsize=11)
    else:
        ax.set_title('Ordinary Hash: Random split (colors random)\nBuckets: {}'.format(len(unique_buckets)), fontsize=11)
    
    ax.set_xlim(X[:,0].min()-1, X[:,0].max()+1)
    ax.set_ylim(X[:,1].min()-1, X[:,1].max()+1)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()