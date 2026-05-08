import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'

def paper_score(q, t=0.05, S_ij=100):
    """论文中的分段评分函数"""
    if t == 1:
        return 0
    elif q <= t:
        return S_ij
    elif t < q <= 1:
        return S_ij * (1 - q + t)
    else:
        return 0

def exp_score(q, t=0.05):
    """指数衰减评分函数"""
    q_mean = q
    if q_mean <= t:
        return 100.0
    elif q_mean - t <= 0.3:
        exceed = q_mean - t
        return 100 * np.exp(-2.5 * exceed)
    elif 0.3 < q_mean - t <= 1:
        exceed = q_mean - t
        return 100 * np.exp(-5 * exceed)
    else:
        return 0.0

S_ij = 100
t = 0.05

# 生成q值范围
q_values = np.linspace(0, 1.5, 500)

# 计算两种评分
paper_scores = [paper_score(q, t, S_ij) for q in q_values]
exp_scores = [exp_score(q, t) for q in q_values]

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# 绘制论文评分曲线
ax.plot(q_values, paper_scores, linewidth=2.5, color='blue', label='论文评分函数 (分段线性)')

# 绘制指数衰减评分曲线
ax.plot(q_values, exp_scores, linewidth=2.5, color='red', label='指数衰减评分函数')

# 标记关键点
ax.plot(t, 100, 'o', color='blue', markersize=8)
ax.plot(1, 100 * t, 's', color='blue', markersize=8)
ax.plot(1, 0, '^', color='blue', markersize=8)

# 指数衰减的关键点
t1 = t + 0.3
score_at_t1 = 100 * np.exp(-2.5 * 0.3)
ax.plot(t1, score_at_t1, 'ro', color='red', markersize=6)
ax.plot(t + 0.3, score_at_t1, 'ro', markersize=6)

ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='满分线 (100分)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 添加区域标注
# ax.fill_between([t, 1], [0, 0], [100, 100 * t], 
#                  alpha=0.1, color='yellow', label='线性衰减区')
# ax.fill_between([t, t+0.3], [0, 0], [100, score_at_t1], 
#                  alpha=0.1, color='orange', label='指数衰减区1 (k=2.5)')
# ax.fill_between([t+0.3, 1], [0, 0], [score_at_t1, 100 * np.exp(-5 * (1-t))], 
#                  alpha=0.1, color='purple', label='指数衰减区2 (k=5)')

# ax.set_xlabel('q (测量误差)', fontsize=12)
# ax.set_ylabel('Score (分数)', fontsize=12)
# ax.set_title(f'评分函数对比 (t = {t})', fontsize=14, fontweight='bold')
# ax.set_xlim(0, 1.5)
# ax.set_ylim(-5, 105)
# ax.set_xticks(np.arange(0, 1.6, 0.2))
# ax.set_yticks(np.arange(0, 101, 10))
# ax.grid(True, alpha=0.3)
# ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

# 打印对比数据
print("="*80)
print(f"评分函数对比 (阈值 t = {t})")
print("="*80)
print("\n不同误差值下的分数对比:")
print("-"*80)
print(f"{'q值':<10} {'论文评分':<15} {'指数衰减评分':<15} {'差异':<10}")
print("-"*80)

test_q = [0, t, t+0.1, t+0.2, t+0.3, t+0.5, t+0.8, 1.0, 1.2]
for q in test_q:
    paper = paper_score(q, t, 100)
    exp = exp_score(q, t)
    diff = paper - exp
    print(f"{q:<10.2f} {paper:<15.1f} {exp:<15.1f} {diff:<+10.1f}")

print("\n" + "="*80)
print("指数衰减函数分段说明:")
print("="*80)
print(f"1. q ≤ {t}: 满分 100 分")
print(f"2. {t} < q ≤ {t+0.3}: 指数衰减, k=2.5")
print(f"3. {t+0.3} < q ≤ 1: 指数衰减, k=5")
print(f"4. q > 1: 0 分")