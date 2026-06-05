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

def frame_score(q, t=0.05, k=2):
    """
    计算帧评分（与calculate_frame_score逻辑一致）
    参数：
        q: 加权平均误差（或直接传入误差值）
        t: 阈值
        k: 衰减系数
    """
    if q <= t:
        return 100.0
    else:
        exceed = q - t
        return 100 * np.exp(-k * exceed)

S_ij = 100
t = 0.05

# 生成q值范围
q_values = np.linspace(0, 1.5, 500)

# 计算三种评分
paper_scores = [paper_score(q, t, S_ij) for q in q_values]
exp_scores = [exp_score(q, t) for q in q_values]
frame_scores = [frame_score(q, t, k=2) for q in q_values]

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# 绘制三条评分曲线（无额外标记）
ax.plot(q_values, paper_scores, linewidth=2.5, color='blue', label='论文评分函数 (分段线性)')
ax.plot(q_values, exp_scores, linewidth=2.5, color='red', label='指数衰减评分函数 (分段k)')
ax.plot(q_values, frame_scores, linewidth=2.5, color='green', label='帧评分函数 (k=4)')

# 只保留最基本的参考线（零线、满分线），无关键点标记
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# 简洁的图例与坐标轴标签
ax.set_xlabel('误差值 q')
ax.set_ylabel('评分')
ax.set_title('评分函数对比')
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# 打印对比数据
print("="*80)
print(f"评分函数对比 (阈值 t = {t})")
print("="*80)
print("\n不同误差值下的分数对比:")
print("-"*80)
print(f"{'q值':<10} {'论文评分':<15} {'指数衰减评分':<15} {'帧评分(k=4)':<15}")
print("-"*80)

test_q = [0, t, t+0.1, t+0.2, t+0.3, t+0.5, t+0.8, 1.0, 1.2]
for q in test_q:
    paper = paper_score(q, t, 100)
    exp = exp_score(q, t)
    frame = frame_score(q, t, k=2)
    print(f"{q:<10.2f} {paper:<15.1f} {exp:<15.1f} {frame:<15.1f}")

print("\n" + "="*80)
print("评分函数说明:")
print("="*80)
print("论文评分函数: 分段线性，q≤t满分，t<q≤1线性下降，>1零分")
print("指数衰减评分函数: 分段指数，q≤t满分，之后按不同衰减率下降")
print("帧评分函数(k=4): 单指数衰减，q≤t满分，否则按 mu = 100 * exp(-k*(q-t)) 计算")