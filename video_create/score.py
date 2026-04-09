import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

def new_score(q_mean, t=0.5, k=3.5):
    """
    新评分函数：平滑指数衰减
    
    参数:
    q_mean: 实际测量值
    t: 阈值（满分临界值）
    k: 衰减速度控制参数（k越大衰减越快）
    """
    if q_mean <= t:
        return 100.0
    else:
        exceed = q_mean - t
        score = 100 * np.exp(-k * exceed)
        return max(0, score)

# 向量化函数
vectorized_score = np.vectorize(new_score)

# 设置参数
t = 0.5  # 阈值
k_values = [2.0, 3.5, 5.0, 8.0]  # 不同衰减速度

# 生成q_mean的值范围
q_mean_values = np.linspace(t - 0.2, t + 2.0, 500)

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：不同k值的对比
for k in k_values:
    scores = vectorized_score(q_mean_values, t, k)
    axes[0].plot(q_mean_values, scores, linewidth=2, label=f'k = {k}')

axes[0].axvline(x=t, color='r', linestyle='--', alpha=0.7, label=f'阈值 t = {t}')
axes[0].axhline(y=36.8, color='gray', linestyle=':', alpha=0.5, label='36.8% (1/e点)')
axes[0].set_xlabel('q_mean', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('不同衰减速度(k值)对比', fontsize=14, fontweight='bold')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-5, 105)

# 右图：单个k值的详细图（默认k=3.5）
k_default = 3.5
scores_default = vectorized_score(q_mean_values, t, k_default)

axes[1].plot(q_mean_values, scores_default, 'b-', linewidth=2.5, label=f'k = {k_default}')
axes[1].axvline(x=t, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'阈值 t = {t}')
axes[1].axhline(y=100, color='green', linestyle=':', alpha=0.5, label='满分区 (100分)')

# 标注关键点
e_point = t + 1/k_default  # 1/e点
score_at_e = 100 * np.exp(-1)
axes[1].axvline(x=e_point, color='orange', linestyle='--', alpha=0.5, label=f'1/e点 (t+1/k={e_point:.2f})')
axes[1].plot(e_point, score_at_e, 'ro', markersize=8)
axes[1].annotate(f'({e_point:.2f}, {score_at_e:.1f})', 
                 xy=(e_point, score_at_e), xytext=(e_point+0.1, score_at_e+5),
                 arrowprops=dict(arrowstyle='->', color='red'))

# 标注半衰点（50分点）
from scipy.optimize import fsolve
def find_half_life(k):
    return fsolve(lambda x: 100 * np.exp(-k * x) - 50, 0.5)[0]

half_point = find_half_life(k_default)
axes[1].axvline(x=t+half_point, color='purple', linestyle='--', alpha=0.5, label=f'半衰点 (t+{half_point:.2f})')
axes[1].plot(t+half_point, 50, 'go', markersize=8)
axes[1].annotate(f'({t+half_point:.2f}, 50)', 
                 xy=(t+half_point, 50), xytext=(t+half_point+0.1, 55),
                 arrowprops=dict(arrowstyle='->', color='green'))

axes[1].set_xlabel('q_mean', fontsize=12)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title(f'指数衰减评分曲线 (t={t}, k={k_default})', fontsize=14, fontweight='bold')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-5, 105)

plt.tight_layout()
plt.show()

# 打印不同k值下的关键点分数
print("="*80)
print(f"阈值 t = {t}")
print("="*80)
print("\n不同衰减速度(k值)下的分数衰减情况:")
print("-"*80)
print(f"{'超出量':<10} {'k=2.0':<12} {'k=3.5':<12} {'k=5.0':<12} {'k=8.0':<12}")
print("-"*80)

exceed_values = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]
for exceed in exceed_values:
    scores = [100 * np.exp(-k * exceed) for k in k_values]
    score_str = [f"{s:.1f}" for s in scores]
    print(f"{exceed:<10.1f} {score_str[0]:<12} {score_str[1]:<12} {score_str[2]:<12} {score_str[3]:<12}")

print("\n" + "="*80)
print("不同k值下的衰减特性:")
print("="*80)
for k in k_values:
    e_point = 1/k
    half_life = np.log(2)/k
    print(f"k = {k}:")
    print(f"  - 1/e点 (36.8分): 超出量 = {e_point:.3f}")
    print(f"  - 半衰点 (50分): 超出量 = {half_life:.3f}")
    print(f"  - 降到10分所需超出量: {np.log(10)/k:.3f}")
    print()

# 交互式参数调整（可选）
def interactive_score_plot():
    """
    交互式函数，可以手动输入参数查看曲线
    """
    print("\n交互模式：输入参数查看评分曲线")
    try:
        t_input = float(input("请输入阈值 t (默认0.5): ") or "0.5")
        k_input = float(input("请输入衰减系数 k (默认3.5，越大衰减越快): ") or "3.5")
        
        q_range = np.linspace(t_input - 0.2, t_input + 2.0, 500)
        scores_interactive = vectorized_score(q_range, t_input, k_input)
        
        plt.figure(figsize=(10, 6))
        plt.plot(q_range, scores_interactive, 'b-', linewidth=2.5)
        plt.axvline(x=t_input, color='r', linestyle='--', linewidth=2, label=f'阈值 t={t_input}')
        plt.xlabel('q_mean', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'指数衰减评分曲线 (t={t_input}, k={k_input})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 105)
        plt.legend()
        
        # 标注示例点
        test_exceeds = [0.2, 0.5, 1.0]
        for exceed in test_exceeds:
            q_point = t_input + exceed
            score_point = new_score(q_point, t_input, k_input)
            plt.plot(q_point, score_point, 'ro', markersize=6)
            plt.annotate(f'({q_point:.2f}, {score_point:.1f})', 
                        xy=(q_point, score_point), xytext=(q_point+0.05, score_point+5),
                        fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
    except ValueError:
        print("输入无效，请使用数字")

# 如果需要交互式体验，取消下面的注释
interactive_score_plot()