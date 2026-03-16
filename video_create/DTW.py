import numpy as np
import os
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from matplotlib import rcParams #字体
rcParams['font.family'] = 'SimHei'

TEMPLE_FILE = 'run_man_features.npy'
TEST_FILE = 'run_man1_features.npy'

def normalize_features(features: np.ndarray) -> np.ndarray:
    """归一化特征到[0,1]范围"""
    normalized = np.zeros_like(features)
    for i in range(features.shape[1]):
        col = features[:, i]
        min_val = np.min(col)
        max_val = np.max(col)
        if max_val > min_val:
            normalized[:, i] = (col - min_val) / (max_val - min_val)
        else:
            normalized[:, i] = col
    return normalized

def calculate_frame_score(test_feat, template_feat, t=0.15):
    """
    计算单帧得分 - 参考文献公式(12)和(13)
    
    Args:
        test_feat: 测试视频当前帧的特征 (32维)
        template_feat: 模板视频当前帧的特征 (32维)
        t: 阈值系数 (先不考虑异常检测，用默认值0.15)
    
    Returns:
        score: 该帧的得分 (0-100)
    """
    # 公式(12): q = |F_test - F_temp| / F_temp
    # 避免除零
    denominator = np.abs(template_feat) + 1e-6
    q = np.abs(test_feat - template_feat) / denominator
    
    # 取平均值作为该帧的差异度
    q_mean = np.mean(q)
    
    # 公式(13): 根据q值计算得分
    if q_mean <= t:
        # q小于阈值，得满分
        score = 100.0
    elif t < q_mean <= 1:
        # q在阈值和1之间，线性扣分
        score = 100.0 * (1 - q_mean + t)
    else:
        # q大于1，得0分
        score = 0.0
    
    return score

def calculate_video_score(test_features, template_features, t=0.15):
    """
    计算整个视频的得分 - 参考文献公式(14)
    
    Args:
        test_features: 测试视频特征 (n_frames, 32)
        template_features: 模板视频特征 (m_frames, 32)
        t: 阈值系数
    
    Returns:
        score: 视频得分 (0-100)
        frame_scores: 每帧的得分
        path: DTW对齐路径
    """
    # 1. 首先用DTW对齐两个序列
    distance, path = fastdtw(test_features, template_features, dist=euclidean)
    path = np.array(path)
    
    print(f"DTW对齐完成: 路径长度 = {len(path)}")
    
    # 2. 沿着对齐路径计算每对帧的得分
    frame_scores = []
    
    for test_idx, template_idx in path:
        test_frame = test_features[test_idx]
        template_frame = template_features[template_idx]
        
        # 计算该对齐对的得分
        score = calculate_frame_score(test_frame, template_frame, t)
        frame_scores.append(score)
    
    # 3. 公式(14): 取平均作为最终得分
    final_score = np.mean(frame_scores)
    
    return final_score, frame_scores, path

def score_video():
    """主函数：对测试视频评分"""
    
    # 设置路径
    features_dir = 'result/features'
    template_file = TEMPLE_FILE     # 标准模板
    test_file = TEST_FILE        # 测试视频
    
    template_path = os.path.join(features_dir, template_file)
    test_path = os.path.join(features_dir, test_file)
    
    # 检查文件
    if not os.path.exists(template_path):
        print(f"错误: 找不到模板文件 {template_path}")
        return
    if not os.path.exists(test_path):
        print(f"错误: 找不到测试文件 {test_path}")
        return
    
    print("="*60)
    print("视频动作质量评分")
    print("="*60)
    
    # 加载特征
    template_features = np.load(template_path)
    test_features = np.load(test_path)
    
    template_name = template_file.replace('_features.npy', '')
    test_name = test_file.replace('_features.npy', '')
    
    print(f"\n标准模板: {template_name}")
    print(f"  帧数: {template_features.shape[0]}")
    print(f"  特征范围: [{template_features.min():.2f}, {template_features.max():.2f}]")
    
    print(f"\n测试视频: {test_name}")
    print(f"  帧数: {test_features.shape[0]}")
    print(f"  特征范围: [{test_features.min():.2f}, {test_features.max():.2f}]")
    
    # 归一化特征
    print("\n归一化特征...")
    norm_template = normalize_features(template_features)
    norm_test = normalize_features(test_features)
    
    # 测试不同的t值
    t_values = [0.1, 0.15, 0.2, 0.25, 0.3]
    results = {}
    
    print("\n" + "-"*40)
    print("评分结果 (t为阈值系数):")
    print("-"*40)
    
    for t in t_values:
        score, frame_scores, path = calculate_video_score(norm_test, norm_template, t)
        results[t] = {
            'score': score,
            'frame_scores': frame_scores,
            'path': path
        }
        print(f"t = {t:.2f}: 得分 = {score:.2f}分")
    
    # 使用默认t=0.15进行详细分析
    t_default = 0.15
    score, frame_scores, path = results[t_default]['score'], results[t_default]['frame_scores'], results[t_default]['path']
    
    print(f"\n" + "="*60)
    print(f"详细分析 (t = {t_default})")
    print("="*60)
    
    print(f"\n最终得分: {score:.2f}分")
    print(f"对齐路径长度: {len(path)}")
    print(f"测试视频帧数: {len(norm_test)}")
    print(f"模板视频帧数: {len(norm_template)}")
    
    # 得分统计
    print(f"\n帧得分统计:")
    print(f"  最高分: {max(frame_scores):.2f}分")
    print(f"  最低分: {min(frame_scores):.2f}分")
    print(f"  平均分: {np.mean(frame_scores):.2f}分")
    print(f"  中位数: {np.median(frame_scores):.2f}分")
    print(f"  标准差: {np.std(frame_scores):.2f}")
    
    # 得分分布
    excellent = sum(1 for s in frame_scores if s >= 90)
    good = sum(1 for s in frame_scores if 80 <= s < 90)
    fair = sum(1 for s in frame_scores if 60 <= s < 80)
    poor = sum(1 for s in frame_scores if s < 60)
    
    print(f"\n得分分布:")
    print(f"  优秀 (90-100分): {excellent} 帧 ({excellent/len(frame_scores)*100:.1f}%)")
    print(f"  良好 (80-89分): {good} 帧 ({good/len(frame_scores)*100:.1f}%)")
    print(f"  及格 (60-79分): {fair} 帧 ({fair/len(frame_scores)*100:.1f}%)")
    print(f"  不及格 (0-59分): {poor} 帧 ({poor/len(frame_scores)*100:.1f}%)")
    
    # 可视化
    print("\n生成可视化图表...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. 对齐路径
    ax = axes[0]
    ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('测试视频帧索引')
    ax.set_ylabel('模板视频帧索引')
    ax.set_title(f'DTW对齐路径 (总得分: {score:.2f})')
    ax.grid(True, alpha=0.3)
    
    # 2. 帧得分曲线
    ax = axes[1]
    ax.plot(frame_scores, 'g-', linewidth=1.5)
    ax.axhline(y=90, color='gold', linestyle='--', alpha=0.7, label='优秀线 (90分)')
    ax.axhline(y=80, color='lime', linestyle='--', alpha=0.7, label='良好线 (80分)')
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='及格线 (60分)')
    ax.axhline(y=score, color='red', linestyle='-', linewidth=2, label=f'平均分 ({score:.1f})')
    
    ax.set_xlabel('对齐路径索引')
    ax.set_ylabel('得分')
    ax.set_title(f'逐帧得分 (t={t_default})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # 3. 得分直方图
    ax = axes[2]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ax.hist(frame_scores, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(x=score, color='red', linestyle='-', linewidth=2, label=f'平均分 ({score:.1f})')
    
    ax.set_xlabel('得分区间')
    ax.set_ylabel('帧数')
    ax.set_title('得分分布直方图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存结果
    output_dir = 'result/scores'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存可视化
    vis_path = f"{output_dir}/{test_name}_vs_{template_name}_score.png"
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存: {vis_path}")
    
    # 保存得分数据
    scores_data = {
        'test_name': test_name,
        'template_name': template_name,
        't_default': t_default,
        'final_score': float(score),
        'frame_scores': [float(s) for s in frame_scores],
        'score_stats': {
            'mean': float(np.mean(frame_scores)),
            'median': float(np.median(frame_scores)),
            'std': float(np.std(frame_scores)),
            'min': float(min(frame_scores)),
            'max': float(max(frame_scores))
        },
        'distribution': {
            'excellent': excellent,
            'good': good,
            'fair': fair,
            'poor': poor
        },
        't_values': {str(t): float(results[t]['score']) for t in t_values}
    }
    
    import json
    with open(f"{output_dir}/{test_name}_score.json", 'w', encoding='utf-8') as f:
        json.dump(scores_data, f, indent=2, ensure_ascii=False)
    print(f"得分数据已保存: {output_dir}/{test_name}_score.json")
    
    # 生成报告
    report_path = f"{output_dir}/{test_name}_score_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("动作质量评分报告\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"测试视频: {test_name}\n")
        f.write(f"标准模板: {template_name}\n")
        f.write(f"阈值系数 t = {t_default}\n\n")
        
        f.write(f"最终得分: {score:.2f} 分\n\n")
        
        f.write("不同阈值下的得分:\n")
        for t, data in results.items():
            f.write(f"  t = {t:.2f}: {data['score']:.2f} 分\n")
        
        f.write("\n帧得分统计:\n")
        f.write(f"  平均分: {np.mean(frame_scores):.2f}\n")
        f.write(f"  中位数: {np.median(frame_scores):.2f}\n")
        f.write(f"  标准差: {np.std(frame_scores):.2f}\n")
        f.write(f"  最高分: {max(frame_scores):.2f}\n")
        f.write(f"  最低分: {min(frame_scores):.2f}\n\n")
        
        f.write("得分分布:\n")
        f.write(f"  优秀 (90-100): {excellent} 帧 ({excellent/len(frame_scores)*100:.1f}%)\n")
        f.write(f"  良好 (80-89): {good} 帧 ({good/len(frame_scores)*100:.1f}%)\n")
        f.write(f"  及格 (60-79): {fair} 帧 ({fair/len(frame_scores)*100:.1f}%)\n")
        f.write(f"  不及格 (0-59): {poor} 帧 ({poor/len(frame_scores)*100:.1f}%)\n")
    
    print(f"评分报告已保存: {report_path}")
    
    print("\n" + "="*60)
    print(f"✅ 评分完成!")
    print(f"   测试视频 '{test_name}' 得分: {score:.2f} 分 (t={t_default})")
    print("="*60)
    
    return score, frame_scores

if __name__ == '__main__':
    score, frame_scores = score_video()