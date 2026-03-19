import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

TEMPLATE_VIDEO = 'run_man.mp4'
TEST_VIDEO = 'run_man1.mp4'
TEMPLATE_FILE = 'run_man_features.npy'
TEST_FILE = 'run_man1_features.npy'
VIEW_FRAME = 100

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

def calculate_frame_score(test_feat, template_feat, t):
    """计算单帧得分"""
    q = np.abs(test_feat - template_feat)
    q_mean = np.mean(q)

    if q_mean <= t:
        score = 100.0
    elif q_mean - t <= 0.3:
        exceed = q_mean - t
        score = 100 * np.exp(-2.5 * exceed)
    elif 0.3 < q_mean - t <= 1:
        exceed = q_mean - t
        score = 100 * np.exp(-5 * exceed)
    else:
        score = 0.0
    
    return score

def calculate_video_score(test_features, template_features):
    """计算整个视频的得分"""
    distance, path = fastdtw(test_features, template_features, dist=euclidean)
    path = np.array(path)
    
    print(f"DTW对齐完成: 路径长度 = {len(path)}")
    
    frame_scores = []
    q_mean_list = []
    
    for test_idx, template_idx in path:
        test_frame = test_features[test_idx]
        template_frame = template_features[template_idx]
        
        q = np.abs(test_frame - template_frame)
        q_mean = np.mean(q)
        q_mean_list.append(q_mean)
        
        score = calculate_frame_score(test_frame, template_frame, t=0.15)
        frame_scores.append(score)
    
    final_score = np.mean(frame_scores)
    return final_score, frame_scores, path, q_mean_list

def score_video():
    """主函数：对测试视频评分"""
    features_dir = 'result/features'
    template_path = os.path.join(features_dir, TEMPLATE_FILE)
    test_path = os.path.join(features_dir, TEST_FILE)
    
    if not os.path.exists(template_path) or not os.path.exists(test_path):
        print("错误: 找不到特征文件")
        return
    
    print("="*60)
    print("视频动作质量评分")
    print("="*60)
    
    template_features = np.load(template_path)
    test_features = np.load(test_path)
    
    print(f"\n模板视频帧数: {template_features.shape[0]}")
    print(f"测试视频帧数: {test_features.shape[0]}")
    
    # 归一化特征
    print("\n归一化特征...")
    norm_template = normalize_features(template_features)
    norm_test = normalize_features(test_features)
    
    score, frame_scores, path, q_mean_list = calculate_video_score(norm_test, norm_template)
    
    print(f"\n最终得分: {score:.2f}")
    
    return score, frame_scores, path, q_mean_list, norm_test, norm_template

def plot_qmean_over_time(q_mean_list):
    """
    只绘制q_mean随时间变化的图
    """
    plt.figure(figsize=(14, 6))
    
    x = np.arange(len(q_mean_list))
    
    # 绘制q_mean曲线
    plt.plot(x, q_mean_list, 'b-', linewidth=2, label='q_mean', alpha=0.8)
    
    # 添加阈值线
    plt.axhline(y=0.15, color='r', linestyle='--', linewidth=2, label='阈值 t=0.15')
    
    # 添加均值线
    mean_q = np.mean(q_mean_list)
    plt.axhline(y=mean_q, color='g', linestyle='--', linewidth=2, 
                label=f'均值: {mean_q:.3f}')
    
    # 填充区域
    plt.fill_between(x, 0, q_mean_list, where=(np.array(q_mean_list) <= 0.15), 
                     color='green', alpha=0.2, label='≤阈值 (合格)')
    plt.fill_between(x, 0, q_mean_list, where=(np.array(q_mean_list) > 0.15), 
                     color='red', alpha=0.2, label='>阈值 (差异较大)')
    
    plt.xlabel('对齐对序号', fontsize=12)
    plt.ylabel('q_mean值 (帧间差异)', fontsize=12)
    plt.title('q_mean随时间变化图', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*40)
    print("q_mean统计")
    print("="*40)
    print(f"均值: {np.mean(q_mean_list):.4f}")
    print(f"标准差: {np.std(q_mean_list):.4f}")
    print(f"中位数: {np.median(q_mean_list):.4f}")
    print(f"最小值: {min(q_mean_list):.4f}")
    print(f"最大值: {max(q_mean_list):.4f}")
    
    exceed_ratio = np.mean(np.array(q_mean_list) > 0.15) * 100
    print(f"\n超过阈值(0.15)的比例: {exceed_ratio:.2f}%")

def visualize_alignment_path(path, test_len, template_len, frame_scores=None):
    """可视化DTW对齐路径"""
    path = np.array(path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：对齐路径
    path_test = path[:, 0]
    path_template = path[:, 1]
    
    cost_matrix = np.full((template_len, test_len), np.nan)
    if frame_scores is not None:
        for i, (test_idx, template_idx) in enumerate(path):
            cost_matrix[template_idx, test_idx] = frame_scores[i]
    
    im = ax1.imshow(cost_matrix, cmap='RdYlGn', aspect='auto', origin='lower',
                    extent=[0, test_len, 0, template_len], vmin=0, vmax=100, alpha=0.8)
    
    ax1.plot(path_test, path_template, 'w-', linewidth=2.5, alpha=0.9, label='DTW对齐路径')
    ax1.plot(path_test, path_template, 'k.', markersize=2, alpha=0.5)
    
    plt.colorbar(im, ax=ax1, label='帧得分')
    ax1.set_xlabel('测试视频帧索引')
    ax1.set_ylabel('模板视频帧索引')
    ax1.set_title('DTW对齐路径')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # 右图：得分分布
    if frame_scores is not None:
        x = np.arange(len(frame_scores))
        ax2.plot(x, frame_scores, 'b-', linewidth=2, label='帧得分')
        ax2.fill_between(x, frame_scores, 0, alpha=0.2, color='blue')
        
        mean_score = np.mean(frame_scores)
        ax2.axhline(y=mean_score, color='r', linestyle='--', linewidth=1.5, 
                    label=f'平均分: {mean_score:.2f}')
        
        ax2.set_xlabel('对齐对序号')
        ax2.set_ylabel('帧得分')
        ax2.set_title('帧得分分布')
        ax2.set_ylim([0, 105])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'DTW对齐分析 (测试:{test_len}帧, 模板:{template_len}帧)')
    plt.tight_layout()
    plt.show()

def visualize_aligned_frames(test_video_path, template_video_path, path, pair_index, frame_scores=None):
    """可视化指定对齐对的两帧画面"""
    test_idx, template_idx = path[pair_index]
    
    cap = cv2.VideoCapture(test_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_idx)
    ret, test_frame = cap.read()
    cap.release()
    
    cap = cv2.VideoCapture(template_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, template_idx)
    ret, template_frame = cap.read()
    cap.release()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    template_frame_rgb = cv2.cvtColor(template_frame, cv2.COLOR_BGR2RGB)
    
    ax1.imshow(test_frame_rgb)
    score_text = f'\n得分: {frame_scores[pair_index]:.3f}' if frame_scores else ''
    ax1.set_title(f'测试视频 - 第{test_idx}帧{score_text}')
    ax1.axis('off')
    
    ax2.imshow(template_frame_rgb)
    ax2.set_title(f'模板视频 - 第{template_idx}帧')
    ax2.axis('off')
    
    plt.suptitle(f'DTW对齐对 #{pair_index}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    score, frame_scores, path, q_mean_list, norm_test, norm_template = score_video()
    
    # 只绘制q_mean随时间变化的图
    print("\n" + "="*60)
    print("q_mean随时间变化")
    print("="*60)
    plot_qmean_over_time(q_mean_list)
    
    # 可选：保留原有的路径可视化
    print("\n" + "="*60)
    print("DTW对齐路径可视化")
    print("="*60)
    visualize_alignment_path(path, len(norm_test), len(norm_template), frame_scores)
    
    # 帧可视化部分
    test_video_path = os.path.join('video_origin/data_video/use', TEST_VIDEO)
    template_video_path = os.path.join('video_origin/data_video/use', TEMPLATE_VIDEO)
    
    if os.path.exists(test_video_path) and os.path.exists(template_video_path):
        if VIEW_FRAME < len(frame_scores):
            visualize_aligned_frames(test_video_path, template_video_path, path, VIEW_FRAME, frame_scores)
    
    print(f"\n最终得分: {score:.2f}")