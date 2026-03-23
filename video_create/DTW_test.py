import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

TEMPLATE_VIDEO = 'run_man.mp4'
TEST_VIDEO = 'run_woman.mp4'
TEMPLATE_FILE_FEATURES = 'run_man_features.npy'
TEST_FILE_FEATURES = 'run_woman_features.npy'
TEMPLATE_FILE_POINT = 'run_man_point.npy'      # 新增：关键点文件
TEST_FILE_POINT = 'run_woman_point.npy'        # 新增：关键点文件
VIEW_FRAME = 91

weight ={"fea":0.7,                            #评分权重
        "point":0.3}

def normalize_features(features: np.ndarray) -> np.ndarray:           #特征归一化
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

# ============= 新增：关键点归一化函数 =============
def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    对关键点进行以人体为中心的归一化
    Args:
        keypoints: 形状为 (帧数, 17, 2) 的关键点数组
    Returns:
        归一化后的关键点数组，形状相同
    """
    normalized = np.zeros_like(keypoints, dtype=np.float32)
    for i in range(keypoints.shape[0]):
        frame_kps = keypoints[i]
        valid_mask = ~((frame_kps[:, 0] == 0) & (frame_kps[:, 1] == 0))
        if np.sum(valid_mask) < 4:
            normalized[i] = frame_kps
            continue
        
        left_hip = frame_kps[11]
        right_hip = frame_kps[12]
        
        if (left_hip[0] == 0 and left_hip[1] == 0) or (right_hip[0] == 0 and right_hip[1] == 0):
            reference = frame_kps[1]
            if reference[0] == 0 and reference[1] == 0:
                normalized[i] = frame_kps
                continue
        else:
            reference = (left_hip + right_hip) / 2
        
        left_shoulder = frame_kps[5]
        right_shoulder = frame_kps[6]
        if (left_shoulder[0] != 0 or left_shoulder[1] != 0) and (right_shoulder[0] != 0 or right_shoulder[1] != 0):
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            scale = shoulder_width
        else:
            neck = frame_kps[1]
            if neck[0] != 0 or neck[1] != 0:
                scale = np.linalg.norm(neck - reference)
            else:
                scale = 1.0
        
        if scale < 1e-6:
            scale = 1.0
        
        for j in range(17):
            if valid_mask[j]:
                normalized[i, j, 0] = (frame_kps[j, 0] - reference[0]) / scale
                normalized[i, j, 1] = (frame_kps[j, 1] - reference[1]) / scale
            else:
                normalized[i, j] = [0, 0]
    
    return normalized
# ============= 新增结束 =============

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

# ============= 新增：关键点评分函数 =============
def calculate_keypoint_score(test_points, template_points):
    """计算关键点的DTW评分"""
    print("\n" + "-"*40)
    print("关键点评分")
    print("-"*40)
    # 归一化关键点
    test_norm = normalize_keypoints(test_points)
    template_norm = normalize_keypoints(template_points)
    # 重塑为适合DTW的形状 (帧数, 34)
    test_flat = test_norm.reshape(test_norm.shape[0], -1)
    template_flat = template_norm.reshape(template_norm.shape[0], -1)
    # 计算DTW距离和路径
    distance, path = fastdtw(test_flat, template_flat, dist=euclidean)
    path = np.array(path)

    print(f"关键点DTW距离: {distance:.4f}")
    print(f"对齐路径长度: {len(path)}")
    
    # 计算每对对齐帧的得分和距离
    frame_scores = []
    frame_distances = []
    for test_idx, template_idx in path:
        test_frame = test_flat[test_idx]
        template_frame = template_flat[template_idx]
        # 计算欧氏距离
        dist = np.linalg.norm(test_frame - template_frame)
        frame_distances.append(dist)
        # 将距离转换为得分
        if dist <= 35.0:
            score = 100 * (1-dist/100)
        else:
            score = 0.0
        frame_scores.append(score)
    
    final_score = np.mean(frame_scores)
    return final_score, frame_scores, frame_distances, path
# ============= 新增结束 =============

# ============= 新增：dist可视化函数 =============
def plot_dist_over_time(dist_list):
    """
    绘制dist随时间变化的图
    """
    plt.figure(figsize=(14, 6))
    
    x = np.arange(len(dist_list))
    
    # 绘制dist曲线
    plt.plot(x, dist_list, 'r-', linewidth=2, label='关键点距离', alpha=0.8)
    
    # 添加均值线
    mean_dist = np.mean(dist_list)
    plt.axhline(y=mean_dist, color='g', linestyle='--', linewidth=2, 
                label=f'均值: {mean_dist:.3f}')
    
    # 添加阈值线（可选）
    plt.axhline(y=3.0, color='orange', linestyle=':', linewidth=1.5, 
                label='参考阈值: 3.0')
    
    plt.xlabel('对齐对序号', fontsize=12)
    plt.ylabel('欧氏距离', fontsize=12)
    plt.title('关键点距离随时间变化图', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*40)
    print("关键点距离统计")
    print("="*40)
    print(f"均值: {np.mean(dist_list):.4f}")
    print(f"标准差: {np.std(dist_list):.4f}")
    print(f"中位数: {np.median(dist_list):.4f}")
    print(f"最小值: {min(dist_list):.4f}")
    print(f"最大值: {max(dist_list):.4f}")
# ============= 新增结束 =============

def score_video():
    """主函数：对测试视频评分"""
    features_dir = 'result/features'
    template_feat_path = os.path.join(features_dir, TEMPLATE_FILE_FEATURES)
    test_feat_path = os.path.join(features_dir, TEST_FILE_FEATURES)
    template_point_path = os.path.join(features_dir, TEMPLATE_FILE_POINT)
    test_point_path = os.path.join(features_dir, TEST_FILE_POINT)
    
    if not os.path.exists(template_feat_path) or not os.path.exists(test_feat_path):
        print("错误: 找不到特征文件")
        return
    print("="*60)
    print("视频动作质量评分")
    print("="*60)
    template_features = np.load(template_feat_path)
    test_features = np.load(test_feat_path)
    
    print(f"\n模板视频帧数: {template_features.shape[0]}")
    print(f"测试视频帧数: {test_features.shape[0]}")
    
    # 归一化特征
    norm_template = normalize_features(template_features)
    norm_test = normalize_features(test_features)
    
    # 特征评分
    feat_score, frame_scores, path, q_mean_list = calculate_video_score(norm_test, norm_template)
    
    print(f"\n特征得分: {feat_score:.2f}")
    
    # ============= 新增：关键点评分 =============
    point_score = None
    point_distances = None
    if os.path.exists(template_point_path) and os.path.exists(test_point_path):
        print("\n" + "="*60)
        print("开始计算关键点评分")
        print("="*60)
        
        template_points = np.load(template_point_path)
        test_points = np.load(test_point_path)
        
        print(f"模板关键点形状: {template_points.shape}")
        print(f"测试关键点形状: {test_points.shape}")
        
        # 确保是 (帧数, 17, 2) 格式
        if template_points.shape[2] == 3:
            template_points = template_points[:, :, :2]
        if test_points.shape[2] == 3:
            test_points = test_points[:, :, :2]
        
        point_score, point_frame_scores, point_distances, point_path = calculate_keypoint_score(test_points, template_points)
        print(f"\n关键点得分: {point_score:.2f}")
    else:
        print("\n警告: 找不到关键点文件，跳过关键点评分")
    # ============= 新增结束 =============
    
    return feat_score, point_score, frame_scores, path, q_mean_list, norm_test, norm_template, point_distances

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
    # 运行评分，获取特征得分和关键点得分
    feat_score, point_score, frame_scores, path, q_mean_list, norm_test, norm_template, point_distances = score_video()
    
    # 原有可视化代码保持不变
    print("\n" + "="*60)
    print("q_mean随时间变化")
    print("="*60)
    plot_qmean_over_time(q_mean_list)
    
    # ============= 新增：dist可视化 =============
    if point_distances is not None:
        print("\n" + "="*60)
        print("关键点距离随时间变化")
        print("="*60)
        plot_dist_over_time(point_distances)
    # ============= 新增结束 =============
    
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
    
    # 最后打印两个得分
    print(f"\n{'='*60}")
    print(f"最终得分:")
    print(f"特征得分: {feat_score:.2f}")
    if point_score is not None:
        print(f"关键点得分: {point_score:.2f}")
    print("总分数：")
    combined_score = feat_score * weight['fea'] + point_score * weight['point']
    print(f"{combined_score:.2f}")
    print(f"{'='*60}")