import cv2
import numpy as np
import matplotlib.pyplot as plt
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

def calculate_frame_score(test_feat, template_feat):
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
    score = q_mean
    
    # # 公式(13): 根据q值计算得分
    # if q_mean <= t:
    #     # q小于阈值，得满分
    #     score = 100.0
    # elif t < q_mean <= 1:
    #     # q在阈值和1之间，线性扣分
    #     score = 100.0 * (1 - q_mean + t)
    # else:
    #     # q大于1，得0分
    #     score = 0.0
    
    return score

def calculate_video_score(test_features, template_features):
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
        score = calculate_frame_score(test_frame, template_frame)
        frame_scores.append(score)
    
    # 3. 公式(14): 取平均作为最终得分
    final_score = np.mean(frame_scores)
    print(path)
    
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
    
    results = {}
    
    print("\n" + "-"*40)
    print("评分结果 (t为阈值系数):")
    print("-"*40)
    
    score, frame_scores, path = calculate_video_score(norm_test, norm_template)
    return score, frame_scores, path  # 修改这里，返回path


def visualize_aligned_frames(test_video_path, template_video_path, path, pair_index, frame_scores=None):
    """
    可视化指定对齐对的两帧画面
    
    Args:
        test_video_path: 测试视频文件路径
        template_video_path: 模板视频文件路径
        path: DTW路径
        pair_index: 要查看的第几个匹配对
        frame_scores: 每对得分（可选）
    """
    test_idx, template_idx = path[pair_index]
    
    # 读取测试视频的指定帧
    cap = cv2.VideoCapture(test_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_idx)
    ret, test_frame = cap.read()
    cap.release()
    
    # 读取模板视频的指定帧
    cap = cv2.VideoCapture(template_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, template_idx)
    ret, template_frame = cap.read()
    cap.release()
    
    # 显示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 转换BGR到RGB
    test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    template_frame_rgb = cv2.cvtColor(template_frame, cv2.COLOR_BGR2RGB)
    
    ax1.imshow(test_frame_rgb)
    score_text = f'\n得分: {frame_scores[pair_index]:.3f}' if frame_scores else ''
    ax1.set_title(f'测试视频 - 第{test_idx}帧{score_text}', fontsize=12)
    ax1.axis('off')
    
    ax2.imshow(template_frame_rgb)
    ax2.set_title(f'模板视频 - 第{template_idx}帧', fontsize=12)
    ax2.axis('off')
    
    plt.suptitle(f'DTW对齐对 #{pair_index} (共{len(path)}对)', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    score, frame_scores, path = score_video()
    # 添加可视化部分
    print("\n" + "="*60)
    print("可视化DTW对齐帧")
    print("="*60)
    
    # 设置视频文件路径（根据你的实际文件位置修改）
    test_video = 'run_man1.mp4'  # 你的测试视频文件名
    template_video = 'run_man.mp4'  # 你的模板视频文件名
    video_dir = 'video_origin/data_video/use'  # 视频文件所在目录，如果是当前目录就设为''
    
    # 构建完整路径
    test_video_path = os.path.join(video_dir, test_video)
    template_video_path = os.path.join(video_dir, template_video)
    
    # 检查视频文件是否存在
    if os.path.exists(test_video_path) and os.path.exists(template_video_path):
        # 自定义查看特定的匹配对
        pair_to_view = 99  # 你可以修改这个数字来查看不同的匹配对
        if pair_to_view < len(frame_scores):
            visualize_aligned_frames(test_video_path, template_video_path, path, pair_to_view, frame_scores)
            print(f"已显示第 {pair_to_view} 对匹配对 (测试帧: {path[pair_to_view][0]}, 模板帧: {path[pair_to_view][1]}, 得分: {frame_scores[pair_to_view]:.3f})")
        else:
            print(f"错误: 匹配对索引 {pair_to_view} 超出范围 (0-{len(frame_scores)-1})")
    else:
        print("警告: 找不到视频文件，请检查视频文件路径")
        if not os.path.exists(test_video_path):
            print(f"  测试视频不存在: {test_video_path}")
        if not os.path.exists(template_video_path):
            print(f"  模板视频不存在: {template_video_path}")