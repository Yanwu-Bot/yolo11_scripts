"""
生成全训练视频两两配对的差分回归数据集

生成规则：
    模板视频 = 31个训练视频中的某一个
    查询视频 = 31个训练视频中的另一个
    要求模板 != 查询

最终样本数：
    31 * 30 = 930

每个样本包含：
    ref_name:     模板视频名
    que_name:     查询视频名
    fea_diff:     角度/姿态特征逐帧差分     (T, 26)
    point_diff:   关键点逐帧差分           (T, 34)
    vector_diff:  位移/速度特征逐帧差分     (T, 向量维度)
    score_diff:   查询视频分数 - 模板视频分数
"""

import os
import sys
import json
import pickle

import numpy as np

# ==================== 请根据实际路径修改 ====================

# 特征文件目录
FEATURES_DIR = 'D:/Dataset/sprint/result/features_train'

# 人工评分文件
SCORES_JSON = 'D:/Dataset/sprint/result/video_point/scores.json'

# 输出数据集路径
OUTPUT_PATH = 'D:/Dataset/sprint/result/pair_dataset.pkl'

# AcDTW.py 所在目录
ACDTW_MODULE_PATH = 'scripts/yolo11_scripts/video_create/AcDTW.pyt'

# 是否对逐帧特征差取绝对值（分差回归必须保留方向，建议保持 False）
USE_ABS_DIFF = False

# ==========================================================


# 将 AcDTW 模块所在目录加入 Python 路径
if ACDTW_MODULE_PATH not in sys.path:
    sys.path.append(ACDTW_MODULE_PATH)

try:
    from AcDTW import acdtw
except ImportError as e:
    raise ImportError(f"无法导入 AcDTW，请检查 ACDTW_MODULE_PATH 是否正确。错误信息: {e}")


# ==================== 与 DTW_Score.py 一致的特征权重 ====================

angle_weights = [1] * 4 + [1] * 4 + [1.6] * 4 + [1.2] * 4
center_weights = [1] * 4
orientation_weight = [1.1]
feet_distance_weight = [1.2]
phase_weights = [0.9] * 4

FEATURE_WEIGHTS = np.array(
    angle_weights + center_weights + orientation_weight + feet_distance_weight + phase_weights,
    dtype=float
)

# 重点增强维度：肘/膝屈伸角
KEY_IDX = np.arange(8, 12)

# `calculate_frame_score` 中的固定参数
T_THRESH = 0.05
K_PENALTY = 4.0


def frame_dist(feat_a, feat_b):
    """
    ACDTW 使用的距离函数。

    与 `VideoScoreEvaluator.calculate_video_score` 中的 `med_dist` 保持完全一致：
        两帧差异越小，距离越接近 0；
        两帧差异越大，距离越大。

    输入：
        feat_a, feat_b: 两帧的一维特征向量
    返回：
        float 距离
    """
    a = np.asarray(feat_a, dtype=np.float32).flatten()
    b = np.asarray(feat_b, dtype=np.float32).flatten()

    if a.shape[0] != len(FEATURE_WEIGHTS):
        raise ValueError(
            f"特征维度 {a.shape[0]} 与 expected feature_weights 长度 {len(FEATURE_WEIGHTS)} 不一致。"
            f"请检查特征文件或调整权重定义。"
        )

    q = np.abs(a - b)

    # 重点维度增强：差异越大，权重被放大越多
    enhance = 1.0 + 2.0 * q[KEY_IDX]
    w = FEATURE_WEIGHTS.copy()
    w[KEY_IDX] = w[KEY_IDX] * enhance

    w = w / np.sum(w)
    q_mean = np.sum(q * w)
    exceed = q_mean - T_THRESH

    if exceed < 0:
        score = 100.0
    else:
        score = 100.0 * np.exp(-K_PENALTY * exceed)

    # 距离越小表示越相似
    return max(0.0, 100.0 - score)


# ==================== 工具函数 ====================

def load_score_map(scores_json):
    """读取 scores.json，兼容键中包含 .mp4 或不包含扩展名的情况"""
    with open(scores_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    score_map = {}
    for k, v in raw.items():
        name = os.path.splitext(str(k))[0]
        score_map[name] = float(v)
    return score_map


def load_feature(name, suffix):
    """
    加载指定特征文件。
    suffix 可以为：'features', 'normalized_points', 'vector'
    """
    path = os.path.join(FEATURES_DIR, f"{name}_{suffix}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"特征文件不存在: {path}")
    return np.load(path)


def load_points(name):
    """加载关键点坐标，如果有第三维置信度则去掉，只保留 x, y"""
    pts = load_feature(name, 'normalized_points')
    if pts.ndim == 3 and pts.shape[2] == 3:
        pts = pts[:, :, :2]
    return pts


def load_vector(name):
    """加载位移/速度特征，如果原始数据是三维则展平成二维"""
    vec = load_feature(name, 'vector')
    if vec.ndim == 3:
        vec = vec.reshape(vec.shape[0], -1)
    return vec


# ==================== 生成数据集 ====================

def main():
    # 自动生成训练视频名称
    train_videos = [f"run_{i}" for i in range(2, 44) if not (9 <= i <= 19)]

    print(f"训练视频总数: {len(train_videos)}")
    print("训练视频列表:")
    print(train_videos)

    # 读取分数
    score_map = load_score_map(SCORES_JSON)

    # 只保留同时满足“有分数 + 有全部特征文件”的视频
    valid_videos = []
    for name in train_videos:
        if name not in score_map:
            print(f"警告: {name} 不在 scores.json 中，跳过")
            continue

        has_features = os.path.exists(os.path.join(FEATURES_DIR, f"{name}_features.npy"))
        has_points = os.path.exists(os.path.join(FEATURES_DIR, f"{name}_normalized_points.npy"))
        has_vector = os.path.exists(os.path.join(FEATURES_DIR, f"{name}_vector.npy"))

        if has_features and has_points and has_vector:
            valid_videos.append(name)
        else:
            print(f"警告: {name} 缺少特征文件，跳过")

    print(f"\n实际有效训练视频数: {len(valid_videos)}")

    if len(valid_videos) < 2:
        raise ValueError("有效训练视频不足 2 个，无法生成样本对")

    # 构造全部有序样本对
    sample_pairs = []
    for ref_name in valid_videos:
        for que_name in valid_videos:
            if ref_name != que_name:
                sample_pairs.append((ref_name, que_name))

    print(f"生成样本对总数: {len(sample_pairs)}")

    # 一次性将所有特征读入内存缓存，避免重复读取文件
    data_cache = {}

    for name in valid_videos:
        try:
            data_cache[name] = {
                'feat': load_feature(name, 'features'),
                'points': load_points(name),
                'vector': load_vector(name),
            }
            print(f"加载特征: {name}")
        except Exception as e:
            raise RuntimeError(f"加载 {name} 特征失败: {e}")

    print("所有特征加载完成，开始逐对 ACDTW 对齐和差分计算...\n")

    dataset = []
    failed_count = 0

    for i, (ref_name, que_name) in enumerate(sample_pairs):
        try:
            ref_data = data_cache[ref_name]
            que_data = data_cache[que_name]

            # 与 DTW_Score.py 保持一致：
            # 第一个参数是查询视频，第二个参数是模板/参考视频
            path, _, _, _ = acdtw(
                que_data['feat'],
                ref_data['feat'],
                dist_func=frame_dist,
                window=None,
            )

            path = np.array(path, dtype=int)

            if path.ndim != 2 or path.shape[1] != 2:
                raise RuntimeError(f"ACDTW 返回的 path 维度异常: {path.shape}")

            que_idx = path[:, 0]
            ref_idx = path[:, 1]

            # 计算逐帧差分（方向：查询 - 模板）
            fea_diff = que_data['feat'][que_idx] - ref_data['feat'][ref_idx]
            point_diff = que_data['points'][que_idx] - ref_data['points'][ref_idx]
            vector_diff = que_data['vector'][que_idx] - ref_data['vector'][ref_idx]

            if USE_ABS_DIFF:
                fea_diff = np.abs(fea_diff)
                point_diff = np.abs(point_diff)
                vector_diff = np.abs(vector_diff)

            # 真实分差
            score_diff = score_map[que_name] - score_map[ref_name]

            record = {
                'ref_name': ref_name,
                'que_name': que_name,
                'fea_diff': fea_diff.astype(np.float32),
                'point_diff': point_diff.astype(np.float32),
                'vector_diff': vector_diff.astype(np.float32),
                'score_diff': float(score_diff),
                'path_len': len(fea_diff),
            }

            dataset.append(record)

            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1} / {len(sample_pairs)} 对")

        except Exception as e:
            failed_count += 1
            print(f"生成失败: {ref_name} -> {que_name}, 错误: {e}")

    # 保存数据集
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(dataset, f)

    print("\n" + "=" * 60)
    print(f"生成完成")
    print(f"成功样本数: {len(dataset)}")
    print(f"失败样本数: {failed_count}")
    print(f"保存路径: {OUTPUT_PATH}")

    if len(dataset) > 0:
        print("\n样例记录字段信息:")
        sample = dataset[0]
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")


if __name__ == '__main__':
    main()