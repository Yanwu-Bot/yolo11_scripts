#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 fastdtw 生成全训练视频两两配对的差分回归数据集

生成规则：
    模板视频 = 31 个训练视频中的某一个
    查询视频 = 31 个训练视频中的另一个
    要求模板 != 查询

最终样本数：
    31 * 30 = 930

每个样本包含：
    ref_name:     模板视频名
    que_name:     查询视频名
    fea_diff:     角度/姿态特征逐帧差分     (T, 特征维度)
    point_diff:   关键点逐帧差分           (T, 17*2 或 17*3)
    vector_diff:  位移/速度特征逐帧差分     (T, 向量维度)
    score_diff:   查询视频分数 - 模板视频分数
"""

import os
import json
import pickle

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ==================== 路径配置 ====================

FEATURES_DIR = 'D:/Dataset/sprint/result/features_train'
SCORES_JSON  = 'D:/Dataset/sprint/result/video_point/scores.json'
OUTPUT_PATH  = 'D:/Dataset/sprint/result/diff_dataset/pair_dataset.pkl'

# 是否对差分取绝对值。分差回归建议保留方向，设为 False
USE_ABS_DIFF = False

# ==================================================


def load_score_map(scores_json):
    """读取 scores.json，兼容键带不带扩展名"""
    with open(scores_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    score_map = {}
    for k, v in raw.items():
        name = os.path.splitext(str(k))[0]
        score_map[name] = float(v)
    return score_map


def load_feature(name, suffix):
    """
    加载指定特征文件
    suffix: 'features', 'normalized_points', 'vector'
    """
    path = os.path.join(FEATURES_DIR, f"{name}_{suffix}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"特征文件不存在: {path}")
    return np.load(path)


def load_points(name):
    """加载关键点，去掉可能的置信度第三维"""
    pts = load_feature(name, 'normalized_points')
    if pts.ndim == 3 and pts.shape[2] == 3:
        pts = pts[:, :, :2]
    return pts


def load_vector(name):
    """加载向量特征，若为三维则展开为二维"""
    vec = load_feature(name, 'vector')
    if vec.ndim == 3:
        vec = vec.reshape(vec.shape[0], -1)
    return vec


def compute_diff_with_fastdtw(ref_data, que_data):
    """
    基于 features 使用 fastdtw 对齐，然后计算三个维度逐帧差分
    返回: (fea_diff, point_diff, vector_diff)
    """
    ref_feat = ref_data['feat']
    que_feat = que_data['feat']

    # fastdtw 第一个参数是模板序列，第二个是查询序列
    # path 中每个元素为 (ref_idx, que_idx)
    _, path = fastdtw(ref_feat, que_feat, dist=euclidean)

    # 转成数组便于索引
    path = np.array(path, dtype=int)
    ref_idx = path[:, 0]
    que_idx = path[:, 1]

    # 逐帧差分：查询 - 模板
    fea_diff = que_data['feat'][que_idx] - ref_data['feat'][ref_idx]
    point_diff = que_data['points'][que_idx] - ref_data['points'][ref_idx]
    vector_diff = que_data['vector'][que_idx] - ref_data['vector'][ref_idx]

    if USE_ABS_DIFF:
        fea_diff = np.abs(fea_diff)
        point_diff = np.abs(point_diff)
        vector_diff = np.abs(vector_diff)

    return fea_diff.astype(np.float32), \
           point_diff.astype(np.float32), \
           vector_diff.astype(np.float32)


def main():
    # ----- 1. 自动确定候选视频名：读取 features_train 下所有 *_features.npy -----
    candidate_videos = []
    for fname in os.listdir(FEATURES_DIR):
        if fname.endswith('_features.npy'):
            # 去掉 "_features.npy" 后缀得到视频名，例如 "run_2"
            name = fname[:-len('_features.npy')]
            candidate_videos.append(name)

    # 去重并排序（可选）
    candidate_videos = sorted(set(candidate_videos))

    print(f"自动发现候选视频数: {len(candidate_videos)}")
    print(candidate_videos)

    # ----- 2. 读取人工评分 -----
    score_map = load_score_map(SCORES_JSON)

    # ----- 3. 只保留“有分数 + 有全部特征文件”的视频 -----
    valid_videos = []
    for name in candidate_videos:
        if name not in score_map:
            print(f"警告: {name} 不在 scores.json 中，跳过")
            continue

        has_feat = os.path.exists(os.path.join(FEATURES_DIR, f"{name}_features.npy"))
        has_pts = os.path.exists(os.path.join(FEATURES_DIR, f"{name}_normalized_points.npy"))
        has_vec = os.path.exists(os.path.join(FEATURES_DIR, f"{name}_vector.npy"))

        if has_feat and has_pts and has_vec:
            valid_videos.append(name)
        else:
            print(f"警告: {name} 缺少特征文件，跳过")

    print(f"\n有效训练视频数: {len(valid_videos)}")

    if len(valid_videos) < 2:
        raise ValueError("有效视频不足 2 个")

    # ----- 4. 构造所有有序样本对 -----
    sample_pairs = []
    for ref in valid_videos:
        for que in valid_videos:
            if ref != que:
                sample_pairs.append((ref, que))

    print(f"生成样本对总数: {len(sample_pairs)}")

    # ----- 5. 缓存所有特征到内存，避免反复读盘 -----
    data_cache = {}
    for name in valid_videos:
        try:
            data_cache[name] = {
                'feat': load_feature(name, 'features'),
                'points': load_points(name),
                'vector': load_vector(name),
            }
            print(f"成功加载: {name}")
        except Exception as e:
            raise RuntimeError(f"加载 {name} 失败: {e}")

    print("所有特征加载完成，开始计算 fastdtw 对齐和差分...\n")

    dataset = []
    failed = 0

    for i, (ref_name, que_name) in enumerate(sample_pairs):
        try:
            ref_data = data_cache[ref_name]
            que_data = data_cache[que_name]

            fea_diff, point_diff, vector_diff = compute_diff_with_fastdtw(ref_data, que_data)

            score_diff = score_map[que_name] - score_map[ref_name]

            record = {
                'ref_name': ref_name,
                'que_name': que_name,
                'fea_diff': fea_diff,
                'point_diff': point_diff,
                'vector_diff': vector_diff,
                'score_diff': float(score_diff),
                'path_len': len(fea_diff),
            }

            dataset.append(record)

            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1} / {len(sample_pairs)}")
        except Exception as e:
            failed += 1
            print(f"生成失败 {ref_name} -> {que_name}: {e}")

    # 保存 pkl
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(dataset, f)

    print("\n" + "=" * 60)
    print(f"生成完成！")
    print(f"成功样本数: {len(dataset)}")
    print(f"失败样本数: {failed}")
    print(f"保存路径: {OUTPUT_PATH}")

    # 展示一个样例的结构
    if len(dataset) > 0:
        sample = dataset[0]
        print("\n样例字段:")
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")

if __name__ == '__main__':
    main()