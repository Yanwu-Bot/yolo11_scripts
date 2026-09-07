#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的 Transformer 模型，通过 run_4/5/6 作为模板，自动预测 features_val 目录下所有视频的绝对分数
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr

# ==================== 路径配置 ====================
TEMPLATE_FEATURES_DIR = 'D:/Dataset/sprint/result/features'      # 模板特征
TEST_FEATURES_DIR     = 'D:/Dataset/sprint/result/features_val'  # 测试特征
SCORES_JSON  = 'D:/Dataset/sprint/result/video_point/scores.json'
MODEL_PATH   = 'D:/Dataset/sprint/result/models/transformer_diff_reg.pth'

# 固定模板
TEMPLATES = ['run_4', 'run_43', 'run_5', 'run_6', 'run_7']

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ==================================================


# ==================== 特征加载与对齐（与训练时完全一致） ====================
def load_feature(name, suffix, features_dir):
    """从指定目录加载特征文件"""
    path = os.path.join(features_dir, f"{name}_{suffix}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"特征文件不存在: {path}")
    return np.load(path)


def load_points(name, features_dir):
    """加载关键点，去掉置信度第三维"""
    pts = load_feature(name, 'normalized_points', features_dir)
    if pts.ndim == 3 and pts.shape[2] == 3:
        pts = pts[:, :, :2]
    return pts


def load_vector(name, features_dir):
    """加载向量特征，若三维则展开为二维"""
    vec = load_feature(name, 'vector', features_dir)
    if vec.ndim == 3:
        vec = vec.reshape(vec.shape[0], -1)
    return vec


def to_2d(arr):
    """三维转二维"""
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    return arr


def build_diff_input(template_name, test_name):
    """
    加载两个视频的特征，用 fastdtw 对齐，返回模型所需的拼接差分矩阵。
    模板从 TEMPLATE_FEATURES_DIR 加载，测试从 TEST_FEATURES_DIR 加载。
    """
    # 加载模板特征（在模板目录中）
    ref_feat = load_feature(template_name, 'features', TEMPLATE_FEATURES_DIR)
    ref_points = load_points(template_name, TEMPLATE_FEATURES_DIR)
    ref_vector = load_vector(template_name, TEMPLATE_FEATURES_DIR)

    # 加载测试特征（在测试目录中）
    que_feat = load_feature(test_name, 'features', TEST_FEATURES_DIR)
    que_points = load_points(test_name, TEST_FEATURES_DIR)
    que_vector = load_vector(test_name, TEST_FEATURES_DIR)

    # 使用 fastdtw 对齐
    _, path = fastdtw(ref_feat, que_feat, dist=euclidean)
    path = np.array(path, dtype=int)

    ref_idx = path[:, 0]
    que_idx = path[:, 1]

    # 逐帧差：测试 - 模板
    fea_diff = que_feat[que_idx] - ref_feat[ref_idx]
    point_diff = que_points[que_idx] - ref_points[ref_idx]
    vector_diff = que_vector[que_idx] - ref_vector[ref_idx]

    # 统一转二维并拼接
    fea_diff = to_2d(fea_diff)
    point_diff = to_2d(point_diff)
    vector_diff = to_2d(vector_diff)

    diff = np.concatenate([fea_diff, point_diff, vector_diff], axis=-1).astype(np.float32)

    return diff


# ==================== 模型定义（与训练脚本完全一致） ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class DiffTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, mask=None):
        # x: (1, T, D)
        x = self.input_proj(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)   # 单样本无需 padding mask
        # 平均池化
        pooled = x.mean(dim=1)
        out = self.reg_head(pooled)
        return out


# ==================== 主流程 ====================
def main():
    # ----- 1. 读取真实分数 -----
    with open(SCORES_JSON, 'r', encoding='utf-8') as f:
        raw_scores = json.load(f)
    scores = {os.path.splitext(str(k))[0]: float(v) for k, v in raw_scores.items()}

    # ----- 2. 自动确定测试视频：读取 features_val 下所有 *_features.npy -----
    all_videos = []
    for fname in os.listdir(TEST_FEATURES_DIR):
        if fname.endswith('_features.npy'):
            name = fname[:-len('_features.npy')]
            all_videos.append(name)

    all_videos = sorted(set(all_videos))

    # 测试视频 = features_val 中存在的视频，排除模板
    test_videos = []
    for name in all_videos:
        if name in TEMPLATES:
            continue

        if name not in scores:
            print(f"跳过（无分数）: {name}")
            continue

        # 检查测试特征文件是否齐全
        feat_path = os.path.join(TEST_FEATURES_DIR, f"{name}_features.npy")
        pts_path = os.path.join(TEST_FEATURES_DIR, f"{name}_normalized_points.npy")
        vec_path = os.path.join(TEST_FEATURES_DIR, f"{name}_vector.npy")

        if os.path.exists(feat_path) and os.path.exists(pts_path) and os.path.exists(vec_path):
            test_videos.append(name)
        else:
            print(f"跳过（缺少特征文件）: {name}")

    if not test_videos:
        raise RuntimeError("没有找到任何可用于测试的视频，请检查 features_val 目录。")

    print(f"自动发现测试视频数: {len(test_videos)}")
    print("测试视频列表:", test_videos)

    # ----- 3. 检查模板特征是否存在并加载模型 -----
    for t in TEMPLATES:
        t_feat = os.path.join(TEMPLATE_FEATURES_DIR, f"{t}_features.npy")
        if not os.path.exists(t_feat):
            raise FileNotFoundError(f"模板特征缺失: {t_feat}")

    # 用第一个测试视频确定输入维度
    sample_diff = build_diff_input(TEMPLATES[0], test_videos[0])
    input_dim = sample_diff.shape[-1]

    model = DiffTransformer(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"模型加载完成，输入维度: {input_dim}")

    # ----- 4. 存放预测结果 -----
    predictions = {}

    with torch.no_grad():
        for test_video in test_videos:
            pred_scores = []

            for template in TEMPLATES:
                # 构造差分输入
                diff = build_diff_input(template, test_video)
                diff_tensor = torch.from_numpy(diff).unsqueeze(0).to(DEVICE)

                # 预测分差
                pred_diff = model(diff_tensor).item()

                # 反推绝对分数
                template_score = scores[template]
                pred_score = template_score + pred_diff
                pred_scores.append(pred_score)

                print(f"{test_video} vs {template}: 模板分={template_score}, 预测分差={pred_diff:.3f}, 预测分={pred_score:.3f}")

            # 三个模板取平均
            avg_pred = float(np.mean(pred_scores))
            predictions[test_video] = avg_pred
            print(f"→ {test_video} 最终预测: {avg_pred:.3f}\n")

    # ----- 5. 计算指标 -----
    true_scores = [scores[v] for v in test_videos]
    pred_scores = [predictions[v] for v in test_videos]

    print("=" * 60)
    print("测试集结果:")
    print(f"{'视频':<10}{'真实分':<10}{'预测分':<10}")
    for v, t, p in zip(test_videos, true_scores, pred_scores):
        print(f"{v:<10}{t:<10.1f}{p:<10.3f}")

    spearman = spearmanr(true_scores, pred_scores).correlation
    mae = np.mean(np.abs(np.array(true_scores) - np.array(pred_scores)))
    print("=" * 60)
    print(f"测试集 Spearman: {spearman:.4f}")
    print(f"测试集 MAE: {mae:.4f}")


if __name__ == '__main__':
    main()