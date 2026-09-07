# generate_semantic_residual_dataset.py
# 用途：为残差修正模型生成“语义差”训练数据（不包含逐帧DTW差值）
import os
import json
import numpy as np
import torch
import module
from AcDTW import acdtw
from DTW_Score import VideoScoreEvaluator   # 仅复用其中的打分函数

# ==================== 配置 ====================
SCORES_FILE = 'D:/Dataset/sprint/result/video_point/scores1.json'
FEATURES_DIR = 'D:/Dataset/sprint/result/features'
OUTPUT_DIR = 'D:/Dataset/sprint/result/semantic_dataset'
MODEL_PATH = 'D:/Dataset/sprint/result/model/CTR-GCN/best_9_2_ctr.pth'
MODEL_NAME = 'CTR'                      # 可选 STGCN/GRU/LSTM/MLP/CTR/TCN
WEIGHT = {"fea": 0.7, "point": 0.15, "displacement": 0.15}

# 手动指定多个模板（不带 .mp4）
TEMPLATE_VIDEOS = ['run_5', 'run_6', 'run_7', 'run_2']

# 滑窗参数（必须与预训练模型一致）
WINDOW_SIZE = 9
STRIDE = 2                              # 可以改成 2/3 使特征更密，但更慢
# =============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------- 加载人工评分 ----------
with open(SCORES_FILE, 'r', encoding='utf-8') as f:
    human_scores = json.load(f)

all_video_names = list(human_scores.keys())
test_video_names = [v for v in all_video_names if v not in TEMPLATE_VIDEOS]
print(f"模板视频: {TEMPLATE_VIDEOS}")
print(f"测试视频数量: {len(test_video_names)}")
print(f"预计样本数: {len(TEMPLATE_VIDEOS) * len(test_video_names)}")


# ---------- 加载预训练编码器 ----------
def build_model(name):
    if name == 'CTR':
        model = module.CTRGCNEncoder(in_channels=2, output_dim=64)
    elif name == 'STGCN':
        model = module.STGCNEncoder(in_channels=2, output_dim=64)
    elif name == 'GRU':
        model = module.GRUEncoder(in_channels=2, output_dim=64)
    elif name == 'LSTM':
        model = module.LSTMEncoder(in_channels=2, output_dim=64)
    elif name == 'MLP':
        model = module.MLPEncoder(in_channels=2, output_dim=64)
    elif name == 'TCN':
        model = module.TCNEncoder(in_channels=2, output_dim=64)
    else:
        raise ValueError(f"未知模型: {name}")
    return model

model = build_model(MODEL_NAME)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()


# ---------- 各种缓存 ----------
_features_cache = {}
_points_cache = {}

def load_npy(video_name, suffix):
    path = os.path.join(FEATURES_DIR, f"{video_name}{suffix}")
    if not os.path.exists(path):
        return None
    return np.load(path)

def get_features(name):
    """加载三个特征文件（缓存）"""
    if name not in _features_cache:
        feat = load_npy(name, '_features.npy')
        point = load_npy(name, '_normalized_points.npy')
        vec = load_npy(name, '_vector.npy')
        _features_cache[name] = (feat, point, vec)
    return _features_cache[name]


def get_points(name):
    """加载并缓存关键点数据，统一为 (T,17,2)"""
    if name not in _points_cache:
        pts = load_npy(name, '_normalized_points.npy')
        if pts is None:
            print(f"警告: {name} 缺少点文件")
            _points_cache[name] = None
        else:
            if pts.ndim == 3 and pts.shape[2] == 3:
                pts = pts[:, :, :2]   # 去掉置信度
            _points_cache[name] = pts.astype(np.float32)
    return _points_cache[name]


def extract_video_vector(video_name):
    """
    用滑窗提取整个视频的语义向量，最终平均池化为一个64维向量。
    """
    points = get_points(video_name)
    if points is None:
        return None

    T = points.shape[0]
    if T < WINDOW_SIZE:
        return None

    window_feats = []

    with torch.no_grad():
        for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
            win = points[start:start + WINDOW_SIZE]        # (9,17,2)
            tensor = torch.FloatTensor(win).permute(2, 0, 1).unsqueeze(0).to(device)
            feat = model(tensor).cpu().numpy()[0]          # (64,)
            window_feats.append(feat)

    video_vec = np.mean(window_feats, axis=0)
    # 重新归一化，保证模长为1
    norm = np.linalg.norm(video_vec)
    if norm > 1e-8:
        video_vec = video_vec / norm
    return video_vec


# 语义向量缓存（每个视频只提取一次）
_vec_cache = {}

def get_video_vec(video_name):
    if video_name not in _vec_cache:
        _vec_cache[video_name] = extract_video_vector(video_name)
    return _vec_cache[video_name]


# ---------- 计算 base_score ----------
evaluator = VideoScoreEvaluator()   # 只用于调用其内部打分方法

def compute_base_score(test_name, template_name):
    """
    基于原始三维特征（feat/point/vector）计算基础分。
    不加载任何窗口修正网络。
    """
    test_feat, test_points, test_vec = get_features(test_name)
    template_feat, template_points, template_vec = get_features(template_name)

    if test_feat is None or template_feat is None:
        return None

    # DTW路径
    def med_dist(tf, pf):
        score = evaluator.calculate_frame_score(tf, pf, t=0.05, k=4)
        return max(0.0, 100.0 - score)

    path, _, _, _ = acdtw(
        test_feat,
        template_feat,
        dist_func=med_dist,
        window=None
    )

    frame_scores = []
    point_scores = []
    disp_scores = []

    for test_idx, template_idx in path:
        # 1) 特征得分
        fs = evaluator.calculate_frame_score(
            test_feat[test_idx],
            template_feat[template_idx],
            t=0.05, k=4
        )
        frame_scores.append(fs)

        # 2) 关键点得分
        if test_points is not None and template_points is not None:
            tp = test_points[test_idx]
            pp = template_points[template_idx]
            if tp.ndim == 2 and tp.shape[1] == 3:
                tp = tp[:, :2]
            if pp.ndim == 2 and pp.shape[1] == 3:
                pp = pp[:, :2]
            ps, _ = evaluator.calculate_keypoint_frame_score(
                tp, pp, threshold=160, k=4
            )
            point_scores.append(ps)

        # 3) 位移得分
        if test_vec is not None and template_vec is not None:
            tv = test_vec[test_idx].reshape(-1)
            pv = template_vec[template_idx].reshape(-1)
            ds = evaluator.calculate_displacement_frame_score(
                tv, pv, t=0.02, k=4
            )
            disp_scores.append(ds)

    if len(frame_scores) == 0:
        return None

    avg_fea = float(np.mean(frame_scores))
    avg_point = float(np.mean(point_scores)) if point_scores else 0.0
    avg_disp = float(np.mean(disp_scores)) if disp_scores else 0.0

    # 重新归一化有效权重
    weights = dict(WEIGHT)
    active = ['fea']
    if point_scores:
        active.append('point')
    if disp_scores:
        active.append('displacement')
    total = sum(weights[k] for k in active)
    for k in active:
        weights[k] /= total

    score = (weights['fea'] * avg_fea +
             weights.get('point', 0.0) * avg_point +
             weights.get('displacement', 0.0) * avg_disp)
    return score

# ---------- 主生成循环 ----------
sample_idx = 0

for template_name in TEMPLATE_VIDEOS:
    vec_template = get_video_vec(template_name)
    if vec_template is None:
        print(f"模板 {template_name} 点文件不可用，跳过")
        continue

    for test_name in test_video_names:
        vec_test = get_video_vec(test_name)
        if vec_test is None:
            print(f"{test_name} 点文件不可用，跳过")
            continue

        # 语义差向量
        semantic_diff = vec_test - vec_template

        # base_score
        base_score = compute_base_score(test_name, template_name)
        if base_score is None:
            print(f"{test_name} vs {template_name} base_score计算失败，跳过")
            continue

        # 真实分
        human = human_scores.get(test_name)
        if human is None:
            continue

        # 保存
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"semantic_sample_{sample_idx:05d}.npz"),
            template=np.array(template_name),
            test=np.array(test_name),
            human_score=np.array(human),
            base_score=np.array(base_score),
            semantic_diff=semantic_diff,           # (64,)
        )
        sample_idx += 1

        if sample_idx % 50 == 0:
            print(f"已生成 {sample_idx} 个样本")

print(f"\n完成！共生成 {sample_idx} 个样本")
print(f"保存位置: {OUTPUT_DIR}")