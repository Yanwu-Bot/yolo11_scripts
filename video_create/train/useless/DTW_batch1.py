# evaluate_single_template_correction.py
# -*- coding: utf-8 -*-
"""
仿照 DTW_batch.py 批量评估：单模板残差修正前后的 Spearman
"""
import os, csv, json
import numpy as np
import torch, pickle
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from module import CTRGCNEncoder
from AcDTW import acdtw
from DTW_Score import VideoScoreEvaluator

# ====================== 配置 ======================
TEMPLATE_VIDEO = 'run_9'                # 模板视频（不带.mp4）
MODEL_PATH = 'D:/Dataset/sprint/result/model/CTR-GCN/best_9_2_ctr.pth'   # 与DTW_1.py一致
PICKLE_PATH = 'template_free_ridge.pkl'
WINDOW_SIZE = 9
STRIDE = 2

SCORES_FILE = 'D:/Dataset/sprint/result/video_point/scores.json'
FEATURES_DIR = 'D:/Dataset/sprint/result/features'
VIDEO_DIR = 'D:/Dataset/sprint/Whole'
WEIGHT = {"fea": 0.7, "point": 0.15, "displacement": 0.15}
OUTPUT_DIR = 'result/plots'
N_TEST_VIDEOS = 30
RANDOM_SEED = 45
# ==================================================

if RANDOM_SEED is not None:
    import random
    random.seed(RANDOM_SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载CTR编码器
encoder = CTRGCNEncoder(in_channels=2, output_dim=64).to(device)
encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
encoder.eval()

# 加载残差回归器
with open(PICKLE_PATH, 'rb') as f:
    regressor = pickle.load(f)

# 读人工评分
with open(SCORES_FILE, 'r', encoding='utf-8') as f:
    human_scores = json.load(f)

candidate_videos = [name for name in human_scores.keys() if name != TEMPLATE_VIDEO]
test_video_names = random.sample(candidate_videos, N_TEST_VIDEOS)
print(f"模板视频：{TEMPLATE_VIDEO}")
print(f"随机测试视频（{len(test_video_names)} 个）：{test_video_names}\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_video_vector(video_name):
    pts = np.load(os.path.join(FEATURES_DIR, f'{video_name}_normalized_points.npy'))
    if pts.ndim == 3 and pts.shape[2] == 3:
        pts = pts[:, :, :2]
    pts = pts.astype(np.float32)
    feats = []
    with torch.no_grad():
        for start in range(0, len(pts) - WINDOW_SIZE + 1, STRIDE):
            win = pts[start:start + WINDOW_SIZE]
            tensor = torch.FloatTensor(win).permute(2, 0, 1).unsqueeze(0).to(device)
            feats.append(encoder(tensor).cpu().numpy()[0])
    vec = np.mean(feats, axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec


# 模板向量，只提取一次
vec_template = extract_video_vector(TEMPLATE_VIDEO)

results = []
for i, name in enumerate(test_video_names, 1):
    try:
        # 1. 原始打分（与DTW_1.py完全一致）
        evaluator = VideoScoreEvaluator(
            template_video=f'{TEMPLATE_VIDEO}.mp4',
            test_video=f'{name}.mp4',
            features_dir=FEATURES_DIR,
            video_dir=VIDEO_DIR,
            weight=WEIGHT,
            output_dir=OUTPUT_DIR
        )
        evaluator.score_video('CTR')   # 若原代码需要选择模型，这里对应你实际使用的模型
        before_score = evaluator.get_combined_score()

        # 2. 语义差 + 残差修正
        vec_test = extract_video_vector(name)
        semantic_diff = vec_test - vec_template
        X_input = np.concatenate([semantic_diff, [before_score]]).reshape(1, -1)
        residual = regressor.predict(X_input)[0]
        after_score = before_score + residual

        human_score = human_scores[name]
        results.append((name, human_score, before_score, after_score))
        print(f"[{i}/{len(test_video_names)}] {name}: "
              f"人工={human_score:.2f}, 修正前={before_score:.2f}, 修正后={after_score:.2f}")

    except Exception as e:
        print(f"[{i}/{len(test_video_names)}] {name} 评分失败: {e}")

valid = [r for r in results if r[1] is not None and r[2] is not None and r[3] is not None]
if len(valid) < 2:
    print("有效样本不足")
    exit()

human_list = [r[1] for r in valid]
before_list = [r[2] for r in valid]
after_list = [r[3] for r in valid]

rho_before, p_before = spearmanr(before_list, human_list)
rho_after, p_after = spearmanr(after_list, human_list)
mae_before = mean_absolute_error(human_list, before_list)
mae_after = mean_absolute_error(human_list, after_list)

print("\n========== 斯皮尔曼相关系数（真实人工评分） ==========")
print(f"修正前 vs 人工：rho = {rho_before:.4f}, p = {p_before:.4f}, MAE = {mae_before:.3f}")
print(f"修正后 vs 人工：rho = {rho_after:.4f}, p = {p_after:.4f}, MAE = {mae_after:.3f}")
print("========================================================")

# 保存CSV
csv_path = os.path.join(OUTPUT_DIR, 'single_template_correction_results.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['test_video', 'human_score', 'before_score', 'after_score'])
    for name, h, b, a in valid:
        writer.writerow([name, f"{h:.3f}", f"{b:.3f}", f"{a:.3f}"])
print(f"结果保存至: {csv_path}")