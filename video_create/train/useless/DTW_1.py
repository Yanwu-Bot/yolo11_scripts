# infer_template_free.py
import os, numpy as np, pickle, torch
from module import CTRGCNEncoder
from AcDTW import acdtw
from DTW_Score import VideoScoreEvaluator

# ============ 配置 ============
TEMPLATE = 'run_5'          # 你可以换成任意模板
TEST_VIDEO = 'run_10'       # 你要打分的视频
MODEL_PATH = 'D:/Dataset/sprint/result/model/CTR-GCN/best_9_2_ctr.pth'
PICKLE_PATH = 'template_free_ridge.pkl'      # 刚训练好的模型
FEATURES_DIR = 'D:/Dataset/sprint/result/features'
WINDOW_SIZE = 9
STRIDE = 2                  # 必须与生成数据集时一致
WEIGHT = {"fea": 0.7, "point": 0.15, "displacement": 0.15}
# ==============================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载编码器
encoder = CTRGCNEncoder(in_channels=2, output_dim=64).to(device)
encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
encoder.eval()

# 加载修正模型
with open(PICKLE_PATH, 'rb') as f:
    regressor = pickle.load(f)


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
    return vec / (np.linalg.norm(vec) + 1e-8)


# 1. 计算基础分（三维DTW打分，与数据集生成完全一致）
evaluator = VideoScoreEvaluator(template_video=f'{TEMPLATE}.mp4',
                                test_video=f'{TEST_VIDEO}.mp4',
                                features_dir=FEATURES_DIR,
                                weight=WEIGHT)
evaluator.score_video()   # 注意：如果你的score_video内部自动加了窗口修正，请绕开或用compute_pairwise_scores
base = evaluator.get_combined_score()

# 2. 计算语义差
vec_test = extract_video_vector(TEST_VIDEO)
vec_template = extract_video_vector(TEMPLATE)
semantic_diff = vec_test - vec_template

# 3. 输入修正模型
X_new = np.concatenate([semantic_diff, [base]]).reshape(1, -1)
residual = regressor.predict(X_new)[0]
final = base + residual

print(f"模板: {TEMPLATE}")
print(f"测试视频: {TEST_VIDEO}")
print(f"基础分: {base:.2f}")
print(f"修正量: {residual:+.2f}")
print(f"最终得分: {final:.2f}")