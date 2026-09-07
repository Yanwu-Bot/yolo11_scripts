# train_template_free_ridge.py
import glob, os, pickle
import numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, GroupKFold
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

# ============ 配置 ============
DATA_DIR = 'D:/Dataset/sprint/result/semantic_dataset'   # 你生成的所有 .npz
SAVE_MODEL = 'template_free_ridge.pkl'
ALPHA = 10.0
# ==============================

files = sorted(glob.glob(os.path.join(DATA_DIR, 'semantic_sample_*.npz')))
print(f"样本对数量: {len(files)}")

X, y, groups, base_list, human_list = [], [], [], [], []
for f in files:
    d = np.load(f, allow_pickle=True)
    diff = d['semantic_diff'].astype(np.float32)     # (64,)
    base = float(d['base_score'])
    human = float(d['human_score'])
    test = d['test'].item()

    X.append(np.concatenate([diff, [base]]))          # (65,)
    y.append(human - base)
    groups.append(test)
    base_list.append(base)
    human_list.append(human)

X = np.array(X)
y = np.array(y)
groups = np.array(groups)
base_list = np.array(base_list)
human_list = np.array(human_list)

print(f"总特征维度: {X.shape[1]} (64 语义差 + 1 基础分)")
print(f"测试视频数: {len(set(groups))}")

# ---------- 交叉验证评估（按视频分组） ----------
gkf = GroupKFold(n_splits=5)
pred_residual = np.zeros_like(y)

for train_idx, val_idx in gkf.split(X, y, groups):
    model = Ridge(alpha=ALPHA)
    model.fit(X[train_idx], y[train_idx])
    pred_residual[val_idx] = model.predict(X[val_idx])

# 按测试视频聚合后再算 Spearman/MAE（更贴近实际打分）
video_base = defaultdict(list)
video_corrected = defaultdict(list)
video_human = {}

for i, test in enumerate(groups):
    video_base[test].append(base_list[i])
    video_corrected[test].append(base_list[i] + pred_residual[i])
    video_human[test] = human_list[i]

tests_sorted = list(video_human.keys())
base_video = np.array([np.mean(video_base[t]) for t in tests_sorted])
corr_video = np.array([np.mean(video_corrected[t]) for t in tests_sorted])
human_video = np.array([video_human[t] for t in tests_sorted])

rho_base = spearmanr(human_video, base_video).correlation
rho_corr = spearmanr(human_video, corr_video).correlation
mae_base = mean_absolute_error(human_video, base_video)
mae_corr = mean_absolute_error(human_video, corr_video)

print("\n========== 视频级留一评估 ==========")
print(f"基础分平均      Spearman: {rho_base:.4f}, MAE: {mae_base:.3f}")
print(f"修正后          Spearman: {rho_corr:.4f}, MAE: {mae_corr:.3f}")
print("=====================================")

# ---------- 用全部数据训练最终模型 ----------
final_model = Ridge(alpha=ALPHA)
final_model.fit(X, y)

with open(SAVE_MODEL, 'wb') as f:
    pickle.dump(final_model, f)
print(f"\n最终修正模型已保存: {SAVE_MODEL}")