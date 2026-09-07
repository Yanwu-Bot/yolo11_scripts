# evaluate_semantic_residual.py
import os
import glob
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

# ================= 配置 =================
DATA_DIR = 'D:/Dataset/sprint/result/semantic_dataset'
ALPHA = 10.0
# ========================================

# 读取所有样本
files = sorted(glob.glob(os.path.join(DATA_DIR, 'semantic_sample_*.npz')))
print(f"找到 {len(files)} 个样本")

X_list = []
y_list = []
groups = []
base_scores = []
true_scores = []

for f in files:
    data = np.load(f, allow_pickle=True)
    semantic_diff = data['semantic_diff'].astype(np.float32)   # (64,)
    base = float(data['base_score'])
    human = float(data['human_score'])
    test = data['test'].item()

    # 拼接语义差和base_score作为特征
    feature = np.concatenate([semantic_diff, [base]])          # (65,)
    X_list.append(feature)
    y_list.append(human - base)                 # 残差标签
    groups.append(test)                         # 用于按视频分组
    base_scores.append(base)
    true_scores.append(human)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
base_scores = np.array(base_scores)
true_scores = np.array(true_scores)
groups = np.array(groups)

print(f"特征维度: {X.shape[1]}")

# GroupKFold 按视频分组，保证同一测试视频的所有模板对只在同一折中
gkf = GroupKFold(n_splits=5)
pred_residual = np.zeros_like(y)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    model = Ridge(alpha=ALPHA)
    model.fit(X[train_idx], y[train_idx])
    pred_residual[val_idx] = model.predict(X[val_idx])
    print(f"Fold {fold+1} 完成，验证集样本数: {len(val_idx)}")

# 修正后的得分
corrected_scores = base_scores + pred_residual

# 评估
rho_base = spearmanr(true_scores, base_scores).correlation
rho_corrected = spearmanr(true_scores, corrected_scores).correlation
mae_base = mean_absolute_error(true_scores, base_scores)
mae_corrected = mean_absolute_error(true_scores, corrected_scores)

print("\n========== 交叉验证结果 ==========")
print(f"原始 base_score    Spearman: {rho_base:.4f}   MAE: {mae_base:.3f}")
print(f"修正后得分         Spearman: {rho_corrected:.4f}   MAE: {mae_corrected:.3f}")
print("==================================")

# 如果想看哪些样本修正得好/差，可以按视频聚合后再计算一次
# 按视频取平均得到每个视频唯一分数
unique_tests = list(set(groups))
true_video = []
base_video = []
corrected_video = []

for t in unique_tests:
    mask = (groups == t)
    true_video.append(np.mean(true_scores[mask]))
    base_video.append(np.mean(base_scores[mask]))
    corrected_video.append(np.mean(corrected_scores[mask]))

rho_base_video = spearmanr(true_video, base_video).correlation
rho_corrected_video = spearmanr(true_video, corrected_video).correlation

print("\n===== 按视频聚合后的结果（更贴近实际使用）=====")
print(f"原始 base_score    Spearman: {rho_base_video:.4f}")
print(f"修正后得分         Spearman: {rho_corrected_video:.4f}")
print("=============================================")

# 保存最终模型（用全部数据训练Ridge）
final_model = Ridge(alpha=ALPHA)
final_model.fit(X, y)
with open('residual_ridge_semantic.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("\n已保存最终模型: residual_ridge_semantic.pkl")