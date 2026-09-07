# build_video_dataset.py
import glob, os, json
import numpy as np
from collections import defaultdict

# ======== 配置 ========
DATA_DIR = 'D:/Dataset/sprint/result/semantic_dataset'     # semantic_sample_*.npz 所在目录
OUTPUT_NPZ = 'D:/Dataset/sprint/result/video_dataset.npz'  # 输出文件

# 手动指定模板顺序（之后训练/推理都必须用这个顺序）
TEMPLATE_ORDER = ['run_6', 'run_7', 'run_8', 'run_10']
# =====================

files = sorted(glob.glob(os.path.join(DATA_DIR, 'semantic_sample_*.npz')))
print(f"找到 {len(files)} 个样本对")

# 按 test 视频和 template 视频存储
data_by_test = defaultdict(dict)
for f in files:
    npz = np.load(f, allow_pickle=True)
    test = npz['test'].item()
    template = npz['template'].item()
    human = float(npz['human_score'])
    base = float(npz['base_score'])
    semantic_diff = npz['semantic_diff'].astype(np.float32)
    data_by_test[test][template] = (human, base, semantic_diff)

# 保留包含所有模板的测试视频，并按照 TEMPLATE_ORDER 拼接特征
X_list = []
y_list = []
test_names = []

for test, template_dict in data_by_test.items():
    # 检查是否包含所有需要的模板
    if not all(t in template_dict for t in TEMPLATE_ORDER):
        print(f"跳过 {test}: 缺少模板")
        continue

    human, _, _ = template_dict[TEMPLATE_ORDER[0]]
    parts = []

    for t in TEMPLATE_ORDER:
        _, base, diff = template_dict[t]
        parts.append(diff)                       # 64 维向量
        parts.append(np.array([base], dtype=np.float32))  # 1 维 base

    X_list.append(np.concatenate(parts))
    y_list.append(human)
    test_names.append(test)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"有效测试视频数: {len(y)}")
print(f"每个视频特征维度: {X.shape[1]} = {len(TEMPLATE_ORDER)} 个模板 × (64语义差 + 1基线)")

# 保存
np.savez_compressed(OUTPUT_NPZ, X=X, y=y, test_names=np.array(test_names))
print(f"已保存至: {OUTPUT_NPZ}")