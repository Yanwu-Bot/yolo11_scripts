# diagnose_coverage.py
import glob, os
from collections import defaultdict

DATA_DIR = 'D:/Dataset/sprint/result/semantic_dataset'
CANDIDATE_TEMPLATES = ['run_5', 'run_6', 'run_7', 'run_8', 'run_10', 'run_11']  # 根据你的实际情况调整

files = sorted(glob.glob(os.path.join(DATA_DIR, 'semantic_sample_*.npz')))
print(f"总样本对: {len(files)}")

# 统计每个测试视频覆盖了哪些模板
coverage = defaultdict(set)
for f in files:
    data = np.load(f, allow_pickle=True)
    test = data['test'].item()
    template = data['template'].item()
    coverage[test].add(template)

print(f"测试视频总数: {len(coverage)}")

# 统计不同模板组合的覆盖情况
from itertools import combinations
for K in [2, 3, 4]:
    best_combo = None
    best_count = 0
    for combo in combinations(CANDIDATE_TEMPLATES, K):
        count = sum(1 for v in coverage.values() if set(combo).issubset(v))
        if count > best_count:
            best_count = count
            best_combo = list(combo)
    print(f"\nK={K}: 最佳模板组合 {best_combo} 覆盖 {best_count} 个测试视频")

# 打印每个模板实际覆盖的视频数
print("\n各模板覆盖的测试视频数:")
for t in CANDIDATE_TEMPLATES:
    cnt = sum(1 for v in coverage.values() if t in v)
    print(f"  {t}: {cnt}")