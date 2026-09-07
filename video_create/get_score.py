# get_score.py
# -*- coding: utf-8 -*-
"""
使用 DTW_Score.VideoScoreEvaluator 对多个视频进行评分，输出为 CSV。
不再使用残差回归模型。
"""
import os
import csv
import json

from DTW_Score import VideoScoreEvaluator

# ===================== 配置 =====================
TEMPLATE_VIDEO = 'run_6'                # 模板视频（不带 .mp4）
SCORES_FILE = 'D:/Dataset/sprint/result/video_point/scores.json'
FEATURES_DIR = 'D:/Dataset/sprint/result/features'
VIDEO_DIR = 'D:/Dataset/sprint/Whole'
WEIGHT = {"fea": 0.7, "point": 0.15, "displacement": 0.15}
OUTPUT_DIR = 'result/plots'
# ==================================================

# 读取人工评分
with open(SCORES_FILE, 'r', encoding='utf-8') as f:
    human_scores = json.load(f)

# 使用除模板外的全部视频作为测试集
test_video_names = [name for name in human_scores.keys() if name != TEMPLATE_VIDEO]
print(f"模板视频：{TEMPLATE_VIDEO}")
print(f"测试视频（{len(test_video_names)} 个）：{test_video_names}\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []
for i, name in enumerate(test_video_names, 1):
    try:
        # 使用 VideoScoreEvaluator 计算该视频的得分
        evaluator = VideoScoreEvaluator(
            template_video=f'{TEMPLATE_VIDEO}.mp4',
            test_video=f'{name}.mp4',
            features_dir=FEATURES_DIR,
            video_dir=VIDEO_DIR,
            weight=WEIGHT,
            output_dir=OUTPUT_DIR
        )
        evaluator.score_video('CTR')          # 若需调整窗口相似度模型，请修改此参数
        score = evaluator.get_combined_score() # 使用加权后分数；若想用窗口调整后分数，可改为 evaluator.mu

        human_score = human_scores.get(name)
        if human_score is None:
            print(f"[{i}/{len(test_video_names)}] {name}: 人工评分为空，跳过")
            continue

        results.append((name, human_score, score))
        print(f"[{i}/{len(test_video_names)}] {name}: 人工={human_score:.2f}, 评分={score:.2f}")

    except Exception as e:
        print(f"[{i}/{len(test_video_names)}] {name} 评分失败: {e}")

# 保存 CSV
csv_path = os.path.join(OUTPUT_DIR, 'single_template_scores.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['test_video', 'human_score', 'score'])
    for name, h, s in results:
        writer.writerow([name, f"{h:.3f}", f"{s:.3f}"])

print(f"\n共 {len(results)} 个有效样本，评分结果已保存至: {csv_path}")