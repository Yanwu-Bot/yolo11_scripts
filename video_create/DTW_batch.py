# -*- coding: utf-8 -*-
"""
固定计算 run_9 ~ run_19 的修正前后斯皮尔曼系数 + MAE（使用真实人工评分）
"""
import os
import csv
import json
from scipy.stats import spearmanr
from DTW_Score import VideoScoreEvaluator


def main():
    # ====================== 配置 ======================
    TEMPLATE_VIDEO = 'run_43'                # 模板视频（不带.mp4）
    SEL_MODEL = 'CTR'                       # 模型：STGCN/GRU/LSTM/MLP/CTR/TCN
    SCORES_FILE = 'D:/Dataset/sprint/result/video_point/scores2.json'             # 人工评分文件
    FEATURES_DIR = 'D:/Dataset/sprint/result/features'
    VIDEO_DIR = 'D:/Dataset/sprint/Whole'
    WEIGHT = {"fea": 0.7, "point": 0.15, "displacement": 0.15}
    OUTPUT_DIR = 'result/plots'

    # 固定测试视频：run_9 ~ run_18
    TEST_PREFIX = 'run'
    START_ID = 8
    END_ID = 18
    test_video_names = [f'{TEST_PREFIX}_{i}' for i in range(START_ID, END_ID + 1)]
    # ==================================================

    # 读入人工评分
    with open(SCORES_FILE, 'r', encoding='utf-8') as f:
        human_scores = json.load(f)

    # 只保留存在人工评分的测试视频，并排除模板视频
    missing = [name for name in test_video_names
               if name not in human_scores or name == TEMPLATE_VIDEO]
    if missing:
        print("下列视频不在人工评分文件中/为模板视频，将被跳过：", missing)

    test_video_names = [name for name in test_video_names
                        if name in human_scores and name != TEMPLATE_VIDEO]

    print(f"模板视频：{TEMPLATE_VIDEO}")
    print(f"计算视频范围：run_{START_ID} ~ run_{END_ID}")
    print(f"实际有效测试视频（{len(test_video_names)} 个）：{test_video_names}\n")

    # 结果列表：视频名, 人工真实评分, 修正前得分, 修正后得分
    results = []

    for i, name in enumerate(test_video_names, 1):
        try:
            evaluator = VideoScoreEvaluator(
                template_video=f'{TEMPLATE_VIDEO}.mp4',
                test_video=f'{name}.mp4',
                features_dir=FEATURES_DIR,
                video_dir=VIDEO_DIR,
                weight=WEIGHT,
                output_dir=OUTPUT_DIR
            )
            evaluator.score_video(SEL_MODEL)

            before_score = evaluator.get_combined_score()
            after_score = evaluator.mu
            human_score = human_scores[name]   # 真实人工评分

            results.append((name, human_score, before_score, after_score))
            print(f"[{i}/{len(test_video_names)}] {name}: "
                  f"人工评分={human_score}, "
                  f"修正前={before_score:.2f}, 修正后={after_score:.2f}")

        except Exception as e:
            print(f"[{i}/{len(test_video_names)}] {name} 评分失败: {e}")

    # 保留有效记录（人工评分和模型得分均不能为 None）
    valid = [r for r in results
             if r[1] is not None and r[2] is not None and r[3] is not None]

    if len(valid) < 2:
        print("\n有效样本不足，无法计算斯皮尔曼系数和 MAE")
        return

    human_scores_list = [r[1] for r in valid]
    before_scores = [r[2] for r in valid]
    after_scores = [r[3] for r in valid]

    # 计算 Spearman 相关系数（使用真实人类评分）
    rho_before, p_before = spearmanr(before_scores, human_scores_list)
    rho_after, p_after = spearmanr(after_scores, human_scores_list)

    print("\n========== 斯皮尔曼相关系数（真实人工评分） ==========")
    print(f"修正前得分 vs 人工评分：rho = {rho_before:.4f}, p = {p_before:.4f}")
    print(f"修正后得分 vs 人工评分：rho = {rho_after:.4f}, p = {p_after:.4f}")
    print("========================================================")

    # ====================== 新增：MAE ======================
    mae_before = sum(abs(before - human) for before, human in zip(before_scores, human_scores_list)) / len(human_scores_list)
    mae_after = sum(abs(after - human) for after, human in zip(after_scores, human_scores_list)) / len(human_scores_list)

    print("\n========== MAE（平均绝对误差，真实人工评分） ==========")
    print(f"修正前得分 vs 人工评分：MAE = {mae_before:.4f}")
    print(f"修正后得分 vs 人工评分：MAE = {mae_after:.4f}")
    print("======================================================")
    # ========================================================

    # 保存详细结果到 CSV
    csv_path = 'result/spearman_results.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['test_video', 'human_score', 'before_score', 'after_score'])
        for name, score, before, after in valid:
            writer.writerow([name, score, f"{before:.3f}", f"{after:.3f}"])
    print(f"\n详细结果已保存至：{csv_path}")


if __name__ == '__main__':
    main()