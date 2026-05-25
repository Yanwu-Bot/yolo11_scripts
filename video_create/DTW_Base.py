#DTW标准运行代码，可实现指定对比对可视化
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from matplotlib import rcParams
import math
rcParams['font.family'] = 'SimHei'
matplotlib.use('TkAgg')

class VideoScoreEvaluator:
    def __init__(self, 
                template_video: str = None,
                test_video: str = None,
                features_dir: str = 'result/features',
                video_dir: str = 'video_origin/data_video/use',
                weight: dict = None,
                output_dir: str = 'result/plots'):
        self.template_video = template_video
        self.test_video = test_video
        self.features_dir = features_dir
        self.video_dir = video_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.weight = weight if weight else {"fea": 0.5, "point": 0.3, "displacement": 0.2}
        self.angle_weights = [1] * 24
        self.center_weights = [1] * 8
        self.orientation_weight = [1]
        self.feet_distance_weight = [1]
        self.phase_weights = [1] * 4
        self.feature_weights = (self.angle_weights + self.center_weights + 
                                self.orientation_weight + self.feet_distance_weight + 
                                self.phase_weights)
        self.feat_score = None
        self.point_score = None
        self.displacement_score = None
        self.frame_scores = None
        self.path = None
        self.q_mean_list = None
        self.point_distances = None
        self.test_features = None
        self.template_features = None
        self.displacement_frame_scores = None
        self.lower_bound = None
        self.upper_bound = None
        self.template_video_path = None
        self.test_video_path = None
        if template_video and test_video:
            self.set_videos(template_video, test_video)

        self.window = 10
        self.mu =None
        self.sigma = None

    def set_videos(self, template_video: str, test_video: str):
        self.template_video = template_video
        self.test_video = test_video
        self.template_video_path = os.path.join(self.video_dir, template_video)
        self.test_video_path = os.path.join(self.video_dir, test_video)
    
    def set_feature_weights(self, 
                            angle_weight: float = 1.8,
                            center_weight: float = 1.4,
                            orientation_weight: float = 0.7,
                            feet_distance_weight: float = 1.2,
                            phase_weight: float = 1.3):
        self.angle_weights = [angle_weight] * 24
        self.center_weights = [center_weight] * 8
        self.orientation_weight = [orientation_weight]
        self.feet_distance_weight = [feet_distance_weight]
        self.phase_weights = [phase_weight] * 4
        self.feature_weights = (self.angle_weights + self.center_weights + 
                                self.orientation_weight + self.feet_distance_weight + 
                                self.phase_weights)
    
    def calculate_frame_score(self, test_feat: np.ndarray, template_feat: np.ndarray, t: float = 0.05, k: float = 4) -> float:
        """只返回特征得分，不再计算方差"""
        f_weights = np.array(self.feature_weights) / np.sum(self.feature_weights)
        q = np.abs(test_feat - template_feat)
        q_mean = np.sum(q * f_weights)
        exceed = q_mean - t
        if q_mean <= t:
            mu = 100.0
        else:
            mu = 100 * np.exp(-k * exceed)
        return mu

    def calculate_keypoint_frame_score(self, test_points: np.ndarray, template_points: np.ndarray) -> tuple:
        dist = np.linalg.norm(test_points - template_points)
        if dist <= 50.0:
            score = 100 
        elif 50 < dist < 150:
            score = (1 - ((dist-50) / 100)) * 100
        else:
            score = 0.0
        return score, dist   

    def calculate_displacement_frame_score(self, test_vec: np.ndarray, template_vec: np.ndarray, t: float = 0.025, k: float = 4) -> float:
        """只返回位移得分，不再计算方差"""
        q = np.abs(test_vec - template_vec)
        q_mean = np.mean(q)
        exceed = q_mean - t
        if q_mean <= t:
            mu = 100.0
        else:
            mu = 100 * np.exp(-k * exceed)
        return mu

    def calculate_video_score(self, test_features: np.ndarray, template_features: np.ndarray) -> np.ndarray:
        """只计算DTW路径，不计算得分"""
        distance, path = fastdtw(test_features, template_features, dist=euclidean)
        path = np.array(path)
        self.path = path
        print(f"DTW对齐完成: 路径长度 = {len(path)}")
        return path

    def compute_pairwise_scores(self, path: np.ndarray, use_window=True):
        template_feat_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.template_video)[0]}_features.npy")
        test_feat_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.test_video)[0]}_features.npy")
        template_features = np.load(template_feat_path)
        test_features = np.load(test_feat_path)

        template_point_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.template_video)[0]}_normalized_points.npy")
        test_point_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.test_video)[0]}_normalized_points.npy")
        template_points = None
        test_points = None
        if os.path.exists(template_point_path) and os.path.exists(test_point_path):
            template_points = np.load(template_point_path)
            test_points = np.load(test_point_path)
            if len(template_points.shape) == 3 and template_points.shape[2] == 3:
                template_points = template_points[:, :, :2]
            if len(test_points.shape) == 3 and test_points.shape[2] == 3:
                test_points = test_points[:, :, :2]
        else:
            print("\n警告: 找不到关键点文件，跳过关键点评分")

        template_vector_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.template_video)[0]}_vector.npy")
        test_vector_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.test_video)[0]}_vector.npy")
        template_vector = None
        test_vector = None
        if os.path.exists(template_vector_path) and os.path.exists(test_vector_path):
            template_vector = np.load(template_vector_path)
            test_vector = np.load(test_vector_path)
        else:
            print("\n警告: 找不到向量文件，跳过位移得分")

        # 逐对计算得分
        frame_scores = []           # 特征得分
        q_mean_list = []
        point_frame_scores = []
        displacement_frame_scores = []
        point_distances = []

        for test_idx, template_idx in path:
            # 特征得分（仅取得分，忽略方差）
            test_frame = test_features[test_idx]
            template_frame = template_features[template_idx]
            score = self.calculate_frame_score(test_frame, template_frame)
            q = np.abs(test_frame - template_frame)
            q_mean = np.mean(q)
            frame_scores.append(score)
            q_mean_list.append(q_mean)

            # 关键点得分
            if test_points is not None and template_points is not None:
                test_point_frame = test_points[test_idx]
                template_point_frame = template_points[template_idx]
                point_score_frame, point_dist = self.calculate_keypoint_frame_score(
                    test_point_frame, template_point_frame)
                point_frame_scores.append(point_score_frame)
                point_distances.append(point_dist)

            # 位移得分（仅取得分）
            if test_vector is not None and template_vector is not None:
                if len(test_vector.shape) == 3:
                    test_vec_frame = test_vector[test_idx].reshape(-1)
                    template_vec_frame = template_vector[template_idx].reshape(-1)
                else:
                    test_vec_frame = test_vector[test_idx]
                    template_vec_frame = template_vector[template_idx]
                displacement_score = self.calculate_displacement_frame_score(
                    test_vec_frame, template_vec_frame)
                displacement_frame_scores.append(displacement_score)

        # 保存逐对得分
        self.frame_scores = frame_scores
        self.q_mean_list = q_mean_list
        self.point_frame_scores = point_frame_scores
        self.point_distances = point_distances
        self.displacement_frame_scores = displacement_frame_scores

        # 窗口聚合
        if use_window and self.window > 0:
            L = len(path)
            window_fea_scores = []
            window_point_scores = []
            window_disp_scores = []

            for start in range(0, L, self.window):
                end = min(start + self.window, L)
                win_fea = frame_scores[start:end]
                win_point = point_frame_scores[start:end] if point_frame_scores else []
                win_disp = displacement_frame_scores[start:end] if displacement_frame_scores else []

                avg_fea = np.mean(win_fea) if win_fea else 0.0
                avg_point = np.mean(win_point) if win_point else 0.0
                avg_disp = np.mean(win_disp) if win_disp else 0.0

                window_fea_scores.append(avg_fea)
                window_point_scores.append(avg_point)
                window_disp_scores.append(avg_disp)
            self.mu_sigma(window_fea_scores,window_point_scores,window_disp_scores)
            # 最终得分 = 所有窗口平均分的平均
            self.feat_score = np.mean(window_fea_scores) if window_fea_scores else 0.0
            self.point_score = np.mean(window_point_scores) if window_point_scores else 0.0
            self.displacement_score = np.mean(window_disp_scores) if window_disp_scores else 0.0

            self.window_scores = {
                'feature': window_fea_scores,
                'point': window_point_scores,
                'displacement': window_disp_scores
            }

    def mu_sigma(self, fea, point, vec):
        scores = []
        if len(fea) == len(point) == len(vec):
            length = len(fea)
            for i in range(length):
                score = self.weight['fea'] * fea[i] + \
                        self.weight['point'] * point[i] + \
                        self.weight['displacement'] * vec[i]
                scores.append(score)
            self.mu = sum(scores) / length

            total = 0
            for i in range(length):
                x = (scores[i] - self.mu) ** 2
                total += x
            self.sigma = math.sqrt(total / (length - 1)) if length > 1 else 0.0
            self.sigma = self.sigma / math.sqrt(length)

            self.lower_bound = max(0, self.mu - 1.96 * self.sigma)
            self.upper_bound = min(100, self.mu + 1.96 * self.sigma)
        else:
            print("Error")


    def score_video(self) -> tuple:
        template_feat_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.template_video)[0]}_features.npy")
        test_feat_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.test_video)[0]}_features.npy")
        if not os.path.exists(template_feat_path) or not os.path.exists(test_feat_path):
            print("错误: 找不到特征文件")
            return None, None, None, None, None, None, None, None
        
        self.template_features = np.load(template_feat_path)
        self.test_features = np.load(test_feat_path)
        print(f"\n模板视频帧数: {self.template_features.shape[0]}")
        print(f"测试视频帧数: {self.test_features.shape[0]}")
        
        path = self.calculate_video_score(self.test_features, self.template_features)
        
        self.compute_pairwise_scores(path, use_window=True)
        

    def visualize_aligned_frames(self, pair_index: int):
        if self.path is None:
            print("没有对齐数据")
            return
        if not self.template_video_path or not self.test_video_path:
            print("请先设置视频路径")
            return
        if pair_index >= len(self.path):
            print(f"pair_index {pair_index} 超出范围 (最大 {len(self.path)-1})")
            return
        test_idx, template_idx = self.path[pair_index]
        cap = cv2.VideoCapture(self.test_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, test_idx)
        ret, test_frame = cap.read()
        cap.release()
        cap = cv2.VideoCapture(self.template_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, template_idx)
        ret, template_frame = cap.read()
        cap.release()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
        template_frame_rgb = cv2.cvtColor(template_frame, cv2.COLOR_BGR2RGB)
        ax1.imshow(test_frame_rgb)
        score_text = f'\n得分: {self.frame_scores[pair_index]:.3f}' if self.frame_scores else ''
        ax1.set_title(f'测试视频 - 第{test_idx}帧{score_text}')
        ax1.axis('off')
        ax2.imshow(template_frame_rgb)
        ax2.set_title(f'模板视频 - 第{template_idx}帧')
        ax2.axis('off')
        plt.suptitle(f'DTW对齐对 #{pair_index}')
        plt.tight_layout()
        plt.show()
    
    def get_combined_score(self) -> float:
        if self.feat_score is None:
            print("请先运行 score_video()")
            return 0.0
        combined_score = (self.feat_score * self.weight['fea'] + 
                         self.point_score * self.weight['point'] +
                         self.displacement_score * self.weight['vec'])
        return combined_score
    
    def print_summary(self):
        if self.feat_score is None:
            print("请先运行 score_video()")
            return
        print(f"\n{'='*60}")
        print(f"DTW对比结果 (窗口大小: {self.window})")
        print(f"{'='*60}")
        print(f"模板视频: {self.template_video}")
        print(f"测试视频: {self.test_video}")
        print(f"{'='*60}")
        print(f"特征得分: {self.feat_score:.2f}")
        if self.point_score is not None:
            print(f"关键点得分: {self.point_score:.2f}")
        if self.displacement_score is not None:
            print(f"位移得分: {self.displacement_score:.2f}")
        print(f"{'='*60}")
        print(f"综合得分: {self.mu:.2f}" if hasattr(self, 'mu') else f"综合得分: {self.get_combined_score():.2f}")
        print(f"得分区间: {self.lower_bound:.2f} --- {self.upper_bound:.2f}")
        print(f"评分权重: fea={self.weight['fea']}, point={self.weight['point']}, displacement={self.weight['displacement']}")
        print(f"{'='*60}")

if __name__ == '__main__':
    evaluator = VideoScoreEvaluator(
        template_video='run_man.mp4',
        test_video='run_woman.mp4',
        features_dir='result/features',
        video_dir='video_origin/data_video/use',
        weight={"fea": 0.5, "point": 0.3, "displacement": 0.2},
        output_dir='result/plots'
    )
    evaluator.score_video()
    
    VIEW_FRAME = 20
    if evaluator.frame_scores and VIEW_FRAME < len(evaluator.frame_scores):
        evaluator.visualize_aligned_frames(VIEW_FRAME)
    
    evaluator.print_summary()