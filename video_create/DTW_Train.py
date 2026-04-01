import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import euclidean
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'


class VideoScoreEvaluator:
    """
    视频动作质量评分器（仅特征评分，支持加权DTW）
    """
    
    def __init__(self, 
                template_video: str = None,
                test_video: str = None,
                features_dir: str = 'result/features',
                video_dir: str = 'video_origin/data_video/use',
                weight: dict = None):
        
        self.template_video = template_video
        self.test_video = test_video
        self.features_dir = features_dir
        self.video_dir = video_dir
        
        self.weight = weight if weight else {"fea": 0.7, "point": 0.3}
        
        self.angle_weights = [1] * 24
        self.center_weights = [1] * 8
        self.orientation_weight = [1]
        self.feet_distance_weight = [1]
        self.phase_weights = [1] * 4
        self.feature_weights = (self.angle_weights + self.center_weights + 
                                self.orientation_weight + self.feet_distance_weight + 
                                self.phase_weights)
        
        self.feat_score = None
        self.frame_scores = None
        self.path = None
        self.q_mean_list = None
        self.norm_test = None
        self.norm_template = None
        
        self.trained_weights = None
        self.trained_distance = None
        self.trained_path = None
        self.training_history = None
        
        self.template_video_path = None
        self.test_video_path = None
        
        if template_video and test_video:
            self.set_videos(template_video, test_video)
    
    def set_videos(self, template_video: str, test_video: str):
        self.template_video = template_video
        self.test_video = test_video
        self.template_video_path = os.path.join(self.video_dir, template_video)
        self.test_video_path = os.path.join(self.video_dir, test_video)
    
    def dtw(self, seq1, seq2, dist_func):
        n1, n2 = len(seq1), len(seq2)
        
        cost_matrix = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                cost_matrix[i, j] = dist_func(seq1[i], seq2[j])
        
        dtw_matrix = np.full((n1 + 1, n2 + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                cost = cost_matrix[i-1, j-1]
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                              dtw_matrix[i, j-1],
                                              dtw_matrix[i-1, j-1])
        
        path = []
        i, j = n1, n2
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            min_prev = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            if dtw_matrix[i-1, j-1] == min_prev:
                i, j = i-1, j-1
            elif dtw_matrix[i-1, j] == min_prev:
                i = i-1
            else:
                j = j-1
        
        path.reverse()
        return dtw_matrix[n1, n2], path
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(features)
        for i in range(features.shape[1]):
            col = features[:, i]
            min_val = np.min(col)
            max_val = np.max(col)
            if max_val > min_val:
                normalized[:, i] = (col - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = col
        return normalized
    
    def calculate_frame_score(self, test_feat: np.ndarray, template_feat: np.ndarray, t: float = 0.10) -> float:
        f_weights = np.array(self.feature_weights) / np.sum(self.feature_weights)
        q = np.abs(test_feat - template_feat)
        q_mean = np.sum(q * f_weights)
        
        if q_mean <= t:
            score = 100.0
        elif q_mean - t <= 0.3:
            exceed = q_mean - t
            score = 100 * np.exp(-2.5 * exceed)
        elif 0.3 < q_mean - t <= 1:
            exceed = q_mean - t
            score = 100 * np.exp(-5 * exceed)
        else:
            score = 0.0
        return score
    
    def calculate_video_score(self, test_features: np.ndarray, template_features: np.ndarray, 
                              use_weighted: bool = False, weights: np.ndarray = None) -> tuple:
        if use_weighted and weights is not None:
            def weighted_euclidean(a, b):
                diff = a - b
                w = np.array(weights).flatten()
                return np.sqrt(np.sum(w * (diff ** 2)))
            distance, path = self.dtw(test_features, template_features, weighted_euclidean)
            print(f"加权DTW对齐完成: 路径长度 = {len(path)}, 距离 = {distance:.4f}")
        else:
            distance, path = self.dtw(test_features, template_features, euclidean)
            print(f"标准DTW对齐完成: 路径长度 = {len(path)}, 距离 = {distance:.4f}")
        
        frame_scores = []
        q_mean_list = []
        
        for test_idx, template_idx in path:
            test_frame = test_features[test_idx]
            template_frame = template_features[template_idx]
            
            q = np.abs(test_frame - template_frame)
            q_mean = np.mean(q)
            q_mean_list.append(q_mean)
            
            score = self.calculate_frame_score(test_frame, template_frame, t=0.10)
            frame_scores.append(score)
        
        final_score = np.mean(frame_scores)
        return final_score, frame_scores, path, q_mean_list, distance
    
    def score_video(self, use_weighted: bool = False, weights: np.ndarray = None) -> tuple:
        template_feat_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.template_video)[0]}_features.npy")
        test_feat_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.test_video)[0]}_features.npy")
        
        if not os.path.exists(template_feat_path) or not os.path.exists(test_feat_path):
            print("错误: 找不到特征文件")
            return None, None, None, None, None
        
        print("="*60)
        print("视频动作质量评分")
        print("="*60)
        
        template_features = np.load(template_feat_path)
        test_features = np.load(test_feat_path)
        
        print(f"\n模板视频帧数: {template_features.shape[0]}")
        print(f"测试视频帧数: {test_features.shape[0]}")
        print(f"特征维度: {test_features.shape[1]}")
        
        norm_template = self.normalize_features(template_features)
        norm_test = self.normalize_features(test_features)
        
        self.feat_score, self.frame_scores, self.path, self.q_mean_list, distance = \
            self.calculate_video_score(norm_test, norm_template, use_weighted, weights)
        
        print(f"\n特征得分: {self.feat_score:.2f}")
        
        self.norm_test = norm_test
        self.norm_template = norm_template
        
        return self.feat_score, self.frame_scores, self.path, self.q_mean_list, distance
    
    def train_weights(self, max_epoch: int = 20, learning_rate: float = 0.5, verbose: bool = True) -> dict:
        """
        训练特征权重
        
        原理：差异小的特征权重增大，差异大的特征权重减小
        """
        self.score_video()
        
        if self.norm_test is None or self.norm_template is None:
            print("数据获取错误")
            return None
        
        test_features = self.norm_test
        template_features = self.norm_template
        features_dim = test_features.shape[1]

        print('\n' + "="*60)
        print("训练开始")
        print("="*60)
        print(f"特征维度: {features_dim}")
        print(f"最大迭代次数: {max_epoch}")
        print(f"学习率: {learning_rate}")
        print("-"*60)
        
        # 初始化权重（均匀分布）
        weights = np.ones(features_dim) / features_dim
        
        history = {
            'weights': [weights.copy()],
            'distances': [],
            'paths': [],
            'path_changes': []
        }
        
        prev_path = None
        
        for epoch in range(max_epoch):
            def weighted_euclidean(a, b):
                diff = a - b
                return np.sqrt(np.sum(weights * (diff ** 2)))
            
            distance, path = self.dtw(test_features, template_features, weighted_euclidean)
            
            if np.isnan(distance) or np.isinf(distance):
                print(f"警告: 迭代 {epoch+1} 距离无效，停止训练")
                break
            
            # 计算路径上每个特征的平均差异
            features_diffs = np.zeros(features_dim)
            for test_idx, temp_idx in path:
                diff = np.abs(test_features[test_idx] - template_features[temp_idx])
                features_diffs += diff
            features_diffs = features_diffs / max(len(path), 1)
            
            # 防止除零
            features_diffs = np.maximum(features_diffs, 1e-8)
            
            # 关键修正：差异越小，权重应该越大
            # 使用倒数作为权重（差异小→倒数大→权重大）
            new_weights = 1.0 / features_diffs
            
            # 归一化
            new_weights = new_weights / np.sum(new_weights)
            
            # 平滑更新（避免剧烈变化）
            alpha = 0.3  # 平滑系数，新权重占比
            weights = (1 - alpha) * weights + alpha * new_weights
            weights = weights / np.sum(weights)  # 再次归一化
            
            # 计算路径变化率
            path_change = 1.0
            if prev_path is not None:
                set_current = set((p[0], p[1]) for p in path)
                set_prev = set((p[0], p[1]) for p in prev_path)
                overlap = len(set_current & set_prev)
                union = max(len(set_current), len(set_prev))
                if union > 0:
                    path_change = 1 - (overlap / union)
            
            weight_change = np.max(np.abs(weights - history['weights'][-1]))
            
            # 记录历史
            history['distances'].append(distance)
            history['paths'].append(path)
            history['path_changes'].append(path_change)
            history['weights'].append(weights.copy())
            
            # 打印
            if verbose:
                print(f"迭代 {epoch+1:2d}: 距离={distance:.4f}, 路径变化={path_change:.2%}, "
                      f"权重变化={weight_change:.6f}, 最大权重={np.max(weights):.4f}")
            
            # 收敛判断
            if epoch >= 3:
                path_stable = path_change < 0.05
                weight_stable = weight_change < 0.01
                
                if len(history['distances']) >= 2:
                    dist_change = abs(history['distances'][-1] - history['distances'][-2])
                    dist_change /= (history['distances'][-2] + 1e-6)
                    dist_stable = dist_change < 0.001
                else:
                    dist_stable = False
                
                if (path_stable or weight_stable) and dist_stable:
                    if verbose:
                        print("-"*70)
                        print(f"✓ 训练收敛于第 {epoch+1} 次迭代")
                    break
            
            prev_path = path
        
        # 保存结果
        self.trained_weights = weights
        self.trained_distance = distance
        self.trained_path = path
        self.training_history = history
        
        if verbose:
            print("="*60)
            print("训练完成")
            print(f"最终距离: {distance:.4f}")
            print(f"最终迭代次数: {epoch + 1}")
            print(f"权重统计: 均值={np.mean(weights):.4f}, 最大值={np.max(weights):.4f}, "
                  f"最小值={np.min(weights):.4f}")
            # 显示权重最大的前5个特征
            top5_idx = np.argsort(weights)[-5:][::-1]
            print(f"权重最大的5个特征: {top5_idx}")
            print(f"对应权重: {weights[top5_idx]}")
            print("权重：")
            print(weights)
            print("="*60)
        
        return {
            'weights': weights,
            'distance': distance,
            'path': path,
            'history': history,
            'iterations': epoch + 1
        }
    
    def save_trained_weights(self, save_path: str):
        if self.trained_weights is None:
            print("错误：请先运行 train_weights()")
            return
        np.save(save_path, self.trained_weights)
        print(f"权重已保存到: {save_path}")
    
    def load_trained_weights(self, load_path: str):
        self.trained_weights = np.load(load_path)
        print(f"权重已加载: {load_path}")
        print(f"权重维度: {self.trained_weights.shape}")
    
    def get_weighted_score(self, anomaly_coefficient: float = 0) -> float:
        """使用训练好的权重计算加权评分"""
        if self.trained_weights is None:
            print("错误：请先运行 train_weights()")
            return 0.0
        
        if self.norm_test is None or self.norm_template is None:
            print("错误：请先运行 score_video()")
            return 0.0
        
        def weighted_euclidean(a, b):
            diff = a - b
            return np.sqrt(np.sum(self.trained_weights * (diff ** 2)))
        
        distance, path = self.dtw(self.norm_test, self.norm_template, weighted_euclidean)
        
        frame_scores = []
        for test_idx, template_idx in path:
            test_frame = self.norm_test[test_idx]
            template_frame = self.norm_template[template_idx]
            score = self.calculate_frame_score(test_frame, template_frame, t=0.10)
            frame_scores.append(score)
        
        final_score = np.mean(frame_scores)
        
        if anomaly_coefficient > 0:
            final_score = final_score * (1 - anomaly_coefficient * 0.1)
        
        return final_score


if __name__ == '__main__':
    evaluator = VideoScoreEvaluator(
        template_video='run_man.mp4',
        test_video='run_woman.mp4',
        features_dir='result/features',
        video_dir='video_origin/data_video/use'
    )
    
    # 原始评分
    print("\n" + "="*60)
    print("标准DTW评分")
    print("="*60)
    evaluator.score_video(use_weighted=False)
    original_score = evaluator.feat_score
    print(f"\n标准特征得分: {original_score:.2f}")
    
    # 训练权重
    result = evaluator.train_weights(max_epoch=20, learning_rate=0.1)
    
    # 加权评分
    if result is not None:
        print("\n" + "="*60)
        print("加权DTW评分")
        print("="*60)
        weighted_score = evaluator.get_weighted_score(anomaly_coefficient=0)
        print(f"加权特征得分: {weighted_score:.2f}")
        
        print("\n" + "="*60)
        print("结果对比")
        print("="*60)
        print(f"标准DTW得分: {original_score:.2f}")
        print(f"加权DTW得分: {weighted_score:.2f}")
        print(f"差异: {weighted_score - original_score:.2f}")