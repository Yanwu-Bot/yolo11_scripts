import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'


class VideoScoreEvaluator:
    """
    视频动作质量评分器
    
    用于比较两个视频的动作相似度，支持：
    - 特征向量评分（基于动作特征）
    - 关键点评分（基于骨骼关键点）
    - DTW对齐和可视化
    """
    
    def __init__(self, 
                template_video: str = None,
                test_video: str = None,
                features_dir: str = 'result/features',
                video_dir: str = 'video_origin/data_video/use',
                weight: dict = None):
        """
        初始化评分器
        
        Args:
            template_video: 模板视频文件名
            test_video: 测试视频文件名
            features_dir: 特征文件目录
            video_dir: 视频文件目录
            weight: 评分权重 {'fea': float, 'point': float, 'displacement': float}
        """
        self.template_video = template_video
        self.test_video = test_video
        self.features_dir = features_dir
        self.video_dir = video_dir
        
        # 设置默认权重
        self.weight = weight if weight else {"fea": 0.7, "point": 0.3, "displacement": 0.2}
        
        # 特征权重配置（用于计算帧得分时的加权）
        self.angle_weights = [1] * 24
        self.center_weights = [1] * 8
        self.orientation_weight = [1]
        self.feet_distance_weight = [1]
        self.phase_weights = [1] * 4
        self.feature_weights = (self.angle_weights + self.center_weights + 
                                self.orientation_weight + self.feet_distance_weight + 
                                self.phase_weights)
        
        # 存储结果
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
        
        # 视频路径
        self.template_video_path = None
        self.test_video_path = None
        
        if template_video and test_video:
            self.set_videos(template_video, test_video)
    
    def set_videos(self, template_video: str, test_video: str):
        """设置要比较的视频"""
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
        """设置特征权重"""
        self.angle_weights = [angle_weight] * 24
        self.center_weights = [center_weight] * 8
        self.orientation_weight = [orientation_weight]
        self.feet_distance_weight = [feet_distance_weight]
        self.phase_weights = [phase_weight] * 4
        self.feature_weights = (self.angle_weights + self.center_weights + 
                                self.orientation_weight + self.feet_distance_weight + 
                                self.phase_weights)

    def calculate_frame_score(self, test_feat: np.ndarray, template_feat: np.ndarray, t: float = 0.05,k:float = 4) -> tuple:
        """计算单帧得分（特征已经归一化，直接使用）"""
        f_weights = np.array(self.feature_weights) / np.sum(self.feature_weights)
        q = np.abs(test_feat - template_feat)
        q_mean = np.sum(q * f_weights)
        exceed = q_mean - t
        if q_mean <= t:
            mu = 100.0
        else:
            # 调整k值控制衰减速度，k越大衰减越快
            mu = 100 * np.exp(-k * exceed)
        sigma_squared = 9.0 + 0.016 * mu * (100 - mu)
        # 确保方差不为0
        sigma_squared = max(sigma_squared, 1.0)
        return mu, sigma_squared
    
    def calculate_keypoint_score(self, test_points: np.ndarray, template_points: np.ndarray) -> tuple:
        """
        计算关键点的DTW评分
        test_points和template_points应该是已经从*_normalized_points.npy加载的数据
        """
        print("\n" + "-"*40)
        print("关键点评分")
        print("-"*40)
        # 重塑为适合DTW的形状 (帧数, 34)
        test_flat = test_points.reshape(test_points.shape[0], -1)
        template_flat = template_points.reshape(template_points.shape[0], -1)
        # 计算DTW距离和路径
        distance, path = fastdtw(test_flat, template_flat, dist=euclidean)
        path = np.array(path)
    
        # 计算每对对齐帧的得分和距离
        frame_scores = []
        frame_distances = []
        for test_idx, template_idx in path:
            test_frame = test_flat[test_idx]
            template_frame = template_flat[template_idx]
            dist = np.linalg.norm(test_frame - template_frame)
            frame_distances.append(dist)
            # 将距离转换为得分（阈值根据归一化后的尺度调整）
            if dist <= 50.0:
                score = 100 
            elif 50 < dist < 150:
                score = (1 - ((dist-50) / 100)) * 100
            else:
                score = 0.0
            frame_scores.append(score)
        
        final_score = np.mean(frame_scores)
        return final_score, frame_scores, frame_distances, path
    
    def calculate_displacement_frame_score(self, test_vec: np.ndarray, template_vec: np.ndarray, t: float = 0.025, k: float = 4) -> tuple:
        """计算单帧位移得分（与特征评分相同的逻辑）"""
        q = np.abs(test_vec - template_vec)
        q_mean = np.mean(q)
        exceed = q_mean - t
        if q_mean <= t:
            mu = 100.0
        else:
            mu = 100 * np.exp(-k * exceed)
        sigma_squared = 9.0 + 0.016 * mu * (100 - mu)
        sigma_squared = max(sigma_squared, 1.0)
        return mu, sigma_squared
    
    def calculate_video_score(self, test_features: np.ndarray, template_features: np.ndarray) -> tuple:
        """计算整个视频的得分（特征已经归一化，直接使用）"""
        distance, path = fastdtw(test_features, template_features, dist=euclidean)
        path = np.array(path)
        
        print(f"DTW对齐完成: 路径长度 = {len(path)}")
        
        frame_scores = []
        q_mean_list = []
        mu_list = []
        sigma_squared_list = []
        
        for test_idx, template_idx in path:
            test_frame = test_features[test_idx]
            template_frame = template_features[template_idx]
            
            q = np.abs(test_frame - template_frame)
            q_mean = np.mean(q)
            q_mean_list.append(q_mean)
            
            score, sigma_squared = self.calculate_frame_score(test_frame, template_frame)
            mu_list.append(score)
            sigma_squared_list.append(sigma_squared)
            frame_scores.append(score)
        
        mu_array = np.array(mu_list)
        var_array = np.array(sigma_squared_list)  # 每帧的方差
        precision = 1.0 / var_array  # 精度 = 1/方差
        total_precision = np.sum(precision)
        mu_fuse = np.sum(mu_array * precision) / total_precision
        sigma_fuse_squared = 1.0 / total_precision
        action_std = min(np.std(mu_array), 3.0)  # 限制最大影响
        sigma_video = np.sqrt(sigma_fuse_squared + action_std**2)
        # 3. 95% 置信区间
        lower_bound = mu_fuse - 1.96 * sigma_video
        upper_bound = mu_fuse + 1.96 * sigma_video
        final_score = mu_fuse
        return final_score, frame_scores, path, q_mean_list, lower_bound, upper_bound
    
    def score_video(self) -> tuple:
        """主函数：对测试视频评分"""
        # 构建文件路径
        template_feat_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.template_video)[0]}_features.npy")
        test_feat_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.test_video)[0]}_features.npy")
        template_point_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.template_video)[0]}_normalized_points.npy")
        test_point_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.test_video)[0]}_normalized_points.npy")
        template_vector_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.template_video)[0]}_vector.npy")
        test_vector_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.test_video)[0]}_vector.npy")
        
        # 检查特征文件是否存在
        if not os.path.exists(template_feat_path) or not os.path.exists(test_feat_path):
            print("错误: 找不到特征文件")
            return None, None, None, None, None, None, None, None
        
        print("="*60)
        print("视频动作质量评分")
        print("="*60)
        # 加载特征（特征已经在Feature类中归一化好了）
        self.template_features = np.load(template_feat_path)
        self.test_features = np.load(test_feat_path)
        
        print(f"\n模板视频帧数: {self.template_features.shape[0]}")
        print(f"测试视频帧数: {self.test_features.shape[0]}")
        # 特征评分
        self.feat_score, self.frame_scores, self.path, self.q_mean_list, self.lower_bound, self.upper_bound = \
            self.calculate_video_score(self.test_features, self.template_features)
        
        print(f"\n特征得分: {self.feat_score:.2f}")
        
        # 关键点评分 - 直接使用已归一化的关键点，不再调用normalize_keypoints
        self.point_score = 0.0
        self.point_distances = None
        
        if os.path.exists(template_point_path) and os.path.exists(test_point_path):
            print("\n" + "="*60)
            print("开始计算关键点评分")
            print("="*60)
            
            template_points = np.load(template_point_path)
            test_points = np.load(test_point_path)
            # 确保是 (帧数, 17, 2) 格式，如果包含置信度则只取前两维
            if len(template_points.shape) == 3:
                if template_points.shape[2] == 3:
                    template_points = template_points[:, :, :2]
            if len(test_points.shape) == 3:
                if test_points.shape[2] == 3:
                    test_points = test_points[:, :, :2]
            
            # 直接调用calculate_keypoint_score，不再进行二次归一化
            self.point_score, point_frame_scores, self.point_distances, point_path = \
                self.calculate_keypoint_score(test_points, template_points)
            print(f"\n关键点得分: {self.point_score:.2f}")
        else:
            print("\n警告: 找不到关键点文件，跳过关键点评分")
            if not os.path.exists(template_point_path):
                print(f"  缺失: {template_point_path}")
            if not os.path.exists(test_point_path):
                print(f"  缺失: {test_point_path}")
        
        self.displacement_score = 0.0
        self.displacement_frame_scores = None
        
        if os.path.exists(template_vector_path) and os.path.exists(test_vector_path):
            template_vector = np.load(template_vector_path)
            test_vector = np.load(test_vector_path)
            test_vector_flat = test_vector.reshape(test_vector.shape[0], -1)
            template_vector_flat = template_vector.reshape(template_vector.shape[0], -1)   
            # 复用特征评分的DTW路径，计算位移得分
            displacement_scores = []
            for test_idx, template_idx in self.path:
                test_frame = test_vector_flat[test_idx]
                template_frame = template_vector_flat[template_idx]
                score, _ = self.calculate_displacement_frame_score(test_frame, template_frame)
                displacement_scores.append(score)
            # 直接平均
            self.displacement_score = np.mean(displacement_scores)
            self.displacement_frame_scores = displacement_scores
            print(f"位移得分: {self.displacement_score:.2f}")

        
        return (self.feat_score, self.point_score, self.frame_scores, self.path, 
                self.q_mean_list, self.test_features, self.template_features, self.point_distances)
    
    def plot_qmean_over_time(self, t: float = 0.05):
        """绘制q_mean随时间变化的图"""
        if self.q_mean_list is None:
            print("没有q_mean数据，请先运行 score_video()")
            return
        
        plt.figure(figsize=(14, 6))
        
        x = np.arange(len(self.q_mean_list))
        
        # 绘制q_mean曲线
        plt.plot(x, self.q_mean_list, 'b-', linewidth=2, label='q_mean', alpha=0.8)
        
        # 添加阈值线
        plt.axhline(y=t, color='r', linestyle='--', linewidth=2, label=f'阈值 t={t}')
        
        # 添加均值线
        mean_q = np.mean(self.q_mean_list)
        plt.axhline(y=mean_q, color='g', linestyle='--', linewidth=2, 
                    label=f'均值: {mean_q:.3f}')
        
        # 填充区域
        q_array = np.array(self.q_mean_list)
        plt.fill_between(x, 0, q_array, where=(q_array <= t), 
                        color='green', alpha=0.2, label='≤阈值 (合格)')
        plt.fill_between(x, 0, q_array, where=(q_array > t), 
                        color='red', alpha=0.2, label='>阈值 (差异较大)')
        
        plt.xlabel('对齐对序号', fontsize=12)
        plt.ylabel('q_mean值 (帧间差异)', fontsize=12)
        plt.title('q_mean随时间变化图', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print("\n" + "="*40)
        print("q_mean统计")
        print("="*40)
        print(f"均值: {np.mean(self.q_mean_list):.4f}")
        print(f"标准差: {np.std(self.q_mean_list):.4f}")
        print(f"中位数: {np.median(self.q_mean_list):.4f}")
        print(f"最小值: {min(self.q_mean_list):.4f}")
        print(f"最大值: {max(self.q_mean_list):.4f}")
        
        exceed_ratio = np.mean(q_array > t) * 100
        print(f"\n超过阈值({t})的比例: {exceed_ratio:.2f}%")
    
    def plot_dist_over_time(self):
        """绘制关键点距离随时间变化的图"""
        if self.point_distances is None:
            print("没有关键点距离数据，请先运行 score_video()")
            return
        
        plt.figure(figsize=(14, 6))
        
        x = np.arange(len(self.point_distances))
        
        # 绘制dist曲线
        plt.plot(x, self.point_distances, 'r-', linewidth=2, label='关键点距离', alpha=0.8)
        
        # 添加均值线
        mean_dist = np.mean(self.point_distances)
        plt.axhline(y=mean_dist, color='g', linestyle='--', linewidth=2, 
                    label=f'均值: {mean_dist:.3f}')
        
        # 添加阈值线
        plt.axhline(y=3.0, color='orange', linestyle=':', linewidth=1.5, 
                    label='参考阈值: 3.0')
        
        plt.xlabel('对齐对序号', fontsize=12)
        plt.ylabel('欧氏距离', fontsize=12)
        plt.title('关键点距离随时间变化图', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print("\n" + "="*40)
        print("关键点距离统计")
        print("="*40)
        print(f"均值: {np.mean(self.point_distances):.4f}")
        print(f"标准差: {np.std(self.point_distances):.4f}")
        print(f"中位数: {np.median(self.point_distances):.4f}")
        print(f"最小值: {min(self.point_distances):.4f}")
        print(f"最大值: {max(self.point_distances):.4f}")
    
    def plot_displacement_over_time(self):
        """绘制位移得分随时间变化的图"""
        if self.displacement_frame_scores is None:
            print("没有位移得分数据，请先运行 score_video()")
            return
        
        plt.figure(figsize=(14, 6))
        
        x = np.arange(len(self.displacement_frame_scores))
        
        plt.plot(x, self.displacement_frame_scores, 'g-', linewidth=2, label='位移帧得分', alpha=0.8)
        
        mean_score = np.mean(self.displacement_frame_scores)
        plt.axhline(y=mean_score, color='orange', linestyle='--', linewidth=2, 
                    label=f'均值: {mean_score:.2f}')
        
        plt.xlabel('对齐对序号', fontsize=12)
        plt.ylabel('位移得分', fontsize=12)
        plt.title('位移向量得分随时间变化图', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*40)
        print("位移得分统计")
        print("="*40)
        print(f"均值: {np.mean(self.displacement_frame_scores):.2f}")
        print(f"标准差: {np.std(self.displacement_frame_scores):.2f}")
        print(f"中位数: {np.median(self.displacement_frame_scores):.2f}")
        print(f"最小值: {min(self.displacement_frame_scores):.2f}")
        print(f"最大值: {max(self.displacement_frame_scores):.2f}")
    
    def visualize_alignment_path(self):
        """可视化DTW对齐路径"""
        if self.path is None or self.test_features is None or self.template_features is None:
            print("没有对齐数据，请先运行 score_video()")
            return
        
        path = np.array(self.path)
        test_len = len(self.test_features)
        template_len = len(self.template_features)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：对齐路径
        path_test = path[:, 0]
        path_template = path[:, 1]
        
        cost_matrix = np.full((template_len, test_len), np.nan)
        if self.frame_scores is not None:
            for i, (test_idx, template_idx) in enumerate(path):
                cost_matrix[template_idx, test_idx] = self.frame_scores[i]
        
        im = ax1.imshow(cost_matrix, cmap='RdYlGn', aspect='auto', origin='lower',
                        extent=[0, test_len, 0, template_len], vmin=0, vmax=100, alpha=0.8)
        
        ax1.plot(path_test, path_template, 'w-', linewidth=2.5, alpha=0.9, label='DTW对齐路径')
        ax1.plot(path_test, path_template, 'k.', markersize=2, alpha=0.5)
        
        plt.colorbar(im, ax=ax1, label='帧得分')
        ax1.set_xlabel('测试视频帧索引')
        ax1.set_ylabel('模板视频帧索引')
        ax1.set_title('DTW对齐路径')
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        
        # 右图：得分分布
        if self.frame_scores is not None:
            x = np.arange(len(self.frame_scores))
            ax2.plot(x, self.frame_scores, 'b-', linewidth=2, label='帧得分')
            ax2.fill_between(x, self.frame_scores, 0, alpha=0.2, color='blue')
            
            mean_score = np.mean(self.frame_scores)
            ax2.axhline(y=mean_score, color='r', linestyle='--', linewidth=1.5, 
                        label=f'平均分: {mean_score:.2f}')
            
            ax2.set_xlabel('对齐对序号')
            ax2.set_ylabel('帧得分')
            ax2.set_title('帧得分分布')
            ax2.set_ylim([0, 105])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'DTW对齐分析 (测试:{test_len}帧, 模板:{template_len}帧)')
        plt.tight_layout()
        plt.show()
    
    def visualize_aligned_frames(self, pair_index: int):
        """可视化指定对齐对的两帧画面"""
        if self.path is None:
            print("没有对齐数据，请先运行 score_video()")
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
        """获取综合得分"""
        if self.feat_score is None:
            print("请先运行 score_video()")
            return 0.0
        
        point_score = self.point_score if self.point_score is not None else 0.0
        displacement_score = self.displacement_score if self.displacement_score is not None else 0.0
        combined_score = (self.feat_score * self.weight['fea'] + 
                         point_score * self.weight['point'] +
                         displacement_score * self.weight['displacement'])
        return combined_score
    
    def print_summary(self):
        """打印评分摘要"""
        if self.feat_score is None:
            print("请先运行 score_video()")
            return
        
        print(f"\n{'='*60}")
        print(f"评分摘要")
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
        print(f"综合得分: {self.get_combined_score():.2f}")
        print(f"区间范围：{(self.lower_bound*self.weight['fea']+self.point_score*self.weight['point']+self.displacement_score*self.weight['displacement']):.2f}--{(self.upper_bound*self.weight['fea']+self.point_score*self.weight['point']+self.displacement_score*self.weight['displacement']):.2f}")
        print(f"评分权重: fea={self.weight['fea']}, point={self.weight['point']}, displacement={self.weight['displacement']}")
        print(f"{'='*60}")


if __name__ == '__main__':
    # 创建评分器实例
    evaluator = VideoScoreEvaluator(
        template_video='run_man.mp4',
        test_video='run_woman.mp4',
        features_dir='result/features',
        video_dir='video_origin/data_video/use',
        weight={"fea": 0.5, "point": 0.3, "displacement": 0.2}
    )
    # 运行评分
    evaluator.score_video()
    # 可视化结果
    evaluator.plot_qmean_over_time(t=0.05)
    evaluator.plot_dist_over_time()
    evaluator.plot_displacement_over_time()
    evaluator.visualize_alignment_path()
    # 可视化指定对齐帧
    VIEW_FRAME = 100
    if evaluator.frame_scores and VIEW_FRAME < len(evaluator.frame_scores):
        evaluator.visualize_aligned_frames(VIEW_FRAME)
    # 打印摘要
    evaluator.print_summary()