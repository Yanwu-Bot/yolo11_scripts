import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
matplotlib.use('TkAgg')  # 使用非交互式后端

class VideoScoreEvaluator:
    """
    视频动作质量评分器
    用于比较两个视频的动作相似度，支持：
    - 特征评分（基于动作特征）
    - 关键点评分（基于骨骼关键点）
    - 向量评分
    - DTW对齐和可视化
    """
    def __init__(self, 
                template_video: str = None,
                test_video: str = None,
                features_dir: str = 'result/features',
                video_dir: str = 'video_origin/data_video/use',
                weight: dict = None,
                output_dir: str = 'result/plots'):
        """
        初始化评分器
        Args:
            template_video: 模板视频文件名
            test_video: 测试视频文件名
            features_dir: 特征文件目录
            video_dir: 视频文件目录
            weight: 评分权重 {'fea': float, 'point': float, 'displacement': float}
            output_dir: 图片输出目录
        """
        self.template_video = template_video
        self.test_video = test_video
        self.features_dir = features_dir
        self.video_dir = video_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        # 设置默认权重
        self.weight = weight if weight else {"fea": 0.5, "point": 0.3, "displacement": 0.2}
        
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
        self.lower_bound = None
        self.upper_bound = None
        
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
    
    #设置特征的权重
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
    #计算单帧特征得分
    def calculate_frame_score(self, test_feat: np.ndarray, template_feat: np.ndarray, t: float = 0.05, k: float = 4) -> tuple:
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
    #计算单帧关键点得分
    def calculate_keypoint_frame_score(self, test_points: np.ndarray, template_points: np.ndarray) -> tuple:
        # 计算欧氏距离
        dist = np.linalg.norm(test_points - template_points)
        # 将距离转换为得分
        if dist <= 50.0:
            score = 100 
        elif 50 < dist < 150:
            score = (1 - ((dist-50) / 100)) * 100
        else:
            score = 0.0
        return score, dist   
    #计算单帧向量得分
    def calculate_displacement_frame_score(self, test_vec: np.ndarray, template_vec: np.ndarray, t: float = 0.025, k: float = 4) -> tuple:
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
        """计算整个视频特征得分和DTW对齐"""
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
        var_array = np.array(sigma_squared_list)
        precision = 1.0 / var_array
        total_precision = np.sum(precision)
        mu_fuse = np.sum(mu_array * precision) / total_precision
        sigma_fuse_squared = 1.0 / total_precision
        action_std = min(np.std(mu_array), 3.0)
        sigma_video = np.sqrt(sigma_fuse_squared + action_std**2)
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
        # 加载特征
        self.template_features = np.load(template_feat_path)
        self.test_features = np.load(test_feat_path)
        
        print(f"\n模板视频帧数: {self.template_features.shape[0]}")
        print(f"测试视频帧数: {self.test_features.shape[0]}")
        # 特征评分（同时获取DTW路径和帧得分）
        self.feat_score, self.frame_scores, self.path, self.q_mean_list, self.lower_bound, self.upper_bound = \
            self.calculate_video_score(self.test_features, self.template_features)
        # 加载关键点数据
        template_points = None
        test_points = None
        if os.path.exists(template_point_path) and os.path.exists(test_point_path):
            template_points = np.load(template_point_path)
            test_points = np.load(test_point_path)
            # 确保是 (帧数, 17, 2) 格式
            if len(template_points.shape) == 3:
                if template_points.shape[2] == 3:
                    template_points = template_points[:, :, :2]
            if len(test_points.shape) == 3:
                if test_points.shape[2] == 3:
                    test_points = test_points[:, :, :2]
        else:
            print("\n警告: 找不到关键点文件，跳过关键点评分")
        
        # 加载位移向量数据
        template_vector = None
        test_vector = None
        
        if os.path.exists(template_vector_path) and os.path.exists(test_vector_path):
            template_vector = np.load(template_vector_path)
            test_vector = np.load(test_vector_path)
        else:
            print("\n警告: 找不到向量文件，跳过位移得分")
        # 使用相同的DTW路径计算关键点和位移得分
        point_frame_scores = []
        displacement_frame_scores = []
        point_distances = []
        for test_idx, template_idx in self.path:
            # 关键点得分
            if test_points is not None and template_points is not None:
                test_point_frame = test_points[test_idx]
                template_point_frame = template_points[template_idx]
                point_score_frame, point_dist = self.calculate_keypoint_frame_score(test_point_frame, template_point_frame)
                point_frame_scores.append(point_score_frame)
                point_distances.append(point_dist)
            # 位移得分
            if test_vector is not None and template_vector is not None:
                if len(test_vector.shape) == 3:
                    test_vec_frame = test_vector[test_idx].reshape(-1)
                    template_vec_frame = template_vector[template_idx].reshape(-1)
                else:
                    test_vec_frame = test_vector[test_idx]
                    template_vec_frame = template_vector[template_idx]
                displacement_score_frame, _ = self.calculate_displacement_frame_score(test_vec_frame, template_vec_frame)
                displacement_frame_scores.append(displacement_score_frame)
        # 计算最终得分
        if point_frame_scores:
            self.point_score = np.mean(point_frame_scores)
            self.point_distances = point_distances
        else:
            self.point_score = 0.0
        
        if displacement_frame_scores:
            self.displacement_score = np.mean(displacement_frame_scores)
            self.displacement_frame_scores = displacement_frame_scores
        else:
            self.displacement_score = 0.0
        
        return (self.feat_score, self.point_score, self.frame_scores, self.path, 
                self.q_mean_list, self.test_features, self.template_features, self.point_distances)
    
    def plot_all_analysis(self, t: float = 0.05, save_path: str = None):
        """
        综合分析图：将四个分析图合并在一个2x2画布中
        Args:
            t: q_mean的阈值
            save_path: 保存路径，如果为None则显示图片
        """
        if self.q_mean_list is None or self.path is None:
            print("没有分析数据，请先运行 score_video()")
            return
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # ========== 子图1: q_mean随时间变化 ==========
        ax1 = axes[0, 0]
        x1 = np.arange(len(self.q_mean_list))
        ax1.plot(x1, self.q_mean_list, 'b-', linewidth=2, label='q_mean', alpha=0.8)
        ax1.axhline(y=t, color='r', linestyle='--', linewidth=2, label=f'阈值 t={t}')
        mean_q = np.mean(self.q_mean_list)
        ax1.axhline(y=mean_q, color='g', linestyle='--', linewidth=2, label=f'均值: {mean_q:.3f}')
        
        q_array = np.array(self.q_mean_list)
        ax1.fill_between(x1, 0, q_array, where=(q_array <= t), 
                        color='green', alpha=0.2, label='≤阈值 (合格)')
        ax1.fill_between(x1, 0, q_array, where=(q_array > t), 
                        color='red', alpha=0.2, label='>阈值 (差异较大)')
        
        ax1.set_xlabel('对齐对序号', fontsize=12)
        ax1.set_ylabel('q_mean值 (帧间差异)', fontsize=12)
        ax1.set_title('1. q_mean随时间变化图', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息到子图1
        exceed_ratio = np.mean(q_array > t) * 100
        ax1.text(0.98, 0.95, f'超过阈值: {exceed_ratio:.1f}%', 
                transform=ax1.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # ========== 子图2: 关键点距离随时间变化 ==========
        ax2 = axes[0, 1]
        if self.point_distances is not None:
            x2 = np.arange(len(self.point_distances))
            ax2.plot(x2, self.point_distances, 'r-', linewidth=2, label='关键点距离', alpha=0.8)
            mean_dist = np.mean(self.point_distances)
            ax2.axhline(y=mean_dist, color='g', linestyle='--', linewidth=2, label=f'均值: {mean_dist:.3f}')
            ax2.axhline(y=3.0, color='orange', linestyle=':', linewidth=1.5, label='参考阈值: 3.0')
            ax2.set_title('2. 关键点距离随时间变化图', fontsize=14, fontweight='bold')
            
            # 添加统计信息
            ax2.text(0.98, 0.95, f'均值: {mean_dist:.3f}\n标准差: {np.std(self.point_distances):.3f}', 
                    transform=ax2.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, '无关键点数据', ha='center', va='center', fontsize=16)
            ax2.set_title('2. 关键点距离随时间变化图 (无数据)', fontsize=14, fontweight='bold')
        
        ax2.set_xlabel('对齐对序号', fontsize=12)
        ax2.set_ylabel('欧氏距离', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # ========== 子图3: 位移得分随时间变化 ==========
        ax3 = axes[1, 0]
        if self.displacement_frame_scores is not None:
            x3 = np.arange(len(self.displacement_frame_scores))
            ax3.plot(x3, self.displacement_frame_scores, 'g-', linewidth=2, label='位移帧得分', alpha=0.8)
            mean_score = np.mean(self.displacement_frame_scores)
            ax3.axhline(y=mean_score, color='orange', linestyle='--', linewidth=2, label=f'均值: {mean_score:.2f}')
            ax3.set_title('3. 位移向量得分随时间变化图', fontsize=14, fontweight='bold')
            
            # 添加统计信息
            ax3.text(0.98, 0.95, f'均值: {mean_score:.2f}\n标准差: {np.std(self.displacement_frame_scores):.2f}', 
                    transform=ax3.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, '无位移数据', ha='center', va='center', fontsize=16)
            ax3.set_title('3. 位移向量得分随时间变化图 (无数据)', fontsize=14, fontweight='bold')
        
        ax3.set_xlabel('对齐对序号', fontsize=12)
        ax3.set_ylabel('位移得分', fontsize=12)
        ax3.set_ylim([0, 105])
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # ========== 子图4: DTW对齐路径 ==========
        ax4 = axes[1, 1]
        path = np.array(self.path)
        test_len = len(self.test_features)
        template_len = len(self.template_features)
        
        path_test = path[:, 0]
        path_template = path[:, 1]
        
        cost_matrix = np.full((template_len, test_len), np.nan)
        if self.frame_scores is not None:
            for i, (test_idx, template_idx) in enumerate(path):
                cost_matrix[template_idx, test_idx] = self.frame_scores[i]
        
        im = ax4.imshow(cost_matrix, cmap='RdYlGn', aspect='auto', origin='lower',
                        extent=[0, test_len, 0, template_len], vmin=0, vmax=100, alpha=0.8) 
        ax4.plot(path_test, path_template, 'w-', linewidth=2.5, alpha=0.9, label='DTW对齐路径')
        ax4.plot(path_test, path_template, 'k.', markersize=2, alpha=0.5)
        
        plt.colorbar(im, ax=ax4, label='帧得分')
        ax4.set_xlabel('测试视频帧索引', fontsize=12)
        ax4.set_ylabel('模板视频帧索引', fontsize=12)
        ax4.set_title('4. DTW对齐路径', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=10)
        ax4.grid(True, alpha=0.2)
    
        # 添加总体标题
        video_name = f"{os.path.splitext(self.template_video)[0]} vs {os.path.splitext(self.test_video)[0]}"
        fig.suptitle(f'视频动作分析报告 - {video_name}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"综合分析图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        

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
        combined_score = (self.feat_score * self.weight['fea'] + 
                         self.point_score * self.weight['point'] +
                         self.displacement_score * self.weight['displacement'])
        return combined_score
    
    def print_summary(self):
        """打印评分摘要"""
        if self.feat_score is None:
            print("请先运行 score_video()")
            return
        
        print(f"\n{'='*60}")
        print(f"DTW对比结果")
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
        weight={"fea": 0.5, "point": 0.3, "displacement": 0.2},
        output_dir='result/plots'
    )
    # 运行评分
    evaluator.score_video()
    # 生成综合分析图（2x2布局）
    evaluator.plot_all_analysis(t=0.05, save_path=None)
    # 可视化指定对齐帧（可选）
    VIEW_FRAME = 100
    if evaluator.frame_scores and VIEW_FRAME < len(evaluator.frame_scores):
        evaluator.visualize_aligned_frames(VIEW_FRAME)
    # 打印摘要
    evaluator.print_summary()