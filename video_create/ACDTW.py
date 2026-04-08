import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端
import os
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
    - ACDTW（自适应约束动态时间规整）
    """
    
    def __init__(self, 
                template_video: str = None,
                test_video: str = None,
                features_dir: str = 'result/features',
                video_dir: str = 'video_origin/data_video/use',
                weight: dict = None,
                use_acdtw: bool = True):
        """
        初始化评分器
        
        Args:
            template_video: 模板视频文件名
            test_video: 测试视频文件名
            features_dir: 特征文件目录
            video_dir: 视频文件目录
            weight: 评分权重 {'fea': float, 'point': float}
            use_acdtw: 是否使用ACDTW（True=ACDTW, False=标准DTW）
        """
        self.template_video = template_video
        self.test_video = test_video
        self.features_dir = features_dir
        self.video_dir = video_dir
        self.use_acdtw = use_acdtw
        
        # 设置默认权重
        self.weight = weight if weight else {"fea": 0.7, "point": 0.3}
        
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
        self.frame_scores = None
        self.path = None
        self.q_mean_list = None
        self.point_distances = None
        self.test_features = None
        self.template_features = None
        
        # ACDTW 额外存储
        self.S_core_matrix = None
        self.MED_matrix = None
        self.P_matrix = None
        self.Q_matrix = None
        
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
    
    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        对关键点进行以人体为中心的归一化
        
        Args:
            keypoints: 形状为 (帧数, 17, 2) 的关键点数组
            
        Returns:
            归一化后的关键点数组，形状相同
        """
        normalized = np.zeros_like(keypoints, dtype=np.float32)
        for i in range(keypoints.shape[0]):
            frame_kps = keypoints[i]
            valid_mask = ~((frame_kps[:, 0] == 0) & (frame_kps[:, 1] == 0))
            
            if np.sum(valid_mask) < 4:
                normalized[i] = frame_kps
                continue
            
            left_hip = frame_kps[11]
            right_hip = frame_kps[12]
            
            # 确定参考点（髋部中心）
            if (left_hip[0] == 0 and left_hip[1] == 0) or (right_hip[0] == 0 and right_hip[1] == 0):
                reference = frame_kps[1]
                if reference[0] == 0 and reference[1] == 0:
                    normalized[i] = frame_kps
                    continue
            else:
                reference = (left_hip + right_hip) / 2
            
            # 确定缩放因子
            left_shoulder = frame_kps[5]
            right_shoulder = frame_kps[6]
            if (left_shoulder[0] != 0 or left_shoulder[1] != 0) and (right_shoulder[0] != 0 or right_shoulder[1] != 0):
                shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                scale = shoulder_width
            else:
                neck = frame_kps[1]
                if neck[0] != 0 or neck[1] != 0:
                    scale = np.linalg.norm(neck - reference)
                else:
                    scale = 1.0
            
            if scale < 1e-6:
                scale = 1.0
            
            # 归一化关键点
            for j in range(17):
                if valid_mask[j]:
                    normalized[i, j, 0] = (frame_kps[j, 0] - reference[0]) / scale
                    normalized[i, j, 1] = (frame_kps[j, 1] - reference[1]) / scale
                else:
                    normalized[i, j] = [0, 0]
        
        return normalized
    
    def calculate_frame_score(self, test_feat: np.ndarray, template_feat: np.ndarray, t: float = 0.05) -> float:
        """计算单帧得分（特征已经归一化，直接使用）"""
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
    
    def calculate_acdtw(self, test_features: np.ndarray, template_features: np.ndarray, t_param: float = 0.05) -> tuple:
        """
        使用 ACDTW（自适应约束动态时间规整）计算视频得分
        论文方法：Multi-dimensional Adaptive Constrained Dynamic Time Warping
        """
        a = len(template_features)  # 模板帧数
        b = len(test_features)       # 测试帧数
        
        print(f"ACDTW计算中: 模板帧数={a}, 测试帧数={b}")
        
        # ========== 步骤1: 计算帧间分数矩阵 S_core 和距离矩阵 MED ==========
        print("步骤1/4: 计算帧间分数矩阵...")
        S_core = np.zeros((a, b))
        MED = np.zeros((a, b))
        
        for i in range(a):
            for j in range(b):
                score = self.calculate_frame_score(
                    test_features[j], template_features[i], t=t_param
                )
                S_core[i, j] = score
                if score > 0:
                    MED[i, j] = 100.0 / score - 1.0
                else:
                    MED[i, j] = 1e6
        
        self.S_core_matrix = S_core
        self.MED_matrix = MED
        
        # ========== 步骤2: ACDTW 动态规划 ==========
        print("步骤2/4: ACDTW 动态规划...")
        
        P = np.zeros((a, b), dtype=int)
        Q = np.zeros((a, b), dtype=int)
        
        ACDTW_dist = np.full((a, b), np.inf)
        ACDTW_dist[0, 0] = MED[0, 0]
        P[0, 0] = 1
        Q[0, 0] = 1
        
        def calc_penalty(a_len, b_len, n_usage):
            return (2 * max(a_len, b_len) / (a_len + b_len)) * n_usage
        
        # 填充第一行
        for i in range(1, a):
            usage = Q[i-1, 0] + 1
            p = calc_penalty(a, b, usage)
            ACDTW_dist[i, 0] = ACDTW_dist[i-1, 0] + p * MED[i, 0]
            P[i, 0] = 1
            Q[i, 0] = Q[i-1, 0] + 1
        
        # 填充第一列
        for j in range(1, b):
            usage = P[0, j-1] + 1
            p = calc_penalty(a, b, usage)
            ACDTW_dist[0, j] = ACDTW_dist[0, j-1] + p * MED[0, j]
            P[0, j] = P[0, j-1] + 1
            Q[0, j] = 1
        
        # 填充剩余矩阵
        for i in range(1, a):
            for j in range(1, b):
                candidates = []
                
                # 对角线移动
                candidates.append((ACDTW_dist[i-1, j-1] + MED[i, j], 'diag', i-1, j-1))
                
                # 垂直移动
                usage = Q[i-1, j] + 1
                p = calc_penalty(a, b, usage)
                candidates.append((ACDTW_dist[i-1, j] + p * MED[i, j], 'vert', i-1, j))
                
                # 水平移动
                usage = P[i, j-1] + 1
                p = calc_penalty(a, b, usage)
                candidates.append((ACDTW_dist[i, j-1] + p * MED[i, j], 'horiz', i, j-1))
                
                best = min(candidates, key=lambda x: x[0])
                ACDTW_dist[i, j] = best[0]
                
                if best[1] == 'diag':
                    P[i, j] = 1
                    Q[i, j] = 1
                elif best[1] == 'vert':
                    P[i, j] = 1
                    Q[i, j] = Q[best[2], best[3]] + 1
                else:
                    P[i, j] = P[best[2], best[3]] + 1
                    Q[i, j] = 1
        
        self.P_matrix = P
        self.Q_matrix = Q
        
        # ========== 步骤3: 回溯得到最优路径 ==========
        print("步骤3/4: 回溯最优路径...")
        path = []
        i, j = a - 1, b - 1
        
        while i > 0 or j > 0:
            path.append((j, i))  # (test_idx, template_idx)
            
            if i > 0 and j > 0:
                diag_cost = ACDTW_dist[i-1, j-1]
                vert_cost = ACDTW_dist[i-1, j]
                horiz_cost = ACDTW_dist[i, j-1]
                
                if diag_cost <= vert_cost and diag_cost <= horiz_cost:
                    i -= 1
                    j -= 1
                elif vert_cost <= horiz_cost:
                    i -= 1
                else:
                    j -= 1
            elif i > 0:
                i -= 1
            else:
                j -= 1
        
        path.append((0, 0))
        path.reverse()
        path = np.array(path)
        
        print(f"ACDTW对齐完成: 路径长度 = {len(path)}")
        
        # ========== 步骤4: 计算最终得分 ==========
        print("步骤4/4: 计算最终得分...")
        frame_scores = []
        q_mean_list = []
        
        for test_idx, template_idx in path:
            frame_scores.append(S_core[template_idx, test_idx])
            
            test_frame = test_features[test_idx]
            template_frame = template_features[template_idx]
            q = np.abs(test_frame - template_frame)
            q_mean_list.append(np.mean(q))
        
        final_score = np.mean(frame_scores)
        
        return final_score, frame_scores, path, q_mean_list
    
    def calculate_video_score(self, test_features: np.ndarray, template_features: np.ndarray) -> tuple:
        """计算整个视频的得分（支持标准DTW和ACDTW）"""
        if self.use_acdtw:
            print("\n" + "="*40)
            print("使用 ACDTW (自适应约束动态时间规整)")
            print("="*40)
            return self.calculate_acdtw(test_features, template_features, t_param=0.05)
        else:
            print("\n" + "="*40)
            print("使用标准 DTW")
            print("="*40)
            from fastdtw import fastdtw
            distance, path = fastdtw(test_features, template_features, dist=euclidean)
            path = np.array(path)
            
            print(f"DTW对齐完成: 路径长度 = {len(path)}")
            
            frame_scores = []
            q_mean_list = []
            
            for test_idx, template_idx in path:
                test_frame = test_features[test_idx]
                template_frame = template_features[template_idx]
                
                q = np.abs(test_frame - template_frame)
                q_mean = np.mean(q)
                q_mean_list.append(q_mean)
                
                score = self.calculate_frame_score(test_frame, template_frame, t=0.05)
                frame_scores.append(score)
            
            final_score = np.mean(frame_scores)
            return final_score, frame_scores, path, q_mean_list
    
    def calculate_keypoint_score(self, test_points: np.ndarray, template_points: np.ndarray) -> tuple:
        """计算关键点的DTW评分"""
        print("\n" + "-"*40)
        print("关键点评分")
        print("-"*40)
        
        # 归一化关键点
        test_norm = self.normalize_keypoints(test_points)
        template_norm = self.normalize_keypoints(template_points)
        
        # 重塑为适合DTW的形状 (帧数, 34)
        test_flat = test_norm.reshape(test_norm.shape[0], -1)
        template_flat = template_norm.reshape(template_norm.shape[0], -1)
        
        # 计算DTW距离和路径
        from fastdtw import fastdtw
        distance, path = fastdtw(test_flat, template_flat, dist=euclidean)
        path = np.array(path)
        
        print(f"关键点DTW距离: {distance:.4f}")
        print(f"对齐路径长度: {len(path)}")
        
        # 计算每对对齐帧的得分和距离
        frame_scores = []
        frame_distances = []
        for test_idx, template_idx in path:
            test_frame = test_flat[test_idx]
            template_frame = template_flat[template_idx]
            dist = np.linalg.norm(test_frame - template_frame)
            frame_distances.append(dist)
            
            # 将距离转换为得分
            if dist <= 35.0:
                score = 100 * (1 - dist / 100)
            else:
                score = 0.0
            frame_scores.append(score)
        
        final_score = np.mean(frame_scores)
        return final_score, frame_scores, frame_distances, path
    
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
        
        # 检查特征文件是否存在
        if not os.path.exists(template_feat_path) or not os.path.exists(test_feat_path):
            print("错误: 找不到特征文件")
            return None, None, None, None, None, None, None, None
        
        print("="*60)
        print("视频动作质量评分")
        print("="*60)
        
        # 加载特征
        self.template_features = np.load(template_feat_path)
        self.test_features = np.load(test_feat_path)
        
        print(f"\n模板视频帧数: {self.template_features.shape[0]}")
        print(f"测试视频帧数: {self.test_features.shape[0]}")
        
        print("\n特征已预先归一化，直接进行对齐...")
        
        # 特征评分
        self.feat_score, self.frame_scores, self.path, self.q_mean_list = \
            self.calculate_video_score(self.test_features, self.template_features)
        
        print(f"\n特征得分: {self.feat_score:.2f}")
        
        # 关键点评分
        self.point_score = 0.0
        self.point_distances = None
        
        if os.path.exists(template_point_path) and os.path.exists(test_point_path):
            print("\n" + "="*60)
            print("开始计算关键点评分")
            print("="*60)
            
            template_points = np.load(template_point_path)
            test_points = np.load(test_point_path)
            
            print(f"模板关键点形状: {template_points.shape}")
            print(f"测试关键点形状: {test_points.shape}")
            
            # 确保是 (帧数, 17, 2) 格式
            if len(template_points.shape) == 3:
                if template_points.shape[2] == 3:
                    template_points = template_points[:, :, :2]
            if len(test_points.shape) == 3:
                if test_points.shape[2] == 3:
                    test_points = test_points[:, :, :2]
            
            self.point_score, point_frame_scores, self.point_distances, point_path = \
                self.calculate_keypoint_score(test_points, template_points)
            print(f"\n关键点得分: {self.point_score:.2f}")
        else:
            print("\n警告: 找不到关键点文件，跳过关键点评分")
        
        return (self.feat_score, self.point_score, self.frame_scores, self.path, 
                self.q_mean_list, self.test_features, self.template_features, self.point_distances)
    
    def plot_qmean_over_time(self, t: float = 0.05):
        """绘制q_mean随时间变化的图"""
        if self.q_mean_list is None:
            print("没有q_mean数据，请先运行 score_video()")
            return
        
        plt.figure(figsize=(14, 6))
        
        x = np.arange(len(self.q_mean_list))
        
        plt.plot(x, self.q_mean_list, 'b-', linewidth=2, label='q_mean', alpha=0.8)
        plt.axhline(y=t, color='r', linestyle='--', linewidth=2, label=f'阈值 t={t}')
        
        mean_q = np.mean(self.q_mean_list)
        plt.axhline(y=mean_q, color='g', linestyle='--', linewidth=2, 
                    label=f'均值: {mean_q:.3f}')
        
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
        
        plt.plot(x, self.point_distances, 'r-', linewidth=2, label='关键点距离', alpha=0.8)
        
        mean_dist = np.mean(self.point_distances)
        plt.axhline(y=mean_dist, color='g', linestyle='--', linewidth=2, 
                    label=f'均值: {mean_dist:.3f}')
        plt.axhline(y=3.0, color='orange', linestyle=':', linewidth=1.5, 
                    label='参考阈值: 3.0')
        
        plt.xlabel('对齐对序号', fontsize=12)
        plt.ylabel('欧氏距离', fontsize=12)
        plt.title('关键点距离随时间变化图', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*40)
        print("关键点距离统计")
        print("="*40)
        print(f"均值: {np.mean(self.point_distances):.4f}")
        print(f"标准差: {np.std(self.point_distances):.4f}")
        print(f"中位数: {np.median(self.point_distances):.4f}")
        print(f"最小值: {min(self.point_distances):.4f}")
        print(f"最大值: {max(self.point_distances):.4f}")
    
    def visualize_alignment_path(self):
        """可视化DTW/ACDTW对齐路径"""
        if self.path is None or self.test_features is None or self.template_features is None:
            print("没有对齐数据，请先运行 score_video()")
            return
        
        path = np.array(self.path)
        test_len = len(self.test_features)
        template_len = len(self.template_features)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        path_test = path[:, 0]
        path_template = path[:, 1]
        
        cost_matrix = np.full((template_len, test_len), np.nan)
        if self.frame_scores is not None:
            for i, (test_idx, template_idx) in enumerate(path):
                cost_matrix[template_idx, test_idx] = self.frame_scores[i]
        
        im = ax1.imshow(cost_matrix, cmap='RdYlGn', aspect='auto', origin='lower',
                        extent=[0, test_len, 0, template_len], vmin=0, vmax=100, alpha=0.8)
        
        ax1.plot(path_test, path_template, 'w-', linewidth=2.5, alpha=0.9, label='对齐路径')
        ax1.plot(path_test, path_template, 'k.', markersize=2, alpha=0.5)
        
        plt.colorbar(im, ax=ax1, label='帧得分')
        ax1.set_xlabel('测试视频帧索引')
        ax1.set_ylabel('模板视频帧索引')
        ax1.set_title(f'{"ACDTW" if self.use_acdtw else "DTW"}对齐路径')
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        
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
        
        plt.suptitle(f'{"ACDTW" if self.use_acdtw else "DTW"}对齐分析 (测试:{test_len}帧, 模板:{template_len}帧)')
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
        
        plt.suptitle(f'对齐对 #{pair_index}')
        plt.tight_layout()
        plt.show()
    
    def get_combined_score(self) -> float:
        """获取综合得分"""
        if self.feat_score is None:
            print("请先运行 score_video()")
            return 0.0
        
        point_score = self.point_score if self.point_score is not None else 0.0
        combined_score = (self.feat_score * self.weight['fea'] + 
                         point_score * self.weight['point'])
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
        print(f"对齐算法: {'ACDTW (自适应约束DTW)' if self.use_acdtw else '标准DTW'}")
        print(f"{'='*60}")
        print(f"特征得分: {self.feat_score:.2f}")
        if self.point_score is not None:
            print(f"关键点得分: {self.point_score:.2f}")
        print(f"{'='*60}")
        print(f"综合得分: {self.get_combined_score():.2f}")
        print(f"评分权重: fea={self.weight['fea']}, point={self.weight['point']}")
        print(f"{'='*60}")


if __name__ == '__main__':
    # 创建评分器实例
    # use_acdtw=True 使用ACDTW，False 使用标准DTW
    evaluator = VideoScoreEvaluator(
        template_video='run_man.mp4',
        test_video='run_wrong.mp4',
        features_dir='result/features',
        video_dir='video_origin/data_video/use',
        weight={"fea": 0.7, "point": 0.3},
        use_acdtw=True
    )
    
    # 运行评分
    evaluator.score_video()
    
    # 可视化结果
    evaluator.plot_qmean_over_time(t=0.05)
    evaluator.plot_dist_over_time()
    evaluator.visualize_alignment_path()
    
    # 可视化指定对齐帧
    VIEW_FRAME = 232
    if evaluator.frame_scores and VIEW_FRAME < len(evaluator.frame_scores):
        evaluator.visualize_aligned_frames(VIEW_FRAME)
    
    # 打印摘要
    evaluator.print_summary()