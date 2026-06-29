#可用
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from matplotlib import rcParams
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
rcParams['font.family'] = 'SimHei'
matplotlib.use('TkAgg')
WINDOWSIZE = 8 #窗口大小
MODEL = 'result/GCN/model/best_8_2.pth'

class EADM(nn.Module):
    """Energy-based Attention-guided Drop Module"""
    def __init__(self, drop_ratio=0.2, lambda_=1e-4):
        super().__init__()
        self.drop_ratio = drop_ratio
        self.lambda_ = lambda_
    def forward(self, x):
        B, C, T, V = x.shape
        N = T * V
        x_flat = x.view(B, C, N)
        mu = x_flat.mean(dim=2, keepdim=True)
        var = x_flat.var(dim=2, keepdim=True, unbiased=False)
        diff = x_flat - mu
        energy = 4 * (var + self.lambda_) / (diff**2 + 2*var + 2*self.lambda_)
        importance = torch.sigmoid(1.0 / (energy + 1e-10))
        k = int(N * self.drop_ratio)
        if k > 0:
            topk_values, topk_indices = torch.topk(importance, k, dim=2, largest=True, sorted=False)
            mask = torch.ones_like(importance)
            mask.scatter_(2, topk_indices, 0.0)
        else:
            mask = torch.ones_like(importance)
        x_masked = x_flat * mask
        keep_ratio = 1.0 - self.drop_ratio
        if keep_ratio > 0:
            x_masked = x_masked * (N / (N * keep_ratio + 1e-8))
        else:
            x_masked = x_masked * 0
        out = x_masked.view(B, C, T, V)
        return out
class COCOGraph:
    def __init__(self, hop_size=2):
        self.num_node = 17
        self.hop_size = hop_size
        self.get_edge()
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)
        self.get_adjacency()
    def get_edge(self):
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_base = [
            (0,5),(0,6),(5,6),(6,8),(8,10),(6,12),(5,7),(7,9),(5,11),(11,12),
            (11,13),(13,15),(12,14),(14,16),(6,10),(5,9),(12,16),(11,15),(5,12),(6,11)
        ]
        self.edge = self_link + neighbor_base
    def get_hop_distance(self, num_node, edge, hop_size):
        A = np.zeros((num_node, num_node))
        for i,j in edge:
            A[j,i]=1; A[i,j]=1
        hop_dis = np.zeros((num_node,num_node))+np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size+1)]
        arrive_mat = (np.stack(transfer_mat)>0)
        for d in range(hop_size,-1,-1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis
    def get_adjacency(self):
        valid_hop = range(0, self.hop_size+1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis==hop]=1
        normalize_adjacency = self.normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis==hop] = normalize_adjacency[self.hop_dis==hop]
        self.A = A
    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        Dn = np.zeros((A.shape[0], A.shape[0]))
        for i in range(A.shape[0]):
            if Dl[i]>0:
                Dn[i,i] = Dl[i]**(-1)
        return np.dot(A, Dn)
class SpatialGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels*s_kernel_size, 1)
    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()
class STGC_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.5):
        super().__init__()
        self.sgc = SpatialGraphConvolution(in_channels, out_channels, A_size[0])
        self.M = nn.Parameter(torch.ones(A_size))
        self.B = nn.Parameter(torch.zeros(A_size))
        self.tgc = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, (t_kernel_size,1), (stride,1),
                    ((t_kernel_size-1)//2,0)),
            nn.BatchNorm2d(out_channels), nn.ReLU())
    def forward(self, x, A):
        return self.tgc(self.sgc(x, A * self.M + self.B))
class ContrastiveEncoder(nn.Module):
    def __init__(self, in_channels=2, t_kernel_size=9, hop_size=2, output_dim=128):
        super().__init__()
        graph = COCOGraph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()
        self.bn = nn.BatchNorm1d(in_channels * graph.num_node)
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size, dropout=0.1)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size, dropout=0.1)
        # self.eadm = EADM(drop_ratio=0.2)
        self.projection = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0,3,1,2).contiguous().view(N, V*C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0,2,3,1).contiguous()
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        # x = self.eadm(x)
        x = F.adaptive_avg_pool2d(x, (1,1)).view(N, -1)
        x = self.projection(x)
        return F.normalize(x, dim=1)
#评分类
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
        #角度特征
        self.angle_weights = [1.2] * 24
        #四肢中心点位置
        self.center_weights = [1] * 8
        #身体前倾角度
        self.orientation_weight = [1.1]
        #两脚间距
        self.feet_distance_weight = [1.2]
        #四肢相对躯干位移
        self.phase_weights = [0.9] * 4
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

        self.window = WINDOWSIZE
        self.mu =None
        self.sigma = None

        self.kps1 = []
        self.kps2 = []
        self.window_sim_scores = None  # 保存每个窗口的ST-GCN相似度得分

    def set_videos(self, template_video: str, test_video: str):
        self.template_video = template_video
        self.test_video = test_video
        self.template_video_path = os.path.join(self.video_dir, template_video)
        self.test_video_path = os.path.join(self.video_dir, test_video)
    
    def calculate_frame_score(self, test_feat: np.ndarray, template_feat: np.ndarray, t: float = 0.05, k: float = 4) -> float:
        f_weights = np.array(self.feature_weights) / np.sum(self.feature_weights)
        q = np.abs(test_feat - template_feat)
        q_mean = np.sum(q * f_weights)
        exceed = q_mean - t
        if exceed < 0:
            score = 100.0
        else:
            score = 100 * np.exp(-k * exceed)
        return score

    def calculate_keypoint_frame_score(self, test_points: np.ndarray, template_points: np.ndarray, threshold=125 , k = 4) -> tuple:
        body_indices = list(range(5, 17))  # 14个关键点
        test_body = test_points[body_indices]
        template_body = template_points[body_indices]
        dist = np.linalg.norm(test_body - template_body)
        exceed = dist - threshold
        if exceed < 0:
            score = 100
        else:
            score = 100 * np.exp(-k * exceed)
        return score, dist

    def calculate_displacement_frame_score(self, test_vec: np.ndarray, template_vec: np.ndarray, t: float = 0.025, k: float = 4) -> float:
        q = np.abs(test_vec - template_vec)
        q_mean = np.mean(q)
        exceed = q_mean - t
        if exceed < 0:
            mu = 100.0
        else:
            mu = 100 * np.exp(-k * exceed)
        return mu

    def calculate_video_score(self, test_features: np.ndarray, template_features: np.ndarray) -> np.ndarray:
        distance, path = fastdtw(test_features, template_features, dist=euclidean)
        path = np.array(path)
        self.path = path
        print(f"DTW对齐完成: 路径长度 = {len(path)}")
        return path

    def compute_pairwise_scores(self, path: np.ndarray, use_window=True):
        #导入特征文件
        template_feat_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.template_video)[0]}_features.npy")
        test_feat_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.test_video)[0]}_features.npy")
        template_features = np.load(template_feat_path)
        test_features = np.load(test_feat_path)
        #导入关键点文件
        template_point_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.template_video)[0]}_normalized_points.npy")
        test_point_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.test_video)[0]}_normalized_points.npy")
        #关键点重新排序
        if os.path.exists(template_point_path) and os.path.exists(test_point_path):
            template_points = np.load(template_point_path)
            test_points = np.load(test_point_path)
            self.kps1 = template_points.tolist()
            self.kps2 = test_points.tolist()
            if len(template_points.shape) == 3 and template_points.shape[2] == 3:
                template_points = template_points[:, :, :2]
            if len(test_points.shape) == 3 and test_points.shape[2] == 3:
                test_points = test_points[:, :, :2]
            # 利用 path 生成对齐后的关键点序列
            self.kps1 = [self.kps1[idx] for idx in path[:, 1]]  
            self.kps2 = [self.kps2[idx] for idx in path[:, 0]]  
        else:
            print("\nPOINT ERR")
        #导入向量文件
        template_vector_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.template_video)[0]}_vector.npy")
        test_vector_path = os.path.join(self.features_dir, 
                                        f"{os.path.splitext(self.test_video)[0]}_vector.npy")

        if os.path.exists(template_vector_path) and os.path.exists(test_vector_path):
            template_vector = np.load(template_vector_path)
            test_vector = np.load(test_vector_path)
        else:
            print("\nVEC ERR")

        # 逐对计算得分
        frame_scores = []           # 特征得分
        point_frame_scores = []
        displacement_frame_scores = []
        point_distances = []

        #计算各个特征各个对应得分
        for test_idx, template_idx in path:
            test_frame = test_features[test_idx]
            template_frame = template_features[template_idx]
            score = self.calculate_frame_score(test_frame, template_frame)
            frame_scores.append(score)

            test_point_frame = test_points[test_idx]
            template_point_frame = template_points[template_idx]
            point_score_frame, point_dist = self.calculate_keypoint_frame_score(
                test_point_frame, template_point_frame)
            point_frame_scores.append(point_score_frame)
            point_distances.append(point_dist)

            if len(test_vector.shape) == 3:
                test_vec_frame = test_vector[test_idx].reshape(-1)
                template_vec_frame = template_vector[template_idx].reshape(-1)
            else:
                test_vec_frame = test_vector[test_idx]
                template_vec_frame = template_vector[template_idx]
            displacement_score = self.calculate_displacement_frame_score(
                test_vec_frame, template_vec_frame)
            displacement_frame_scores.append(displacement_score)

        self.frame_scores = frame_scores
        self.point_frame_scores = point_frame_scores
        self.point_distances = point_distances
        self.displacement_frame_scores = displacement_frame_scores

        # 窗口聚合
        if use_window and self.window > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ContrastiveEncoder(output_dim=128).to(device)
            model.load_state_dict(torch.load(MODEL, map_location=device))
            scores = compare_videos(self.kps1, self.kps2, model, device, self.window) #同序列窗口进行相似度比对
            self.window_sim_scores = scores # 记录窗口相似度得分
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
            self.mu_sigma(window_fea_scores, window_point_scores, window_disp_scores)

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
            scores = np.array(scores)
            self.window_sim_scores = np.array(self.window_sim_scores)/100 #缩放

            #最终得分计算，结合相似度
            alpha = 0.25  # 可调
            scores = scores * (1 - alpha * (1 - self.window_sim_scores))
            scores = np.delete(scores, -1)
            self.mu = scores.mean()

            diff = scores - self.mu
            total = np.sum(diff ** 2)
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
        combined_score = (self.feat_score * self.weight['fea'] + 
                         self.point_score * self.weight['point'] +
                         self.displacement_score * self.weight['displacement'])
        return combined_score
    
    def print_summary(self):
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
        print(f"综合得分（窗口加权前）: {self.get_combined_score():.2f}")
        print(f"综合得分（窗口加权后）: {self.mu:.2f}")
        print(f"得分区间: {self.lower_bound:.2f} --- {self.upper_bound:.2f}")
        print(f"评分权重: fea={self.weight['fea']}, point={self.weight['point']}, displacement={self.weight['displacement']}")
        print(f"{'='*60}")
        print(self.window_sim_scores)

#  窗口相似度对比
def compare_videos(kps1, kps2, model, device, window_size):
    model.eval()
    arr1 = np.array(kps1, dtype=np.float32)  # (L, 17, 2)
    arr2 = np.array(kps2, dtype=np.float32)  # (L, 17, 2)
    L = len(arr1)  
    scores = []

    for start in range(0, L, window_size):
        end = min(start + window_size, L)
        win_len = end - start
        win1 = np.zeros((window_size, 17, 2), dtype=np.float32) #创建全零数组
        win2 = np.zeros((window_size, 17, 2), dtype=np.float32)
        win1[:win_len] = arr1[start:end] #填充数组，长度不够补0
        win2[:win_len] = arr2[start:end]

        def get_feature(arr_win):
            tensor = torch.FloatTensor(arr_win).permute(2,0,1).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(tensor)
            return feat.cpu().numpy()[0]

        feat1 = get_feature(win1) #对每个窗口取特征
        feat2 = get_feature(win2)
        sim = np.dot(feat1, feat2)
        score = (sim + 1) / 2 * 100
        scores.append(score)
    return scores

# 窗口可视化函数
def visualize_window(evaluator, window_idx):
    if evaluator.window_sim_scores is None or window_idx >= len(evaluator.window_sim_scores):
        print(f"窗口索引 {window_idx} 超出范围 (共 {len(evaluator.window_sim_scores) if evaluator.window_sim_scores else 0} 个窗口)")
        return
    if evaluator.path is None:
        print("没有路径数据")
        return

    window_size = evaluator.window
    L = len(evaluator.path)
    start = window_idx * window_size
    end = min(start + window_size, L)
    window_path = evaluator.path[start:end]  # (实际帧数, 2)

    # 读取对应帧
    cap_t = cv2.VideoCapture(evaluator.template_video_path)
    cap_test = cv2.VideoCapture(evaluator.test_video_path)

    n_frames = end - start
    fig, axes = plt.subplots(2, n_frames, figsize=(3*n_frames, 6))
    if n_frames == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(f'Window {window_idx} (ST-GCN Similarity: {evaluator.window_sim_scores[window_idx]:.2f})',
                    fontsize=14)

    for i in range(n_frames):
        t_idx = int(window_path[i, 1])   # 模板帧索引
        test_idx = int(window_path[i, 0]) # 测试帧索引

        # 模板帧
        cap_t.set(cv2.CAP_PROP_POS_FRAMES, t_idx)
        ret_t, frame_t = cap_t.read()
        if ret_t:
            frame_t_rgb = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(frame_t_rgb)
            axes[0, i].set_title(f'Temp {t_idx}')
        else:
            axes[0, i].text(0.5, 0.5, 'missing', ha='center')
        axes[0, i].axis('off')

        # 测试帧
        cap_test.set(cv2.CAP_PROP_POS_FRAMES, test_idx)
        ret_te, frame_te = cap_test.read()
        if ret_te:
            frame_te_rgb = cv2.cvtColor(frame_te, cv2.COLOR_BGR2RGB)
            axes[1, i].imshow(frame_te_rgb)
            axes[1, i].set_title(f'Test {test_idx}')
        else:
            axes[1, i].text(0.5, 0.5, 'missing', ha='center')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
    cap_t.release()
    cap_test.release()

if __name__ == '__main__':
    evaluator = VideoScoreEvaluator(
        template_video='run_5.mp4',
        test_video='run_8.mp4',
        features_dir='result/features',
        video_dir='D:/Dataset/sprint/Whole',
        weight={"fea": 0.5, "point": 0.3, "displacement": 0.2},
        output_dir='result/plots'
    )
    evaluator.score_video()
    VIEW_FRAME = 200
    if evaluator.frame_scores and VIEW_FRAME < len(evaluator.frame_scores):
        evaluator.visualize_aligned_frames(VIEW_FRAME)
    # 可视化第i个窗口
    visualize_window(evaluator, window_idx=6)
    evaluator.print_summary()