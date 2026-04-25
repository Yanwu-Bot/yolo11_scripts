import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj_norm):
        # x: (B, N, F_in), adj_norm: (N, N)
        out = torch.matmul(adj_norm, x)  # 聚合邻居信息
        out = self.linear(out)            # 线性变换
        return F.relu(out)


class PoseGCN(nn.Module):
    def __init__(self, num_nodes=17, in_features=2, hidden_dim=16, out_features=16):
        super(PoseGCN, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        
        # 构建邻接矩阵（COCO 17关键点）
        self.register_buffer('adj_norm', self._build_normalized_adjacency())
        # 两层图卷积
        self.gcn1 = GraphConvLayer(in_features, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, in_features)
        # MLP 映射到输出特征
        self.mlp = nn.Sequential(
            nn.Linear(num_nodes * in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_features),
        )
        
    def _build_normalized_adjacency(self):
        """构建归一化的邻接矩阵"""
        # 1. 创建邻接矩阵（带自环）
        adj = torch.eye(self.num_nodes)  # 自环
        
        # COCO 17关键点骨骼连接
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 面部
            (0, 5), (0, 6), (5, 6),           # 肩膀
            (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
            (5, 11), (6, 12), (11, 12),       # 躯干
            (11, 13), (13, 15), (12, 14), (14, 16)  # 腿
        ]
        
        for i, j in connections:
            adj[i, j] = 1
            adj[j, i] = 1
        
        # 2. 归一化: A_hat = D^(-1/2) @ A @ D^(-1/2)
        D = torch.diag(torch.sum(adj, dim=1))
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        return adj_norm
    
    def forward(self, p_pos):
        """
        提取姿态特征
        
        输入: p_pos - (17, 2) 或 (batch, 17, 2) 归一化后的关键点坐标
              注意：输入数据应该是已经经过躯干归一化的坐标
        输出: (16,) 或 (batch, 16) 特征向量（L2归一化）
        """
        # 记录是否需要压缩输出
        squeeze_out = (p_pos.dim() == 2)
        
        # 输入数据已经是归一化好的（躯干归一化），不再做内部归一化
        x = p_pos
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # 图卷积
        x = self.gcn1(x, self.adj_norm)   # (B, N, hidden_dim)
        x = self.gcn2(x, self.adj_norm)   # (B, N, 2)
        
        # 展平
        x = x.reshape(x.size(0), -1)      # (B, N*2)
        
        # MLP 映射
        features = self.mlp(x)             # (B, out_features)
        
        # L2 归一化（便于后续计算余弦距离）
        features = F.normalize(features, p=2, dim=1)
        
        if squeeze_out:
            features = features.squeeze(0)
        
        return features


class GCNFeatureExtractor:
    """GCN特征提取器"""
    
    def __init__(self, device='cpu', weights_path=None):
        """
        初始化特征提取器
        
        Args:
            device: 'cpu' 或 'cuda'
            weights_path: 预训练权重路径（可选）
        """
        self.device = device
        self.model = PoseGCN().to(device)
        
        # 加载预训练权重
        if weights_path is not None and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"GCN特征提取器初始化成功，已加载预训练权重: {weights_path}")
        else:
            if weights_path is not None:
                print(f"警告: 未找到预训练权重文件 {weights_path}，使用随机初始化")
            else:
                print("GCN特征提取器初始化成功，使用随机初始化")
        
        self.model.eval()
        print(f"  输出维度: 16")
    
    def extract(self, keypoints):
        """
        提取单帧特征
        
        Args:
            keypoints: numpy array, shape (17, 2), dtype=np.float32
                       关键点坐标，顺序为 COCO 格式
                       （应该是已经归一化的坐标）
        
        Returns:
            numpy array, shape (16,)
        """
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.from_numpy(np.array(keypoints, dtype=np.float32))
        
        keypoints = keypoints.to(self.device)
        
        with torch.no_grad():
            feature = self.model(keypoints)
        
        return feature.cpu().numpy()
    
    def extract_batch(self, keypoints_batch):
        """
        批量提取特征
        
        Args:
            keypoints_batch: numpy array, shape (batch, 17, 2)
        
        Returns:
            numpy array, shape (batch, 16)
        """
        if not isinstance(keypoints_batch, torch.Tensor):
            keypoints_batch = torch.from_numpy(np.array(keypoints_batch, dtype=np.float32))
        
        keypoints_batch = keypoints_batch.to(self.device)
        
        with torch.no_grad():
            features = self.model(keypoints_batch)
        
        return features.cpu().numpy()
    
    def extract_video(self, keypoints_sequence):
        """
        提取视频所有帧的特征
        
        Args:
            keypoints_sequence: list of (17,2) 或 numpy array (T, 17, 2)
        
        Returns:
            numpy array (T, 16)
        """
        if isinstance(keypoints_sequence, list):
            keypoints_sequence = np.array(keypoints_sequence, dtype=np.float32)
        
        return self.extract_batch(keypoints_sequence)
