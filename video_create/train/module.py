#所有可用于训练的模型
import time
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  
from Feature import Feature

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
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size+1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(hop_size, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis
    def get_adjacency(self):
        valid_hop = range(0, self.hop_size+1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        self.A = A
    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        Dn = np.zeros((A.shape[0], A.shape[0]))
        for i in range(A.shape[0]):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        return np.dot(A, Dn)

class SpatialGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * s_kernel_size, 1)
    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class STGC_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size,
                 dropout=0.2, dilation=1):
        super().__init__()
        self.sgc = SpatialGraphConvolution(in_channels, out_channels, A_size[0])
        self.M = nn.Parameter(torch.ones(A_size))          # 可学习缩放
        self.B = nn.Parameter(torch.zeros(A_size))         # 自学习边
        self.tgc = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, (t_kernel_size, 1), (stride, 1),
                      ((t_kernel_size - 1) // 2 * dilation, 0),   # padding 随 dilation 放大
                      dilation=(dilation, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, A):
        return self.tgc(self.sgc(x, A * self.M + self.B))

class STGCNEncoder(nn.Module):
    def __init__(self, in_channels=2, t_kernel_size=3, hop_size=2, output_dim=64):
        super().__init__()
        graph = COCOGraph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()
        self.bn = nn.BatchNorm1d(in_channels * graph.num_node)
        # 时序感受野: RF = 1 + (3-1) * (1+1+2) = 9，正好覆盖9帧
        self.stgc1 = STGC_block(in_channels, 16, 1, t_kernel_size, A_size,
                                dropout=0.1, dilation=1)
        self.stgc2 = STGC_block(16, 32, 1, t_kernel_size, A_size,
                                dropout=0.1, dilation=1)
        self.stgc3 = STGC_block(32, 64, 1, t_kernel_size, A_size,
                                dropout=0.1, dilation=2)
        # 方案B：时间注意力池化（输入是stgc3输出，128通道）
        self.att_fc = nn.Linear(64, 1)
        nn.init.constant_(self.att_fc.bias, 0.0)   # 初始≈平均池化，更稳
        # 64维输出
        self.projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        # 输入: (N, C, T, V)，T=9，V=17
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)          # (N, 128, 9, 17)
        # 空间池化（平均17个关节），保留时间维
        x = x.mean(dim=3)                  # (N, 128, 9)
        # 时间注意力池化
        att = torch.sigmoid(self.att_fc(x.permute(0, 2, 1)))  # (N, 9, 1)
        att = att.permute(0, 2, 1)         # (N, 1, 9)
        x = (x * att).sum(dim=2)           # (N, 128)
        x = self.projection(x)             # (N, 64)
        return F.normalize(x, dim=1)       # L2归一化，用于余弦相似度

class MLPEncoder(nn.Module):
    """
    展平后为 N*238 维，映射到 output_dim 维并 L2 归一化。
    """
    def __init__(self, in_channels=2, window_size=7, num_joints=17,
                hidden_dim=256, output_dim=64, dropout=0.2):
        super().__init__()
        self.input_dim = in_channels * window_size * num_joints   # 2*7*17 = 238
        # 对展平输入做 BN，对应 ST-GCN 里开头的 BatchNorm1d
        self.bn = nn.BatchNorm1d(self.input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (N, C, T, V)
        N = x.size(0)
        x = x.reshape(N, -1)        
        x = self.bn(x)
        x = self.mlp(x)
        return F.normalize(x, dim=1)

class TCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, t_kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = ((t_kernel_size - 1) * dilation) // 2
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels,
                        (t_kernel_size, 1), padding=(pad, 0), dilation=(dilation, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class TCNEncoder(nn.Module):
    def __init__(self, in_channels=2, t_kernel_size=3, output_dim=64, num_joints=17):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels * num_joints)   # 34
        self.tcn1 = TCN_block(34, 64, t_kernel_size)
        self.tcn2 = TCN_block(64, 64, t_kernel_size)
        self.tcn3 = TCN_block(64, 128, t_kernel_size)
        self.projection = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)  # (N, 34, T) 关节拼进通道
        x = self.bn(x)
        x = x.unsqueeze(-1)                     # (N, 34, T, 1)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        x = x.squeeze(-1)                       # (N, 128, T)
        x = x.mean(dim=2)                       # 时间池化
        x = self.projection(x)
        return F.normalize(x, dim=1)

class LSTMEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_size=128, num_layers=2,output_dim=64, num_joints=17):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 每个时间步的输入 = 所有关节点坐标展平 (V * C)
        self.input_size = num_joints * in_channels
        # 保持原代码的输入归一化习惯（对每帧关节特征做 BN）
        self.bn = nn.BatchNorm1d(self.input_size)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # 多层 LSTM 才启用层间 dropout，作用类似原 ST-GCN 的 dropout
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        # 与原代码相同的投影头结构（输入维度按 hidden_size 调整）
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        # (N, C, T, V) -> (N, T, V*C)，把每帧所有关节拼成一个向量
        x = x.permute(0, 2, 3, 1).contiguous().view(N, T, V * C)
        # 逐帧 BatchNorm（等价于原代码对关节特征做 BN 的意图）
        x = self.bn(x.view(N * T, -1)).view(N, T, -1)
        # LSTM 编码时序，out: (N, T, hidden_size)
        out, _ = self.lstm(x)
        # 若想用最后时刻状态，可改为 out[:, -1, :]
        x = out.mean(dim=1)               # (N, hidden_size)
        x = self.projection(x)
        return F.normalize(x, dim=1)

class GRUEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_size=128, num_layers=2,output_dim=64, num_joints=17):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 每个时间步的输入 = 所有关节点坐标展平 (V * C)
        self.input_size = num_joints * in_channels
        # 保持原代码的输入归一化习惯（对每帧关节特征做 BN）
        self.bn = nn.BatchNorm1d(self.input_size)
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        # 与原代码相同的投影头结构（输入维度按 hidden_size 调整）
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        # (N, C, T, V) -> (N, T, V*C)，把每帧所有关节拼成一个向量
        x = x.permute(0, 2, 3, 1).contiguous().view(N, T, V * C)
        # 逐帧 BatchNorm（等价于原代码对关节特征做 BN 的意图）
        x = self.bn(x.view(N * T, -1)).view(N, T, -1)
        # LSTM 编码时序，out: (N, T, hidden_size)
        out, _ = self.gru(x)
        # 若想用最后时刻状态，可改为 out[:, -1, :]
        x = out.mean(dim=1)               # (N, hidden_size)
        x = self.projection(x)
        return F.normalize(x, dim=1)

class CTRGC(nn.Module):
    """
    对齐官方 CTR-GCN 的核心机制：
    对每个通道 c 和每个 hop k，都生成一张自己的精炼邻接图
        A_refined[k,c] = A[k] + P[k] + alpha[k] * corr[c]
    其中 corr[c] 由输入特征的双分支相关性生成。
    """
    def __init__(self, in_channels, out_channels, A_shape):
        super().__init__()
        self.K, self.V = A_shape[0], A_shape[1]   # K=3(hop), V=17
        self.out_channels = out_channels
        # 特征变换：输出 K*C'，供每个 hop 用一份
        self.conv = nn.Conv2d(in_channels, out_channels * self.K, 1)
        # 双分支：生成通道相关性
        self.conv_a = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_b = nn.Conv2d(in_channels, out_channels, 1)
        # 可学习拓扑残差（叠加在基础 A 上）
        self.P = nn.Parameter(torch.zeros(self.K, self.V, self.V))
        # 每个 hop 的缩放系数
        self.alpha = nn.Parameter(torch.ones(self.K))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x, A):
        N, C, T, V = x.shape
        # 1) 时间池化，得到 (N, C, 1, V)
        xp = x.mean(dim=2, keepdim=True)
        za = self.conv_a(xp).squeeze(2)          # (N, C', V)
        zb = self.conv_b(xp).squeeze(2)          # (N, C', V)
        # 2) 逐通道相关性图：每个通道 c 有自己的 V×V 精炼项
        corr = torch.einsum('ncv,ncw->ncvw', za, zb)   # (N, C', V, V)
        corr = torch.tanh(corr / (V ** 0.5))
        # 3) 图卷积
        x = self.conv(x)                         # (N, K*C', T, V)
        x = x.view(N, self.K, self.out_channels, T, V)
        out = 0
        for k in range(self.K):
            base = A[k] + self.P[k]              # (V, V)
            # 每通道每hop的精炼图 (N, C', V, V)
            refined = base.unsqueeze(0).unsqueeze(0) + self.alpha[k] * corr
            # 每个通道用自己的图做卷积
            out = out + torch.einsum('nctv,ncvw->nctw', x[:, k], refined)
        return self.relu(self.bn(out))
# ================= 干净的 TCN 残差块 =================
class TCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, t_kernel_size=3, stride=1,
                 dilation=1, dropout=0.2):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, (t_kernel_size, 1), (stride, 1),
                      ((t_kernel_size - 1) // 2 * dilation, 0),
                      dilation=(dilation, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.residual = None
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        res = x if self.residual is None else self.residual(x)
        return self.tcn(x) + res
# ================= 编码器：输入 (N,2,9,17) → 输出 (N,64) =================
class CTRGCNEncoder(nn.Module):
    def __init__(self, in_channels=2, output_dim=64, t_kernel_size=3, hop_size=2):
        super().__init__()
        graph = COCOGraph(hop_size)                    # 用你现有的 COCOGraph（17点）
        A = torch.tensor(graph.A, dtype=torch.float32)
        self.register_buffer('A', A)                   # (3, 17, 17)
        self.bn = nn.BatchNorm1d(in_channels * graph.num_node)
        # 感受野: dilation 1,1,2 → RF = 1+2*(1+1+2) = 9，匹配9帧
        self.gc1 = CTRGC(in_channels, 16, A.shape)
        self.tcn1 = TCN_block(16, 16, t_kernel_size, 1, dilation=1)
        self.gc2 = CTRGC(16, 32, A.shape)
        self.tcn2 = TCN_block(32, 32, t_kernel_size, 1, dilation=1)
        self.gc3 = CTRGC(32, 64, A.shape)
        self.tcn3 = TCN_block(64, 64, t_kernel_size, 1, dilation=2)
        # 时间注意力池化
        self.att_fc = nn.Linear(64, 1)
        nn.init.constant_(self.att_fc.bias, 0.0)
        # 64维输出
        self.projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    def forward(self, x):
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        x = self.tcn1(self.gc1(x, self.A))    # (N,16,9,17)
        x = self.tcn2(self.gc2(x, self.A))    # (N,32,9,17)
        x = self.tcn3(self.gc3(x, self.A))    # (N,64,9,17)
        x = x.mean(dim=3)                     # 空间平均 → (N,64,9)
        att = torch.sigmoid(self.att_fc(x.permute(0, 2, 1)))  # (N,9,1)
        att = att.permute(0, 2, 1)            # (N,1,9)
        x = (x * att).sum(dim=2)              # (N,64)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)