# -*- coding: utf-8 -*-
"""
输入 npy 路径 -> 自动输出回归得分
用法:
  1. 把要预测的 npy 路径填到 CONFIG['npy_path']（单个文件或文件夹）
  2. 运行: python predict.py
  3. 程序自动加载模型并输出得分
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 配置（必须和训练时一致） ====================
CONFIG = {
    # ===== 在这里填 npy 路径：单个文件 或 文件夹 =====
    'npy_path': 'D:/Dataset/sprint/result/video_point/run_2.npy',
    # 例如:
    # 'npy_path': 'D:/Dataset/sprint/keypoints/run_2.npy',   # 单个视频
    # 'npy_path': 'D:/Dataset/sprint/keypoints',             # 文件夹（预测全部）

    'model_path': 'best_model.pth',      # 训练保存的模型权重
    'in_channels': 2,                    # 训练时 add_velocity=True 则为 4
    'target_len': 300,                   # 训练时统一帧数
    'add_velocity': False,               # 训练时是否加了速度通道
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 得分反归一化统计量：二选一
    # 方式1：提供 score_stats.json（推荐，训练时自动保存）
    'stats_path': 'score_stats.json',
    # 方式2：手动填训练集上的 mean / std（stats_path 不存在时使用）
    'score_mean': 6.5,
    'score_std': 1.414,
}

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
    def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.2):
        super().__init__()
        self.sgc = SpatialGraphConvolution(in_channels, out_channels, A_size[0])
        self.M = nn.Parameter(torch.ones(A_size))
        self.tgc = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, (t_kernel_size,1), (stride,1),
                    ((t_kernel_size-1)//2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, A):
        return self.tgc(self.sgc(x, A * self.M))


class STGCNEncoder(nn.Module):
    def __init__(self, in_channels=2, t_kernel_size=3, hop_size=2):
        super().__init__()
        graph = COCOGraph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()
        self.bn = nn.BatchNorm1d(in_channels * graph.num_node)
        self.stgc1 = STGC_block(in_channels, 16, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc2 = STGC_block(16, 32, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc3 = STGC_block(32, 64, 2, t_kernel_size, A_size, dropout=0.1)

        self.reg_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0,3,1,2).contiguous().view(N, V*C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0,2,3,1).contiguous()
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = F.adaptive_avg_pool2d(x, (1,1)).view(N, -1)
        return self.reg_head(x)


# ==================== 预处理（与训练一致） ====================

def compute_velocity(seq):
    """训练时 add_velocity=True 才用：拼接帧间速度 (T,V,2)->(T,V,4)"""
    T = seq.shape[0]
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, vel], axis=-1)


def fix_length_infer(seq, T):
    """推理用均匀采样（不做随机裁剪，保证结果可复现）"""
    n = seq.shape[0]
    if n >= T:
        idx = np.linspace(0, n - 1, T).astype(int)
        return seq[idx]
    # 不足 T 帧：最近帧插值
    idx = np.linspace(0, n - 1, T)
    return np.stack([seq[min(int(i), n - 1)] for i in idx])


def preprocess(npy_path, cfg):
    """加载 npy 并转成模型输入 (1, C, T, V)"""
    seq = np.load(npy_path).astype(np.float32)
    if seq.ndim != 3:
        raise ValueError(f"{npy_path} 形状应为 (T, V, C)，实际 {seq.shape}")
    if cfg['add_velocity']:
        seq = compute_velocity(seq)
    seq = fix_length_infer(seq, cfg['target_len'])
    seq = seq.transpose(2, 0, 1)                    # (C, T, V)
    x = torch.from_numpy(seq).unsqueeze(0)          # (1, C, T, V)
    return x


# ==================== 预测 ====================

def predict_one(model, npy_path, mean, std, cfg):
    """预测单个 npy 文件的得分"""
    x = preprocess(npy_path, cfg).to(cfg['device'])
    model.eval()
    with torch.no_grad():
        pred_norm = model(x).squeeze(-1).item()
    pred = pred_norm * std + mean                   # 反归一化
    return pred


def load_stats(cfg):
    """优先读 score_stats.json，否则用配置里手填的 mean/std"""
    if os.path.exists(cfg['stats_path']):
        with open(cfg['stats_path'], 'r', encoding='utf-8') as f:
            stats = json.load(f)
        return float(stats['mean']), float(stats['std'])
    if cfg['score_mean'] is not None and cfg['score_std'] is not None:
        return cfg['score_mean'], cfg['score_std']
    raise ValueError("缺少得分统计量：请提供 score_stats.json 或配置 score_mean/score_std")


def main():
    cfg = CONFIG

    # ===== 自动从配置里取路径，不需要命令行参数 =====
    path = cfg['npy_path']
    if not path:
        print("请在 CONFIG['npy_path'] 里填写 npy 文件或文件夹路径")
        return

    mean, std = load_stats(cfg)
    print(f"device: {cfg['device']} | 反归一化 mean={mean:.4f}, std={std:.4f}")

    model = STGCNEncoder(in_channels=cfg['in_channels']).to(cfg['device'])
    model.load_state_dict(torch.load(cfg['model_path'], map_location=cfg['device']))
    print(f"模型已加载: {cfg['model_path']}")

    # 单个文件 or 文件夹
    if os.path.isdir(path):
        npy_files = sorted([os.path.join(path, f) for f in os.listdir(path)
                            if f.endswith('.npy')])
    else:
        npy_files = [path]

    print(f"共预测 {len(npy_files)} 个视频\n")
    for f in npy_files:
        try:
            score = predict_one(model, f, mean, std, cfg)
            name = os.path.basename(f)
            print(f"{name:20s} -> 得分 {score:.2f}")
        except Exception as e:
            print(f"{os.path.basename(f):20s} -> 预测失败: {e}")


if __name__ == '__main__':
    main()