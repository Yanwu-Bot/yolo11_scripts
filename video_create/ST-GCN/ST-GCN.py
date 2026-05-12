import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ========== ST-GCN 相关定义 ==========
class COCOGraph:
    """COCO 17关键点图结构"""
    def __init__(self, hop_size=2):
        self.num_node = 17
        self.hop_size = hop_size
        self.get_edge()
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)
        self.get_adjacency()
    
    def get_edge(self):
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_base = [
            (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)
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
    def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.5):
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

class ContrastiveEncoder(nn.Module):
    """对比学习编码器，输出128维归一化特征向量"""
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
        x = F.adaptive_avg_pool2d(x, (1,1)).view(N, -1)
        x = self.projection(x)
        return F.normalize(x, dim=1)

# ========== 损失函数 (NT-Xent) ==========
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T)
    mask = torch.eye(2*batch_size, device=sim.device).bool()
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    for i in range(batch_size):
        pos_mask[i, i+batch_size] = True
        pos_mask[i+batch_size, i] = True
    sim = sim[~mask].view(2*batch_size, -1)
    pos_sim = sim[pos_mask[~mask].view(2*batch_size, -1)].view(2*batch_size, 1)
    neg_sim = sim[~pos_mask[~mask].view(2*batch_size, -1)].view(2*batch_size, -1)
    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(2*batch_size, dtype=torch.long, device=sim.device)
    return F.cross_entropy(logits, labels)

# ========== 数据集类 (从 .npz 加载) ==========
class ContrastiveDatasetFromFile(Dataset):
    """从已保存的 .npz 数据集文件加载窗口，动态生成三元组"""
    def __init__(self, npz_path, window_size=10,
                 neg_cross_ratio=0.7, temporal_threshold=20,
                 transform_params=None):
        data = np.load(npz_path, allow_pickle=True)
        self.windows = data['windows']                  # (N, W, 17, 2)
        self.video_indices = data['video_indices']      # (N,)
        self.start_frames = data['start_frames']        # (N,)
        self.window_size = window_size
        self.neg_cross_ratio = neg_cross_ratio
        self.temporal_threshold = temporal_threshold
        self.transform_params = transform_params or {
            'rotation': 5, 'scale': 0.05, 'noise': 0.02, 'mask': 0.1
        }
        # 建立视频->窗口索引映射
        self.video_window_indices = {}
        for idx, vi in enumerate(self.video_indices):
            vi = int(vi)
            if vi not in self.video_window_indices:
                self.video_window_indices[vi] = []
            self.video_window_indices[vi].append((idx, int(self.start_frames[idx])))
        print(f"加载数据集: {npz_path}, 共 {len(self.windows)} 个窗口")

    def _random_transform(self, window):
        w = window.copy()
        T, V, C = w.shape
        # 旋转
        if 'rotation' in self.transform_params and self.transform_params['rotation'] > 0:
            angle = random.uniform(-self.transform_params['rotation'],
                                    self.transform_params['rotation'])
            rad = math.radians(angle)
            cos, sin = math.cos(rad), math.sin(rad)
            hip_center = w[:, 11:13, :].mean(axis=(0,1))
            w_centered = w - hip_center
            rot = np.zeros_like(w)
            rot[..., 0] = w_centered[..., 0]*cos - w_centered[..., 1]*sin
            rot[..., 1] = w_centered[..., 0]*sin + w_centered[..., 1]*cos
            w = rot + hip_center
        # 缩放
        if 'scale' in self.transform_params and self.transform_params['scale'] > 0:
            scale = 1.0 + random.uniform(-self.transform_params['scale'],
                                          self.transform_params['scale'])
            w = w * scale
        # 噪声
        if 'noise' in self.transform_params and self.transform_params['noise'] > 0:
            noise = np.random.normal(0, self.transform_params['noise'], w.shape)
            w = w + noise
        # 遮挡
        if 'mask' in self.transform_params and self.transform_params['mask'] > 0:
            mask = np.random.binomial(1, 1-self.transform_params['mask'],
                                      size=(T, V, 1))
            w = w * mask
        return w

    def _get_negative_sample(self, anchor_idx, anchor_vi, anchor_start):
        # 跨视频
        if random.random() < self.neg_cross_ratio:
            other_videos = [vi for vi in self.video_window_indices if vi != anchor_vi]
            if other_videos:
                neg_vi = random.choice(other_videos)
                cand = self.video_window_indices[neg_vi]
                if cand:
                    idx, _ = random.choice(cand)
                    return self.windows[idx]
        # 同视频远距离
        own = self.video_window_indices[anchor_vi]
        far = [(idx, s) for idx, s in own if abs(s - anchor_start) >= self.temporal_threshold]
        if far:
            idx, _ = random.choice(far)
            return self.windows[idx]
        return np.zeros((self.window_size, 17, 2), dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        anchor = self.windows[idx]
        anchor_vi = int(self.video_indices[idx])
        anchor_start = int(self.start_frames[idx])
        positive = self._random_transform(anchor)
        negative = self._get_negative_sample(idx, anchor_vi, anchor_start)
        def to_stgcn(data):
            return torch.FloatTensor(data).permute(2, 0, 1)
        return to_stgcn(anchor), to_stgcn(positive), to_stgcn(negative)

def train_contrastive(dataset, epochs=100, batch_size=32, lr=1e-3, temperature=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveEncoder(output_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    best_loss = float('inf')
    save_dir = 'result/GCN/model'
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            z1 = model(anchor)
            z2 = model(positive)
            loss = nt_xent_loss(z1, z2, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            print(f"  -> 保存最佳模型，loss={avg_loss:.6f}")
    print("训练完成")

if __name__ == '__main__':
    # 设置数据集路径（请根据实际修改）
    npz_path = 'result/GCN/dataset/dataset.npz'
    # 创建数据集
    dataset = ContrastiveDatasetFromFile(
        npz_path,
        window_size=10,          # 需与数据集生成时的值一致
        neg_cross_ratio=0.7,
        temporal_threshold=20,
        transform_params={'rotation':5, 'scale':0.05, 'noise':0.02, 'mask':0.1}
    )
    # 开始训练
    train_contrastive(dataset, epochs=100, batch_size=32, lr=1e-3, temperature=0.5)