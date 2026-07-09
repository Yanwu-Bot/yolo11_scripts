#增加了batch间距离过滤的ST-GCN与ST-GCN其他无异
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

MODEL_SAVE_N = 'best_7_1_f.pth'

def show_time(start_time,current_time):
    start_time = time.localtime(start_time)
    current_time = time.localtime(current_time)
    tm_hour = current_time.tm_hour - start_time.tm_hour 
    tm_min = current_time.tm_min - start_time.tm_min  
    tm_sec = current_time.tm_sec - start_time.tm_sec 
    time_string = f"{tm_hour}时{tm_min}分{tm_sec}秒"
    return time_string

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
    def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.3):
        super().__init__()
        self.sgc = SpatialGraphConvolution(in_channels, out_channels, A_size[0])
        self.M = nn.Parameter(torch.ones(A_size))
        self.B = nn.Parameter(torch.zeros(A_size))
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
        return self.tgc(self.sgc(x, A * self.M + self.B))

class EADM(nn.Module):
    def __init__(self, drop_ratio=0.3, lambda_=1e-4):
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

class ContrastiveEncoder(nn.Module):
    def __init__(self, in_channels=2, t_kernel_size=7, hop_size=2, output_dim=128):
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
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
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

def nt_xent_loss(z1, z2, temperature=0.5, raw_windows=None, batch_indices=None, threshold=0.0):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T)
    mask = torch.eye(2*batch_size, device=sim.device).bool()
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    for i in range(batch_size):
        pos_mask[i, i+batch_size] = True
        pos_mask[i+batch_size, i] = True

    if threshold > 0 and raw_windows is not None and batch_indices is not None:
        cur_raw = raw_windows[batch_indices]  # (B, T, V, 2)
        # 计算原始窗口间距离矩阵 (B, B)
        dists = np.zeros((batch_size, batch_size), dtype=np.float32)
        for i in range(batch_size):
            w_i = cur_raw[i]
            for j in range(i+1, batch_size):
                w_j = cur_raw[j]
                diff = np.linalg.norm(w_i - w_j, axis=-1)
                d = np.mean(diff)
                dists[i, j] = d
                dists[j, i] = d

        # 扩展为 (2B, 2B) 与 sim 形状一致
        full_dists = np.zeros((2*batch_size, 2*batch_size), dtype=np.float32)
        full_dists[:batch_size, :batch_size] = dists
        full_dists[:batch_size, batch_size:] = dists
        full_dists[batch_size:, :batch_size] = dists
        full_dists[batch_size:, batch_size:] = dists

        invalid_neg_mask = (full_dists < threshold) & (~pos_mask.cpu().numpy())
        invalid = torch.tensor(invalid_neg_mask, device=sim.device)
        sim[invalid] = -1e9

    sim = sim[~mask].view(2*batch_size, -1)
    pos_sim = sim[pos_mask[~mask].view(2*batch_size, -1)].view(2*batch_size, 1)
    neg_sim = sim[~pos_mask[~mask].view(2*batch_size, -1)].view(2*batch_size, -1)
    pos_sim_raw = pos_sim.clone()
    neg_sim_raw = neg_sim.clone()
    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(2*batch_size, dtype=torch.long, device=sim.device)
    loss = F.cross_entropy(logits, labels)
    pos_avg = pos_sim_raw.mean().item()
    neg_valid = neg_sim_raw[neg_sim_raw > -1e8]
    if neg_valid.numel() > 0:
        neg_avg = neg_valid.mean().item()
    else:
        neg_avg = 0.0
    diff = pos_avg - neg_avg
    return loss, pos_avg, neg_avg, diff

class ContrastiveDatasetFromFile(Dataset):
    def __init__(self, npz_path, window_size=6, transform_params=None):
        data = np.load(npz_path, allow_pickle=True)
        self.windows = data['windows']
        self.window_size = window_size
        self.transform_params = transform_params or {
            'rotation': 5, 'scale': 0.05, 'noise': 0.02, 'mask': 0.1,
            'reverse': 0.2, 'GB': 0.3, 'shear': 0.05, 'flip': 0.2, 'delete':0.1
        }
        print(f"加载数据集: {npz_path}, 共 {len(self.windows)} 个窗口")

    def _random_transform(self, window):
        w = window.copy()
        T, V, C = w.shape
        if 'rotation' in self.transform_params and self.transform_params['rotation'] > 0:
            angle = random.uniform(-self.transform_params['rotation'], self.transform_params['rotation'])
            rad = math.radians(angle)
            cos, sin = math.cos(rad), math.sin(rad)
            hip_center = w[:, 11:13, :].mean(axis=(0,1))
            w_centered = w - hip_center
            rot = np.zeros_like(w)
            rot[..., 0] = w_centered[..., 0]*cos - w_centered[..., 1]*sin
            rot[..., 1] = w_centered[..., 0]*sin + w_centered[..., 1]*cos
            w = rot + hip_center
        if 'scale' in self.transform_params and self.transform_params['scale'] > 0:
            scale = 1.0 + random.uniform(-self.transform_params['scale'], self.transform_params['scale'])
            w = w * scale
        if 'noise' in self.transform_params and self.transform_params['noise'] > 0:
            noise = np.random.normal(0, self.transform_params['noise'], w.shape)
            w = w + noise
        if 'mask' in self.transform_params and self.transform_params['mask'] > 0:
            mask = np.random.binomial(1, 1 - self.transform_params['mask'], size=(T, V, 1))
            w = w * mask
        if 'reverse' in self.transform_params and self.transform_params['reverse'] > 0:
            if random.random() < self.transform_params['reverse']:
                w = w[::-1].copy()
        if 'GB' in self.transform_params and self.transform_params['GB'] > 0:
            if random.random() < self.transform_params['GB']:
                sigma = random.uniform(0.3, 0.7)
                radius = int(4 * sigma + 0.5)
                max_radius = (T - 1) // 2
                if radius > max_radius:
                    radius = max_radius
                    if radius <= 0:
                        pass
                    else:
                        sigma = radius / 4.0
                if radius > 0:
                    t = np.arange(-radius, radius + 1)
                    kernel = np.exp(-0.5 * (t / sigma) ** 2)
                    kernel /= kernel.sum()
                    w_smooth = np.zeros_like(w)
                    for v in range(V):
                        for c in range(C):
                            w_smooth[:, v, c] = np.convolve(w[:, v, c], kernel, mode='same')
                    w = w_smooth
        if 'shear' in self.transform_params and self.transform_params['shear'] > 0:
            shx = random.uniform(-self.transform_params['shear'], self.transform_params['shear'])
            shy = random.uniform(-self.transform_params['shear'], self.transform_params['shear'])
            x_old = w[..., 0].copy()
            y_old = w[..., 1].copy()
            w[..., 0] = x_old + shx * y_old
            w[..., 1] = y_old + shy * x_old
        if 'flip' in self.transform_params and random.random() < self.transform_params['flip']:
            swap_pairs = [
                (1,2), (3,4), (5,6), (7,8), (9,10),
                (11,12), (13,14), (15,16), (0,0)
            ]
            w_flipped = w.copy()
            for a, b in swap_pairs:
                w_flipped[:, a, :] = -w[:, b, :]
                w_flipped[:, b, :] = -w[:, a, :]
            w = w_flipped
        if 'delete' in self.transform_params and random.random() < self.transform_params['delete']:
            drop_ratio = random.uniform(0.05, 0.2)
            num_drop = max(1, int(T * drop_ratio))
            drop_indices = random.sample(range(T), num_drop)
            w[drop_indices, :, :] = 0.0
        return w

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        anchor = self.windows[idx]
        positive = self._random_transform(anchor)
        def to_stgcn(data):
            return torch.FloatTensor(data).permute(2, 0, 1)
        return to_stgcn(anchor), to_stgcn(positive), idx

def train_contrastive(dataset, epochs=100, batch_size=32, lr=1e-3, temperature=0.5, diversity_threshold=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveEncoder(output_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    best_loss = float('inf')
    save_dir = 'result/GCN/model'
    os.makedirs(save_dir, exist_ok=True)
    loss_history = []
    pos_history = []
    neg_history = []
    diff_history = []

    raw_windows = dataset.windows

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        total_diff = 0.0
        num_batches = 0
        for anchor, positive, indices in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            batch_indices = indices.tolist()
            z1 = model(anchor)
            z2 = model(positive)
            loss, pos_avg, neg_avg, diff = nt_xent_loss(
                z1, z2, temperature,
                raw_windows=raw_windows,
                batch_indices=batch_indices,
                threshold=diversity_threshold
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_pos += pos_avg
            total_neg += neg_avg
            total_diff += diff
            num_batches += 1
        avg_loss = total_loss / num_batches
        avg_pos = total_pos / num_batches
        avg_neg = total_neg / num_batches
        avg_diff = total_diff / num_batches
        loss_history.append(avg_loss)
        pos_history.append(avg_pos)
        neg_history.append(avg_neg)
        diff_history.append(avg_diff)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, "
              f"PosSim: {avg_pos:.4f}, NegSim: {avg_neg:.4f}, Diff: {avg_diff:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, MODEL_SAVE_N))
            print(f"  -> 保存最佳模型，loss={avg_loss:.6f}")
    print("训练完成")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(loss_history, color='black')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    ax2.plot(pos_history, label='Pos Sim', color='blue')
    ax2.plot(neg_history, label='Neg Sim', color='red')
    ax2.plot(diff_history, label='Diff (Pos-Neg)', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Similarity')
    ax2.set_title('Positive vs Negative Similarity')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    npz_path = 'result/GCN/dataset/dataset_7_1.npz'
    dataset = ContrastiveDatasetFromFile(
        npz_path,
        window_size=7,
        transform_params={'rotation':15, 'scale':0.15, 'noise':0.05, 'mask':0.1,
                        'reverse':0.15, 'GB':0.25, 'shear':0.1, 'flip':0.15, 'delete':0.15}
    )
    train_contrastive(dataset, epochs=100, batch_size=64, lr=0.001, temperature=0.1, diversity_threshold=25)
    elapsed = show_time(start_time, time.time())
    print(f"Total time: {elapsed}")