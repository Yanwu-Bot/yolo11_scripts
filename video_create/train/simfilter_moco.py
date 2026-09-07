#用于训练窗口相似度 —— MoCo v2 形式
# moco_v2_filter.py
# MoCo v2 = MoCo v1 + 2层MLP投影头
# query/key 编码器各自带投影头，loss 用投影输出；下游保存纯 encoder（不含投影头）
import time
import os
import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from Feature import Feature
import module

SEED = 0

def set_seed(seed):
    global SEED
    SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    random.seed(SEED + worker_id)
    np.random.seed(SEED + worker_id)

def show_time(start_time, current_time):
    sec = int(current_time - start_time)
    return f"{sec // 3600}时{sec % 3600 // 60}分{sec % 60}秒"

# ===== MoCo v2 核心：encoder + 2层MLP投影头 =====
class EncoderWithProj(nn.Module):
    def __init__(self, encoder, feat_dim=64, proj_hidden=128):
        super().__init__()
        self.encoder = encoder                 # 下游特征来源，保存时只存它
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, feat_dim),
        )

    def forward(self, x):
        h = self.encoder(x)          # encoder 输出（下游用，不进 loss）
        z = F.normalize(self.proj(h), dim=1)   # 投影输出，L2归一化后进 loss/队列
        return z

def moco_v2_loss(q, k_pos, queue, queue_idx, batch_indices, temperature=0.07,
                 dist_matrix=None, threshold=0.0):
    # q/k_pos: (B, d) 已归一化；queue: (K, d)；queue_idx: (K,) 记录原始窗口id
    B = q.size(0)
    K = queue.size(0)
    l_pos = (q * k_pos).sum(dim=1, keepdim=True)          # (B, 1)
    l_neg = torch.mm(q, queue.t())                        # (B, K)
    invalid = torch.zeros_like(l_neg, dtype=torch.bool)
    # 1) 队列未填满位置 / 与当前样本同原始窗口 不能当负样本
    q_idx = torch.tensor(batch_indices, dtype=torch.long, device=q.device)
    invalid |= (queue_idx[None, :] < 0)
    invalid |= (queue_idx[None, :] == q_idx[:, None])
    if threshold > 0:
        # 2) 距离过滤：anchor窗口 与 队列元素原始窗口 距离 < threshold 的负样本去掉（查表）
        q_list = [max(int(i), 0) for i in batch_indices]
        k_list = queue_idx.cpu().tolist()
        d = dist_matrix[np.ix_(q_list, k_list)]           # (B, K)
        d = torch.as_tensor(d, dtype=torch.float32, device=q.device)
        d[queue_idx[None, :] < 0] = 1e9
        invalid |= (d < threshold)
    l_neg = l_neg.masked_fill(invalid, -1e9)
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature   # (B, K+1)
    labels = torch.zeros(B, dtype=torch.long, device=q.device)
    loss = F.cross_entropy(logits, labels)
    pos_avg = l_pos.mean().item()
    neg_v = l_neg[l_neg > -1e8]
    neg_avg = neg_v.mean().item() if neg_v.numel() > 0 else 0.0
    masked_count = invalid.sum().item()
    return loss, pos_avg, neg_avg, pos_avg - neg_avg, masked_count

def precompute_frame_features(raw_windows, cache_path):
    if os.path.exists(cache_path):
        feats = np.load(cache_path)
        print(f"加载帧特征缓存: {feats.shape}")
        return feats
    N, T = raw_windows.shape[0], raw_windows.shape[1]
    feats = np.zeros((N, T, 26), dtype=np.float32)
    for n in range(N):
        for t in range(T):
            try:
                feats[n, t] = Feature(raw_windows[n, t].tolist()).get_all_features()
            except Exception as e:
                feats[n, t] = 0.0
                print(f"帧特征提取失败 n={n},t={t}: {e}")
    np.save(cache_path, feats)
    print(f"帧特征预计算完成: {feats.shape} -> {cache_path}")
    return feats

def precompute_dist_matrix(frame_features, cache_path):
    if os.path.exists(cache_path):
        D = np.load(cache_path)
        print(f"加载距离矩阵缓存: {D.shape}")
        return D
    N, T, _ = frame_features.shape
    D = np.zeros((N, N), dtype=np.float32)
    for t in range(T):                                     # 逐帧 L2 -> 对 T 取平均
        x = frame_features[:, t]
        sq = (x * x).sum(1, keepdims=True)
        dd = sq + sq.T - 2.0 * (x @ x.T)
        D += np.sqrt(np.maximum(dd, 0))
    D /= T
    np.fill_diagonal(D, 0)
    np.save(cache_path, D)
    print(f"距离矩阵计算完成: {D.shape} -> {cache_path}")
    return D

class ContrastiveDatasetFromFile(Dataset):
    def __init__(self, npz_path, window_size=6, transform_params=None):
        data = np.load(npz_path, allow_pickle=True)
        self.windows = data['windows']
        self.window_size = window_size
        self.transform_params = transform_params or {
            'rotation': 5, 'scale': 0.05, 'noise': 0.02, 'mask': 0.1,
            'reverse': 0.2, 'GB': 0.3, 'shear': 0.05, 'flip': 0.2, 'delete': 0.1
        }
        print(f"加载数据集: {npz_path}, 共 {len(self.windows)} 个窗口")

    def _random_transform(self, window):
        w = window.copy()
        T, V, C = w.shape
        if 'rotation' in self.transform_params and self.transform_params['rotation'] > 0:
            angle = random.uniform(-self.transform_params['rotation'], self.transform_params['rotation'])
            rad = math.radians(angle)
            cos, sin = math.cos(rad), math.sin(rad)
            hip_center = w[:, 11:13, :].mean(axis=(0, 1))
            w_centered = w - hip_center
            rot = np.zeros_like(w)
            rot[..., 0] = w_centered[..., 0] * cos - w_centered[..., 1] * sin
            rot[..., 1] = w_centered[..., 0] * sin + w_centered[..., 1] * cos
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
                (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                (11, 12), (13, 14), (15, 16), (0, 0)
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

def build_encoder(select, device):
    if select == 'STGCN':
        return module.STGCNEncoder(output_dim=64).to(device)
    elif select == 'GRU':
        return module.GRUEncoder(output_dim=64).to(device)
    elif select == 'LSTM':
        return module.LSTMEncoder(output_dim=64).to(device)
    elif select == 'MLP':
        return module.MLPEncoder(output_dim=64).to(device)
    elif select == 'TCN':
        return module.TCNEncoder(output_dim=64).to(device)
    elif select == 'CTR':
        return module.CTRGCNEncoder(in_channels=2, output_dim=64).to(device)

def train_moco_v2(dataset, epochs=200, batch_size=256, lr=0.001, temperature=0.07,
                  momentum=0.999, queue_size=8192, diversity_threshold=0.9,
                  select='CTR', num_workers=4,
                  feat_cache='frame_features_cache.npy', dist_cache='dist_matrix_cache.npy'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    model = EncoderWithProj(build_encoder(select, device)).to(device)   # query（含投影头）
    key_model = copy.deepcopy(model)                                    # key（含投影头）
    for p in key_model.parameters():
        p.requires_grad = False
    optimizer = optim.Adam(model.parameters(), lr=lr)
    out_dim = 64
    queue = torch.zeros(queue_size, out_dim).to(device)   # 存 key 的投影输出（已归一化）
    queue_idx = torch.full((queue_size,), -1, dtype=torch.long, device=device)
    ptr = 0

    save_dir = {'STGCN': 'D:/Dataset/sprint/result/model/ST-GCN/best_mocov2_stgcn.pth',
                'GRU':   'D:/Dataset/sprint/result/model/GRU/best_mocov2_gru.pth',
                'LSTM':  'D:/Dataset/sprint/result/model/LSTM/best_mocov2_lstm.pth',
                'MLP':   'D:/Dataset/sprint/result/model/MLP/best_mocov2_mlp.pth',
                'TCN':   'D:/Dataset/sprint/result/model/TCN/best_mocov2_tcn.pth',
                'CTR':   'D:/Dataset/sprint/result/model/CTR-GCN/best_mocov2_ctr.pth'}[select]
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
                        persistent_workers=(num_workers > 0), worker_init_fn=worker_init_fn)

    raw_windows = dataset.windows
    print(f"使用模型：{select}（MoCo v2）| queue_size={queue_size} | momentum={momentum}")
    frame_features = precompute_frame_features(raw_windows, feat_cache)
    dist_matrix = precompute_dist_matrix(frame_features, dist_cache)

    best_loss = float('inf')
    loss_history, pos_history, neg_history, diff_history = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss = total_pos = total_neg = total_diff = 0.0
        total_masked = total_neg_pairs = num_batches = 0
        for anchor, positive, indices in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            batch_indices = indices.tolist()
            B = anchor.size(0)

            q = model(anchor)                             # query 投影输出（反传）
            with torch.no_grad():
                k = key_model(positive)                   # key 投影输出（不反传）

            loss, pos_avg, neg_avg, diff, masked_count = moco_v2_loss(
                q, k, queue, queue_idx, batch_indices,
                temperature=temperature,
                dist_matrix=dist_matrix,
                threshold=diversity_threshold
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 动量更新 key 编码器 + key 投影头（整体 EMA）
            with torch.no_grad():
                for p_q, p_k in zip(model.parameters(), key_model.parameters()):
                    p_k.data = momentum * p_k.data + (1 - momentum) * p_q.data

            # 队列 FIFO：当前 batch 的 key 投影特征入队
            with torch.no_grad():
                idx = torch.arange(ptr, ptr + B) % queue_size
                queue[idx] = k.detach()
                queue_idx[idx] = torch.tensor(batch_indices, device=device)
                ptr = int((ptr + B) % queue_size)

            total_loss += loss.item()
            total_pos += pos_avg
            total_neg += neg_avg
            total_diff += diff
            total_masked += masked_count
            total_neg_pairs += B * queue_size
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_pos = total_pos / num_batches
        avg_neg = total_neg / num_batches
        avg_diff = total_diff / num_batches
        loss_history.append(avg_loss)
        pos_history.append(avg_pos)
        neg_history.append(avg_neg)
        diff_history.append(avg_diff)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, "
                  f"PosSim: {avg_pos:.4f}, NegSim: {avg_neg:.4f}, Diff: {avg_diff:.4f}, "
                  f"MaskedNeg: {total_masked}/{total_neg_pairs} "
                  f"({total_masked/total_neg_pairs*100:.1f}%)")
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 只保存纯 encoder（不含投影头），下游加载方式与之前一致
            torch.save(model.encoder.state_dict(), save_dir)
            print(f"  -> 保存最佳模型，loss={avg_loss:.6f}")
    print("训练完成")

    from thop import profile, clever_format
    model.encoder.eval()
    dummy = torch.randn(1, 2, 9, 17).to(device)
    flops, params = profile(model.encoder, inputs=(dummy,), verbose=False)
    flops_f, params_f = clever_format([flops, params], "%.3f")
    print(f"参数量(encoder): {params_f} | FLOPs(encoder): {flops_f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(loss_history, color='black')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss'); ax1.grid(True)
    ax2.plot(pos_history, label='Pos Sim', color='blue')
    ax2.plot(neg_history, label='Neg Sim', color='red')
    ax2.plot(diff_history, label='Diff (Pos-Neg)', color='green')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Similarity')
    ax2.set_title('Positive vs Negative Similarity')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    set_seed(0)
    npz_path = 'D:/Dataset/sprint/result/window_data/dataset_9_2.npz'
    dataset = ContrastiveDatasetFromFile(
        npz_path,
        window_size=9,
        transform_params={'rotation': 15, 'scale': 0.15, 'noise': 0.05, 'mask': 0.1,
                          'reverse': 0.15, 'GB': 0.25, 'shear': 0.1, 'flip': 0.15,
                          'delete': 0.15}
    )
    train_moco_v2(dataset, epochs=200, batch_size=256, lr=0.001, temperature=0.07,
                  momentum=0.999, queue_size=4096, diversity_threshold=0.9,
                  select='CTR', num_workers=4,
                  feat_cache=npz_path.replace('.npz', '_frame_features.npy'),
                  dist_cache=npz_path.replace('.npz', '_dist.npy'))
    elapsed = show_time(start_time, time.time())
    print(f"Total time: {elapsed}")