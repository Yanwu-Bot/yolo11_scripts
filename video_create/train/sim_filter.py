#用于训练窗口相似度
# ST_filter_fea.py
# 增加了batch间距离过滤的ST-GCN与ST-GCN其他无异
# 帧特征/距离均每次现算，不落盘；距离按当前 batch 现算，避免 NxN 全量矩阵
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
import module

MODEL_SAVE_N = 'best_7_1_f.pth'
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

def nt_xent_loss(z1, z2, temperature=0.5, frame_features=None, batch_indices=None, threshold=0.0):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T)
    # 向量化构建 pos/neg mask
    eye = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
    pos_mask = torch.zeros_like(eye)
    idx = torch.arange(batch_size, device=sim.device)
    pos_mask[idx, idx + batch_size] = True
    pos_mask[idx + batch_size, idx] = True
    neg_mask = ~(eye | pos_mask)
    # 每次现算当前 batch 距离：逐帧 L2 -> 对 T 取平均（度量与原版一致）
    invalid = torch.zeros_like(eye)
    if threshold > 0:
        fb = torch.as_tensor(frame_features[batch_indices],
                             dtype=torch.float32, device=sim.device)   # (B,T,26)
        d = torch.zeros(batch_size, batch_size, device=sim.device)
        for t in range(fb.shape[1]):
            d += torch.cdist(fb[:, t], fb[:, t])
        d /= fb.shape[1]
        invalid = (d.repeat(2, 2) < threshold) & neg_mask
    s = sim / temperature
    s[invalid] = -1e9
    logits = torch.cat([s[pos_mask].view(2 * batch_size, 1),
                        s[neg_mask].view(2 * batch_size, -1)], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=sim.device)
    loss = F.cross_entropy(logits, labels)
    pos_avg = sim[pos_mask].mean().item()
    neg_v = sim[neg_mask][~invalid[neg_mask]]
    neg_avg = neg_v.mean().item() if neg_v.numel() > 0 else 0.0
    diff = pos_avg - neg_avg
    masked_count = invalid[neg_mask].sum().item()
    return loss, pos_avg, neg_avg, diff, masked_count

def precompute_frame_features(raw_windows):
    # 每次都现算，不读缓存不落盘
    N, T = raw_windows.shape[0], raw_windows.shape[1]
    feats = np.zeros((N, T, 26), dtype=np.float32)
    for n in range(N):
        for t in range(T):
            try:
                feats[n, t] = Feature(raw_windows[n, t].tolist()).get_all_features()
            except Exception as e:
                feats[n, t] = 0.0
                print(f"帧特征提取失败 n={n},t={t}: {e}")
    print(f"帧特征计算完成: {feats.shape}")
    return feats

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

def train_contrastive(dataset, epochs=100, batch_size=32, lr=1e-3, temperature=0.5,
                      diversity_threshold=0.0, select='STGCN', num_workers=0,
                      resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    #根据选择模型更换存储路径
    if select == 'STGCN':
        save_dir = 'D:/Dataset/sprint/result/model/ST-GCN'
        MODEL_SAVE_N = 'best_9_2_stgcn.pth'
        model = module.STGCNEncoder(output_dim=64).to(device)
    elif select == 'GRU':
        save_dir = 'D:/Dataset/sprint/result/model/GRU'
        MODEL_SAVE_N = 'best_7_1_gru.pth'
        model = module.GRUEncoder(output_dim=64).to(device) 
    elif select == 'LSTM':
        save_dir = 'D:/Dataset/sprint/result/model/LSTM'
        MODEL_SAVE_N = 'best_7_1_lstm.pth'
        model = module.LSTMEncoder(output_dim=64).to(device) 
    elif select == 'MLP':
        save_dir = 'D:/Dataset/sprint/result/model/MLP'
        MODEL_SAVE_N = 'best_7_1_mlp.pth'
        model = module.MLPEncoder(output_dim=64).to(device) 
    elif select == 'TCN':
        save_dir = 'D:/Dataset/sprint/result/model/TCN'
        MODEL_SAVE_N = 'best_7_1_tcn.pth'
        model = module.TCNEncoder(output_dim=64).to(device) 
    elif select == 'CTR':
        save_dir = 'D:/Dataset/sprint/result/model/CTR-GCN'
        MODEL_SAVE_N = 'best_9_2_ctr.pth'
        model = module.CTRGCNEncoder(in_channels=2, output_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ckpt_path = os.path.join(save_dir, 'last_checkpoint.pth')
    start_epoch = 0
    best_loss = float('inf')
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt['best_loss']
        print(f"续训: 从 epoch {start_epoch} 开始, best_loss={best_loss:.6f}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
                        persistent_workers=(num_workers > 0), worker_init_fn=worker_init_fn)
    os.makedirs(save_dir, exist_ok=True)
    loss_history = []
    pos_history = []
    neg_history = []
    diff_history = []

    raw_windows = dataset.windows
    print(f"使用模型：{select}")
    frame_features = precompute_frame_features(raw_windows)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        total_diff = 0.0
        total_masked = 0
        total_neg_pairs = 0
        num_batches = 0
        for anchor, positive, indices in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            batch_indices = indices.tolist()
            z1 = model(anchor)
            z2 = model(positive)
            loss, pos_avg, neg_avg, diff, masked_count = nt_xent_loss(
                z1, z2, temperature,
                frame_features=frame_features,
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
            total_masked += masked_count
            total_neg_pairs += 2 * batch_size * (2 * batch_size - 2)
            num_batches += 1
        avg_loss = total_loss / num_batches
        avg_pos = total_pos / num_batches
        avg_neg = total_neg / num_batches
        avg_diff = total_diff / num_batches
        loss_history.append(avg_loss)
        pos_history.append(avg_pos)
        neg_history.append(avg_neg)
        diff_history.append(avg_diff)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, "
                f"PosSim: {avg_pos:.4f}, NegSim: {avg_neg:.4f}, Diff: {avg_diff:.4f}, "
                f"MaskedNeg: {total_masked}/{total_neg_pairs} ({total_masked/total_neg_pairs*100:.1f}%)")
        # best 模型保持纯 state_dict，兼容原下游加载；完整状态存 last_checkpoint 供续训
        torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(),
                    'epoch': epoch, 'best_loss': best_loss}, ckpt_path)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, MODEL_SAVE_N))
            print(f"  -> 保存最佳模型，loss={avg_loss:.6f}")
    print("训练完成")
    from thop import profile, clever_format
    model.eval()
    dummy = torch.randn(1, 2, 9, 17).to(device)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    flops_f, params_f = clever_format([flops, params], "%.3f")
    print(f"参数量: {params_f} | FLOPs: {flops_f}")

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
    set_seed(0) #设置种子
    npz_path = 'D:/Dataset/sprint/result/window_data/dataset_9_2.npz'
    dataset = ContrastiveDatasetFromFile(
        npz_path,
        window_size=9,
        transform_params={'rotation':15, 'scale':0.15, 'noise':0.05, 'mask':0.1,
                        'reverse':0.15, 'GB':0.25, 'shear':0.1, 'flip':0.15, 'delete':0.15}
    )

    train_contrastive(dataset, epochs=100, batch_size=512, lr=0.001, temperature=0.3,
                    diversity_threshold=0.9, select='CTR', num_workers=4)
                    # resume_path='D:/Dataset/sprint/result/model/CTR-GCN/last_checkpoint.pth'
    """
    STGCN,GRU,LSTM,MLP,TCN,CTR
    """
    elapsed = show_time(start_time, time.time())
    print(f"Total time: {elapsed}")