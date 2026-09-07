import os
import re
import json
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ======================= 需要修改的区域 =======================
SCORES_PATH   = r'D:/Dataset/sprint/result/video_point/scores1.json'  # 得分文件
KEYPOINTS_DIR = r'D:/Dataset/sprint/result/features'                 # run_X_normalized_points.npy 所在目录
# ==============================================================

CONFIG = {
    'target_len': 300,          # 统一帧数（不足则最近帧重复补齐）
    'min_frames': 100,          # 有效帧少于该值的视频直接丢弃
    'in_channels': 2,           # 关键点 npy 是 (T,17,2)，保持 2
    'batch_size': 4,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'epochs': 200,
    'patience': 40,
    'seed': 42,
    'train_ratio': 0.9,
    'val_ratio': 0.1,
    'augment': True,
    'test_used': False,          # 训练后是否评估 test 集
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': r'D:/Dataset/sprint/result/model/T-STGCN/best_model.pth',
}


# -------------------- COCO 图与 STGCN 模块 --------------------
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
    """
    在原有 STGC_block 基础上增加可选的残差连接
    """
    def __init__(self, in_channels, out_channels, stride, t_kernel_size,
                 A_size, dropout=0.2, use_residual=True):
        super().__init__()
        self.sgc = SpatialGraphConvolution(in_channels, out_channels, A_size[0])
        self.M = nn.Parameter(torch.ones(A_size))
        self.tgc = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, (t_kernel_size, 1), (stride, 1),
                      ((t_kernel_size - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.use_residual = use_residual
        if use_residual:
            if in_channels != out_channels or stride != 1:
                # 通道数或时间长度变化时，用 1x1 卷积对齐残差分支
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=(stride, 1), bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()

    def forward(self, x, A):
        out = self.tgc(self.sgc(x, A * self.M))
        if self.use_residual:
            out = out + self.shortcut(x)
        return out


class STGCNEncoder(nn.Module):
    def __init__(self, in_channels=2, t_kernel_size=3, hop_size=2):
        super().__init__()
        graph = COCOGraph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        self.bn = nn.BatchNorm1d(in_channels * graph.num_node)

        # 注意：第一层不开残差（因为输入 2 通道直接跳到 64，变化太大）
        self.stgc1 = STGC_block(in_channels, 64, 1, t_kernel_size, A_size,
                                 dropout=0.1, use_residual=False)
        self.stgc2 = STGC_block(64, 64, 1, t_kernel_size, A_size,
                                 dropout=0.15, use_residual=True)
        self.stgc3 = STGC_block(64, 128, 2, t_kernel_size, A_size,
                                 dropout=0.15, use_residual=True)   # 300 -> 150
        self.stgc4 = STGC_block(128, 128, 1, t_kernel_size, A_size,
                                 dropout=0.15, use_residual=True)
        self.stgc5 = STGC_block(128, 128, 1, t_kernel_size, A_size,
                                 dropout=0.15, use_residual=True)

        self.reg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)     # 时间维下采样
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(N, -1)   # (N, 128)
        return self.reg_head(x)
# -------------------- 数据工具 --------------------
def fix_length(seq, T, random_crop=True):
    """统一到 T 帧。训练用随机裁剪，验证/测试用均匀采样。"""
    n = seq.shape[0]
    if n >= T:
        if random_crop:
            start = np.random.randint(0, n - T + 1)
            return seq[start:start + T]
        idx = np.linspace(0, n - 1, T).astype(int)
        return seq[idx]
    idx = np.linspace(0, n - 1, T)
    return np.stack([seq[min(int(i), n - 1)] for i in idx])


def augment_sequence(seq):
    """seq: (T,V,C)。旋转/缩放基于髋中心坐标系。"""
    T, V, C = seq.shape
    seq = seq.copy()
    angle = np.random.uniform(-10, 10) * np.pi / 180.0
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    seq[..., :2] = seq[..., :2] @ R.T
    scale = np.random.uniform(0.9, 1.1)
    seq[..., :2] *= scale
    if np.random.rand() < 0.3:
        num_mask = np.random.randint(1, 3)
        joints = np.random.choice(V, num_mask, replace=False)
        seq[:, joints, :] = 0.0
    return seq


class SkeletonScoreDataset(Dataset):
    """样本 = (vid, seq(T,17,2), score)"""
    def __init__(self, samples, target_len=300, train=True, augment=True):
        self.samples = samples
        self.target_len = target_len
        self.train = train
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, seq, score = self.samples[idx]
        seq = fix_length(seq, self.target_len, random_crop=self.train)
        if self.train and self.augment:
            seq = augment_sequence(seq)
        seq = seq.transpose(2, 0, 1)          # (C,T,V)
        return torch.from_numpy(seq.copy()), torch.tensor(score, dtype=torch.float32)


def pearson(pred, y):
    pred = pred - pred.mean()
    y = y - y.mean()
    denom = np.sqrt((pred**2).sum() * (y**2).sum()) + 1e-8
    return float((pred * y).sum() / denom)


def evaluate(model, loader, mean, std, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for seq, y in loader:
            pred = model(seq.to(device)).squeeze(-1).cpu()
            pred = pred * std + mean
            preds.append(pred)
            targets.append(y)
    if not preds:
        return float('nan'), float('nan'), float('nan')
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mae = float(np.abs(preds - targets).mean())
    rmse = float(np.sqrt(((preds - targets)**2).mean()))
    r = pearson(preds, targets)
    return mae, rmse, r


# -------------------- 加载与清洗关键点 --------------------
def load_samples(scores_path, kp_dir, min_frames):
    with open(scores_path, 'r', encoding='utf-8') as f:
        raw_scores = json.load(f)
    scores = {str(k).strip(): float(v) for k, v in raw_scores.items()}

    # 兼容 key 写成 "run_2.mp4" 的情况
    for k in list(scores.keys()):
        scores.setdefault(os.path.splitext(k)[0], scores[k])

    pattern = re.compile(r'^(.*)_normalized_points\.npy$')
    npy_files = sorted(glob.glob(os.path.join(kp_dir, '*_normalized_points.npy')))

    samples, skipped = [], []
    for f in npy_files:
        name = os.path.basename(f)
        m = pattern.match(name)
        if not m:
            continue
        vid = m.group(1)
        if vid not in scores:
            skipped.append(f"跳过 {name}: 无对应分数")
            continue

        arr = np.load(f).astype(np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (17, 2):
            skipped.append(f"跳过 {name}: 形状异常 {arr.shape}")
            continue

        # 剔除无人帧（全零行）
        mask = np.any(arr.reshape(arr.shape[0], -1) != 0, axis=1)
        seq = arr[mask]

        if len(seq) < min_frames:
            skipped.append(f"跳过 {name}: 有效帧 {len(seq)} < {min_frames}")
            continue

        samples.append((vid, seq, scores[vid]))

    return samples, skipped


# -------------------- 主流程 --------------------
def main(scores_path, kp_dir):
    cfg = CONFIG
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    model_dir = os.path.dirname(cfg['save_path']) or '.'
    os.makedirs(model_dir, exist_ok=True)
    print(f"device: {cfg['device']}")

    # 1. 加载 + 清洗
    samples, skipped = load_samples(scores_path, kp_dir, cfg['min_frames'])
    for s in skipped:
        print(s)
    if not samples:
        raise RuntimeError("没有可用样本：请检查 scores 文件名与 npy 命名是否一致")
    print(f"\n可用样本 {len(samples)} 个")

    # ===== 新增：打印有效帧长度分布，判断 target_len 是否合理 =====
    lengths = [len(s[1]) for s in samples]
    print(f"有效帧长度: min {min(lengths)}, median {int(np.median(lengths))}, max {max(lengths)}")
    # =============================================================

    # 2. 按视频划分
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * cfg['train_ratio'])
    n_val   = int(n * cfg['val_ratio'])
    train_samples = samples[:n_train]
    val_samples   = samples[n_train:n_train + n_val]
    test_samples  = samples[n_train + n_val:]
    print(f"train {len(train_samples)} | val {len(val_samples)} | test {len(test_samples)}")

    # 3. 只用训练集算归一化统计量
    train_scores = np.array([s[2] for s in train_samples])
    mean, std = float(train_scores.mean()), float(train_scores.std()) + 1e-6
    print(f"score mean {mean:.3f}, std {std:.3f}")

    # 4. 保存统计量（predict.py 从这里读取，避免手填错误）
    stats_path = os.path.join(model_dir, 'score_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({'mean': mean, 'std': std}, f, indent=2)
    print(f"得分统计量已保存: {stats_path}")

    # 5. 数据集
    train_ds = SkeletonScoreDataset(train_samples, cfg['target_len'],
                                    train=True,  augment=cfg['augment'])
    val_ds   = SkeletonScoreDataset(val_samples,   cfg['target_len'],
                                    train=False, augment=False)
    test_ds  = SkeletonScoreDataset(test_samples,  cfg['target_len'],
                                    train=False, augment=False) if test_samples else None

    drop_last = len(train_ds) > cfg['batch_size']
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True, drop_last=drop_last, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False, num_workers=0) if test_ds else None

    # 6. 模型
    model = STGCNEncoder(in_channels=cfg['in_channels']).to(cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = nn.SmoothL1Loss()

    # 7. 训练 + 早停
    best_val_mae = float('inf')
    patience = 0
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for seq, y in train_loader:
            seq, y = seq.to(cfg['device']), y.to(cfg['device'])
            pred = model(seq).squeeze(-1)
            loss = criterion(pred, (y - mean) / std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq.size(0)
        train_loss = total_loss / max(len(train_ds), 1)

        val_mae, val_rmse, val_r = evaluate(model, val_loader, mean, std, cfg['device'])
        print(f"[Epoch {epoch:3d}] train_loss {train_loss:.4f} | "
              f"val MAE {val_mae:.4f} RMSE {val_rmse:.4f} R {val_r:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience = 0
            torch.save(model.state_dict(), cfg['save_path'])
            print(f"  -> saved best model (val MAE {val_mae:.4f})")
        else:
            patience += 1
            if patience >= cfg['patience']:
                print(f"Early stop at epoch {epoch}")
                break

    print(f"\n最优模型已保存: {cfg['save_path']} (val MAE {best_val_mae:.4f})")

    # 8. 测试集评估（可选）
    if cfg['test_used'] and test_loader is not None:
        model.load_state_dict(torch.load(cfg['save_path'], map_location=cfg['device']))
        t_mae, t_rmse, t_r = evaluate(model, test_loader, mean, std, cfg['device'])
        print(f"Test  MAE {t_mae:.4f} RMSE {t_rmse:.4f} R {t_r:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', default=SCORES_PATH)
    parser.add_argument('--keypoints', default=KEYPOINTS_DIR)
    args = parser.parse_args()
    main(args.scores, args.keypoints)