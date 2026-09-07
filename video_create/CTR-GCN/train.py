import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================== 配置 ====================
CONFIG = {
    'skeleton_dir': 'D:/Dataset/sprint/result/video_point',  # npy 文件目录
    'scores_path': 'D:/Dataset/sprint/result/video_point/scores.json',                   # 得分文件
    'target_len': 300,                              # 统一帧数
    'in_channels': 2,                               # 提取时 add_velocity=True 则改 4
    'batch_size': 8,                                # 160 样本建议 8~16
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 100,
    'patience': 30,                                 # 早停
    'seed': 42,
    'train_ratio': 0.8,
    'val_ratio': 0.2,
    'augment': True,                                # 训练数据增强
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': 'D:/Dataset/sprint/result/model/T-STGCN/best_model.pth',
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
        return F.normalize(x, p=2, dim=1)                           # (N, 1)

def fix_length(seq, T, random_crop=True):
    """把序列统一到 T 帧。训练用随机裁剪，验证/测试用均匀采样。"""
    n = seq.shape[0]
    if n >= T:
        if random_crop:
            start = np.random.randint(0, n - T + 1)
            return seq[start:start + T]
        idx = np.linspace(0, n - 1, T).astype(int)
        return seq[idx]
    # 不足 T 帧：插值（重复最近帧）
    idx = np.linspace(0, n - 1, T)
    return np.stack([seq[min(int(i), n - 1)] for i in idx])


def augment_sequence(seq):
    """seq: (T, V, C)，返回增强后的序列。只在训练时用。"""
    T, V, C = seq.shape
    seq = seq.copy()
    # 随机旋转（绕原点，人体绕重力轴旋转对跑步语义不变）
    angle = np.random.uniform(-10, 10) * np.pi / 180.0
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    seq[..., :2] = seq[..., :2] @ R.T
    # 随机缩放
    scale = np.random.uniform(0.9, 1.1)
    seq[..., :2] *= scale
    # 随机遮蔽 1~2 个关节
    if np.random.rand() < 0.3:
        num_mask = np.random.randint(1, 3)
        joints = np.random.choice(V, num_mask, replace=False)
        seq[:, joints, :] = 0.0
    return seq


class RunningVideoDataset(Dataset):
    def __init__(self, video_ids, scores, skeleton_dir,
                 target_len=300, train=True, augment=True):
        # 只保留 npy 文件真实存在的视频
        self.video_ids = [v for v in video_ids
                          if os.path.exists(os.path.join(skeleton_dir, v + '.npy'))]
        self.scores = scores
        self.skeleton_dir = skeleton_dir
        self.target_len = target_len
        self.train = train
        self.augment = augment

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        seq = np.load(os.path.join(self.skeleton_dir, vid + '.npy')).astype(np.float32)  # (T,V,C)
        seq = fix_length(seq, self.target_len, random_crop=self.train)
        if self.train and self.augment:
            seq = augment_sequence(seq)
        seq = seq.transpose(2, 0, 1)          # (C, T, V)
        score = float(self.scores[vid])
        return torch.from_numpy(seq), torch.tensor(score, dtype=torch.float32)

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
            pred = pred * std + mean           # 反归一化
            preds.append(pred)
            targets.append(y)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mae = float(np.abs(preds - targets).mean())
    rmse = float(np.sqrt(((preds - targets)**2).mean()))
    r = pearson(preds, targets)
    return mae, rmse, r

def main():
    cfg = CONFIG
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    print(f"device: {cfg['device']}")

    # 1. 加载得分
    with open(cfg['scores_path'], 'r', encoding='utf-8') as f:
        scores = json.load(f)
    scores = {k.strip(): float(v) for k, v in scores.items()}
    print(f"共 {len(scores)} 个视频有得分")

    # 2. 按视频划分：只分 train / val（test 暂时不碰）
    video_ids = list(scores.keys())
    random.shuffle(video_ids)
    n = len(video_ids)
    n_train = int(n * cfg['train_ratio'])
    n_val = int(n * cfg['val_ratio'])
    train_ids = video_ids[:n_train]
    val_ids   = video_ids[n_train:n_train + n_val]
    # test_ids 不在这里用，等模型定型后再单独做最终验收
    test_ids  = video_ids[n_train + n_val:]
    print(f"train {len(train_ids)} | val {len(val_ids)} | "
          f"test(暂不使用) {len(test_ids)}")

    # 3. 得分归一化（只在训练集上算统计量）
    train_scores = np.array([scores[v] for v in train_ids])
    mean, std = train_scores.mean(), train_scores.std() + 1e-6
    print(f"score mean {mean:.3f}, std {std:.3f}")

    # 4. 数据集：只建 train / val
    train_ds = RunningVideoDataset(train_ids, scores, cfg['skeleton_dir'],
                                    cfg['target_len'], train=True,  augment=cfg['augment'])
    val_ds   = RunningVideoDataset(val_ids,   scores, cfg['skeleton_dir'],
                                    cfg['target_len'], train=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                            shuffle=True, drop_last=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    # 5. 模型 / 优化器 / 损失
    model = CTRGCNEncoder(in_channels=cfg['in_channels']).to(cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = nn.SmoothL1Loss()

    # 6. 训练 + 早停（只看 val）
    best_val_mae = float('inf')
    patience = 0
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for seq, y in train_loader:
            seq, y = seq.to(cfg['device']), y.to(cfg['device'])
            pred = model(seq).squeeze(-1)                     # (N,)
            loss = criterion(pred, (y - mean) / std)          # 归一化后回归
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq.size(0)
        train_loss = total_loss / len(train_ds)

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

    print(f"\n训练完成，最优模型已保存至 {cfg['save_path']} (val MAE {best_val_mae:.4f})")

if __name__ == '__main__':
    main()