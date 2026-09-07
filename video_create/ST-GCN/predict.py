# -*- coding: utf-8 -*-
"""
STGCN 跑步评分批量预测 + 计算斯皮尔曼系数

用法:
  1. 将需要预测的 run_X_normalized_points.npy 放在一个文件夹内
  2. 配置 CONFIG 中的路径、人工评分文件、测试数量等
  3. 运行: python predict.py
"""
import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# ==================== 配置 ====================
CONFIG = {
    # 可指向：
    #   1) 单个 .npy 文件（只能输出单个样本分数，不能算斯皮尔曼）
    #   2) 一个文件夹（里面含多个 run*_normalized_points.npy，将批量预测）
    'npy_path': r'D:/Dataset/sprint/result/features',

    'model_path': r'D:/Dataset/sprint/result/model/T-STGCN/best_model.pth',
    # 反归一化统计量：train.py 训练时保存的 score_stats.json
    'stats_path': r'D:/Dataset/sprint/result/model/T-STGCN/score_stats.json',
    # 仅当 stats_path 不存在时使用手填值（必须等于训练时打印）
    'score_mean': 68.864,
    'score_std': 9.903,

    # 人工评分文件，格式: {"run_1": 88.2, "run_2": 91.0, ...}
    'scores_file': r'D:/Dataset/sprint/result/video_point/scores2.json',

    # 特征文件的固定后缀（只处理该后缀的 .npy）
    'feature_suffix': '_normalized_points.npy',

    # 固定选取的视频编号范围：9 到 19，含 9 和 19
    'FIXED_VIDEO_NUM_RANGE': (9, 19),

    # 原随机抽样参数，固定范围模式下不会使用
    'N_TEST_VIDEOS': None,
    'RANDOM_SEED': None,

    # 需要排除的视频名称（比如模板 "run_6"），可留空列表
    'EXCLUDE_VIDEOS': [],

    # 输出 CSV 路径
    'output_csv': r'result/spearman_predict_results.csv',

    'in_channels': 2,          # 与训练一致
    'target_len': 300,         # 与训练一致
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# ==================== 模型结构（与训练完全一致） ====================
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

        self.stgc1 = STGC_block(in_channels, 64, 1, t_kernel_size, A_size,
                                 dropout=0.1, use_residual=False)
        self.stgc2 = STGC_block(64, 64, 1, t_kernel_size, A_size,
                                 dropout=0.15, use_residual=True)
        self.stgc3 = STGC_block(64, 128, 2, t_kernel_size, A_size,
                                 dropout=0.15, use_residual=True)
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
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(N, -1)
        return self.reg_head(x)

# -------------------- 工具函数 --------------------
def fix_length_infer(seq, T):
    n = seq.shape[0]
    if n >= T:
        idx = np.linspace(0, n - 1, T).astype(int)
        return seq[idx]
    idx = np.linspace(0, n - 1, T)
    return np.stack([seq[min(int(i), n - 1)] for i in idx])


def load_and_clean(npy_path):
    seq = np.load(npy_path).astype(np.float32)
    if seq.ndim != 3 or seq.shape[1:] != (17, 2):
        raise ValueError(f"{npy_path} 形状应为 (T,17,2)，实际 {seq.shape}")
    mask = np.any(seq.reshape(seq.shape[0], -1) != 0, axis=1)
    seq = seq[mask]
    if len(seq) == 0:
        raise ValueError(f"{npy_path} 剔除无人帧后没有有效帧")
    return seq


def preprocess(npy_path, cfg):
    seq = load_and_clean(npy_path)
    seq = fix_length_infer(seq, cfg['target_len'])
    seq = seq.transpose(2, 0, 1)
    return torch.from_numpy(seq.copy()).unsqueeze(0)


def load_stats(cfg):
    if os.path.exists(cfg['stats_path']):
        with open(cfg['stats_path'], 'r', encoding='utf-8') as f:
            stats = json.load(f)
        return float(stats['mean']), float(stats['std'])
    if cfg['score_mean'] is not None and cfg['score_std'] is not None:
        return float(cfg['score_mean']), float(cfg['score_std'])
    raise ValueError("缺少 score_stats.json，且未配置手填 mean/std")


def extract_video_name(npy_filename, suffix):
    name = os.path.basename(npy_filename)
    if name.endswith(suffix):
        return name[:-len(suffix)]
    if name.endswith('.npy'):
        return name[:-4]
    return name


def load_human_scores(scores_file):
    with open(scores_file, 'r', encoding='utf-8') as f:
        scores = json.load(f)
    return {k: float(v) for k, v in scores.items() if v is not None}


def predict_one(model, npy_path, mean, std, cfg):
    x = preprocess(npy_path, cfg).to(cfg['device'])
    model.eval()
    with torch.no_grad():
        pred_norm = model(x).squeeze(-1).item()
    return pred_norm * std + mean


def parse_video_num(video_name):
    """从 'run_9' 中提取数字 9，失败返回 None"""
    tail = video_name.rsplit('_', 1)[-1]
    return int(tail) if tail.isdigit() else None


# -------------------- 主函数 --------------------
def main():
    cfg = CONFIG
    path = cfg['npy_path']

    # 1. 准备模型
    if not os.path.exists(cfg['model_path']):
        raise FileNotFoundError(f"模型不存在: {cfg['model_path']}")
    mean, std = load_stats(cfg)
    print(f"device: {cfg['device']} | 反归一化 mean={mean:.4f}, std={std:.4f}")

    model = STGCNEncoder(in_channels=cfg['in_channels']).to(cfg['device'])
    model.load_state_dict(torch.load(cfg['model_path'], map_location=cfg['device']))
    print(f"模型已加载: {cfg['model_path']}")

    # 2. 确定候选 .npy 文件列表
    if os.path.isdir(path):
        suffix = cfg['feature_suffix']
        npy_files = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.endswith(suffix)
        )
        if not npy_files:
            npy_files = sorted(
                os.path.join(path, f) for f in os.listdir(path)
                if f.endswith('.npy')
            )
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在: {path}")
        npy_files = [path]

    if not npy_files:
        print("没有找到任何 .npy 文件")
        return

    # 3. 加载人工评分
    human_scores = load_human_scores(cfg['scores_file'])

    # 4. 关联视频名，并过滤无人评分或需要排除的视频
    paired_files = []
    for f in npy_files:
        video_name = extract_video_name(f, cfg.get('feature_suffix', '_normalized_points.npy'))
        if video_name in cfg['EXCLUDE_VIDEOS']:
            continue
        if video_name not in human_scores:
            print(f"跳过 {video_name}（没有人工评分）")
            continue
        paired_files.append((video_name, f))

    if not paired_files:
        print("没有找到既有特征文件又有人工评分的样本")
        return

    print(f"候选可测试样本数: {len(paired_files)}")

    # 5. 固定选择 run_9 到 run_19
    range_cfg = cfg.get('FIXED_VIDEO_NUM_RANGE')
    if range_cfg is not None:
        min_id, max_id = range_cfg
        selected = []
        for video_name, npy_file in paired_files:
            num = parse_video_num(video_name)
            if num is not None and min_id <= num <= max_id:
                selected.append((video_name, npy_file, num))

        # 按编号排序输出
        selected.sort(key=lambda x: x[2])
        selected = [(name, f) for name, f, _ in selected]

        if not selected:
            print(f"固定视频编号 {min_id}~{max_id} 范围内没有可测试样本")
            return

        print(f"固定测试范围: {min_id}~{max_id}，实际测试样本数: {len(selected)}")
    else:
        # 如果没设置固定范围，则使用原来的随机抽样逻辑
        import random
        n_test = cfg.get('N_TEST_VIDEOS')
        if n_test is not None:
            if n_test > len(paired_files):
                n_test = len(paired_files)
                print(f"指定数量大于候选数，自动使用全部 {n_test} 个样本")
            seed = cfg.get('RANDOM_SEED')
            if seed is not None:
                random.seed(seed)
            selected = random.sample(paired_files, n_test)
        else:
            selected = paired_files

        print(f"实际测试样本数: {len(selected)}")

    # 6. 逐个预测并收集结果
    records = []
    for idx, (video_name, npy_file) in enumerate(selected, 1):
        try:
            pred = predict_one(model, npy_file, mean, std, cfg)
            true = human_scores[video_name]
            records.append((video_name, true, pred))
            print(f"[{idx}/{len(selected)}] {video_name}: "
                  f"人工={true:.2f}, 预测={pred:.2f}, 差={pred-true:+.2f}")
        except Exception as e:
            print(f"[{idx}/{len(selected)}] {video_name} 预测失败: {e}")

    # 7. 计算斯皮尔曼系数
    if len(records) < 2:
        print("\n有效样本不足 2 个，无法计算斯皮尔曼系数")
        return

    human_list = [r[1] for r in records]
    pred_list = [r[2] for r in records]
    rho, p = spearmanr(pred_list, human_list)

    print("\n========== 斯皮尔曼相关系数（模型预测 vs 人工评分） ==========")
    print(f"预测得分 vs 人工评分：rho = {rho:.4f}, p = {p:.4f}")
    print("==============================================================")

    # 8. 保存结果
    os.makedirs(os.path.dirname(cfg['output_csv']), exist_ok=True)
    with open(cfg['output_csv'], 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'human_score', 'pred_score'])
        for name, true, pred in records:
            writer.writerow([name, f"{true:.3f}", f"{pred:.3f}"])
    print(f"\n详细结果已保存至：{cfg['output_csv']}")


if __name__ == '__main__':
    main()