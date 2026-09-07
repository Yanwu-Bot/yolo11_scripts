#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分差回归 Transformer 训练脚本（修复版）
数据集：pair_dataset_all_train_930.pkl
"""

import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

# ==================== 可配置区域 ====================

# 数据路径
DATA_PATH = 'D:/Dataset/sprint/result/diff_dataset/pair_dataset.pkl'
OUTPUT_DIR = 'D:/Dataset/sprint/result/models'
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'transformer_diff_reg.pth')

# 训练配置
SEED = 42
VAL_QUERY_NUM = 6              # 从31个查询视频中取出6个作为验证
EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Transformer 结构参数
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# 是否使用 GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 关键修复：统一将特征转为二维 (T, D) ====================

def to_2d(arr):
    """
    将数组统一处理成二维形状 (T, D)。
    如果输入是三维 (T, H, W)，则展平为 (T, H*W)。
    如果输入已经是二维，保持不变。
    """
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    return arr


# ==================== 1. 数据集定义 ====================

class PairDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # 对每个通道都做 to_2d 处理，防止某些特征是三维的
        fea_diff = to_2d(rec['fea_diff'])
        point_diff = to_2d(rec['point_diff'])
        vector_diff = to_2d(rec['vector_diff'])

        # 重新调整为 float32
        fea_diff = np.asarray(fea_diff, dtype=np.float32)
        point_diff = np.asarray(point_diff, dtype=np.float32)
        vector_diff = np.asarray(vector_diff, dtype=np.float32)

        # 拼接三个通道的逐帧差分 -> (T, D_total)
        diff = np.concatenate([fea_diff, point_diff, vector_diff], axis=-1)

        label = float(rec['score_diff'])

        return diff, label


def pad_collate(batch):
    """动态padding到batch内最长序列，返回 (padded, mask, labels)"""
    seqs, labels = zip(*batch)

    lens = [len(s) for s in seqs]
    max_len = max(lens)
    feat_dim = seqs[0].shape[-1]

    padded = np.zeros((len(seqs), max_len, feat_dim), dtype=np.float32)
    masks = np.ones((len(seqs), max_len), dtype=bool)  # True表示padding位置

    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
        masks[i, :len(s)] = False

    padded = torch.from_numpy(padded).to(DEVICE)
    masks = torch.from_numpy(masks).to(DEVICE)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1).to(DEVICE)

    return padded, masks, labels


def load_and_split(data_path, val_query_num=6):
    with open(data_path, 'rb') as f:
        records = pickle.load(f)

    # 按查询视频分组
    query_to_records = defaultdict(list)
    for rec in records:
        query_to_records[rec['que_name']].append(rec)

    all_query_names = sorted(query_to_records.keys())
    print(f"查询视频总数: {len(all_query_names)}")
    print(f"总样本数: {len(records)}")

    # 随机抽取 val_query_num 个查询视频作为验证集
    random.shuffle(all_query_names)
    val_queries = set(all_query_names[:val_query_num])
    train_queries = set(all_query_names[val_query_num:])

    train_records = []
    val_records = []

    for rec in records:
        if rec['que_name'] in val_queries:
            val_records.append(rec)
        else:
            train_records.append(rec)

    print(f"训练样本数: {len(train_records)} (查询视频数: {len(train_queries)})")
    print(f"验证样本数: {len(val_records)} (查询视频数: {val_query_num})")
    print("验证查询视频:", sorted(val_queries))

    return train_records, val_records


# ==================== 2. Transformer模型 ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0)]


class DiffTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, mask):
        # x: (batch, seq_len, input_dim)
        # mask: (batch, seq_len), True表示padding位置

        x = self.input_proj(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # 对非padding位置做平均池化
        mask_expand = ~mask.unsqueeze(-1)          # (batch, seq, 1)
        sum_x = (x * mask_expand).sum(dim=1)
        count = mask_expand.float().sum(dim=1).clamp(min=1.0)
        pooled = sum_x / count

        out = self.reg_head(pooled)
        return out


# ==================== 3. 评估函数 ====================

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, mask, label in dataloader:
            pred = model(x, mask)
            all_preds.append(pred.cpu().numpy().flatten())
            all_labels.append(label.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    spearman = spearmanr(all_preds, all_labels).correlation
    if np.isnan(spearman):
        spearman = 0.0

    mse = np.mean((all_preds - all_labels) ** 2)

    return spearman, mse


# ==================== 4. 主训练循环 ====================

def main():
    print("=" * 50)
    print("加载数据...")
    train_records, val_records = load_and_split(DATA_PATH, VAL_QUERY_NUM)

    train_dataset = PairDataset(train_records)
    val_dataset = PairDataset(val_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=0,
    )

    # 确定输入维度
    sample_diff, _ = train_dataset[0]
    input_dim = sample_diff.shape[-1]
    print(f"输入特征维度: {input_dim}")

    model = DiffTransformer(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_spearman = -1.0
    best_mse = float('inf')
    best_epoch = 0

    print("\n开始训练...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for x, mask, label in train_loader:
            optimizer.zero_grad()
            pred = model(x, mask)
            loss = criterion(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_train_loss = total_loss / num_batches

        val_spearman, val_mse = evaluate(model, val_loader)

        if (epoch % 5 == 0) or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val MSE: {val_mse:.4f} | "
                  f"Val Spearman: {val_spearman:.4f}")

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            best_mse = val_mse
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  → 保存最佳模型 (Spearman={best_spearman:.4f}) 到 {MODEL_SAVE_PATH}")

    print("\n训练结束。")
    print(f"最佳验证 Spearman: {best_spearman:.4f} (Epoch {best_epoch})")
    print(f"最佳验证 MSE: {best_mse:.4f}")


if __name__ == '__main__':
    main()