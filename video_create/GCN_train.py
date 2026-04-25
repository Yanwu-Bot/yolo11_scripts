import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from GCN import PoseGCN

# ========== 数据集类 ==========
class PosePairDataset(Dataset):
    def __init__(self, kp1, kp2, labels):
        self.kp1 = torch.from_numpy(kp1).float()
        self.kp2 = torch.from_numpy(kp2).float()
        self.labels = torch.from_numpy(labels).float()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.kp1[idx], self.kp2[idx], self.labels[idx]

# ========== 对比损失函数 ==========
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_pos = label * torch.pow(distance, 2)
        loss_neg = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss = torch.mean(loss_pos + loss_neg) / 2
        return loss

# ========== 加载数据集 ==========
def load_dataset(load_path='result/GCN/training_data.npz'):
    data = np.load(load_path)
    return data['keypoints1'], data['keypoints2'], data['labels']

# ========== 训练函数 ==========
def train_gcn(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_loss = float('inf')
    
    print(f"\n开始训练 GCN...")
    print(f"设备: {device}")
    print(f"学习率: {lr}")
    print(f"训练轮数: {epochs}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for kp1, kp2, labels in train_loader:
            kp1 = kp1.to(device)
            kp2 = kp2.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            feat1 = model(kp1)
            feat2 = model(kp2)
            loss = criterion(feat1, feat2, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for kp1, kp2, labels in val_loader:
                kp1 = kp1.to(device)
                kp2 = kp2.to(device)
                labels = labels.to(device)
                feat1 = model(kp1)
                feat2 = model(kp2)
                val_loss += criterion(feat1, feat2, labels).item()
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs('result/GCN', exist_ok=True)
            torch.save(model.state_dict(), 'result/GCN/best_model.pth')
            print(f"  -> 保存最佳模型 (Val Loss: {best_loss:.4f})")
    
    print(f"{'='*60}")
    print(f"训练完成！最佳模型保存至: result/GCN/best_model.pth")

# ========== 主函数 ==========
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 检查数据集
    dataset_path = 'result/GCN/training_data.npz'
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集不存在 {dataset_path}")
        exit()
    
    # 2. 加载数据集
    print(f"加载数据集: {dataset_path}")
    kp1, kp2, labels = load_dataset(dataset_path)
    print(f"  关键点1形状: {kp1.shape}")
    print(f"  关键点2形状: {kp2.shape}")
    print(f"  标签形状: {labels.shape}")
    print(f"  正样本: {np.sum(labels==1)} | 负样本: {np.sum(labels==0)}")
    
    # 3. 划分训练集和验证集
    n_samples = len(labels)
    n_train = int(n_samples * 0.8)
    indices = np.random.permutation(n_samples)
    
    train_kp1 = kp1[indices[:n_train]]
    train_kp2 = kp2[indices[:n_train]]
    train_labels = labels[indices[:n_train]]
    
    val_kp1 = kp1[indices[n_train:]]
    val_kp2 = kp2[indices[n_train:]]
    val_labels = labels[indices[n_train:]]
    
    print(f"训练集: {len(train_labels)} 对 | 验证集: {len(val_labels)} 对")
    
    # 4. 创建数据加载器
    train_dataset = PosePairDataset(train_kp1, train_kp2, train_labels)
    val_dataset = PosePairDataset(val_kp1, val_kp2, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 5. 创建模型
    model = PoseGCN(out_features=16)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. 训练
    train_gcn(model, train_loader, val_loader, epochs=50, lr=0.01, device=device)