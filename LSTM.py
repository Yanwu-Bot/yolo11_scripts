import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from matplotlib import rcParams #字体
rcParams['font.family'] = 'SimHei'

# ============================ 配置项 ============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")

DATASET_DIR = "video_dataset"
MODEL_SAVE_PATH = os.path.join("model", "running_anomaly_lstm_pytorch.pth")
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 8
FACTOR = 0.5
MIN_LR = 1e-6
INPUT_DIM = 34
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
THRESHOLD = 0.4 #检测阈值
# =================================================================

# -------------------------- 1. 修复数据集类 --------------------------
class RunningPoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N,1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------- 2. 修复数据加载（类别权重处理） --------------------------
def load_dataset():
    # 加载数据
    train_data = np.load(os.path.join(DATASET_DIR, "train.npz"))
    X_train = train_data["X"]
    y_train = train_data["y"]
    
    test_data = np.load(os.path.join(DATASET_DIR, "test.npz"))
    X_test = test_data["X"]
    y_test = test_data["y"]
    
    print("✅ 数据集加载完成：")
    print(f"   训练集：X={X_train.shape}, y={y_train.shape}")
    print(f"   测试集：X={X_test.shape}, y={y_test.shape}")
    
    # 构建Dataset
    train_dataset = RunningPoseDataset(X_train, y_train)
    test_dataset = RunningPoseDataset(X_test, y_test)
    
    # 修复：计算类别权重（用于WeightedRandomSampler，而非Loss）
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    # 生成每个样本的权重（用于采样）
    sample_weights = np.array([class_weights_np[int(label)] for label in y_train])
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    
    # 加权采样器（平衡训练集类别）
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return train_loader, test_loader

# -------------------------- 3. 修复LSTM模型（无改动） --------------------------
class RunningAnomalyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(RunningAnomalyLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = lstm_out[:, -1, :]  # 取最后一个时间步
        out = self.batch_norm1(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out

# -------------------------- 4. 修复训练函数（Loss权重问题） --------------------------
def train_model(model, train_loader, test_loader):
    # 修复：BCELoss不传入class_weights（权重已通过采样器平衡）
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=FACTOR,
        patience=4,
        min_lr=MIN_LR,
        verbose=True
    )
    
    best_val_loss = float("inf")
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    print("\n🚀 开始训练LSTM模型（PyTorch）...")
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # 修复：输出和标签维度匹配（均为[N,1]）
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            pred = (outputs > 0.5).float()
            train_correct += (pred == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                pred = (outputs > 0.5).float()
                val_correct += (pred == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        # 计算平均指标
        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # 记录历史
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 打印结果
        print(f"\nEpoch {epoch+1} 结果：")
        print(f"训练损失：{avg_train_loss:.4f} | 训练准确率：{train_acc:.4f}")
        print(f"验证损失：{avg_val_loss:.4f} | 验证准确率：{val_acc:.4f}")
        
        # 早停逻辑
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存模型
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_val_loss,
            }, MODEL_SAVE_PATH)
            print(f"✅ 保存最优模型（验证损失：{best_val_loss:.4f}）")
        else:
            patience_counter += 1
            print(f"⚠️ 早停计数器：{patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("🛑 验证损失不再下降，触发早停")
                break
    
    # 绘制训练曲线
    plot_training_history(train_loss_history, val_loss_history, train_acc_history, val_acc_history)
    
    return model

# -------------------------- 5. 评估/可视化/推理（无改动） --------------------------
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            outputs = model(X_batch)
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            labels = y_batch.cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print("\n📊 模型评估结果（测试集）：")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["正常(0)", "异常(1)"],
        digits=4
    ))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\n🔍 混淆矩阵：")
    print(f"          预测正常  预测异常")
    print(f"实际正常   {cm[0][0]}        {cm[0][1]}")
    print(f"实际异常   {cm[1][0]}        {cm[1][1]}")
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"\n📈 AUC值：{roc_auc:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率（FPR）')
    plt.ylabel('真阳性率（TPR）')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(DATASET_DIR, "roc_curve.png"))
    plt.close()
    
    return all_preds, all_probs

def plot_training_history(train_loss, val_loss, train_acc, val_acc):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='训练损失', color='blue')
    plt.plot(val_loss, label='验证损失', color='red')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='训练准确率', color='blue')
    plt.plot(val_acc, label='验证准确率', color='red')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_DIR, "training_history.png"))
    plt.close()
    print(f"\n✅ 训练曲线已保存至：{os.path.join(DATASET_DIR, 'training_history.png')}")

def predict_new_data(model_path, new_X):
    model = RunningAnomalyLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    new_X_tensor = torch.tensor(new_X, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(new_X_tensor)
        probs = outputs.cpu().numpy().flatten()
        preds = (outputs > THRESHOLD).float().cpu().numpy().flatten()
    
    print("\n🔮 新数据预测结果：")
    for i in range(len(preds)):
        print(f"样本{i}：异常概率={probs[i]:.4f} → {'异常(1)' if preds[i]==1 else '正常(0)'}")
    
    return probs, preds

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 加载数据
    train_loader, test_loader = load_dataset()
    
    # 初始化模型
    model = RunningAnomalyLSTM(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    print("\n📌 LSTM模型结构：")
    print(model)
    
    # 训练模型
    model = train_model(model, train_loader, test_loader)
    
    # 加载最优模型评估
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n📌 加载最优模型（Epoch {checkpoint['epoch']}，验证损失 {checkpoint['best_loss']:.4f}）")
    evaluate_model(model, test_loader)
    
    # 预测示例
    print("\n==================== 预测示例 ====================")
    test_data = np.load(os.path.join(DATASET_DIR, "test.npz"))
    X_test = test_data["X"]
    sample_X = X_test[:10]
    predict_new_data(MODEL_SAVE_PATH, sample_X)
    
    print(f"\n🎉 训练完成！模型保存至：{MODEL_SAVE_PATH}")