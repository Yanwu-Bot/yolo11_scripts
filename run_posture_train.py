import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import json
import os

SAVE_DIR = "run_posture_model" 
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, 'resnet18_run_posture.pth')
# ==================== 1. 加载数据 ====================
print("加载数据...")
train_data = np.load("run_posture_dataset/train.npz")
test_data = np.load("run_posture_dataset/test.npz")

X_train, y_train = train_data["X"], train_data["y"]
X_test, y_test = test_data["X"], test_data["y"]

print(f"训练集: {X_train.shape} -> {y_train.shape}")
print(f"测试集: {X_test.shape} -> {y_test.shape}")

# ==================== 2. 重塑数据为图像格式 (34维->[1, 17, 2]) ====================
def reshape_keypoints(X):
    """将34维关键点转换为图像格式 [batch, 1, 17, 2]"""
    # X shape: (n_samples, 34)
    # 先reshape为 (n_samples, 17, 2)
    X_reshaped = X.reshape(-1, 17, 2)
    
    # 归一化到0-1范围
    X_normalized = (X_reshaped - X_reshaped.min(axis=(1,2), keepdims=True)) / \
                    (X_reshaped.max(axis=(1,2), keepdims=True) - X_reshaped.min(axis=(1,2), keepdims=True) + 1e-8)
    
    # 扩展为 [n_samples, 1, 17, 2]
    X_expanded = np.expand_dims(X_normalized, axis=1)
    
    # 上采样到 [n_samples, 1, 224, 224] 以适配ResNet
    # 使用最近邻插值上采样
    import cv2
    X_resized = []
    for i in range(X_expanded.shape[0]):
        img = X_expanded[i, 0]  # (17, 2)
        # 扩展为 (17, 2, 1) 然后转置为 (2, 17, 1) -> 扩展维度
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        X_resized.append(img_resized)
    
    X_resized = np.array(X_resized).reshape(-1, 1, 224, 224)
    return X_resized

X_train_img = reshape_keypoints(X_train)
X_test_img = reshape_keypoints(X_test)

print(f"重塑后的训练集: {X_train_img.shape}")
print(f"重塑后的测试集: {X_test_img.shape}")

# ==================== 3. 创建数据加载器 ====================
batch_size = 32
train_dataset = TensorDataset(torch.FloatTensor(X_train_img), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test_img), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== 4. 使用ResNet18模型 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载预训练的ResNet18
model = models.resnet18(pretrained=True)

# 修改第一层以接受单通道输入
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# 修改最后一层以适应3个类别
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

model = model.to(device)

# ==================== 5. 训练配置 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ==================== 6. 训练循环 ====================
epochs = 50
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

print("\n开始训练...")
for epoch in range(epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # 测试阶段
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    # 更新学习率
    scheduler.step()
    
    print(f'Epoch [{epoch+1}/{epochs}] - '
        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
        f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# ==================== 7. 绘制损失和准确率曲线 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(train_losses, label='Train Loss', linewidth=2)
ax1.plot(test_losses, label='Test Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True)

# 准确率曲线
ax2.plot(train_accuracies, label='Train Accuracy', linewidth=2)
ax2.plot(test_accuracies, label='Test Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 8. 保存模型 ====================
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies,
}, save_path)

print(f"\n模型已保存为: resnet18_run_posture.pth")
print(f"训练结果图已保存为: training_results.png")

# ==================== 9. 最终评估 ====================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 计算分类报告
from sklearn.metrics import classification_report, confusion_matrix
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=['起跑', '摆动', '落地']))

# 保存训练统计数据
stats = {
    'final_train_acc': train_acc,
    'final_test_acc': test_acc,
    'best_test_acc': max(test_accuracies),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'train_accuracies': train_accuracies,
    'test_accuracies': test_accuracies
}

with open('training_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"训练统计数据已保存为: training_stats.json")