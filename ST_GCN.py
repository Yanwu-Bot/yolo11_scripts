import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

print('Use CUDA:', torch.cuda.is_available())

# 种子值固定
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

# 创建模型保存目录（只增加这一行）
MODEL_SAVE_DIR = "model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 加载你的数据集（根据你的data_create.py格式）
def load_coco_dataset(data_dir="video_dataset"):
    """加载你的COCO格式数据集"""
    train_data = np.load(f"{data_dir}/train.npz")
    test_data = np.load(f"{data_dir}/test.npz")
    
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]
    
    print(f"训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_test, y_test

# 加载数据集
X_train, y_train, X_test, y_test = load_coco_dataset()

# 数据加载器
class COCOFeeder(torch.utils.data.Dataset):
    def __init__(self, X_data, y_data):
        super().__init__()
        self.data = X_data
        self.label = y_data
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        data = torch.FloatTensor(self.data[index])  # shape: [WINDOW_SIZE, 34]
        label = torch.LongTensor([self.label[index]])[0]
        
        # 重新组织数据形状为 [C, T, V]
        # 你的数据是 [时间窗口, 34] = [时间窗口, 17*2]
        # 需要转换为 [通道数=2, 时间窗口, 节点数=17]
        T, total_dims = data.shape
        V = 17  # COCO关键点数
        C = total_dims // V  # 应该是2 (x,y)
        
        # 重塑数据
        data = data.reshape(T, V, C)  # [T, V, C]
        data = data.permute(2, 0, 1)  # [C, T, V]
        
        return data, label

# COCO图结构（17个关键点）
class COCOGraph():
    def __init__(self, hop_size=2):
        self.num_node = 17  # COCO 17个关键点
        self.hop_size = hop_size
        self.get_edge()
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)
        self.get_adjacency()
    
    def get_edge(self):
        """定义COCO关键点的连接关系"""
        self_link = [(i, i) for i in range(self.num_node)]  # 自连接
        
        # COCO骨架连接（基于人体结构）
        neighbor_base = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
            (11, 12), (5, 11), (6, 12),  # 躯干
            (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
        ]
        
        self.edge = self_link + neighbor_base
    
    def get_adjacency(self):
        valid_hop = range(0, self.hop_size + 1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        
        normalize_adjacency = self.normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        
        self.A = A
    
    def get_hop_distance(self, num_node, edge, hop_size):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        
        for d in range(hop_size, -1, -1):
            hop_dis[arrive_mat[d]] = d
        
        return hop_dis
    
    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        
        DAD = np.dot(A, Dn)
        return DAD

class SpatialGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels * s_kernel_size,
                            kernel_size=1)
    
    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class STGC_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.5):
        super().__init__()
        self.sgc = SpatialGraphConvolution(in_channels=in_channels,
                                        out_channels=out_channels,
                                        s_kernel_size=A_size[0])
        
        self.M = nn.Parameter(torch.ones(A_size))
        
        self.tgc = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels,
                     out_channels,
                     (t_kernel_size, 1),
                     (stride, 1),
                     ((t_kernel_size - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    
    def forward(self, x, A):
        x = self.tgc(self.sgc(x, A * self.M))
        return x

class COCO_ST_GCN(nn.Module):
    def __init__(self, num_classes=2, in_channels=2, t_kernel_size=9, hop_size=2):
        super().__init__()
        # 创建COCO图
        graph = COCOGraph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()
        
        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])
        
        # STGC blocks - 调整通道数以适应你的数据
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size, dropout=0.1)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size, dropout=0.1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # x shape: [N, C, T, V] = [batch, 2, 时间窗口, 17]
        N, C, T, V = x.size()
        
        # Batch Normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        # STGC blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        
        # 全局平均池化
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x

# ============= 只保存最佳模型的功能 =============
def save_best_model(model, optimizer, epoch, val_acc, filename='best_model.pth'):
    """只保存最佳模型"""
    checkpoint = {
        'epoch': epoch,
        'val_acc': val_acc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, filename))
    print(f"🎉 保存最佳模型到: {os.path.join(MODEL_SAVE_DIR, filename)} (验证准确率: {val_acc:.2f}%)")

def load_best_model(model, optimizer=None, filename='best_model.pth'):
    """加载最佳模型"""
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, filename)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ 从 {checkpoint_path} 加载最佳模型")
        print(f"   轮次: {checkpoint['epoch']}, 验证准确率: {checkpoint['val_acc']:.2f}%")
        return model, optimizer, checkpoint['epoch'], checkpoint['val_acc']
    else:
        print(f"⚠️ 最佳模型文件不存在: {checkpoint_path}")
        return model, optimizer, 0, 0.0
# ==============================================

# 训练函数（修改为只保存最佳模型）
def train_model(model, train_loader, test_loader, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 跟踪最佳验证准确率
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 调整学习率
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 只在验证准确率提高时保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"🌟 发现新的最佳模型! (准确率: {val_acc:.2f}%)")
            save_best_model(model, optimizer, epoch+1, val_acc)
        
        print('-' * 50)
    
    print(f"\n🏆 训练完成!")
    print(f"最佳模型在第 {best_epoch} 轮，验证准确率: {best_val_acc:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs

# 主程序
if __name__ == "__main__":
    # 创建数据集
    train_dataset = COCOFeeder(X_train, y_train)
    test_dataset = COCOFeeder(X_test, y_test)
    
    # 创建数据加载器
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # 检查一批数据
    for data, labels in train_loader:
        print(f"Batch data shape: {data.shape}")  # 应该是 [batch, 2, 时间窗口, 17]
        print(f"Batch labels shape: {labels.shape}")
        break
    
    # 创建模型
    model = COCO_ST_GCN(num_classes=2, in_channels=2, t_kernel_size=3, hop_size=2)
    
    # 打印模型结构
    print("\n" + "="*50)
    print("模型结构:")
    print("="*50)
    print(model)
    print("="*50)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    test_input = torch.randn(2, 2, X_train.shape[1], 17)  # [batch, channels, time_window, nodes]
    test_output = model(test_input)
    print(f"\n测试输入形状: {test_input.shape}")
    print(f"测试输出形状: {test_output.shape}")
    
    # 开始训练
    print("\n" + "="*50)
    print("开始训练...")
    print("="*50)
    print("注意: 只保存最佳模型到 saved_models/best_model.pth")
    print("="*50)
    
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, epochs=50, lr=0.01
    )
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "training_curve.png"))
    plt.show()
    
    # 加载并测试最佳模型
    print("\n" + "="*50)
    print("加载最佳模型进行测试...")
    print("="*50)
    
    best_model, _, best_epoch, best_acc = load_best_model(model)
    
    if best_epoch > 0:
        # 测试最佳模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_model = best_model.to(device)
        best_model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = best_model(data)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        print(f"最佳模型测试准确率: {test_acc:.2f}%")
        print(f"保存时的验证准确率: {best_acc:.2f}%")