import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 字体设置
rcParams['font.family'] = 'SimHei'

# ============================ 配置项 ============================
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 模型路径（训练好的模型）
MODEL_PATH = "model/best_model.pth"

# 模型参数（这些应该与训练时完全一致，从checkpoint中读取）
# 注意：我们不在这里硬编码，而是从checkpoint中读取

# 时序窗口参数
WINDOW_SIZE = 12         # 时间窗口大小（必须与训练时一致）

# 关键点提取参数
CONF_THRESHOLD = 0.5     # YOLO关键点置信度阈值

# 预测阈值
PRED_THRESHOLD = 0.5     # 概率>阈值判定为异常

# ==============================================================================

# -------------------------- 1. 直接加载训练好的模型 --------------------------
def load_trained_model(model_path):
    """直接加载训练好的STGCN模型"""
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在：{model_path}")
        return None
    
    # 直接加载整个checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # 检查checkpoint的内容
    print(f"📋 Checkpoint包含的键：{list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        # 从checkpoint中加载模型
        print("✅ 从checkpoint中加载模型状态字典")
        
        # 我们需要知道模型的配置，可以从checkpoint中读取或使用默认值
        model_config = checkpoint.get('model_config', {
            'num_classes': 2,
            'in_channels': 2,
            't_kernel_size': 3,
            'hop_size': 2
        })
        
        # 打印模型配置
        print(f"📊 模型配置：{model_config}")
        
        # 动态导入训练时使用的模型定义
        # 注意：这里假设你的训练代码在同一目录下
        try:
            # 方法1：直接导入训练代码中的模型类
            from ST_GCN import COCO_ST_GCN  # 需要改成你的训练文件名
            print("✅ 从训练文件导入模型类")
        except ImportError:
            # 方法2：使用一个简化的模型定义（如果知道确切结构）
            print("⚠️ 无法导入训练模型，使用简化模型定义")
            
            # 这里需要定义和训练时完全一样的模型结构
            # 这部分应该从你的训练代码中复制过来
            class COCOGraph():
                def __init__(self, hop_size=2):
                    self.num_node = 17
                    self.hop_size = hop_size
                    self.get_edge()
                    self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)
                    self.get_adjacency()
                
                def get_edge(self):
                    self_link = [(i, i) for i in range(self.num_node)]
                    neighbor_base = [
                        (0, 1), (0, 2), (1, 3), (2, 4),
                        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                        (11, 12), (5, 11), (6, 12),
                        (11, 13), (13, 15), (12, 14), (14, 16)
                    ]
                    neighbor_link = [(i, j) for (i, j) in neighbor_base]
                    self.edge = self_link + neighbor_link
                
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
                        nn.Conv2d(out_channels, out_channels,
                                (t_kernel_size, 1), (stride, 1),
                                ((t_kernel_size - 1) // 2, 0)),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
                
                def forward(self, x, A):
                    x = self.tgc(self.sgc(x, A * self.M))
                    return x
            
            class COCO_ST_GCN(nn.Module):
                def __init__(self, num_classes=2, in_channels=2, t_kernel_size=9, hop_size=2):
                    super().__init__()
                    from ST_GCN import COCOGraph  # 假设图定义在单独文件
                    graph = COCOGraph(hop_size)
                    A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
                    self.register_buffer('A', A)
                    A_size = A.size()
                    
                    self.bn = nn.BatchNorm1d(in_channels * A_size[1])
                    self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size, dropout=0.1)
                    self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size, dropout=0.1)
                    self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size, dropout=0.1)
                    self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size, dropout=0.1)
                    self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size, dropout=0.1)
                    self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size, dropout=0.1)
                    self.fc = nn.Sequential(
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(32, num_classes)
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
                    x = self.stgc6(x, self.A)
                    x = F.avg_pool2d(x, x.size()[2:])
                    x = x.view(N, -1)
                    x = self.fc(x)
                    return x
        
        # 创建模型实例
        model = COCO_ST_GCN(**model_config).to(DEVICE)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 从checkpoint读取其他信息
        epoch = checkpoint.get('epoch', '未知')
        val_acc = checkpoint.get('val_acc', '未知')
        
        print(f"✅ 模型加载成功！")
        print(f"   训练轮次: {epoch}")
        print(f"   验证准确率: {val_acc}%")
        
    else:
        # 如果checkpoint中没有model_state_dict，假设整个文件就是模型
        print("⚠️ Checkpoint中没有model_state_dict，尝试直接加载为模型")
        
        # 这里需要根据你的实际模型结构来调整
        # 最简单的方法：从训练代码导入
        try:
            from ST_GCN import COCO_ST_GCN
            # 需要知道模型参数，这里使用默认值
            model = COCO_ST_GCN(
                num_classes=2,
                in_channels=2,
                t_kernel_size=3,
                hop_size=2
            ).to(DEVICE)
            model.load_state_dict(checkpoint)
            print("✅ 直接加载模型权重成功")
        except:
            print("❌ 无法加载模型，请检查模型结构是否匹配")
            return None
    
    # 设置为评估模式
    model.eval()
    print("✅ 模型已切换到评估模式")
    
    return model

# -------------------------- 2. 关键点提取工具 --------------------------
# 初始化YOLO姿态模型
try:
    yolo_pose = YOLO("weights/yolo11m-pose.pt")  # 根据你的路径调整
    print("✅ YOLO姿态模型加载成功")
except:
    print("⚠️ 无法加载YOLO模型，请检查路径")
    yolo_pose = None

def extract_pose_from_frame(frame, normalize=True):
    """
    从单帧提取姿态关键点
    """
    if yolo_pose is None:
        print("❌ YOLO模型未加载")
        return np.zeros(34)
    
    h, w = frame.shape[:2]
    results = yolo_pose(frame, conf=CONF_THRESHOLD)
    
    norm_pose = np.zeros((17, 2))
    
    if len(results[0].keypoints) > 0:
        kpts = results[0].keypoints.data[0].cpu().numpy()
        for i in range(17):
            x, y, conf = kpts[i]
            if conf >= CONF_THRESHOLD:
                if normalize:
                    norm_pose[i] = [x / w, y / h]
                else:
                    norm_pose[i] = [x, y]
    
    return norm_pose.flatten()

# -------------------------- 3. 时序窗口构建 --------------------------
class PoseWindowBuffer:
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.buffer = []
    
    def add_pose(self, pose):
        self.buffer.append(pose)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
    
    def get_window(self):
        if len(self.buffer) < self.window_size:
            pad_len = self.window_size - len(self.buffer)
            pad_pose = np.zeros((pad_len, 34))
            window = np.concatenate([pad_pose, np.array(self.buffer)], axis=0)
        else:
            window = np.array(self.buffer)
        
        # 转换为STGCN格式: [C, T, V] = [2, WINDOW_SIZE, 17]
        T, total_dims = window.shape
        V = 17
        C = total_dims // V
        
        window_reshaped = window.reshape(T, V, C).transpose(2, 0, 1)
        window_reshaped = window_reshaped[np.newaxis, :, :, :]
        
        return window_reshaped

# -------------------------- 4. 核心推理函数 --------------------------
def predict_frame_sequence(model, pose_window):
    """对时序姿态窗口进行预测"""
    pose_tensor = torch.tensor(pose_window, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        output = model(pose_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_prob = probabilities[0, 1].cpu().numpy()
    
    pred_label = 1 if pred_prob > PRED_THRESHOLD else 0
    
    return pred_prob, pred_label

# -------------------------- 5. 可视化绘制 --------------------------
def draw_pred_result(frame, pred_prob, pred_label):
    """在帧上绘制预测结果"""
    h, w = frame.shape[:2]
    
    if pred_label == 1:
        bg_color = (0, 0, 255)
        text = f"Abnromal: {pred_prob:.3f}"
    else:
        bg_color = (0, 255, 0)
        text = f"Normal: {pred_prob:.3f}"
    
    # 绘制状态文字
    cv2.rectangle(frame, (10, 10), (250, 60), bg_color, -1)
    cv2.putText(frame, text, (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

# -------------------------- 6. 简化版本：只做视频推理 --------------------------
def infer_video_simple(model_path, video_path, save_output=False):
    """
    简化版本：直接加载模型并进行视频推理
    """
    print("="*60)
    print("COCO-STGCN视频推理工具")
    print("="*60)
    
    # 加载模型
    model = load_trained_model(model_path)
    if model is None:
        print("❌ 模型加载失败，程序退出")
        return
    
    # 检查视频文件
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在：{video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{video_path}")
        return
    
    # 获取视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 视频信息:")
    print(f"  尺寸: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
    
    # 保存输出
    if save_output:
        output_path = os.path.splitext(video_path)[0] + "_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 初始化缓冲区
    pose_buffer = PoseWindowBuffer(WINDOW_SIZE)
    
    print(f"\n🚀 开始推理... (按ESC键退出)")
    
    frame_count = 0
    abnormal_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 提取姿态
        pose = extract_pose_from_frame(frame, normalize=True)
        pose_buffer.add_pose(pose)
        
        # 至少收集了1帧才开始预测
        if frame_count >= 1:
            pose_window = pose_buffer.get_window()
            pred_prob, pred_label = predict_frame_sequence(model, pose_window)
            
            if pred_label == 1:
                abnormal_count += 1
            
            # 绘制结果
            frame = draw_pred_result(frame, pred_prob, pred_label)
            
            # 显示进度
            progress = frame_count / total_frames * 100
            cv2.putText(frame, f"PROCESS: {progress:.1f}%", 
                        (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示帧数
            cv2.putText(frame, f"FRAME: {frame_count}", 
                        (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示
        cv2.imshow("STGCN异常检测", frame)
        
        # 保存
        if save_output:
            out_writer.write(frame)
        
        # 进度显示
        if frame_count % 30 == 0:
            print(f"处理: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # ESC退出
        if cv2.waitKey(1) & 0xFF == 27:
            print("⚠️ 用户提前退出")
            break
    
    # 清理
    cap.release()
    if save_output:
        out_writer.release()
        print(f"✅ 结果保存到: {output_path}")
    
    cv2.destroyAllWindows()
    
    # 统计
    print(f"\n📊 推理统计:")
    print(f"总帧数: {frame_count}")
    print(f"异常帧数: {abnormal_count}")
    if frame_count > 0:
        print(f"异常率: {abnormal_count/frame_count*100:.2f}%")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 使用示例
    model_path = "model/best_model.pth"
    video_path = "video_origin//run_woman2.mp4"  # 你的视频路径
    
    # 直接调用简化版本
    infer_video_simple(
        model_path=model_path,
        video_path=video_path,
        save_output=True  # 是否保存结果视频
    )