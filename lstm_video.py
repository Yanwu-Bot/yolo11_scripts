import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from matplotlib import rcParams #字体
rcParams['font.family'] = 'SimHei'

# ============================ 配置项（需与训练时一致） ============================
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型路径
MODEL_PATH = "model\\running_anomaly_lstm_pytorch.pth"
# 模型参数（必须和训练时完全一致）
INPUT_DIM = 34       # 17关节×2坐标
HIDDEN_DIM = 64      # LSTM隐藏层维度
NUM_LAYERS = 2       # LSTM层数
DROPOUT = 0.2        # Dropout比例
# 时序窗口参数（必须和训练时一致）
WINDOW_SIZE = 12     # LSTM时间步长（连续12帧）
STEP = 4             # 窗口滑动步长（推理时可设为1，实时性更高）
# 关键点提取参数
CONF_THRESHOLD = 0.5 # YOLO关键点置信度阈值
# 预测阈值（可调整：越低越灵敏，越高越严格）
PRED_THRESHOLD = 0.3 # 概率>阈值判定为异常
# ==================================================================================

# -------------------------- 1. 定义LSTM模型（与训练时完全一致） --------------------------
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
        out = lstm_out[:, -1, :]  # 取最后一个时间步输出
        out = self.batch_norm1(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out

# -------------------------- 2. 加载LSTM模型 --------------------------
def load_lstm_model(model_path):
    """加载训练好的LSTM模型"""
    # 初始化模型
    model = RunningAnomalyLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    # 加载权重（忽略优化器等无关参数）
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    # 设置为评估模式（禁用Dropout/BatchNorm训练行为）
    model.eval()
    print(f"✅ 模型加载完成：{model_path}")
    return model

# -------------------------- 3. 关键点提取工具（与数据集生成一致） --------------------------
# 初始化YOLO姿态模型
yolo_pose = YOLO("yolo11m-pose.pt")

def extract_pose_from_frame(frame, normalize=True):
    """
    从单帧提取归一化的姿态关键点
    :param frame: 原始视频帧 (H,W,3)
    :param normalize: 是否归一化（必须和训练时一致）
    :return: norm_pose (34,) → 归一化后的关键点，无关键点则返回全0
    """
    h, w = frame.shape[:2]
    results = yolo_pose(frame, conf=CONF_THRESHOLD)
    
    # 初始化关键点
    norm_pose = np.zeros((17, 2))  # (17关节, x/y)
    
    if len(results[0].keypoints) > 0:
        kpts = results[0].keypoints.data[0].cpu().numpy()  # (17, 3) x,y,conf
        for i in range(17):
            x, y, conf = kpts[i]
            if conf >= CONF_THRESHOLD:
                # 原始像素坐标
                raw_x, raw_y = x, y
                # 归一化（和训练时的预处理逻辑完全一致）
                if normalize:
                    # 基于人体包围盒归一化
                    non_zero_kpts = kpts[kpts[:,2]>=CONF_THRESHOLD, :2]
                    if len(non_zero_kpts) > 0:
                        min_xy = np.min(non_zero_kpts, axis=0)
                        max_xy = np.max(non_zero_kpts, axis=0)
                        bbox_w = max_xy[0] - min_xy[0] if max_xy[0] > min_xy[0] else 1
                        bbox_h = max_xy[1] - min_xy[1] if max_xy[1] > min_xy[1] else 1
                        norm_x = (raw_x - min_xy[0]) / bbox_w
                        norm_y = (raw_y - min_xy[1]) / bbox_h
                        norm_pose[i] = [norm_x, norm_y]
                else:
                    norm_pose[i] = [raw_x/w, raw_y/h]  # 基于画面归一化
    
    return norm_pose.flatten()  # (34,)

# -------------------------- 4. 时序窗口构建（核心：匹配LSTM输入） --------------------------
class PoseWindowBuffer:
    """姿态窗口缓冲区：维护最近N帧的姿态，构建LSTM输入的时序窗口"""
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.buffer = []  # 存储最近的姿态序列
    
    def add_pose(self, pose):
        """添加单帧姿态到缓冲区"""
        self.buffer.append(pose)
        # 保持缓冲区长度不超过窗口大小
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
    
    def get_window(self):
        """获取完整的时序窗口（不足则补0）"""
        if len(self.buffer) < self.window_size:
            # 不足窗口大小：前面补0
            pad_len = self.window_size - len(self.buffer)
            pad_pose = np.zeros((pad_len, INPUT_DIM))
            window = np.concatenate([pad_pose, np.array(self.buffer)], axis=0)
        else:
            window = np.array(self.buffer)
        return window[np.newaxis, :, :]  # 增加batch维度 → (1, window_size, 34)

# -------------------------- 5. 核心推理函数 --------------------------
def predict_frame_sequence(model, pose_window):
    """
    对时序姿态窗口进行预测
    :param model: 加载好的LSTM模型
    :param pose_window: 姿态窗口 (1, window_size, 34)
    :return: pred_prob (异常概率), pred_label (0=正常/1=异常)
    """
    # 转换为Tensor并移至设备
    pose_tensor = torch.tensor(pose_window, dtype=torch.float32).to(DEVICE)
    
    # 预测（禁用梯度计算，提升速度）
    with torch.no_grad():
        pred_prob = model(pose_tensor).cpu().numpy().flatten()[0]
    pred_label = 1 if pred_prob > PRED_THRESHOLD else 0
    
    return pred_prob, pred_label

# -------------------------- 6. 可视化绘制（实时显示结果） --------------------------
def draw_pred_result(frame, pred_prob, pred_label):
    """在帧上绘制预测结果"""
    h, w = frame.shape[:2]
    # 绘制背景框
    if pred_label == 1:  # 异常：红色背景
        bg_color = (0, 0, 255)
        text = f"Abnormal: {pred_prob:.3f}"
    else:  # 正常：绿色背景
        bg_color = (0, 255, 0)
        text = f"Normal: {pred_prob:.3f}"
    
    # 绘制文字
    cv2.rectangle(frame, (10, 10), (300, 60), bg_color, -1)
    cv2.putText(frame, text, (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# -------------------------- 7. 场景1：实时视频流推理（摄像头） --------------------------
def infer_realtime_camera(model):
    """实时摄像头画面推理"""
    cap = cv2.VideoCapture(0)  # 0=默认摄像头，可改为视频路径
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    # 初始化姿态缓冲区
    pose_buffer = PoseWindowBuffer(WINDOW_SIZE)
    
    print("\n🚀 实时推理中（按ESC退出）...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. 提取当前帧姿态
        pose = extract_pose_from_frame(frame)
        # 2. 添加到缓冲区，构建时序窗口
        pose_buffer.add_pose(pose)
        pose_window = pose_buffer.get_window()
        # 3. 预测
        pred_prob, pred_label = predict_frame_sequence(model, pose_window)
        # 4. 绘制结果
        frame_with_result = draw_pred_result(frame, pred_prob, pred_label)
        # 5. 显示画面
        cv2.imshow("Running Anomaly Detection (Real-time)", frame_with_result)
        
        # 按ESC退出
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# -------------------------- 8. 场景2：本地视频文件推理 --------------------------
def infer_local_video(model, video_path, save_output=False):
    """
    本地视频文件推理
    :param video_path: 视频路径
    :param save_output: 是否保存推理结果视频
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{video_path}")
        return
    
    # 获取视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 保存视频配置
    if save_output:
        output_path = os.path.splitext(video_path)[0] + "_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 初始化姿态缓冲区
    pose_buffer = PoseWindowBuffer(WINDOW_SIZE)
    
    print(f"\n🚀 视频推理中：{video_path}（共{total_frames}帧，按ESC提前退出）...")
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. 提取姿态
        pose = extract_pose_from_frame(frame)
        # 2. 构建窗口
        pose_buffer.add_pose(pose)
        pose_window = pose_buffer.get_window()
        # 3. 预测
        pred_prob, pred_label = predict_frame_sequence(model, pose_window)
        # 4. 绘制结果
        frame_with_result = draw_pred_result(frame, pred_prob, pred_label)
        # 5. 显示
        cv2.imshow("Running Anomaly Detection (Video)", frame_with_result)
        pose = extract_pose_from_frame(frame)
        print(f"当前帧姿态前5维：{pose[:5]}")  # 打印前5维，看是否变化
        # 6. 保存（可选）
        if save_output:
            out_writer.write(frame_with_result)
        
        # 按ESC退出
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # 释放资源
    cap.release()
    if save_output:
        out_writer.release()
        print(f"✅ 推理结果视频已保存：{output_path}")
    cv2.destroyAllWindows()

# -------------------------- 9. 场景3：单帧图片推理 --------------------------
def infer_single_image(model, img_path):
    """单帧图片推理（需模拟时序窗口，补前11帧为0）"""
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"❌ 无法读取图片：{img_path}")
        return
    
    # 提取当前帧姿态
    pose = extract_pose_from_frame(frame)
    # 构建窗口（仅当前帧有效，其余补0）
    pose_buffer = PoseWindowBuffer(WINDOW_SIZE)
    pose_buffer.add_pose(pose)
    pose_window = pose_buffer.get_window()
    # 预测
    pred_prob, pred_label = predict_frame_sequence(model, pose_window)
    
    # 绘制结果并保存
    frame_with_result = draw_pred_result(frame, pred_prob, pred_label)
    output_path = os.path.splitext(img_path)[0] + "_result.jpg"
    cv2.imwrite(output_path, frame_with_result)
    
    # 打印结果
    print("\n📊 单帧推理结果：")
    print(f"图片路径：{img_path}")
    print(f"异常概率：{pred_prob:.4f}")
    print(f"判定结果：{'异常' if pred_label==1 else '正常'}")
    print(f"结果保存：{output_path}")
    
    # 显示结果
    cv2.imshow("Single Frame Result", frame_with_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------- 主函数：选择推理场景 --------------------------
if __name__ == "__main__":
    # 1. 加载模型
    lstm_model = load_lstm_model(MODEL_PATH)
    
    # 2. 选择推理场景（取消注释对应场景）
    # 场景1：实时摄像头推理
    # infer_realtime_camera(lstm_model)
    
    # 场景2：本地视频推理（替换为你的视频路径）
    infer_local_video(lstm_model, "video_origin\\run_woman2.mp4", save_output=True)
    
    # 场景3：单帧图片推理（替换为你的图片路径）
    # infer_single_image(lstm_model, "running_videos/test_frame.jpg")