import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import simpledialog
from matplotlib import rcParams #字体
rcParams['font.family'] = 'SimHei'

# ============================ 核心配置 ============================
VIDEO_DIR = "video_origin\data_video"          
LABEL_SAVE_DIR = "video_labels"  
SAVE_DIR = "video_dataset"     
SAMPLE_FPS = 10                       
WINDOW_SIZE = 10                      
STEP = 4                             
# COCO 17个关键点的标准索引+名称（精准对应）
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
KEY_JOINTS = list(range(17))  # 保持17个关键点，但绘制时按标准索引来
CONF_THRESHOLD = 0.5                  
# 关键修复：先提取原始坐标，归一化只用于模型输入，绘制用原始坐标
NORMALIZE = True                      
TEST_SIZE = 0.2                       
RANDOM_SEED = 42                      
# =================================================================

# 初始化YOLO11-Pose模型（确保加载官方权重，关键点更准）
model = YOLO("weights\yolo11l-pose.pt")

# 创建目录
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LABEL_SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------- 修复：精准提取关键点（原始+归一化） --------------------------
def extract_pose_from_frame(frame, return_original=True):
    """
    提取17个关键点的原始坐标+归一化坐标
    :param frame: 原始帧
    :param return_original: 是否返回原始像素坐标
    :return: norm_pose（归一化）, original_pose（原始像素坐标）
    """
    h, w = frame.shape[:2]  #获取帧高宽
    results = model(frame, conf=CONF_THRESHOLD)
    
    # 初始化原始坐标和归一化坐标
    original_pose = np.zeros((17, 2))  # (17, 2) 原始像素坐标
    norm_pose = np.zeros((17, 2))      # (17, 2) 归一化坐标
    
    if len(results[0].keypoints) > 0:
        # 提取YOLO输出的关键点（x,y,conf）
        kpts = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
        for i in range(17):
            x, y, conf = kpts[i]
            if conf >= CONF_THRESHOLD:
                # 原始像素坐标
                original_pose[i] = [x, y]
                # 归一化坐标（0-1）
                norm_pose[i] = [x/w, y/h]
    
    # 展平返回
    norm_pose_flat = norm_pose.flatten()  # (34,)
    original_pose_flat = original_pose.flatten()  # (34,)
    
    if return_original:
        return norm_pose_flat, original_pose_flat
    else:
        return norm_pose_flat

# -------------------------- 修复：精准绘制关键点 --------------------------
def draw_keypoints(frame, original_pose, thickness=2):
    """
    在原始帧上绘制精准的关键点+骨骼连线
    :param frame: 原始帧
    :param original_pose: 原始像素坐标的关键点 (34,) → 展平的(17,2)
    :return: 带关键点的帧
    """
    frame_copy = frame.copy()
    kpts = original_pose.reshape(-1, 2)  # 重新变成（17，2）
    
    # COCO关键点骨骼连线（按人体结构）
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
    ]
    
    # 1. 绘制骨骼连线
    for (i, j) in skeleton:
        x1, y1 = int(kpts[i][0]), int(kpts[i][1])
        x2, y2 = int(kpts[j][0]), int(kpts[j][1])
        # 只绘制有效关键点（坐标>0）
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    
    # 2. 绘制关键点（不同关节不同颜色）
    colors = [
        (0, 0, 255),  # 头部（红）
        (255, 0, 0),  # 上肢（蓝）
        (0, 255, 0)   # 下肢（绿）
    ]
    # 头部关键点（0-4）
    for i in range(5):
        x, y = int(kpts[i][0]), int(kpts[i][1])
        if x > 0 and y > 0:
            cv2.circle(frame_copy, (x, y), 5, colors[0], -1)
    # 上肢关键点（5-10）
    for i in range(5, 11):
        x, y = int(kpts[i][0]), int(kpts[i][1])
        if x > 0 and y > 0:
            cv2.circle(frame_copy, (x, y), 5, colors[1], -1)
    # 下肢关键点（11-16）
    for i in range(11, 17):
        x, y = int(kpts[i][0]), int(kpts[i][1])
        if x > 0 and y > 0:
            cv2.circle(frame_copy, (x, y), 5, colors[2], -1)
    
    return frame_copy

# -------------------------- 修复：手动标注（精准关键点可视化） --------------------------
def manual_label_frames(video_path, norm_pose_seq, original_pose_seq, frame_list):
    """
    可视化带精准关键点的帧，手动标注0/1（优化：键盘直接输入，无需点击输入框）
    :param video_path: 视频路径
    :param norm_pose_seq: 归一化姿态序列 (帧数, 34)
    :param original_pose_seq: 原始像素坐标姿态序列 (帧数, 34)
    :param frame_list: 原始帧列表
    :return: labels (帧数,)
    """
    video_name = os.path.basename(video_path).split('.')[0]
    label_save_path = os.path.join(LABEL_SAVE_DIR, f"{video_name}.txt")
    
    # 加载已有标注
    if os.path.exists(label_save_path):
        print(f"加载已有标注：{video_name}.txt")
        labels = np.loadtxt(label_save_path).astype(int)
        return labels
    
    labels = []
    total_frames = len(frame_list)
    
    # 移除tkinter依赖（改用键盘输入）
    print(f"\n========== 标注视频：{video_name} ==========")
    print("      标注操作说明：      ")
    print("   按 0 键 → 标注为正常")
    print("   按 1 键 → 标注为异常")
    print("   按 Q 键 → 上一帧标注为正常")
    print("   按 P 键 → 上一帧标注为异常")
    print("   按 ESC 键 → 跳过剩余帧（默认标0）")
    print("   按 空格键 → 暂停/继续标注（可选）")
    print("==============================================")
    
    # 创建显示窗口（固定名称，避免多窗口）
    cv2.namedWindow(f"精准关键点标注 - {video_name}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"精准关键点标注 - {video_name}", 800, 600)
    
    pause = False  # 暂停标志
    for i in range(total_frames):
        if pause:
            # 暂停状态：等待空格键继续
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # 空格键继续
                    pause = False
                    break
                elif key == 27:  # ESC退出
                    print(" ESC跳过，剩余帧标为0")
                    labels.extend([0] * (total_frames - len(labels)))
                    cv2.destroyAllWindows()
                    break
            if key == 27:
                break
        
        frame = frame_list[i]
        original_pose = original_pose_seq[i]
        
        # 绘制精准的关键点+骨骼
        frame_with_kpts = draw_keypoints(frame, original_pose)
        # 缩放窗口（适配屏幕）
        h, w = frame_with_kpts.shape[:2]
        #缩放比例
        scale = min(1200/w, 900/h) 
        display_frame = cv2.resize(frame_with_kpts, (int(w*scale), int(h*scale)))
        
        # 显示帧信息和操作提示
        cv2.putText(display_frame, f"Frame {i+1}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, "0=normal  1=abnoraml  ESC=skip  space=stop", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示当前帧
        cv2.imshow(f"关键点标注 - {video_name}", display_frame)
        
        # 等待键盘输入（无超时，直到按下有效键）
        while True:
            key = cv2.waitKey(0) & 0xFF  # 0表示无限等待输入
            if key == 48:  # 数字键0
                labels.append(0)
                print(f"帧{i+1}/{total_frames} → 标注为正常(0)")
                break
            elif key == 49:  # 数字键1
                labels.append(1)
                print(f"帧{i+1}/{total_frames} → 标注为异常(1)")
                break
            elif key == 81:
                labels[-1] == 0
                print(f"帧{i}/{total_frames} → 标注为正常(0)")
                continue
            elif key == 80:
                labels[-1] == 1
                print(f"帧{i}/{total_frames} → 标注为异常(1)")
                continue
            elif key == 27:  # ESC键
                print(" ESC跳过，剩余帧标为0")
                labels.extend([0] * (total_frames - len(labels)))
                cv2.destroyAllWindows()
                labels = np.array(labels, dtype=int)
                np.savetxt(label_save_path, labels, fmt="%d")
                print(f"✅ 标注完成：{label_save_path} | 正常={np.sum(labels==0)}, 异常={np.sum(labels==1)}")
                return labels
            elif key == 32:  # 空格键
                pause = True
                print(f"帧{i+1}/{total_frames} → 已暂停（按空格键继续）")
                break
            else:
                print(f" 无效按键！请按 0/1/ESC/空格，当前按键：{key}")
                continue
    
    # 关闭窗口
    cv2.destroyAllWindows()
    
    # 转换为数组并保存
    labels = np.array(labels, dtype=int)
    np.savetxt(label_save_path, labels, fmt="%d")
    print(f" 标注完成：{label_save_path} | 正常帧数={np.sum(labels==0)}, 异常帧数={np.sum(labels==1)}")
    
    return labels

# -------------------------- 修复：视频处理流程（保存原始关键点） --------------------------
def process_single_video(video_path):
    """处理单视频：抽帧→提精准关键点→手动标注"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频：{video_path}")
        return None, None
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #采样帧数，视频变为SAMPLE_FPS帧
    frame_interval = max(1, int(video_fps / SAMPLE_FPS))
    
    # 保存三类数据：归一化姿态、原始姿态、原始帧
    norm_pose_list = []
    original_pose_list = []
    frame_list = []
    
    for frame_idx in range(0, total_frames, frame_interval):
        #转移到索引帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        # 提取精准的关键点（原始+归一化）
        norm_pose, original_pose = extract_pose_from_frame(frame)
        norm_pose_list.append(norm_pose)
        original_pose_list.append(original_pose)
        frame_list.append(frame)
    
    cap.release()
    
    if len(norm_pose_list) < WINDOW_SIZE:
        print(f"帧数不足：{len(norm_pose_list)} < {WINDOW_SIZE}")
        return None, None
    
    # 预处理归一化姿态（用于模型输入）
    norm_pose_seq = np.array(norm_pose_list)
    original_pose_seq = np.array(original_pose_list)
    
    # 手动标注（用原始关键点绘制）
    labels = manual_label_frames(video_path, norm_pose_seq, original_pose_seq, frame_list)
    
    # 滑动窗口切分（用归一化姿态）
    X_video, y_video = sliding_window_split(norm_pose_seq, labels)
    return X_video, y_video

def preprocess_pose_sequence(pose_seq):
    """预处理归一化姿态序列（填充缺失值）"""
    processed_seq = pose_seq.copy()
    zero_frames = np.all(processed_seq == 0, axis=1)
    for i in range(len(processed_seq)):
        if zero_frames[i]:
            neighbors = []
            if i > 0 and not zero_frames[i-1]:
                neighbors.append(processed_seq[i-1])
            if i < len(processed_seq)-1 and not zero_frames[i+1]:
                neighbors.append(processed_seq[i+1])
            if neighbors:
                processed_seq[i] = np.mean(neighbors, axis=0)
            else:
                non_zero = processed_seq[~zero_frames]
                if len(non_zero) > 0:
                    processed_seq[i] = np.mean(non_zero, axis=0)
    return processed_seq

def sliding_window_split(pose_seq, labels=None):
    """滑动窗口切分"""
    X, y = [], []
    #滑动窗口，STEP为滑动步长
    for i in range(0, len(pose_seq) - WINDOW_SIZE + 1, STEP):
        window = pose_seq[i:i+WINDOW_SIZE]
        X.append(window)
        if np.any(labels[i:i+WINDOW_SIZE] == 1):  # 假设1是异常
            y.append(1)
        else:
            y.append(0)
    X = np.array(X)
    y = np.array(y) if labels is not None else None
    return X, y

def print_dataset_info(X_train, y_train, X_test, y_test):
    """打印数据集信息"""
    print("\n==================== 数据集信息 ====================")
    print(f"特征维度：17关节×2坐标=34维（归一化）")
    print(f"时间步长：{WINDOW_SIZE}帧")
    print(f"\n训练集：总数={len(X_train)} | 正常={np.sum(y_train==0)} | 异常={np.sum(y_train==1)}")
    print(f"测试集：总数={len(X_test)} | 正常={np.sum(y_test==0)} | 异常={np.sum(y_test==1)}")
    print("====================================================")

def main():
    video_ext = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(video_ext)]
    if not video_files:
        print("未找到视频文件！")
        return
    
    all_X, all_y = [], []
    video_sample_counts = []
    for video_file in tqdm(video_files, desc="处理视频"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        X_video, y_video = process_single_video(video_path)
        if X_video is None:
            continue
        all_X.append(X_video)
        all_y.append(y_video)
        video_sample_counts.append(len(X_video))
    
    if not all_X:
        print("无有效数据！")
        return
    
    # 合并+划分数据集
    X_total = np.concatenate(all_X, axis=0)
    y_total = np.concatenate(all_y, axis=0)
    video_indices = []
    for vid_idx, count in enumerate(video_sample_counts):
        video_indices.extend([vid_idx] * count)
    video_indices = np.array(video_indices)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_total, y_total, test_size=TEST_SIZE,
        stratify=video_indices, random_state=RANDOM_SEED
    )
    
    # 保存数据集
    np.savez(os.path.join(SAVE_DIR, "train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(SAVE_DIR, "test.npz"), X=X_test, y=y_test)
    
    # 打印信息
    print_dataset_info(X_train, y_train, X_test, y_test)
    print(f" 数据集保存至：{SAVE_DIR}")
    print(f" 关键点已精准校准，标注标签保存至：{LABEL_SAVE_DIR}")

def load_dataset():
    """加载数据集"""
    train_data = np.load(os.path.join(SAVE_DIR, "train.npz"))
    X_train = train_data["X"]
    y_train = train_data["y"]
    
    test_data = np.load(os.path.join(SAVE_DIR, "test.npz"))
    X_test = test_data["X"]
    y_test = test_data["y"]
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    main()
    # 验证加载
    X_train, y_train, X_test, y_test = load_dataset()
    print(f"\n数据集加载验证：")
    print(f"训练集形状：{X_train.shape} | 标签形状：{y_train.shape}")