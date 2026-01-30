import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import transforms  
from utill import *

# ============================ 核心配置 ============================
VIDEO_DIR = "video_origin/data_video/use"          
LABEL_SAVE_DIR = "video_labels"  
SAVE_DIR = "video_dataset"     
SAMPLE_FPS = 15                       
WINDOW_SIZE_LIB = [10,20,30] #不同的窗口大小识别过程不同         
WINDOW_SIZE = WINDOW_SIZE_LIB[0]                       
STEP = 15                             
CONF_THRESHOLD = 0.5                  
NORMALIZE = True                      
TEST_SIZE = 0.2                       
RANDOM_SEED = 42    
#24帧每秒
VIDEO_FRAME_SPEED = 24
#每帧时间
TIME_GAP = round(1/VIDEO_FRAME_SPEED,3)   
Key_point_list = []
#加速度列表
Key_point_acceleration = []    
Max_acc = []  # 每帧的最大加速度     
cycle = 0     
# 新增加速度阈值配置
ACCELERATION_THRESHOLD = 50.0  # 加速度阈值，可根据实际情况调整
# =================================================================

# 初始化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载YOLO模型
yolo_model = YOLO("weights/yolo11n.pt")

# 加载HRNet模型
weights_path = "HRnet/pytorch/pose_coco/pose_hrnet_w32_256x192.pth"
keypoint_json_path = "HRnet/person_keypoints.json"

# 读取JSON配置
with open(keypoint_json_path, "r") as f:
    person_info = json.load(f)

# 创建HRNet模型
from HRNet_model import HighResolutionNet  # 导入你的HRNet模型
hrnet_model = HighResolutionNet(base_channel=32)
weights = torch.load(weights_path, map_location=device)
weights = weights if "model" not in weights else weights["model"]
hrnet_model.load_state_dict(weights)
hrnet_model.to(device)
hrnet_model.eval()

# HRNet数据转换
resize_hw = (256, 192)
hrnet_transform = transforms.Compose([
    transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_person_with_yolo(frame, conf_threshold=0.5):
    """使用YOLO检测人物位置"""
    results = yolo_model(frame, verbose=False)
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # YOLO中person的类别ID通常是0
                if cls_id == 0 and conf >= conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    return [float(x1), float(y1), float(x2), float(y2)], conf
    return None, 0

def extract_pose_from_frame(frame):
    """使用YOLO+HRNet提取17个关键点坐标"""
    global Key_point_list
    global Key_point_acceleration
    global Max_acc
    global cycle
    
    cycle += 1
    h, w = frame.shape[:2]
    
    # 初始化坐标数组
    original_pose = np.zeros((17, 2))  # 原始像素坐标
    norm_pose = np.zeros((17, 2))      # 归一化坐标
    
    # 1. 使用YOLO检测人物
    bbox, conf = detect_person_with_yolo(frame, CONF_THRESHOLD)
    
    if bbox is None:
        # 没有检测到人物，返回零值
        Max_acc.append(0.0)  # 记录零加速度
        return norm_pose.flatten(), original_pose.flatten()
    
    # 确保边界框在图像范围内
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    
    # 2. 裁剪人物区域（添加少量padding）
    padding = 10
    roi_x1 = max(0, int(x1) - padding)
    roi_y1 = max(0, int(y1) - padding)
    roi_x2 = min(w, int(x2) + padding)
    roi_y2 = min(h, int(y2) + padding)
    
    # 裁剪ROI
    person_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    if person_roi.size == 0:
        Max_acc.append(0.0)  # 记录零加速度
        return norm_pose.flatten(), original_pose.flatten()
    
    # 确保图像是RGB格式
    if len(person_roi.shape) == 3:
        if person_roi.shape[2] == 3:
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        else:
            person_rgb = person_roi
    else:
        person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_GRAY2RGB)
    
    roi_height, roi_width = person_rgb.shape[:2]
    
    # 3. 在ROI上使用HRNet检测关键点
    try:
        img_tensor, target = hrnet_transform(
            person_rgb, 
            {"box": [0, 0, roi_width - 1, roi_height - 1]}
        )
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        
        with torch.no_grad():
            outputs = hrnet_model(img_tensor.to(device))
            
            # 翻转测试增强
            flip_test = True
            if flip_test:
                flip_tensor = transforms.flip_images(img_tensor)
                flip_outputs = torch.squeeze(
                    transforms.flip_back(hrnet_model(flip_tensor.to(device)), person_info["flip_pairs"]),
                )
                flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
                outputs = (outputs + flip_outputs) * 0.5
            
            keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
            
            # 处理输出结果
            keypoints = np.squeeze(keypoints)
            scores = np.squeeze(scores)
            
            if scores.ndim == 0:
                scores = np.array([scores])
            
            # 4. 将关键点坐标转换回原始图像
            for i, (kp, score_val) in enumerate(zip(keypoints, scores)):
                if hasattr(kp, '__iter__') and len(kp) >= 2:
                    # HRNet输出的坐标（相对于ROI）
                    roi_x, roi_y = float(kp[0]), float(kp[1])
                    
                    # 转换到原始图像坐标
                    orig_x = roi_x1 + roi_x
                    orig_y = roi_y1 + roi_y
                    
                    # 确保坐标在图像范围内
                    orig_x = max(0, min(orig_x, w - 1))
                    orig_y = max(0, min(orig_y, h - 1))
                    
                    # 保存坐标
                    original_pose[i] = [orig_x, orig_y]
                    norm_pose[i] = [orig_x/w, orig_y/h] if NORMALIZE else [orig_x, orig_y]
    except Exception as e:
        print(f"HRNet处理出错: {e}")
    
    # 计算加速度
    p_pos = norm_pose.tolist()
    current_max_accel = 0.0
    
    if not Key_point_list:  # 如果历史列表为空（第一帧）
        Key_point_list = p_pos.copy()
    else:
        # 计算加速度
        Key_point_acceleration.clear()
        for j in range(17):
            # 使用utill.py中的acceleration函数
            accel = acceleration(p_pos[j], Key_point_list[j], TIME_GAP)
            Key_point_acceleration.append(accel)
        
        # 获取最大加速度
        if Key_point_acceleration:
            current_max_accel = max(Key_point_acceleration)
            print(f"帧 {cycle}: 最大加速度 = {current_max_accel:.2f}")
        
        # 更新历史关键点
        Key_point_list = p_pos.copy()
    
    # 记录当前帧的最大加速度
    Max_acc.append(current_max_accel)
    
    return norm_pose.flatten(), original_pose.flatten()

def draw_keypoints(frame, original_pose, current_max_accel=None):
    """在帧上绘制关键点和骨架，显示加速度信息"""
    frame_copy = frame.copy()
    kpts = original_pose.reshape(-1, 2)
    
    # COCO关键点骨骼连线
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
    ]
    
    # 绘制骨骼连线
    for (i, j) in skeleton:
        x1, y1 = int(kpts[i][0]), int(kpts[i][1])
        x2, y2 = int(kpts[j][0]), int(kpts[j][1])
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 绘制关键点
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # 红、蓝、绿
    
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
    
    # 显示加速度信息（如果提供）
    if current_max_accel is not None:
        # 设置颜色：超过阈值显示红色，否则显示绿色
        accel_color = (0, 0, 255) if current_max_accel > ACCELERATION_THRESHOLD else (0, 255, 0)
        
        # 显示当前最大加速度
        cv2.putText(frame_copy, f"Max Accel: {current_max_accel:.2f}", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, accel_color, 2)
        
        # 显示加速度阈值
        cv2.putText(frame_copy, f"Threshold: {ACCELERATION_THRESHOLD:.1f}", 
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 如果超过阈值，显示警告
        if current_max_accel > ACCELERATION_THRESHOLD:
            cv2.putText(frame_copy, "WARNING: High Acceleration!", 
                        (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame_copy

def manual_label_frames(video_path, norm_pose_seq, original_pose_seq, frame_list):
    """手动标注，显示加速度信息"""
    video_name = os.path.basename(video_path).split('.')[0]
    label_save_path = os.path.join(LABEL_SAVE_DIR, f"{video_name}.txt")
    
    if os.path.exists(label_save_path):
        print(f"加载已有标注：{video_name}.txt")
        labels = np.loadtxt(label_save_path).astype(int)
        return labels
    
    labels = []
    total_frames = len(frame_list)
    
    # 确保加速度列表长度与帧数匹配
    if len(Max_acc) < total_frames:
        print(f"警告：加速度列表长度({len(Max_acc)}) < 总帧数({total_frames})")
        # 补零
        while len(Max_acc) < total_frames:
            Max_acc.append(0.0)
    
    print(f"\n========== 标注视频：{video_name} ==========")
    print("      标注操作说明：      ")
    print("   按 0 键 → 标注为正常")
    print("   按 1 键 → 标注为异常")
    print("   按 Q 键 → 上一帧标注为正常")
    print("   按 P 键 → 上一帧标注为异常")
    print("   按 ESC 键 → 跳过剩余帧（默认标0）")
    print("   按 空格键 → 暂停/继续标注（可选）")
    print("==============================================")
    
    cv2.namedWindow(f"关键点标注 - {video_name}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"关键点标注 - {video_name}", 800, 600)
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
        
        # 获取当前帧的加速度（确保索引不越界）
        current_max_accel = Max_acc[i] if i < len(Max_acc) else 0.0
        
        # 绘制关键点+骨架+加速度信息
        frame_with_kpts = draw_keypoints(frame, original_pose, current_max_accel)
        
        # 缩放窗口（适配屏幕）
        h, w = frame_with_kpts.shape[:2]
        scale = min(1200/w, 900/h) 
        display_frame = cv2.resize(frame_with_kpts, (int(w*scale), int(h*scale)))
        
        # 显示帧信息和操作提示
        cv2.putText(display_frame, f"Frame {i+1}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, "0=normal 1=abnoraml ESC=skip space=stop q:<normal p<abnormal", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示当前帧
        cv2.imshow(f"关键点标注 - {video_name}", display_frame)
        
        # 等待键盘输入（无超时，直到按下有效键）
        while True:
            key = cv2.waitKey(0) & 0xFF  # 0表示无限等待输入
            if key == 48:  # 数字键0
                labels.append(0)
                print(f"帧{i+1}/{total_frames} → 标注为正常(0), 加速度={current_max_accel:.2f}")
                break
            elif key == 49:  # 数字键1
                labels.append(1)
                print(f"帧{i+1}/{total_frames} → 标注为异常(1), 加速度={current_max_accel:.2f}")
                break
            elif key == 81:  # Q键
                if len(labels) > 0:
                    labels[-1] = 0
                    print(f"帧{i}/{total_frames} → 上一帧标注为正常(0)")
                break
            elif key == 80:  # P键
                if len(labels) > 0:
                    labels[-1] = 1
                    print(f"帧{i}/{total_frames} → 上一帧标注为异常(1)")
                break
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

def process_single_video(video_path):
    """处理单视频"""
    global Max_acc, Key_point_list, Key_point_acceleration, cycle
    
    # 重置全局变量
    Max_acc.clear()
    Key_point_list.clear()
    Key_point_acceleration.clear()
    cycle = 0
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频：{video_path}")
        return None, None
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / SAMPLE_FPS))
    
    # 计算实际要处理的帧数
    num_frames_to_process = (total_frames + frame_interval - 1) // frame_interval
    
    norm_pose_list = []
    original_pose_list = []
    frame_list = []
    
    # 使用 tqdm 显示帧处理进度
    for frame_idx in tqdm(range(0, total_frames, frame_interval), 
                        desc=f"处理帧", 
                        total=num_frames_to_process,
                        leave=False,
                        ncols=80):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # 使用YOLO+HRNet提取关键点
        norm_pose, original_pose = extract_pose_from_frame(frame)
        norm_pose_list.append(norm_pose)
        original_pose_list.append(original_pose)
        frame_list.append(frame)
    
    cap.release()
    
    if len(norm_pose_list) < WINDOW_SIZE:
        print(f"帧数不足：{len(norm_pose_list)} < {WINDOW_SIZE}")
        return None, None
    
    norm_pose_seq = np.array(norm_pose_list)
    original_pose_seq = np.array(original_pose_list)
    
    # 手动标注
    labels = manual_label_frames(video_path, norm_pose_seq, original_pose_seq, frame_list)
    
    # 滑动窗口切分（新增加速度过滤）
    X_video, y_video = sliding_window_split(norm_pose_seq, labels)
    return X_video, y_video

def sliding_window_split(pose_seq, labels=None):
    """滑动窗口切分 - 改进版：包含加速度过滤"""
    X, y = [], []
    
    if labels is None:
        # 如果没有标签，返回全0
        for i in range(0, len(pose_seq) - WINDOW_SIZE + 1, STEP):
            # 检查窗口内是否有加速度超过阈值
            window_accel = Max_acc[i:i+WINDOW_SIZE] if i+WINDOW_SIZE <= len(Max_acc) else []
            if window_accel:
                max_accel_in_window = max(window_accel)
                if max_accel_in_window > ACCELERATION_THRESHOLD:
                    print(f"窗口 {i}-{i+WINDOW_SIZE} 被过滤，最大加速度 {max_accel_in_window:.2f} > 阈值 {ACCELERATION_THRESHOLD:.1f}")
                    continue  # 跳过这个窗口
            
            X.append(pose_seq[i:i+WINDOW_SIZE])
            y.append(0)
    else:
        for i in range(0, len(pose_seq) - WINDOW_SIZE + 1, STEP):
            window = pose_seq[i:i+WINDOW_SIZE]
            window_labels = labels[i:i+WINDOW_SIZE]
            
            # 检查窗口内是否有加速度超过阈值
            window_accel = Max_acc[i:i+WINDOW_SIZE] if i+WINDOW_SIZE <= len(Max_acc) else []
            if window_accel:
                max_accel_in_window = max(window_accel)
                if max_accel_in_window > ACCELERATION_THRESHOLD:
                    print(f"窗口 {i}-{i+WINDOW_SIZE} 被过滤，最大加速度 {max_accel_in_window:.2f} > 阈值 {ACCELERATION_THRESHOLD:.1f}")
                    continue  # 跳过这个窗口
            
            X.append(window)
            
            # 核心改进：需要连续异常才标为异常
            abnormal_count = np.sum(window_labels == 1)
            # 策略1：连续异常检测
            max_consecutive = 0
            current_streak = 0
            for label in window_labels:
                if label == 1:
                    current_streak += 1
                    max_consecutive = max(max_consecutive, current_streak)
                else:
                    current_streak = 0
            # 判断条件：
            # 1. 异常帧比例 > 20%
            # 2. 有至少3帧连续异常（摔倒通常不是孤立帧）
            abnormal_ratio = abnormal_count / WINDOW_SIZE
            if abnormal_ratio > 0.2 and max_consecutive >= 3:
                y.append(1)
            else:
                y.append(0)
    
    if len(X) == 0:
        print("警告：所有窗口都因加速度超过阈值被过滤！")
        return np.array([]), np.array([])
    
    X = np.array(X)
    y = np.array(y)
    
    # 打印统计信息
    print(f"生成 {len(X)} 个窗口，其中异常窗口: {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
    return X, y

# 在 main() 函数中修改视频处理循环
def main():
    """主函数"""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(LABEL_SAVE_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    video_ext = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(video_ext)]
    
    if not video_files:
        print("未找到视频文件！")
        return
    
    all_X, all_y = [], []
    video_sample_counts = []
    
    # 添加 tqdm 包装器
    for video_file in tqdm(video_files, desc="处理视频", unit="video", ncols=80):
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
    
    # 合并数据集
    X_total = np.concatenate(all_X, axis=0)
    y_total = np.concatenate(all_y, axis=0)
    video_indices = []
    for vid_idx, count in enumerate(video_sample_counts):
        video_indices.extend([vid_idx] * count)
    video_indices = np.array(video_indices)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_total, y_total, test_size=TEST_SIZE,
        stratify=video_indices, random_state=RANDOM_SEED
    )
    
    # 保存数据集
    np.savez(os.path.join(SAVE_DIR, "train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(SAVE_DIR, "test.npz"), X=X_test, y=y_test)
    
    # 打印信息
    print(f"\n==================== 数据集信息 ====================")
    print(f"特征维度：17关节×2坐标=34维")
    print(f"时间步长：{WINDOW_SIZE}帧")
    print(f"加速度阈值：{ACCELERATION_THRESHOLD}")
    print(f"训练集：总数={len(X_train)} | 正常={np.sum(y_train==0)} | 异常={np.sum(y_train==1)}")
    print(f"测试集：总数={len(X_test)} | 正常={np.sum(y_test==0)} | 异常={np.sum(y_test==1)}")
    print(f"数据集保存至：{SAVE_DIR}")
    print(f"标注标签保存至：{LABEL_SAVE_DIR}")
    print(f"关键点检测：YOLO + HRNet")
    print("====================================================")

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