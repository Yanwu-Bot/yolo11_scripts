import os
import cv2
import math
import json
import torch
import numpy as np
from HRNet_model import HighResolutionNet
import transforms
from ultralytics import YOLO
import random
import time
from collections import deque
import matplotlib.pyplot as plt
from utill import *
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

MAX_LEN = 170

def fix_endpoint_by_angle(mid_point, end_point, max_deg, deg, max_len, statue):
    """
    角度修正函数
    statue=1: 用于手臂（角度应小于 max_deg）
    statue=2: 用于腿（角度应大于 max_deg）
    """
    dx = end_point[0] - mid_point[0]
    dy = end_point[1] - mid_point[1]
    length = math.sqrt(dx**2 + dy**2)

    if statue == 1:
        if deg > max_deg:
            diff = deg - max_deg
            diff_rad = math.radians(diff)
            cos_a = math.cos(-diff_rad)
            sin_a = math.sin(-diff_rad)
            new_dx = dx * cos_a - dy * sin_a
            new_dy = dx * sin_a + dy * cos_a
            new_x = mid_point[0] + new_dx
            new_y = mid_point[1] + new_dy
            new_length = math.sqrt(new_dx**2 + new_dy**2)
        else:
            new_length = length
            new_dx = dx
            new_dy = dy
            new_x = end_point[0]
            new_y = end_point[1]
        if new_length > max_len:
            scale = max_len / new_length
            return (mid_point[0] + new_dx * scale, mid_point[1] + new_dy * scale)
        return (new_x, new_y)

    elif statue == 2:
        if deg < max_deg:
            diff = deg - max_deg
            diff_rad = math.radians(diff)
            cos_a = math.cos(-diff_rad)
            sin_a = math.sin(-diff_rad)
            new_dx = dx * cos_a - dy * sin_a
            new_dy = dx * sin_a + dy * cos_a
            new_x = mid_point[0] + new_dx
            new_y = mid_point[1] + new_dy
            new_length = math.sqrt(new_dx**2 + new_dy**2)
        else:
            new_length = length
            new_dx = dx
            new_dy = dy
            new_x = end_point[0]
            new_y = end_point[1]
        if new_length > max_len:
            scale = max_len / new_length
            return (mid_point[0] + new_dx * scale, mid_point[1] + new_dy * scale)
        return (new_x, new_y)

def _apply_keypoint_fix(p_pos):
    """对关键点应用角度和距离修正"""
    if p_pos is None or len(p_pos) < 17:
        return p_pos

    p_pos = np.array(p_pos, dtype=np.float32)

    # 计算角度
    r_a_a = calculate_angle(p_pos[6], p_pos[8], p_pos[10])   # 右臂角度
    l_a_a = calculate_angle(p_pos[5], p_pos[7], p_pos[9])   # 左臂角度
    r_l_a = calculate_angle(p_pos[12], p_pos[14], p_pos[16]) # 右腿角度
    l_l_a = calculate_angle(p_pos[11], p_pos[13], p_pos[15]) # 左腿角度

    # 计算末端长度
    r_a_l = distance(p_pos[8], p_pos[10])   # 右小臂长度
    l_a_l = distance(p_pos[7], p_pos[9])    # 左小臂长度
    r_l_l = distance(p_pos[14], p_pos[16])  # 右小腿长度
    l_l_l = distance(p_pos[13], p_pos[15])  # 左小腿长度

    # 右臂修正（角度应 ≤170 且长度 <150）
    if not (0 <= r_a_a <= 170 and r_a_l < MAX_LEN):
        p_pos[10] = fix_endpoint_by_angle(p_pos[8], p_pos[10], 170, r_a_a, MAX_LEN, 1)

    # 左臂修正（角度应 ≤170 且长度 <150）
    if not (0 <= l_a_a <= 170 and l_a_l < MAX_LEN):
        p_pos[9] = fix_endpoint_by_angle(p_pos[7], p_pos[9], 170, l_a_a, MAX_LEN, 1)

    # 右腿修正（角度应 ≥180 且长度 <150）
    if not (180 <= r_l_a <= 360 and r_l_l < MAX_LEN):
        p_pos[16] = fix_endpoint_by_angle(p_pos[14], p_pos[16], 170, r_l_a, MAX_LEN, 2)

    # 左腿修正（角度应 ≥180 且长度 <150）
    if not (180 <= l_l_a <= 360 and l_l_l < MAX_LEN):
        p_pos[15] = fix_endpoint_by_angle(p_pos[13], p_pos[15], 170, l_l_a, MAX_LEN, 2)

    return p_pos

device = None
hrnet_model = None
yolo_model = None
person_info = None
hrnet_transform = None
models_initialized = False

def _init_models():
    global device, hrnet_model, yolo_model, person_info, hrnet_transform, models_initialized
    if models_initialized:
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    yolo_model = YOLO('weights/yolo11n.pt')
    weights_path = "HRnet/pytorch/pose_coco/pose_hrnet_w32_256x192.pth"
    keypoint_json_path = "HRnet/person_keypoints.json"
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    hrnet_model = HighResolutionNet(base_channel=32)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    hrnet_model.load_state_dict(weights)
    hrnet_model.to(device)
    hrnet_model.eval()

    resize_hw = (256, 192)
    hrnet_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    _models_initialized = True
    print("模型初始化完成")

def _detect_person(frame):
    results = yolo_model(frame, verbose=False)
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) >= 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    return [float(x1), float(y1), float(x2), float(y2)]
    return None

def _extract_single_frame_keypoints(frame):
    bbox = _detect_person(frame)
    if bbox is None:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    padding = 10
    roi_x1 = max(0, int(x1) - padding)
    roi_y1 = max(0, int(y1) - padding)
    roi_x2 = min(width, int(x2) + padding)
    roi_y2 = min(height, int(y2) + padding)
    person_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if person_roi.size == 0:
        return None
    if len(person_roi.shape) == 3 and person_roi.shape[2] == 3:
        person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
    else:
        person_rgb = person_roi
    roi_height, roi_width = person_rgb.shape[:2]
    img_tensor, target = hrnet_transform(person_rgb, {"box": [0, 0, roi_width - 1, roi_height - 1]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    with torch.no_grad():
        outputs = hrnet_model(img_tensor.to(device))
        keypoints, _ = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
    result = []
    for kp in keypoints:
        if len(kp) >= 2:
            orig_x = roi_x1 + float(kp[0])
            orig_y = roi_y1 + float(kp[1])
            orig_x = max(0, min(orig_x, width - 1))
            orig_y = max(0, min(orig_y, height - 1))
            result.append([orig_x, orig_y])
        else:
            result.append([0, 0])
    return np.array(result, dtype=np.float32)

def _normalize_keypoints(p_pos, target_torso_length=100):
    if len(p_pos) < 17:
        return p_pos
    shoulder_center_x = (p_pos[5][0] + p_pos[6][0]) / 2
    shoulder_center_y = (p_pos[5][1] + p_pos[6][1]) / 2
    hip_center_x = (p_pos[11][0] + p_pos[12][0]) / 2
    hip_center_y = (p_pos[11][1] + p_pos[12][1]) / 2
    torso_length = math.sqrt((shoulder_center_x - hip_center_x)**2 +
                            (shoulder_center_y - hip_center_y)**2)
    if torso_length < 1e-6:
        return p_pos
    scale = target_torso_length / torso_length
    center_x, center_y = hip_center_x, hip_center_y
    normalized = []
    for i in range(17):
        norm_x = (p_pos[i][0] - center_x) * scale
        norm_y = (p_pos[i][1] - center_y) * scale
        normalized.append([norm_x, norm_y])
    return np.array(normalized, dtype=np.float32)

def _extract_video_keypoints(video_path, normalize=True):
    _init_models()
    cap = cv2.VideoCapture(video_path)
    all_kp = []
    frame_count = 0
    print(f"PROCSSING VIDEO: {os.path.basename(video_path)}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        kp = _extract_single_frame_keypoints(frame)
        if kp is not None and len(kp) == 17:
            # ===== 新增：应用关键点和距离修正（在归一化之前） =====
            kp = _apply_keypoint_fix(kp)
            # ====================================================
            if normalize:
                kp = _normalize_keypoints(kp)
            all_kp.append(kp)
        if frame_count % 50 == 0:
            print(f"processing {frame_count} frames", end='\r')
    cap.release()
    print(f"\nTotal frames: {frame_count}, Valid frames:{len(all_kp)}")
    return np.array(all_kp, dtype=np.float32) if all_kp else None

class ContrastiveWindowDataset(torch.utils.data.Dataset):
    """
    仅提取原始骨架窗口并保存为 .npz，不执行任何数据增强。
    增强由训练时的加载器（如 ContrastiveDatasetFromFile）在线完成。
    """
    def __init__(self,
                video_source,
                window_size=10,
                stride=5,
                normalize=True,
                save_path='result/GCN/dataset/dataset.npz'):
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize

        if os.path.isdir(video_source):
            video_files = [os.path.join(video_source, f)
                            for f in os.listdir(video_source)
                            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        else:
            video_files = [video_source]
        print(f"找到 {len(video_files)} 个视频")

        self.video_keypoints = []
        self.window_list = []

        for vfile in video_files:
            seq = _extract_video_keypoints(vfile, normalize=self.normalize)
            if seq is None or len(seq) < self.window_size:
                print(f"  跳过 {os.path.basename(vfile)}: 有效帧不足")
                continue
            self.video_keypoints.append((os.path.basename(vfile), seq))
            video_idx = len(self.video_keypoints) - 1
            for start in range(0, len(seq) - self.window_size + 1, self.stride):
                window = seq[start:start+self.window_size]
                self.window_list.append((video_idx, start, window))

        print(f"共 {len(self.video_keypoints)} 个视频，生成 {len(self.window_list)} 个窗口")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.save_dataset(save_path)

    def save_dataset(self, save_path):
        num_windows = len(self.window_list)
        window_array = np.zeros((num_windows, self.window_size, 17, 2), dtype=np.float32)
        video_indices = np.zeros(num_windows, dtype=np.int32)
        start_frames = np.zeros(num_windows, dtype=np.int32)
        for i, (vi, start, w) in enumerate(self.window_list):
            window_array[i] = w
            video_indices[i] = vi
            start_frames[i] = start
        np.savez(save_path,
                windows=window_array,
                video_indices=video_indices,
                start_frames=start_frames,
                window_size=self.window_size,
                stride=self.stride)
        print(f"数据集已保存至 {save_path}")

    def __len__(self):
        return len(self.window_list)

    def __getitem__(self, idx):
        _, _, anchor_data = self.window_list[idx]
        # 转换为 (C, T, V) 格式
        return torch.FloatTensor(anchor_data).permute(2, 0, 1)

if __name__ == '__main__':
    video_folder = 'D:/Dataset/sprint/Whole'
    dataset = ContrastiveWindowDataset(
        video_source=video_folder,
        window_size=7,
        stride=1,
        normalize=True,
        save_path='result/GCN/dataset/dataset_7_1.npz'
    )
    # 测试输出形状
    sample = dataset[0]
    print("Sample shape:", sample.shape)   # 应为 (2, 10, 17)
    print(f"数据集总窗口数: {len(dataset)}")