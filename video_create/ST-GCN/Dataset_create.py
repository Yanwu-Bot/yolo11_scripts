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
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

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
    """提取单帧关键点（原始坐标）"""
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
    """归一化关键点（以髋部中心为原点，躯干长度缩放）"""
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
    """提取整个视频的归一化关键点序列"""
    _init_models()  # 确保模型已加载
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
    正样本：同一窗口的随机数据增强。
    负样本：策略混合（跨视频随机窗口 + 同视频远距离窗口）。
    """
    def __init__(self,
                video_source,               # 视频文件路径或包含视频的文件夹路径
                window_size=10,             # 每个窗口的帧数
                stride=5,                   # 滑窗步长
                neg_cross_ratio=0.7,        # 负样本中跨视频占比
                temporal_threshold=20,      # 同视频负样本最小帧距离差
                transform_params=None,      # 数据增强参数
                normalize=True,             # 是否归一化
                save_path='result/GCN/dataset/dataset.npz'):  # 数据集保存路径
        self.window_size = window_size
        self.stride = stride
        self.neg_cross_ratio = neg_cross_ratio
        self.temporal_threshold = temporal_threshold
        self.transform_params = transform_params or {
            'rotation': 5, 'scale': 0.05, 'noise': 0.02, 'mask': 0.1
        }
        self.normalize = normalize
        self.dist_th = None  #距离阈值，控制负样本生成

        if os.path.isdir(video_source):
            video_files = [os.path.join(video_source, f)
                            for f in os.listdir(video_source)
                            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        else:
            video_files = [video_source]
        print(f"找到 {len(video_files)} 个视频")

        self.video_keypoints = []       # 每个元素: (video_name, seq: (N,17,2))
        self.window_list = []           # 每个元素: (video_idx, start_frame, window_data (W,17,2))

        for vfile in video_files:
            seq = _extract_video_keypoints(vfile, normalize=self.normalize) #提取归一化关键点
            if seq is None or len(seq) < self.window_size:
                print(f"  跳过 {os.path.basename(vfile)}: 有效帧不足")
                continue
            self.video_keypoints.append((os.path.basename(vfile), seq))
            video_idx = len(self.video_keypoints) - 1
            for start in range(0, len(seq) - self.window_size + 1, self.stride):
                window = seq[start:start+self.window_size]  # (W,17,2)
                self.window_list.append((video_idx, start, window))

        print(f"共 {len(self.video_keypoints)} 个视频，生成 {len(self.window_list)} 个窗口")

        self.video_window_indices = {vi: [] for vi in range(len(self.video_keypoints))}
        for wi, (vi, start, _) in enumerate(self.window_list):
            self.video_window_indices[vi].append((wi, start))  #窗口索引，第i个视频中第x个窗口，起始帧y

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.save_dataset(save_path)
        self.get_threshold(len(self.window_list))


    def save_dataset(self, save_path):
        """将窗口列表打包保存为 .npz 文件"""
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

    def _random_transform(self, window):
        """对窗口 (W,17,2) 进行随机数据增强，返回增强后的副本"""
        w = window.copy() #(window_size, 17, 2)
        T, V, C = w.shape
        # 旋转
        if 'rotation' in self.transform_params:
            angle_deg = random.uniform(-self.transform_params['rotation'], #获取范围内随机值
                                        self.transform_params['rotation'])
            rad = math.radians(angle_deg)
            cos = math.cos(rad)
            sin = math.sin(rad)

            rot = np.zeros_like(w)
            rot[..., 0] = w[..., 0] * cos - w[..., 1] * sin #旋转公式
            rot[..., 1] = w[..., 0] * sin + w[..., 1] * cos
            w = rot
        # 缩放
        if 'scale' in self.transform_params:
            scale = 1.0 + random.uniform(-self.transform_params['scale'],
                                        self.transform_params['scale'])
            w = w * scale 
        # 高斯噪声
        if 'noise' in self.transform_params:
            noise = np.random.normal(0, self.transform_params['noise'], w.shape) #噪声，均值0，标准差，形状
            w = w + noise
        # 随机遮挡关键点
        if 'mask' in self.transform_params:
            mask_prob = self.transform_params['mask']
            mask = np.random.binomial(1, 1 - mask_prob, size=(T, V, 1)) #生成遮挡，1-mask_prob概率为1
            w = w * mask
        return w

    def _get_negative_sample(self, anchor_video_idx, anchor_start):
        """获取一个负样本窗口"""
        if random.random() < self.neg_cross_ratio:
            # 选择不同视频
            other_videos = [vi for vi in range(len(self.video_keypoints))
                            if vi != anchor_video_idx]
            if not other_videos:
                pass
            else:
                neg_video_idx = random.choice(other_videos)
                cand_win_infos = self.video_window_indices[neg_video_idx] #取随机视频所有窗口起始信息
                if cand_win_infos:
                    wi, _ = random.choice(cand_win_infos)
                    return self.window_list[wi][2]   # window_data

        # 同视频远距离负样本
        own_indices = self.video_window_indices[anchor_video_idx]
        far_indices = [(wi, start) for wi, start in own_indices
                    if abs(start - anchor_start) >= self.temporal_threshold] #满足条件的窗口
        if far_indices:
            wi, _ = random.choice(far_indices)
            return self.window_list[wi][2]
        # 如果都不满足，返回全零窗口作为占位
        return np.zeros((self.window_size, 17, 2), dtype=np.float32)

    def __len__(self):
        return len(self.window_list)

    def __getitem__(self, idx):
        anchor_info = self.window_list[idx]
        anchor_video_idx, anchor_start, anchor_data = anchor_info

        # 正样本：对 anchor 做一次随机增强
        positive = self._random_transform(anchor_data)
        # 负样本
        negative = self._get_negative_sample(anchor_video_idx, anchor_start) 
        # 转换为 (C, T, V) 格式以适用于 ST-GCN
        dist = np.mean(np.linalg.norm(anchor_data - negative, axis=-1))
        #根据阈值选取负样本
        for i in range(10):
            if(dist < self.dist_th):
                negative = self._get_negative_sample(anchor_video_idx, anchor_start) 
                dist = np.mean(np.linalg.norm(anchor_data - negative, axis=-1))
            else:
                break
        def to_stgcn(data):
            # data: (T, V, C) -> (C, T, V)
            return torch.FloatTensor(data).permute(2, 0, 1)

        return to_stgcn(anchor_data), to_stgcn(positive), to_stgcn(negative)
    
    def get_threshold(self,length):
        distances = []
        # 随机抽样一定数量的负样本对计算距离
        for _ in range(length):  # 抽样1000次
            idx = random.randint(0, len(self.window_list)-1)
            anchor_info = self.window_list[idx]
            anchor_video_idx, anchor_start, anchor_data = anchor_info
            negative = self._get_negative_sample(anchor_video_idx, anchor_start)
            dist = np.mean(np.linalg.norm(anchor_data - negative, axis=-1))
            distances.append(dist)
        # 统计并打印
        distances = np.array(distances)
        self.dist_th = np.percentile(distances,25)
        print(f"距离统计: 均值={distances.mean():.2f}, 中位数={np.median(distances):.2f}, "
            f"25%分位数={np.percentile(distances,25):.2f}, 75%分位数={np.percentile(distances,75):.2f}")

if __name__ == '__main__':
    video_folder = 'video_origin/data_video/dataset/'
    dataset = ContrastiveWindowDataset(
        video_source=video_folder,
        window_size=10,
        stride=5,
        neg_cross_ratio=0.7,
        temporal_threshold=20,
        transform_params={'rotation': 5, 'scale': 0.05, 'noise': 0.02, 'mask': 0.1},
        normalize=True,
        save_path='result/GCN/dataset/dataset.npz'
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for anchor, pos, neg in loader:
        print("Anchor shape:", anchor.shape)   # (B, C, T, V)
        print("Positive shape:", pos.shape)
        print("Negative shape:", neg.shape)
        print(len(dataset))
        break
