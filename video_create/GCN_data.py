#输入视频生成帧比对数据集
import os
import cv2
import math
import json
import torch
import numpy as np
from HRNet_model import HighResolutionNet
import transforms
from ultralytics import YOLO
from collections import deque
import time
import random
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

# ========== 全局变量 ==========
device = None
hrnet_model = None
yolo_model = None
person_info = None
hrnet_transform = None

def init_models():
    """初始化YOLO和HRNet模型"""
    global device, hrnet_model, yolo_model, person_info, hrnet_transform
    
    if device is None:
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
        print("模型初始化完成")

def detect_person(frame):
    results = yolo_model(frame, verbose=False)
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) >= 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    return [float(x1), float(y1), float(x2), float(y2)]
    return None

def extract_keypoints(frame):
    global device, hrnet_model, yolo_model, person_info, hrnet_transform
    
    bbox = detect_person(frame)
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

def normalize_keypoints(p_pos, target_torso_length=100):
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
    center_x = hip_center_x
    center_y = hip_center_y
    
    normalized = []
    for i in range(17):
        norm_x = (p_pos[i][0] - center_x) * scale
        norm_y = (p_pos[i][1] - center_y) * scale
        normalized.append([norm_x, norm_y])
    
    return np.array(normalized, dtype=np.float32)

def extract_video_keypoints(video_path, normalize=True):
    print(f"处理: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        kp = extract_keypoints(frame)
        if kp is not None and len(kp) == 17:
            if normalize:
                kp = normalize_keypoints(kp)
            all_keypoints.append(kp)
        
        if frame_count % 50 == 0:
            print(f"  已处理 {frame_count} 帧", end='\r')
    
    cap.release()
    print(f"\n  共提取 {len(all_keypoints)} 帧有效关键点")
    return np.array(all_keypoints, dtype=np.float32)

def build_dataset(video_paths, threshold=50, max_pairs_per_video=500, normalize=True,
                cross_video_ratio=0.3, analyze=True):
    """
    自动构建训练数据集，支持跨视频对比，自动分析距离分布并给出阈值建议
    
    Args:
        video_paths: 视频路径列表
        threshold: 关键点距离阈值
        max_pairs_per_video: 每个视频最多生成多少对样本
        normalize: 是否归一化关键点
        cross_video_ratio: 跨视频样本占比 (0-1)
        analyze: 是否进行距离分布分析
    """
    import matplotlib.pyplot as plt
    
    init_models()
    
    # 提取所有视频的关键点
    video_keypoints = []
    video_names = []
    
    for video_path in video_paths:
        print(f"\n{'='*50}")
        kps = extract_video_keypoints(video_path, normalize=normalize)
        if len(kps) >= 2:
            video_keypoints.append(kps)
            video_names.append(os.path.basename(video_path))
    
    if len(video_keypoints) == 0:
        print("没有有效的视频数据")
        return []
    
    num_videos = len(video_keypoints)
    
    # ========== 距离分布分析（帮助确定阈值） ==========
    if analyze:
        print(f"\n{'='*50}")
        
        same_distances = []
        cross_distances = []
        
        for _ in range(500):
            vi = random.randint(0, num_videos - 1)
            vj = random.randint(0, num_videos - 1)
            kps_i = video_keypoints[vi]
            kps_j = video_keypoints[vj]
            i = random.randint(0, len(kps_i) - 1)
            j = random.randint(0, len(kps_j) - 1)
            dist = np.mean(np.linalg.norm(kps_i[i] - kps_j[j], axis=1))
            
            if vi == vj:
                same_distances.append(dist)
            else:
                cross_distances.append(dist)
        
        # 计算建议阈值
        if same_distances and cross_distances:
            q1_same = np.percentile(same_distances, 75)
            q3_cross = np.percentile(cross_distances, 25)
            suggested_threshold1 = (q1_same + q3_cross) / 2
            suggested_threshold2 = np.max(same_distances) * 1.2
            
            print(f"  同视频内距离: 均值={np.mean(same_distances):.2f}, "
                  f"中位数={np.median(same_distances):.2f}, 最大值={np.max(same_distances):.2f}")
            print(f"  跨视频距离:   均值={np.mean(cross_distances):.2f}, "
                  f"中位数={np.median(cross_distances):.2f}, 最小值={np.min(cross_distances):.2f}")
            print(f"  建议阈值: {suggested_threshold1:.2f} 或 {suggested_threshold2:.2f}")
            
            # 绘图
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.hist(same_distances, bins=30, alpha=0.7, label='同视频内', color='blue')
            plt.hist(cross_distances, bins=30, alpha=0.7, label='跨视频', color='orange')
            plt.axvline(x=threshold, color='red', linestyle='--', label=f'当前阈值={threshold}')
            plt.xlabel('距离')
            plt.ylabel('频次')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.boxplot([same_distances, cross_distances], labels=['同视频内', '跨视频'])
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'阈值={threshold}')
            plt.ylabel('距离')
            plt.legend()
            
            plt.suptitle(f'距离分布分析 (建议阈值: {suggested_threshold1:.1f} 或 {suggested_threshold2:.1f})')
            os.makedirs('result/GCN', exist_ok=True)
            plt.savefig('result/GCN/distance_analysis.png', dpi=150)
            plt.close()
            print(f"  距离分析图已保存: result/GCN/distance_analysis.png")
    
    # ========== 生成样本 ==========
    target_same = int(max_pairs_per_video * num_videos * (1 - cross_video_ratio))
    target_cross = int(max_pairs_per_video * num_videos * cross_video_ratio)
    
    print(f"\n目标样本: 同视频 {target_same}, 跨视频 {target_cross}")
    
    all_pairs = []
    same_video_pairs = []
    cross_video_pairs = []
    
    # 同视频样本
    if target_same > 0:
        print(f"\n生成同视频样本...")
        for vi, kps in enumerate(video_keypoints):
            T = len(kps)
            pairs = []
            target_per_video = target_same // num_videos
            
            for i in range(min(T-1, target_per_video)):
                dist = np.mean(np.linalg.norm(kps[i] - kps[i+1], axis=1))
                pairs.append((kps[i], kps[i+1], 1 if dist < threshold else 0))
            
            step = max(2, T // max(1, target_per_video))
            for i in range(0, T - step, step * 2):
                if len(pairs) >= target_per_video:
                    break
                j = min(i + step, T - 1)
                if j > i:
                    dist = np.mean(np.linalg.norm(kps[i] - kps[j], axis=1))
                    pairs.append((kps[i], kps[j], 1 if dist < threshold else 0))
            
            while len(pairs) < target_per_video and T >= 2:
                i, j = random.sample(range(T), 2)
                dist = np.mean(np.linalg.norm(kps[i] - kps[j], axis=1))
                pairs.append((kps[i], kps[j], 1 if dist < threshold else 0))
            
            all_pairs.extend(pairs)
            same_video_pairs.extend(pairs)
            pos = sum(1 for _, _, l in pairs if l == 1)
            print(f"  {video_names[vi]}: {len(pairs)} 对 (正:{pos}, 负:{len(pairs)-pos})")
    
    # 跨视频样本
    if target_cross > 0 and num_videos >= 2:
        print(f"\n生成跨视频样本...")
        num_pairs_video = num_videos * (num_videos - 1) // 2
        per_pair_target = target_cross // max(1, num_pairs_video)
        
        for vi in range(num_videos):
            for vj in range(vi + 1, num_videos):
                kps_i, kps_j = video_keypoints[vi], video_keypoints[vj]
                pairs = []
                
                for _ in range(per_pair_target):
                    i = random.randint(0, len(kps_i) - 1)
                    j = random.randint(0, len(kps_j) - 1)
                    dist = np.mean(np.linalg.norm(kps_i[i] - kps_j[j], axis=1))
                    pairs.append((kps_i[i], kps_j[j], 1 if dist < threshold else 0))
                
                all_pairs.extend(pairs)
                cross_video_pairs.extend(pairs)
                pos = sum(1 for _, _, l in pairs if l == 1)
                print(f"  {video_names[vi]} vs {video_names[vj]}: {len(pairs)} 对 (正:{pos}, 负:{len(pairs)-pos})")
    
    # ========== 详细统计 ==========
    total_pos = sum(1 for _, _, l in all_pairs if l == 1)
    total_neg = len(all_pairs) - total_pos
    same_pos = sum(1 for _, _, l in same_video_pairs if l == 1)
    same_neg = len(same_video_pairs) - same_pos
    cross_pos = sum(1 for _, _, l in cross_video_pairs if l == 1)
    cross_neg = len(cross_video_pairs) - cross_pos
    
    print(f"\n{'='*60}")
    print(f"数据集构建完成")
    print(f"{'='*60}")
    print(f"总样本: {len(all_pairs)} (正:{total_pos}, 负:{total_neg})")
    print(f"  同视频: {len(same_video_pairs)} (正:{same_pos}, 负:{same_neg})")
    print(f"  跨视频: {len(cross_video_pairs)} (正:{cross_pos}, 负:{cross_neg})")
    print(f"{'='*60}")
    
    # 阈值建议
    if same_neg > len(same_video_pairs) * 0.3:
        print(f"⚠️ 同视频负样本偏多({same_neg/len(same_video_pairs)*100:.1f}%) → 建议增大阈值")
    if cross_pos > len(cross_video_pairs) * 0.3:
        print(f"⚠️ 跨视频正样本偏多({cross_pos/len(cross_video_pairs)*100:.1f}%) → 建议减小阈值")
    
    return all_pairs
# ========== 保存数据集 ==========
def save_dataset(pairs, save_path='training_data.npz'):
    kp1_list = [p[0] for p in pairs]
    kp2_list = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]
    
    np.savez(save_path,
             keypoints1=np.array(kp1_list, dtype=np.float32),
             keypoints2=np.array(kp2_list, dtype=np.float32),
             labels=np.array(labels, dtype=np.int8))
    print(f"数据集已保存: {save_path}")
    print(f"  keypoints1形状: {np.array(kp1_list).shape}")
    print(f"  keypoints2形状: {np.array(kp2_list).shape}")
    print(f"  labels形状: {np.array(labels).shape}")

# ========== 加载数据集 ==========
def load_dataset(load_path='training_data.npz'):
    data = np.load(load_path)
    return data['keypoints1'], data['keypoints2'], data['labels']

def visualize_distance_distribution(video_paths, sample_size=500, normalize=True):
    """
    分析关键点距离分布，帮助确定阈值
    Args:
        video_paths: 视频路径列表
        sample_size: 随机采样的帧对数量
        normalize: 是否归一化关键点
    """
    import matplotlib.pyplot as plt
    
    init_models()
    
    # 提取所有视频的关键点
    all_keypoints = []
    for video_path in video_paths:
        print(f"处理: {os.path.basename(video_path)}")
        kps = extract_video_keypoints(video_path, normalize=normalize)
        if len(kps) > 0:
            all_keypoints.append(kps)
    
    if len(all_keypoints) == 0:
        print("没有有效的关键点数据")
        return
    
    # 收集距离
    same_video_distances = []  # 同视频内距离
    cross_video_distances = []  # 跨视频距离
    
    print(f"\n采样 {sample_size} 对帧...")
    
    for _ in range(sample_size):
        # 随机选择两个视频
        vi = random.randint(0, len(all_keypoints) - 1)
        vj = random.randint(0, len(all_keypoints) - 1)
        
        kps_i = all_keypoints[vi]
        kps_j = all_keypoints[vj]
        
        i = random.randint(0, len(kps_i) - 1)
        j = random.randint(0, len(kps_j) - 1)
        
        dist = np.mean(np.linalg.norm(kps_i[i] - kps_j[j], axis=1))
        
        if vi == vj:
            # 同视频
            same_video_distances.append(dist)
        else:
            # 不同视频
            cross_video_distances.append(dist)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 子图1：直方图
    plt.subplot(1, 2, 1)
    plt.hist(same_video_distances, bins=30, alpha=0.7, label='同视频内', color='blue')
    plt.hist(cross_video_distances, bins=30, alpha=0.7, label='跨视频', color='orange')
    plt.xlabel('关键点平均欧氏距离')
    plt.ylabel('频次')
    plt.title('距离分布直方图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：箱线图
    plt.subplot(1, 2, 2)
    bp = plt.boxplot([same_video_distances, cross_video_distances], 
                     labels=['同视频内', '跨视频'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('orange')
    plt.ylabel('关键点平均欧氏距离')
    plt.title('距离分布箱线图')
    plt.grid(True, alpha=0.3)
    
    # 计算建议阈值
    if same_video_distances and cross_video_distances:
        q1_same = np.percentile(same_video_distances, 75)
        q3_cross = np.percentile(cross_video_distances, 25)
        suggested_threshold = (q1_same + q3_cross) / 2
        
        # 另一种方法：同视频最大值的1.5倍
        max_same = np.max(same_video_distances)
        suggested_threshold2 = max_same * 1.2
        
        plt.suptitle(f'距离分布分析\n建议阈值: {suggested_threshold:.2f} 或 {suggested_threshold2:.2f}', 
                    fontsize=12, fontweight='bold')
        
        print(f"\n{'='*50}")
        print("距离统计:")
        print(f"  同视频内距离: 均值={np.mean(same_video_distances):.2f}, "
              f"中位数={np.median(same_video_distances):.2f}, "
              f"最大值={np.max(same_video_distances):.2f}")
        print(f"  跨视频距离:   均值={np.mean(cross_video_distances):.2f}, "
              f"中位数={np.median(cross_video_distances):.2f}, "
              f"最小值={np.min(cross_video_distances):.2f}")
        print(f"\n建议阈值选项:")
        print(f"  方法1 (分位数中点): {suggested_threshold:.2f}")
        print(f"  方法2 (同视频最大1.2倍): {suggested_threshold2:.2f}")
        print(f"{'='*50}")
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('result/GCN', exist_ok=True)
    save_path = 'result/GCN/distance_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图片已保存: {save_path}")
    
    plt.show()
    plt.close()
    
    return same_video_distances, cross_video_distances

if __name__ == '__main__':
    video_folder = 'video_origin/data_video/use/'
    
    video_files = [
        'run_man.mp4',
        'run_woman.mp4',
    ]
    
    video_paths = [os.path.join(video_folder, v) for v in video_files if os.path.exists(os.path.join(video_folder, v))]
    
    if not video_paths:
        print("没有找到视频文件")
        exit()
    print(f"找到 {len(video_paths)} 个视频:")

    THRESHOLD = 25  # 根据上面输出的建议值修改这里
    # 构建数据集
    pairs = build_dataset(video_paths, threshold=THRESHOLD, max_pairs_per_video=600, 
                        normalize=True, cross_video_ratio=0.4)  
    if pairs:
        os.makedirs('result/GCN', exist_ok=True)
        save_dataset(pairs, 'result/GCN/training_data.npz')
    else:
        print("没有生成任何训练样本")