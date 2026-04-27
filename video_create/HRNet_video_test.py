#结合了YOLO人体识别和HRnet姿态检测的视频生成代码,生成耗时更少

import math  
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from HRNet_model import HighResolutionNet
import transforms
from run_test import *
from tqdm import tqdm
import time    
import cv2
from utill import *
from collections import deque
from time_utils import show_time
from ultralytics import YOLO  # 添加YOLO库
import sys
from Feature import *
from matplotlib import rcParams #字体
rcParams['font.family'] = 'SimHei'

# ==================== 可配置参数 ====================
# 输出视频尺寸 (宽度, 高度)
OUTPUT_VIDEO_SIZE = (1280, 720)  # 可根据需要修改，例如 (1280, 720)
# ===================================================

#视频输入地址
input_path = 'video_origin/data_video/use/0002.mp4'
video_name = os.path.splitext(os.path.basename(input_path))[0]

trajectory_tracker = KeypointTrajectoryTracker(
    num_keypoints=17,
    history_length=200,  # 保存最近200帧的轨迹
    output_dir="result/track_img"
)

time_current = []
data_buffer = deque(maxlen=50)
weight = [1,1,1,0,0,0]
step=[0]
r_arm = [6,8,10]
l_arm = [5,7,9]
r_leg = [12,14,16]
l_leg = [11,13,15]
step_count = 0  # 改为step_count，避免与循环变量i冲突
score = 0
step_fres = 0
frame_count = 0
#24帧每秒
VIDEO_FRAME_SPEED = 24
TIME_GAP = round(1/VIDEO_FRAME_SPEED,3)
current_frame = 1 
START_TIME = time.time()
Key_point_list = []                      #用于存放当前帧关键点
Key_point_acceleration = []              #用于存放当前所有关键点的加速度
Max_acc = []                             #最大加速度总列表
All_feature = []                         #用于存放特征
All_point = []                           #用于存放关键点
Normal_keypoints = []                    #归一化关键点，用于计算向量
Vector_list = []                        #向量信息

# 全局变量用于模型，避免重复加载
device = None
hrnet_model = None  # 改名为hrnet_model避免混淆
yolo_model = None  # 添加YOLO模型
person_info = None
hrnet_transform = None

# 平滑变量
smooth_center = None
SMOOTH_ALPHA = 0.7  # 平滑系数

def init_models():
    """初始化YOLO和HRNet模型"""
    global device, hrnet_model, yolo_model, person_info, hrnet_transform
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"using device: {device}")
        #初始化YOLO模型
        try:
            yolo_model = YOLO('weights/yolo11n.pt')  # 会自动下载
            print("YOLO模型加载成功")
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            return False
        #初始化HRNet模型
        weights_path = "HRnet\\pytorch\\pose_coco\\pose_hrnet_w32_384x288.pth"
        keypoint_json_path = "HRnet\\person_keypoints.json"
        # 确保路径存在
        if not os.path.exists(weights_path):
            print(f"HRNet权重文件不存在: {weights_path}")
            return False
        if not os.path.exists(keypoint_json_path):
            print(f"关键点JSON文件不存在: {keypoint_json_path}")
            return False
        # read json file
        try:
            with open(keypoint_json_path, "r") as f:
                person_info = json.load(f)
        except Exception as e:
            print(f"加载JSON文件错误: {e}")
            return False
        try:
            # create model
            hrnet_model = HighResolutionNet(base_channel=32)
            weights = torch.load(weights_path, map_location=device)
            weights = weights if "model" not in weights else weights["model"]
            hrnet_model.load_state_dict(weights)
            hrnet_model.to(device)
            hrnet_model.eval()
            print("HRNet模型加载成功")
        except Exception as e:
            print(f"HRNet模型加载错误: {e}")
            return False
        # 3. 初始化数据转换 - 384x288模型
        resize_hw = (288, 384)  # 高度288，宽度384
        hrnet_transform = transforms.Compose([
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("所有模型初始化完成")
    return True

def detect_person_with_yolo(frame, conf_threshold=0.5):
    """
    使用YOLO检测人物位置，返回面积最大的人物
    """
    if yolo_model is None:
        return None, 0
    
    results = yolo_model(frame, verbose=False)
    persons = []  # 存储所有检测到的人物 (bbox, conf, area)
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # YOLO中person的类别ID通常是0
                if cls_id == 0 and conf >= conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)  # 计算面积
                    persons.append(([float(x1), float(y1), float(x2), float(y2)], conf, area))
    
    if not persons:
        return None, 0
    
    # 按面积降序排序，返回面积最大的那个人
    persons.sort(key=lambda x: x[2], reverse=True)
    return persons[0][0], persons[0][1]

def normalize_keypoints(p_pos, target_torso_length=100):
    """
    将关键点坐标归一化
    """
    if len(p_pos) < 17:
        return p_pos
    # 获取躯干中心点（肩膀中点和髋部中点）
    # 左肩(p5)和右肩(p6)的中点
    shoulder_center_x = (p_pos[5][0] + p_pos[6][0]) / 2
    shoulder_center_y = (p_pos[5][1] + p_pos[6][1]) / 2
    # 左髋(p11)和右髋(p12)的中点
    hip_center_x = (p_pos[11][0] + p_pos[12][0]) / 2
    hip_center_y = (p_pos[11][1] + p_pos[12][1]) / 2
    # 计算躯干长度
    torso_length = math.sqrt((shoulder_center_x - hip_center_x)**2 + 
                            (shoulder_center_y - hip_center_y)**2)
    if torso_length < 1e-6:
        return p_pos
    # 计算缩放比例
    scale = target_torso_length / torso_length
    # 以髋部中点为归一化原点
    center_x = hip_center_x
    center_y = hip_center_y
    # 归一化所有关键点
    normalized_points = []
    for i in range(len(p_pos)):
        if i < 17:  # 只处理17个关键点
            norm_x = (p_pos[i][0] - center_x) * scale
            norm_y = (p_pos[i][1] - center_y) * scale
            normalized_points.append([norm_x, norm_y])
        else:
            normalized_points.append(p_pos[i])
    return normalized_points, scale, torso_length, (center_x, center_y)

def predict_frame(frame):
    """处理单帧图像，检测关键点（动态ROI + 人物居中 + 平滑）"""
    global device, hrnet_model, yolo_model, person_info, hrnet_transform, smooth_center
    
    # 初始化模型（如果未初始化）
    if yolo_model is None or hrnet_model is None:
        if not init_models():
            return [[]], []
    
    height, width = frame.shape[:2]
    bbox, conf = detect_person_with_yolo(frame, conf_threshold=0.5)
    
    if bbox is None:
        # 没有检测到人物时重置平滑变量
        smooth_center = None
        return [[]], []
    
    x1, y1, x2, y2 = bbox
    
    # 计算当前帧的人物中心
    current_center_x = (x1 + x2) / 2
    current_center_y = (y1 + y2) / 2
    
    # 平滑处理中心点
    if smooth_center is None:
        smooth_center = (current_center_x, current_center_y)
    else:
        smooth_center_x = SMOOTH_ALPHA * current_center_x + (1 - SMOOTH_ALPHA) * smooth_center[0]
        smooth_center_y = SMOOTH_ALPHA * current_center_y + (1 - SMOOTH_ALPHA) * smooth_center[1]
        smooth_center = (smooth_center_x, smooth_center_y)
    
    # 使用平滑后的中心点重新计算检测框
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    x1 = smooth_center[0] - bbox_w / 2
    x2 = smooth_center[0] + bbox_w / 2
    y1 = smooth_center[1] - bbox_h / 2
    y2 = smooth_center[1] + bbox_h / 2
    
    # 边界裁剪
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    
    # 添加动态padding（人物大小的15-20%）
    padding = int(min(bbox_w, bbox_h) * 0.2)
    padding = max(10, min(padding, 50))  # 限制在10-50之间
    
    roi_x1 = max(0, int(x1) - padding)
    roi_y1 = max(0, int(y1) - padding)
    roi_x2 = min(width, int(x2) + padding)
    roi_y2 = min(height, int(y2) + padding)
    
    # 在图像上绘制YOLO检测框（绿色）
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                    (0, 255, 0), 2)
    cv2.putText(frame, f"Person: {conf:.2f}", 
                (int(x1), max(20, int(y1) - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 裁剪ROI
    person_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    if person_roi.size == 0:
        return [[]], []
    
    # 确保图像是RGB格式
    if len(person_roi.shape) == 3:
        if person_roi.shape[2] == 3:
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        else:
            person_rgb = person_roi
    else:
        person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_GRAY2RGB)
    
    roi_height, roi_width = person_rgb.shape[:2]
    
    # 在ROI上使用HRNet检测关键点
    img_tensor, target = hrnet_transform(
        person_rgb, 
        {"box": [0, 0, roi_width - 1, roi_height - 1]}
    )
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    with torch.no_grad():
        outputs = hrnet_model(img_tensor.to(device))

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
        
        # 确保是一维数组
        if scores.ndim == 0:
            scores = np.array([scores])
        
        # 将关键点坐标转换回原始图像
        keypoints_list = []
        for i, (kp, score_val) in enumerate(zip(keypoints, scores)):
            if hasattr(kp, '__iter__') and len(kp) >= 2:
                # HRNet输出的坐标（相对于ROI）
                roi_x, roi_y = float(kp[0]), float(kp[1])
                
                # 原始坐标 = ROI起点 + ROI内坐标
                orig_x = roi_x1 + roi_x
                orig_y = roi_y1 + roi_y
                # 确保坐标在图像范围内
                orig_x = max(0, min(orig_x, width - 1))
                orig_y = max(0, min(orig_y, height - 1))
                keypoints_list.append([int(orig_x), int(orig_y), float(score_val)])
            else:
                keypoints_list.append([0, 0, 0.0])
        result_list = [keypoints_list]
        
        return result_list, scores

def process_frame(img, preview=True, normalize_for_storage=True):
    """处理视频帧"""
    global current_frame, step_count, step_fres, score, time_current, smooth_center
    
    # 预测单帧状态
    list_p, scores = predict_frame(img)
    p_pos = get_keypoints(list_p)  #获取关键点列表[[x,y],[x,y]..]
    
    # 添加归一化处理
    normalized_points = None
    scale_info = None
    if normalize_for_storage and p_pos:
        normalized_points, scale, torso_length, center = normalize_keypoints(p_pos, target_torso_length=100)
        scale_info = {'scale': scale, 'torso_length': torso_length, 'center': center}
    trajectory_tracker.update(p_pos) #更新轨迹
    feature = Feature(p_pos)         #获取关键特征

    #----------向量-------------
    if normalized_points and len(normalized_points) >= 17:
        if not Normal_keypoints:
            for i in range(17):
                Normal_keypoints.append(normalized_points[i])
            for i in range(17):
                Vector_list.append([0.0, 0.0])
        else:
            for i in range(17):
                dx = normalized_points[i][0] - Normal_keypoints[i][0]
                dy = normalized_points[i][1] - Normal_keypoints[i][1]
                vector_info = [dx, dy]
                Vector_list.append(vector_info)
            Normal_keypoints.clear()
            for i in range(17):
                Normal_keypoints.append(normalized_points[i])

    #----------加速度-------------
    if p_pos and len(p_pos) >= 17:
        Key_point_acceleration.clear()
        if not Key_point_list:
            for j in range(17):
                Key_point_list.append(p_pos[j])
        else:
            for j in range(17):
                Key_point_acceleration.append(acceleration(p_pos[j], Key_point_list[j], TIME_GAP))
            Key_point_list.clear()
            for j in range(17):
                Key_point_list.append(p_pos[j])
        if Key_point_acceleration:
            Max_acc.append(max(Key_point_acceleration) / 1000)

    # 步频检测
    if p_pos and len(p_pos) >= 17:
        p15 = p_pos[15]
        p16 = p_pos[16]
        if change_detector(p15, p16):
            step_count += 1
            time_current.append(current_frame)
            if step_count > 2:
                if len(time_current) >= 2 and time_current[-1] > time_current[-2]:
                    frame_diff = time_current[-1] - time_current[-2]
                    gap = frame_diff * TIME_GAP
                    if gap > 0.1:
                        step_fres = round(1 / gap, 3)

    # 绘制关键点（在原图上）
    draw = Draw(img, list_p)
    draw.draw_select()
    
    # 在原图上绘制信息
    cv2.putText(img, f"Frequency:{str(step_fres)}", (10, 160), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.6, thickness=2, color=(255,0,255))
    cv2.putText(img, f"Steps:{str(step_count)}", (10, 100), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.6, thickness=2, color=(255,0,0))
    
    if scale_info and scale_info['torso_length'] > 0:
        cv2.putText(img, f"Torso: {scale_info['torso_length']:.0f}px", (10, 180), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, thickness=1, color=(0,255,255))
    
    current_frame += 1
    feature_frame = feature.get_all_features() if p_pos and len(p_pos) >= 17 else [0] * 50
    
    # ========== 可视化处理：以人物中心裁剪固定尺寸 ==========
    output_w, output_h = OUTPUT_VIDEO_SIZE
    vis_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    if smooth_center is not None:
        center_x, center_y = smooth_center
        
        # 计算裁剪区域
        crop_x1 = int(center_x - output_w / 2)
        crop_y1 = int(center_y - output_h / 2)
        crop_x2 = crop_x1 + output_w
        crop_y2 = crop_y1 + output_h
        
        # 计算有效区域
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(img.shape[1], crop_x2)
        src_y2 = min(img.shape[0], crop_y2)
        
        # 计算目标区域
        dst_x1 = max(0, -crop_x1)
        dst_y1 = max(0, -crop_y1)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # 复制有效区域
        if src_x2 > src_x1 and src_y2 > src_y1:
            vis_frame[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    
    if preview:
        cv2.imshow('YOLO Detection', vis_frame)
        cv2.waitKey(1)
    
    return vis_frame, p_pos, feature_frame, normalized_points, scale_info

def generate_video(input_path):
    """生成处理后的视频"""
    print('=' * 50)
    print(f'视频开始处理: {input_path}')
    print(f'输出视频尺寸: {OUTPUT_VIDEO_SIZE[0]} x {OUTPUT_VIDEO_SIZE[1]}')
    print('=' * 50)
    
    cap = cv2.VideoCapture(input_path)
    # 获取视频总帧数
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
    cap.release()
    print('视频总帧数：', frame_count-1)
    # 重置到视频开头
    cap = cv2.VideoCapture(input_path)
    
    # 使用自定义输出尺寸
    out_w, out_h = OUTPUT_VIDEO_SIZE
    output_path = f'result/result_video/video/yolo_hrnet-{video_name}.mp4'
    os.makedirs('result/result_video/video/', exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, VIDEO_FRAME_SPEED, (out_w, out_h))
    
    if not out.isOpened():
        print("无法创建视频输出")
        return
    
    # 存储归一化关键点的列表
    All_normalized_points = []
    All_scale_info = []
    
    try:
        frame_index = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_index += 1
            try:
                processed_frame, p_pos, feature_frame, normalized_points, scale_info = process_frame(frame, preview=True, normalize_for_storage=True)
                All_feature.append(feature_frame)
                All_point.append(p_pos)
                # 只保存有效的归一化点（非None）
                if normalized_points is not None:
                    # 转换为numpy数组并确保是float32
                    norm_array = np.array(normalized_points, dtype=np.float32)
                    All_normalized_points.append(norm_array)
                else:
                    # 没有检测到人物时，添加一个全零的占位
                    All_normalized_points.append(np.zeros((17, 2), dtype=np.float32))
                All_scale_info.append(scale_info)
                out.write(processed_frame)
                print(f"处理第 {frame_index}/{frame_count} 帧", end='\r')
                progress_bar(frame_index, frame_count)
            except Exception as e:
                print(f'处理第{frame_index}帧时出错: {str(e)[:50]}...')
                # 输出黑色画面
                blank = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                out.write(blank)
                continue
                
    except KeyboardInterrupt:
        print('用户中断')
    except Exception as e:
        print(f'处理视频时出错: {e}')
    finally:
        cv2.destroyAllWindows()
        if out is not None:
            out.release()
        if cap is not None:
            cap.release()
        print('\n视频处理完成')
        print('Video saved to:', output_path)
    
    print(f"轨迹分析图保存在: {trajectory_tracker.output_dir}")
    frames = list(range(2, frame_count + 1))
    print(len(Max_acc))

    #===================================
    #自动eps
    if len(Max_acc) > 0:
        eps = auto_eps(Max_acc, 8)
        print(f"当前自动eps为{eps}")
        wrong_point = point_acceleration(frames, Max_acc, video_name, use_dbscan=True, eps=eps, min_samples=8)
    #===================================

    features_array = np.array(All_feature)
    point_array = np.array(All_point)
    if len(Vector_list) > 0:
        vector_array = np.array(Vector_list)
        if len(All_feature) > 0:
            vector_array = vector_array.reshape(len(All_feature), 17, 2)
            # 提取 dx, dy
            dxdy = vector_array[:, :, :2]  # (帧数, 17, 2)
            # 计算全局最大最小值
            dxdy_min = dxdy.min(axis=(0, 1), keepdims=True)
            dxdy_max = dxdy.max(axis=(0, 1), keepdims=True)
            # Min-Max 归一化到 [0, 1]
            dxdy_normalized = (dxdy - dxdy_min) / (dxdy_max - dxdy_min + 1e-8)
            # 直接保存 dxdy_normalized 作为向量特征
            vector_normalized = dxdy_normalized
            np.save(f'result/features/{video_name}_vector.npy', vector_normalized)
    
    # 将列表转换为三维numpy数组 (帧数, 17, 2)
    if All_normalized_points:
        normalized_array = np.stack(All_normalized_points, axis=0)
        np.save(f'result/features/{video_name}_normalized_points.npy', normalized_array)
        print(f"归一化关键点已保存: result/features/{video_name}_normalized_points.npy")
    else:
        print("警告: 没有归一化关键点数据")
    
    np.save(f'result/features/{video_name}_features.npy', features_array)
    np.save(f'result/features/{video_name}_points.npy', point_array)

def progress_bar(current, total, bar_length=30, prefix="进度"):
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    sys.stdout.write(f'\r{prefix}: |{bar}| {percent*100:.1f}% ({current}/{total})')
    sys.stdout.flush()
    
    if current == total:
        print()

def change_detector(a, b):
    if a[1] > b[1] and step[-1] != 1:
        step.append(1)
        return True
    elif a[1] < b[1] and step[-1] != 2:
        step.append(2)
        return True
    return False

if __name__ == '__main__':
    START_TIME = time.time()
    generate_video(input_path)
    current_time = time.time()
    print(f"生成视频总耗时：{show_time(START_TIME, current_time)}")
    print(f"步数: {step_count}")