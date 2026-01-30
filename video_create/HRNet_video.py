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
gap = 0
#24帧每秒
VIDEO_FRAME_SPEED = 24
TIME_GAP = round(1/VIDEO_FRAME_SPEED,3)
current_frame = 1 
START_TIME = time.time()

# 全局变量用于模型，避免重复加载
device = None
hrnet_model = None  # 改名为hrnet_model避免混淆
yolo_model = None  # 添加YOLO模型
person_info = None
hrnet_transform = None

def init_models():
    """初始化YOLO和HRNet模型"""
    global device, hrnet_model, yolo_model, person_info, hrnet_transform
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"using device: {device}")
        #初始化YOLO模型
        try:
            yolo_model = YOLO('weights\yolo11n.pt')  # 会自动下载
            print("✅ YOLO模型加载成功")
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")
            return False
        #初始化HRNet模型
        weights_path = "HRnet\\pytorch\\pose_coco\\pose_hrnet_w32_256x192.pth"
        keypoint_json_path = "HRnet\\person_keypoints.json"
        # 确保路径存在
        if not os.path.exists(weights_path):
            print(f"❌ HRNet权重文件不存在: {weights_path}")
            return False
        if not os.path.exists(keypoint_json_path):
            print(f"❌ 关键点JSON文件不存在: {keypoint_json_path}")
            return False
        # read json file
        try:
            with open(keypoint_json_path, "r") as f:
                person_info = json.load(f)
        except Exception as e:
            print(f"❌ 加载JSON文件错误: {e}")
            return False
        try:
            # create model
            hrnet_model = HighResolutionNet(base_channel=32)
            weights = torch.load(weights_path, map_location=device)
            weights = weights if "model" not in weights else weights["model"]
            hrnet_model.load_state_dict(weights)
            hrnet_model.to(device)
            hrnet_model.eval()
            print("✅ HRNet模型加载成功")
        except Exception as e:
            print(f"❌ HRNet模型加载错误: {e}")
            return False
        # 3. 初始化数据转换
        resize_hw = (256, 192)
        hrnet_transform = transforms.Compose([
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("✅ 所有模型初始化完成")
    return True

def detect_person_with_yolo(frame, conf_threshold=0.5):
    """
    使用YOLO检测人物位置
    返回: 第一个人物的边界框 [x1, y1, x2, y2] 或 None
    """
    if yolo_model is None:
        return None
    # YOLO检测
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

def predict_frame(frame):
    """处理单帧图像，检测关键点（使用YOLO+HRNet）"""
    global device, hrnet_model, yolo_model, person_info, hrnet_transform
    # 初始化模型（如果未初始化）
    if yolo_model is None or hrnet_model is None:
        if not init_models():
            return [[]], []
    
    # 1. 使用YOLO检测人物
    bbox, conf = detect_person_with_yolo(frame, conf_threshold=0.5)
    
    if bbox is None:
        # 如果没有检测到人物，返回空结果
        return [[]], []
    
    # 确保边界框在图像范围内
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    bbox = [x1, y1, x2, y2]
    
    # 在图像上绘制YOLO检测框（绿色）
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                    (0, 0, 255), 2)
    cv2.putText(frame, f"Person: {conf:.2f}", 
                (int(x1), max(20, int(y1) - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 2. 裁剪人物区域（添加少量padding）
    padding = 10
    roi_x1 = max(0, int(x1) - padding)
    roi_y1 = max(0, int(y1) - padding)
    roi_x2 = min(width, int(x2) + padding)
    roi_y2 = min(height, int(y2) + padding)
    
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
    
    # 3. 在ROI上使用HRNet检测关键点
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
        
        # 4. 将关键点坐标转换回原始图像
        keypoints_list = []
        for i, (kp, score_val) in enumerate(zip(keypoints, scores)):
            if hasattr(kp, '__iter__') and len(kp) >= 2:
                # HRNet输出的坐标（相对于ROI）
                roi_x, roi_y = float(kp[0]), float(kp[1])
                
                # 公式：原始坐标 = ROI起点 + ROI坐标
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

def process_frame(img,preview=True):
    """处理视频帧"""
    global current_frame, step_count, step_fres, score, gap, time_current
    # 预测单帧状态
    # process_single_image(img) 
    list_p, scores = predict_frame(img)
    
    # # list_p = keypoints.tolist()
    angle_ra = angle_show(list_p, (10,20), (0,0,255), "RightArm", r_arm, img)
    angle_la = angle_show(list_p, (10,40), (0,0,255), "LeftArm", l_arm, img)
    angle_rl = angle_show(list_p, (10,60), (0,0,255), "RightLeg", r_leg, img)
    angle_ll = angle_show(list_p, (10,80), (0,0,255), "LeftLeg", l_leg, img)
    p_pos = get_keypoints(list_p)
    p13 = p_pos[13]
    p14 = p_pos[14]
    p15 = p_pos[15]
    p16 = p_pos[16]
    if change_detector(p15, p16):
            step_count += 1
            time_current.append(current_frame)
            if step_count > 2 :
                gap = ((time_current[-1]-time_current[-2]) * TIME_GAP)
                if gap > 0.1:
                    step_fres = round(1/gap,3)

    cv2.putText(img, f"Frequency:{str(step_fres)}", (10,160), 
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
    fontScale=0.6, thickness=2, color=(255,0,255))
    # 显示步数（使用step_count替代原来的i）
    cv2.putText(img, str(step_count), (10, 100), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.6, thickness=2, color=(255,0,0))
    current_frame += 1
    draw_select(img, list_p)
    if preview:
        cv2.imshow('YOLO Detection', img)
        cv2.waitKey(1)
    return img, list_p

def generate_video(input_path):
    """生成处理后的视频"""
    print('视频开始处理', input_path)
    cap = cv2.VideoCapture(input_path)
    # 获取视频总帧数
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
    cap.release()
    print('视频总帧数：', frame_count)
    # 重置到视频开头
    cap = cv2.VideoCapture(input_path)
    # 生成视频
    out, output_path, cap = video_out(input_path, 'video_origin/result_video/', 'yolo_hrnet-')
    if out is None:
        print("无法创建视频输出")
        return
    try:
        frame_index = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_index += 1
            try:
                processed_frame, list_p = process_frame(frame)
                # 绘制关键点（使用你的draw_select函数）
                out.write(processed_frame)
                print(f"处理第 {frame_index}/{frame_count} 帧", end='\r')
                progress_bar(frame_index,frame_count)
            except Exception as e:
                print(f'处理第{frame_index}帧时出错: {str(e)[:50]}...')
                # 如果出错，写入原始帧
                out.write(frame)
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

def progress_bar(current, total, bar_length=30, prefix="进度"):
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    sys.stdout.write(f'\r{prefix}: |{bar}| {percent*100:.1f}% ({current}/{total})')
    sys.stdout.flush()
    
    if current == total:
        print()

def change_detector(a,b):
    if a>b and step[-1] != 1:
        step.append(1)
        return True
    elif a<b and step[-1] != 2:
        step.append(2)
        return True


if __name__ == '__main__':
    input_path = 'video_origin/data_video/run_woman2.mp4'
    START_TIME = time.time()
    generate_video(input_path)
    current_time = time.time()
    print(f"生成视频总耗时：{show_time(START_TIME, current_time)}")
    print(f"步数: {step_count}")