import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from HRNet_model import HighResolutionNet
import transforms
from ultralytics import YOLO
from collections import deque
import time

# 全局变量
device = None
hrnet_model = None
yolo_model = None
person_info = None
data_transform = None

# 平滑缓冲区
keypoint_buffer = deque(maxlen=5)

def init_models():
    """初始化YOLO和HRNet模型"""
    global device, hrnet_model, yolo_model, person_info, data_transform
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 初始化YOLO模型（人物检测）
    try:
        # 使用YOLOv8预训练模型
        yolo_model = YOLO('weights\yolo11n.pt')  # 或者 yolov8n-pose.pt
        print("YOLO模型加载成功")
    except Exception as e:
        print(f"YOLO模型加载失败: {e}")
        # 尝试其他路径
        yolo_model = YOLO('weights\yolo11n.pt')
    
    # 2. 初始化HRNet模型（关键点检测）
    weights_path = "HRnet\\pytorch\\pose_coco\\pose_hrnet_w32_256x192.pth"
    keypoint_json_path = "HRnet\\person_keypoints.json"
    
    if not os.path.exists(weights_path):
        print(f"HRNet权重文件不存在: {weights_path}")
        return False
    if not os.path.exists(keypoint_json_path):
        print(f"关键点JSON文件不存在: {keypoint_json_path}")
        return False
    
    try:
        with open(keypoint_json_path, "r") as f:
            person_info = json.load(f)
    except Exception as e:
        print(f"加载JSON文件错误: {e}")
        return False
    
    try:
        # 创建HRNet模型
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
    
    # 3. 初始化数据转换
    resize_hw = (256, 192)
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return True

def detect_persons_yolo(frame, conf_threshold=0.5):
    """
    使用YOLO检测人物位置
    返回: 人物边界框列表 [(x1, y1, x2, y2), ...]
    """
    if yolo_model is None:
        return []
    
    # YOLO检测
    results = yolo_model(frame, verbose=False)
    
    person_boxes = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # 获取类别和置信度
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # YOLO中 person 的类别ID通常是0
                if cls_id == 0 and conf >= conf_threshold:
                    # 获取边界框坐标 (xyxy格式)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    person_boxes.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'conf': conf,
                        'cls': 'person'
                    })
    
    return person_boxes

def crop_and_resize_person(frame, bbox, padding_ratio=0.1):  # 减少padding比例
    """
    根据边界框裁剪人物区域
    修复：确保坐标转换正确
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # 将浮点数转换为整数
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # 确保在图像范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    # 计算边界框尺寸
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width <= 0 or bbox_height <= 0:
        return None, (0, 0, 0, 0)
    
    # 添加少量padding
    padding_x = int(bbox_width * padding_ratio)
    padding_y = int(bbox_height * padding_ratio)
    
    # 扩展边界框
    x1_pad = max(0, x1 - padding_x)
    y1_pad = max(0, y1 - padding_y)
    x2_pad = min(width, x2 + padding_x)
    y2_pad = min(height, y2 + padding_y)
    
    # 裁剪人物区域
    person_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
    
    if person_roi.size == 0:
        return None, (0, 0, 0, 0)
    
    return person_roi, (x1_pad, y1_pad, x2_pad, y2_pad)

def detect_keypoints_hrnet(person_roi, roi_bbox, original_bbox):
    """
    在裁剪的人物区域上使用HRNet检测关键点
    修复关键点坐标转换
    """
    global hrnet_model, data_transform, person_info
    
    if hrnet_model is None or person_roi is None or person_roi.size == 0:
        return [], []
    
    # 确保图像是RGB格式
    if len(person_roi.shape) == 3:
        if person_roi.shape[2] == 3:
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        else:
            person_rgb = person_roi
    else:
        person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_GRAY2RGB)
    
    roi_height, roi_width = person_rgb.shape[:2]
    
    if roi_width <= 0 or roi_height <= 0:
        return [], []
    
    # 转换数据
    try:
        img_tensor, target = data_transform(
            person_rgb, 
            {"box": [0, 0, roi_width - 1, roi_height - 1]}
        )
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
    except Exception as e:
        print(f"数据转换错误: {e}")
        return [], []
    
    with torch.no_grad():
        outputs = hrnet_model(img_tensor.to(device))
        
        flip_test = True
        if flip_test:
            flip_tensor = transforms.flip_images(img_tensor)
            flip_outputs = transforms.flip_back(
                hrnet_model(flip_tensor.to(device)), 
                person_info["flip_pairs"]
            )
            flip_outputs[..., 1:] = flip_outputs.clone()[..., 0:-1]
            outputs = (outputs + flip_outputs) * 0.5
        
        # 获取关键点
        keypoints, scores = transforms.get_final_preds(
            outputs, [target["reverse_trans"]], True
        )
        
        # 处理输出
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)
        
        if scores.ndim == 0:
            scores = np.array([scores])
        
        # 关键修复：正确的坐标转换
        converted_keypoints = convert_keypoints_to_original(
            keypoints, roi_bbox, original_bbox, roi_width, roi_height
        )
        
        return converted_keypoints, scores

def convert_keypoints_to_original(keypoints, roi_bbox, original_bbox, roi_width, roi_height):
    """
    修复：将HRNet输出的关键点正确转换到原始图像坐标
    """
    if len(keypoints) == 0:
        return []
    
    # 解包边界框
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox
    orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox
    
    # ROI的实际尺寸
    actual_roi_width = roi_x2 - roi_x1
    actual_roi_height = roi_y2 - roi_y1
    
    if actual_roi_width <= 0 or actual_roi_height <= 0:
        return []
    
    # 原始边界框的尺寸
    orig_width = orig_x2 - orig_x1
    orig_height = orig_y2 - orig_y1
    
    converted_keypoints = []
    
    for kp in keypoints:
        if hasattr(kp, '__iter__') and len(kp) >= 2:
            # HRNet输出的坐标（相对于预处理后的ROI）
            hrnet_x, hrnet_y = float(kp[0]), float(kp[1])
            
            # 关键修复步骤：
            # 1. HRNet输出坐标是相对于预处理后的ROI (roi_width x roi_height)
            # 2. 我们需要先映射到实际ROI尺寸
            # 3. 然后映射到原始图像
            
            # 映射到实际ROI坐标
            roi_x = (hrnet_x / roi_width) * actual_roi_width
            roi_y = (hrnet_y / roi_height) * actual_roi_height
            
            # 映射到原始图像坐标
            # 注意：这里需要加上ROI的偏移，然后调整到原始边界框内
            orig_x = roi_x1 + roi_x
            orig_y = roi_y1 + roi_y
            
            # 确保坐标在原始边界框内（可选）
            # orig_x = max(orig_x1, min(orig_x, orig_x2))
            # orig_y = max(orig_y1, min(orig_y, orig_y2))
            
            converted_keypoints.append([orig_x, orig_y])
        else:
            converted_keypoints.append([0, 0])
    
    return converted_keypoints

def smooth_keypoints(keypoints_list):
    """平滑关键点（移动平均）"""
    global keypoint_buffer
    
    if len(keypoints_list) == 0:
        return keypoints_list
    
    # 添加到缓冲区
    keypoint_buffer.append(keypoints_list)
    
    if len(keypoint_buffer) < 2:
        return keypoints_list
    
    # 计算移动平均
    smoothed_list = []
    num_points = len(keypoints_list)
    
    for i in range(num_points):
        x_sum, y_sum = 0, 0
        count = 0
        
        for buffer_kps in keypoint_buffer:
            if i < len(buffer_kps) and len(buffer_kps[i]) >= 2:
                x, y = buffer_kps[i][0], buffer_kps[i][1]
                if x > 0 and y > 0:  # 只使用有效点
                    x_sum += x
                    y_sum += y
                    count += 1
        
        if count > 0:
            avg_x = x_sum / count
            avg_y = y_sum / count
            smoothed_list.append([avg_x, avg_y])
        else:
            smoothed_list.append([0, 0])
    
    return smoothed_list

def process_frame_with_yolo_hrnet(frame):
    """
    主处理函数：YOLO检测 + HRNet关键点检测
    修复坐标偏移版本
    """
    global device
    
    # 初始化模型（如果未初始化）
    if yolo_model is None or hrnet_model is None:
        if not init_models():
            return frame, []
    
    # 1. 使用YOLO检测人物
    person_boxes = detect_persons_yolo(frame, conf_threshold=0.5)
    
    all_keypoints = []
    frame_height, frame_width = frame.shape[:2]
    
    # 2. 对每个检测到的人物进行关键点检测
    for person_box in person_boxes:
        bbox = person_box['bbox']
        confidence = person_box['conf']
        
        # 确保边界框在图像范围内
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))
        bbox = [x1, y1, x2, y2]
        
        # 3. 裁剪人物区域
        person_roi, roi_bbox = crop_and_resize_person(frame, bbox)
        
        if person_roi is None or person_roi.size == 0:
            continue
        
        # 4. 在裁剪区域上使用HRNet检测关键点
        roi_height, roi_width = person_roi.shape[:2]
        keypoints, scores = detect_keypoints_hrnet(
            person_roi, roi_bbox, bbox
        )
        
        if len(keypoints) > 0:
            # 5. 添加置信度信息
            keypoints_with_scores = []
            for i, kp in enumerate(keypoints):
                if i < len(scores):
                    score_val = float(scores[i])
                else:
                    score_val = 0.0
                
                # 确保坐标在图像范围内
                x, y = kp[0], kp[1]
                x = max(0, min(x, frame_width - 1))
                y = max(0, min(y, frame_height - 1))
                
                keypoints_with_scores.append([x, y, score_val])
            
            # 6. 平滑处理
            smoothed_keypoints = smooth_keypoints(keypoints_with_scores)
            all_keypoints.append(smoothed_keypoints)
            
            # 7. 在图像上绘制边界框和关键点
            draw_detection_results(frame, bbox, smoothed_keypoints, confidence, roi_bbox)
    
    return frame, all_keypoints

def draw_detection_results(frame, bbox, keypoints, confidence, roi_bbox=None):
    """绘制检测结果"""
    height, width = frame.shape[:2]
    
    # 绘制人物边界框
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 绘制置信度
    cv2.putText(frame, f"Person: {confidence:.2f}", 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 2)
    
    # 如果提供了ROI边界框，绘制它（调试用）
    if roi_bbox:
        rx1, ry1, rx2, ry2 = map(int, roi_bbox)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)
    
    # 绘制关键点
    if keypoints and len(keypoints) > 0:
        # COCO关键点连接对
        skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], 
            [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], 
            [1, 3], [2, 4], [3, 5], [4, 6]
        ]
        
        # 绘制骨骼连接
        for connection in skeleton:
            start_idx, end_idx = connection
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                len(keypoints[start_idx]) >= 3 and len(keypoints[end_idx]) >= 3):
                
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]
                
                # 只绘制置信度较高的连接
                if (start_kp[2] > 0.1 and end_kp[2] > 0.1 and
                    start_kp[0] > 0 and start_kp[1] > 0 and 
                    end_kp[0] > 0 and end_kp[1] > 0):
                    
                    start_pt = (int(start_kp[0]), int(start_kp[1]))
                    end_pt = (int(end_kp[0]), int(end_kp[1]))
                    
                    cv2.line(frame, start_pt, end_pt, (255, 0, 0), 2)
        
        # 绘制关键点
        for i, kp in enumerate(keypoints):
            if len(kp) >= 3 and kp[2] > 0.1:  # 置信度阈值
                x, y = int(kp[0]), int(kp[1])
                if x > 0 and y > 0:
                    # 根据置信度改变颜色
                    color_intensity = int(255 * kp[2])
                    color = (0, 0, color_intensity)  # 蓝色，越深置信度越高
                    cv2.circle(frame, (x, y), 3, color, -1)
                    
                    # 可选：显示关键点编号（调试用）
                    cv2.putText(frame, str(i), (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def process_video(input_path, output_path, show_preview=False):
    """处理视频文件"""
    # 初始化模型
    if not init_models():
        print("模型初始化失败")
        return
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_path}")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"开始处理视频: {input_path}")
    print(f"视频信息: {width}x{height}, {fps}FPS, 总帧数: {total_frames}")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 处理帧
        processed_frame, keypoints_list = process_frame_with_yolo_hrnet(frame)
        
        # 写入输出视频
        out.write(processed_frame)
        
        # 显示进度
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps_processed = frame_count / elapsed_time
            print(f"进度: {frame_count}/{total_frames} | "
                    f"处理速度: {fps_processed:.1f} FPS")
        
        # 可选：显示实时结果
        if show_preview:
            cv2.imshow('YOLO+HRNet Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"处理完成!")
    print(f"总帧数: {frame_count}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均速度: {frame_count/total_time:.1f} FPS")
    print(f"输出保存到: {output_path}")

# 测试单张图片
def test_single_image(image_path, output_path=None, debug=True):
    """测试单张图片"""
    if not init_models():
        return
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 处理图像
    processed_img, keypoints_list = process_frame_with_yolo_hrnet(img)
    
    # 显示结果
    cv2.imshow('YOLO+HRNet Detection', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, processed_img)
        print(f"结果保存到: {output_path}")
    else:
        # 自动生成输出路径
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"{name}_detected{ext}"
        cv2.imwrite(output_path, processed_img)
        print(f"结果保存到: {output_path}")
    
    # 打印关键点信息
    if debug and keypoints_list:
        for i, person_kps in enumerate(keypoints_list):
            print(f"\n人物 {i+1} 的关键点:")
            for j, kp in enumerate(person_kps):
                if kp[2] > 0.1:  # 只显示置信度较高的点
                    print(f"  关键点 {j}: ({kp[0]:.1f}, {kp[1]:.1f}), 置信度: {kp[2]:.3f}")

if __name__ == "__main__":
    # 使用示例
    
    # 1. 测试单张图片
    # test_single_image("data\image.png", debug=True)
    
    # 2. 处理视频
    input_video = "video_origin/data_video/run_woman.mp4"
    output_video = "video_origin/result_video/yolo_hrnet_output_fixed.mp4"
    
    process_video(input_video, output_video, show_preview=True)