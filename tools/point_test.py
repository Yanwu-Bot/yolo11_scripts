import json
import cv2
import numpy as np
import os

def draw_mpii_keypoints_single(image_path, json_data=None, json_file_path=None):
    """
    在单张图片上显示MPII格式的关键点（只显示关键点，不显示骨架连接）
    
    MPII关键点索引顺序：
    0 - 头顶 (head top)
    1 - 脖子 (neck)
    2 - 右肩 (right shoulder)
    3 - 右肘 (right elbow)
    4 - 右腕 (right wrist)
    5 - 左肩 (left shoulder)
    6 - 左肘 (left elbow)
    7 - 左腕 (left wrist)
    8 - 右髋 (right hip)
    9 - 右膝 (right knee)
    10 - 右踝 (right ankle)
    11 - 左髋 (left hip)
    12 - 左膝 (left knee)
    13 - 左踝 (left ankle)
    14 - 胸部 (chest)
    15 - 骨盆 (pelvis)
    """
    
    # 读取JSON数据
    if json_data is None and json_file_path is not None:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
    
    if json_data is None:
        print("错误: 请提供JSON数据或JSON文件路径")
        return
    
    # 获取图片文件名
    image_filename = os.path.basename(image_path)
    
    # 查找匹配的关键点数据
    keypoint_data = None
    if isinstance(json_data, list):
        for item in json_data:
            if item.get('image') == image_filename:
                keypoint_data = item
                break
    elif isinstance(json_data, dict):
        keypoint_data = json_data
    
    if keypoint_data is None:
        print(f"错误: 未找到图片 {image_filename} 对应的关键点数据")
        return
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 - {image_path}")
        return
    
    # 放大显示
    scale_factor = 1.5
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # 获取关键点
    joints = keypoint_data.get('joints', [])
    joints_vis = keypoint_data.get('joints_vis', [])
    
    if len(joints) == 0:
        print(f"警告: {image_filename} 没有关键点数据")
        return
    
    # 为每个关键点指定颜色（BGR格式）
    point_color_map = [
        (0, 0, 255),    # 0 - 头顶 (红色)
        (0, 0, 255),    # 1 - 脖子 (红色)
        (255, 0, 0),    # 2 - 右肩 (蓝色)
        (255, 0, 0),    # 3 - 右肘 (蓝色)
        (255, 0, 0),    # 4 - 右腕 (蓝色)
        (255, 0, 0),    # 5 - 左肩 (蓝色)
        (255, 0, 0),    # 6 - 左肘 (蓝色)
        (255, 0, 0),    # 7 - 左腕 (蓝色)
        (0, 255, 255),  # 8 - 右髋 (黄色)
        (0, 255, 255),  # 9 - 右膝 (黄色)
        (0, 255, 255),  # 10 - 右踝 (黄色)
        (0, 255, 255),  # 11 - 左髋 (黄色)
        (0, 255, 255),  # 12 - 左膝 (黄色)
        (0, 255, 255),  # 13 - 左踝 (黄色)
        (0, 255, 0),    # 14 - 胸部 (绿色)
        (0, 255, 0)     # 15 - 骨盆 (绿色)
    ]
    
    # 绘制关键点
    for i, (x, y) in enumerate(joints):
        visible = (i < len(joints_vis) and joints_vis[i] == 1)
        
        if visible:
            # 关键点坐标按相同比例放大
            x_int = int(x * scale_factor)
            y_int = int(y * scale_factor)
            
            if 0 <= x_int < img.shape[1] and 0 <= y_int < img.shape[0]:
                color = point_color_map[i] if i < len(point_color_map) else (0, 255, 0)
                
                # 绘制关键点（小圆点）
                cv2.circle(img, (x_int, y_int), 3, color, -1)
                cv2.circle(img, (x_int, y_int), 3, (255, 255, 255), 1)
            else:
                print(f"警告: 关键点{i}坐标({x_int},{y_int})超出图片范围")
    
    # 添加信息文本
    scale = keypoint_data.get('scale', 1.0)
    center = keypoint_data.get('center', [0, 0])
    info_text = f"MPII Keypoints (Points only) | Scale: {scale:.2f} | Center: ({center[0]:.1f}, {center[1]:.1f})"
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 显示图片
    cv2.namedWindow(f'MPII Keypoints - {image_filename}', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'MPII Keypoints - {image_filename}', new_width, new_height)
    cv2.imshow(f'MPII Keypoints - {image_filename}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    # 指定文件路径
    json_file = "D:/Dataset/posedive1/posedive/annotations/pose_finediv_all.json"   # JSON文件路径
    image_file = "D:/Dataset/posedive1/posedive/images/finediv00063.jpg"  # 要显示的图片路径
    
    # 显示图片（只显示关键点）
    draw_mpii_keypoints_single(image_file, json_file_path=json_file)