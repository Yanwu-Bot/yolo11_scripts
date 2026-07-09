import json
import cv2
import numpy as np
import os

def draw_mpii_keypoints_single(image_path, json_data=None, json_file_path=None):
    """
    在单张图片上显示MPII格式的关键点（显示关键点并标注序号）
    
    MPII关键点索引顺序（自定义顺序）：
    0 - 右脚踝 (right ankle)
    1 - 右膝盖 (right knee)
    2 - 右髋部 (right hip)
    3 - 左髋部 (left hip)
    4 - 左膝盖 (left knee)
    5 - 左脚踝 (left ankle)
    6 - 骨盆 (pelvis)
    7 - 胸部 (chest)
    8 - 上颈部 (upper neck)
    9 - 头顶 (head top)
    10 - 右手腕 (right wrist)
    11 - 右肘部 (right elbow)
    12 - 右肩部 (right shoulder)
    13 - 左肩部 (left shoulder)
    14 - 左肘部 (left elbow)
    15 - 左手腕 (left wrist)
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
    # 按照新的顺序分配颜色
    point_color_map = [
        (0, 255, 255),  # 0 - 右脚踝 (黄色)
        (0, 255, 255),  # 1 - 右膝盖 (黄色)
        (0, 255, 255),  # 2 - 右髋部 (黄色)
        (0, 255, 255),  # 3 - 左髋部 (黄色)
        (0, 255, 255),  # 4 - 左膝盖 (黄色)
        (0, 255, 255),  # 5 - 左脚踝 (黄色)
        (0, 255, 0),    # 6 - 骨盆 (绿色)
        (0, 255, 0),    # 7 - 胸部 (绿色)
        (255, 0, 0),    # 8 - 上颈部 (蓝色)
        (0, 0, 255),    # 9 - 头顶 (红色)
        (255, 0, 0),    # 10 - 右手腕 (蓝色)
        (255, 0, 0),    # 11 - 右肘部 (蓝色)
        (255, 0, 0),    # 12 - 右肩部 (蓝色)
        (255, 0, 0),    # 13 - 左肩部 (蓝色)
        (255, 0, 0),    # 14 - 左肘部 (蓝色)
        (255, 0, 0)     # 15 - 左手腕 (蓝色)
    ]
    
    # 关键点名称（按新顺序）
    point_names = [
        "0:R.Ankle", "1:R.Knee", "2:R.Hip", "3:L.Hip", 
        "4:L.Knee", "5:L.Ankle", "6:Pelvis", "7:Chest",
        "8:U.Neck", "9:Head", "10:R.Wrist", "11:R.Elbow",
        "12:R.Shoulder", "13:L.Shoulder", "14:L.Elbow", "15:L.Wrist"
    ]
    
    # 收集所有可见关键点的坐标和索引
    visible_points = []
    for i, (x, y) in enumerate(joints):
        visible = (i < len(joints_vis) and joints_vis[i] == 1)
        if visible:
            x_int = int(x * scale_factor)
            y_int = int(y * scale_factor)
            if 0 <= x_int < img.shape[1] and 0 <= y_int < img.shape[0]:
                visible_points.append((i, x_int, y_int))
    
    # 绘制关键点和序号（使用多种策略避免重叠）
    for idx, (i, x_int, y_int) in enumerate(visible_points):
        color = point_color_map[i] if i < len(point_color_map) else (0, 255, 0)
        
        # 绘制关键点（大一点以便显示序号）
        cv2.circle(img, (x_int, y_int), 5, color, -1)
        cv2.circle(img, (x_int, y_int), 5, (255, 255, 255), 1)
        
        # 策略1: 计算序号位置 - 使用放射状布局
        angle = (idx * 60) % 360  # 每个点使用不同角度
        radius = 20  # 固定半径
        
        # 根据关键点位置调整角度，避免序号超出图片边界
        if x_int < 30:
            angle = 0  # 在左边就向右
        elif x_int > img.shape[1] - 30:
            angle = 180  # 在右边就向左
        if y_int < 30:
            angle = 90  # 在上边就向下
        elif y_int > img.shape[0] - 30:
            angle = 270  # 在下边就向上
        
        # 计算文本位置（在圆点周围）
        angle_rad = np.radians(angle)
        offset_x = int(radius * np.cos(angle_rad))
        offset_y = int(radius * np.sin(angle_rad))
        
        # 如果与其他点距离太近，使用更大的半径
        for other_i, other_x, other_y in visible_points:
            if other_i != i:
                distance = np.sqrt((x_int - other_x)**2 + (y_int - other_y)**2)
                if distance < 30:  # 如果两个点很近
                    # 增加偏移量并调整角度
                    offset_x = int(radius * 1.5 * np.cos(angle_rad + np.pi/2))
                    offset_y = int(radius * 1.5 * np.sin(angle_rad + np.pi/2))
                    break
        
        text_x = x_int + offset_x
        text_y = y_int + offset_y
        
        # 绘制序号（带背景框）
        label = str(i)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # 背景框位置调整，使序号居中
        box_x = text_x - text_w // 2 - 4
        box_y = text_y - text_h // 2 - 4
        box_w = text_w + 8
        box_h = text_h + 8
        
        # 绘制半透明背景框
        overlay = img.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # 绘制边框
        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 1)
        
        # 绘制序号文本（使用与关键点相同的颜色）
        text_pos = (text_x - text_w // 2, text_y + text_h // 2)
        cv2.putText(img, label, text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制连接线（从关键点到序号）
        cv2.line(img, (x_int, y_int), (text_x, text_y - text_h // 2), color, 1, cv2.LINE_AA)
    
    # 添加信息文本
    scale = keypoint_data.get('scale', 1.0)
    center = keypoint_data.get('center', [0, 0])
    info_text = f"MPII Keypoints with Index | Scale: {scale:.2f} | Center: ({center[0]:.1f}, {center[1]:.1f})"
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 添加图例说明（显示关键点名称和序号对应关系）
    legend_y = 60
    legend_x = 10
    
    # 分两列显示图例
    for idx, name in enumerate(point_names):
        row = idx % 8
        col = idx // 8
        y_pos = legend_y + 20 + row * 20
        x_pos = legend_x + col * 170
        if y_pos < img.shape[0] - 20:
            color = point_color_map[idx] if idx < len(point_color_map) else (0, 255, 0)
            # 显示序号和名称
            label_text = name
            cv2.putText(img, label_text, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
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
    image_file = "D:/Dataset/posedive1/posedive/images/finediv00069.jpg"  # 要显示的图片路径
    
    # 显示图片（显示关键点和序号）
    draw_mpii_keypoints_single(image_file, json_file_path=json_file)