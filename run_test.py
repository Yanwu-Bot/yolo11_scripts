import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

# ==================== 关键点提取函数 ====================
def extract_keypoints_from_image(image_path, yolo_model_path="weights/yolo11l-pose.pt", conf_threshold=0.5,read=True):
    """
    从单张图片中提取人体关键点
    """
    pose_model = YOLO(yolo_model_path)
    if read == True:
        img = image_path
    else:
        img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    h, w = img.shape[:2]
    results = pose_model(img, conf=conf_threshold)
    
    if len(results[0].keypoints) == 0:
        print("未检测到人体关键点")
        return None
    
    # 提取第一个人的关键点
    kpts = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
    keypoints_2d = []
    
    for i in range(17):
        x, y, conf = kpts[i]
        if conf >= conf_threshold:
            norm_x = x / w
            norm_y = y / h
            keypoints_2d.extend([norm_x, norm_y])
        else:
            keypoints_2d.extend([0, 0])
    
    return np.array(keypoints_2d), img, results[0].keypoints.data[0]

# ==================== 预处理函数 ====================
def prepare_for_resnet(keypoints, img_size=224):
    """
    将关键点转换为ResNet输入格式
    """
    kpts_reshaped = keypoints.reshape(17, 2)
    kpts_normalized = (kpts_reshaped - kpts_reshaped.min()) / \
                        (kpts_reshaped.max() - kpts_reshaped.min() + 1e-8)
    img_resized = cv2.resize(kpts_normalized, (img_size, img_size), 
                            interpolation=cv2.INTER_NEAREST)
    img_tensor = torch.FloatTensor(img_resized).unsqueeze(0).unsqueeze(0)
    return img_tensor

# ==================== 加载模型 ====================
def load_model(model_path="saved_models/resnet18_run_posture.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

# ==================== 绘制关键点 ====================
def draw_keypoints_on_image(img, keypoints_data, thickness=2):
    """
    在图片上绘制关键点和骨骼
    keypoints_data: (17, 3) tensor [x, y, confidence]
    """
    h, w = img.shape[:2]
    kpts = keypoints_data.cpu().numpy()
    
    # 骨骼连接线
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
    ]
    
    # 绘制骨骼
    for (i, j) in skeleton:
        x1, y1, conf1 = kpts[i]
        x2, y2, conf2 = kpts[j]
        if conf1 > 0.5 and conf2 > 0.5:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)
    
    # 绘制关键点
    for i in range(17):
        x, y, conf = kpts[i]
        if conf > 0.5:
            color = (0, 0, 255) if i < 5 else (255, 0, 0) if i < 11 else (0, 255, 0)
            cv2.circle(img, (int(x), int(y)), 5, color, -1)
    
    return img

# ==================== 主测试函数 ====================
def test_single_image(image_path, model_path="run_posture_model/resnet18_run_posture.pth", show_window=True):
    """
    测试单张图片：显示图片并在左上角标注预测结果
    """
    # 类别名称
    class_names = {0: "start", 1: "wave", 2: "final"}
    
    print(f"处理图片: {image_path}")
    
    # 1. 提取关键点
    result = extract_keypoints_from_image(image_path,read=False)
    if result is None:
        print("关键点提取失败")
        return None
    
    keypoints, original_img, raw_keypoints = result
    
    # 2. 预处理
    input_tensor = prepare_for_resnet(keypoints)
    
    # 3. 加载模型并预测
    model, device = load_model(model_path)
    
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # 4. 获取预测结果
    predicted_class = predicted.item()
    predicted_name = class_names[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    
    print(f"预测结果: {predicted_name} ({confidence:.2%})")
    
    # 5. 绘制结果到图片
    # 先绘制关键点
    img_with_kpts = draw_keypoints_on_image(original_img.copy(), raw_keypoints)
    
    # 在左上角添加预测结果
    text = f"{predicted_name}: {confidence:.1%}"
    
    # 设置字体和大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # 计算文本大小
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # 背景矩形
    bg_color = (0, 0, 0)  # 黑色背景
    text_color = (255, 255, 255)  # 白色文字
    
    # 根据不同类别使用不同颜色
    if predicted_class == 0:  # 起跑
        text_color = (0, 255, 0)  # 绿色
    elif predicted_class == 1:  # 摆动
        text_color = (255, 255, 0)  # 黄色
    else:  # 落地
        text_color = (0, 165, 255)  # 橙色
    
    # 在左上角绘制背景和文字
    padding = 10
    cv2.rectangle(img_with_kpts, 
                    (10, 10), 
                    (text_size[0] + 20, text_size[1] + 20), 
                    bg_color, -1)
    cv2.putText(img_with_kpts, text, (20, 40), font, font_scale, text_color, thickness)
    
    # 6. 显示图片
    if show_window:
        # 调整窗口大小
        window_name = f"跑步姿势识别: {predicted_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 获取屏幕大小并调整图片大小
        screen_height = 800
        h, w = img_with_kpts.shape[:2]
        scale = screen_height / h
        new_width = int(w * scale)
        new_height = int(h * scale)
        img_display = cv2.resize(img_with_kpts, (new_width, new_height))
        
        cv2.imshow(window_name, img_display)
        
        print("\n操作说明:")
        print("按 's' 键保存结果图片")
        print("按任意键关闭窗口")
        
        # 等待按键
        key = cv2.waitKey(0) & 0xFF
        
        # 如果按 's' 键，保存图片
        if key == ord('s'):
            save_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
            cv2.imwrite(save_path, img_with_kpts)
            print(f"结果已保存到: {save_path}")
        
        cv2.destroyAllWindows()
    
    return {
        'image': img_with_kpts,
        'predicted_class': predicted_class,
        'predicted_name': predicted_name,
        'confidence': confidence
    }

def process_single_image(image, model_path="run_posture_model/resnet18_run_posture.pth"):
    """
    测试单张图片：显示图片并在左上角标注预测结果
    """
    # 类别名称
    class_names = {0: "start", 1: "wave", 2: "final"}
    
    # 1. 提取关键点
    result = extract_keypoints_from_image(image)
    if result is None:
        print("关键点提取失败")
        return None
    
    keypoints, original_img, raw_keypoints = result
    
    # 2. 预处理
    input_tensor = prepare_for_resnet(keypoints)
    
    # 3. 加载模型并预测
    model, device = load_model(model_path)
    
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # 4. 获取预测结果
    predicted_class = predicted.item()
    predicted_name = class_names[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    
    # 在左上角添加预测结果
    # text = f"{predicted_name}: {confidence:.1%}"
    
    # # 设置字体和大小
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.8
    # thickness = 2
    
    # # 计算文本大小
    # text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # 背景矩形
    # bg_color = (0, 0, 0)  # 黑色背景
    # text_color = (255, 255, 255)  # 白色文字
    
    # 根据不同类别使用不同颜色
    if predicted_class == 0:  # 起跑
        text_color = (0, 255, 0)  # 绿色
    elif predicted_class == 1:  # 摆动
        text_color = (255, 0, 0)  # 蓝色
    else:  # 落地
        text_color = (0, 0, 255)  # 红色
    
    # 在左上角绘制背景和文字
    padding = 10
    cv2.circle(image,(10,10),10,text_color,-1)#画圈，圆心，半径，颜色，粗细, 
    return image



if __name__ == "__main__":
    # 方式1：测试单张图片
    print("=" * 50)
    print("跑步姿势识别测试")
    print("=" * 50)
    
    # 指定图片路径
    image_path = "data/jpg/run_final3.png"  # 修改为你的图片路径
    
    # 测试单张图片
    result = test_single_image(image_path)