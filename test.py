import cv2
import numpy as np
import math
import time
from ultralytics import YOLO

class PoseNormalizer:
    def __init__(self, model_path="./weights/yolo11n-pose.pt"):
        """初始化姿态归一化器"""
        self.model = YOLO(model_path)
        # 定义关节点索引
        self.dic_points = {
            0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
            5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
            9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
        }
    
    def get_keypoints(self, list_p):
        """从YOLO输出中提取关键点坐标"""
        p_pos = []
        for p in list_p[0]:
            x, y, conf = p[0], p[1], p[2]
            p_pos.append((int(x), int(y)))
        return p_pos
    
    def process_frame(self, img_path):
        """处理单张图片获取关键点"""
        results = self.model(img_path)
        for result in results:
            keypoints = result.keypoints
            for p in keypoints:
                list_p = p.data.tolist()
                return list_p
        return None
    
    def calculate_torso_info(self, list_p):
        """计算躯干信息：肩膀中点、髋中点、躯干长度"""
        p = self.get_keypoints(list_p)
        
        # 肩膀中点
        p5, p6 = p[5], p[6]  # 左右肩
        shoulder_center = ((p5[0] + p6[0]) / 2, (p5[1] + p6[1]) / 2)
        
        # 髋中点
        p11, p12 = p[11], p[12]  # 左右髋
        hip_center = ((p11[0] + p12[0]) / 2, (p11[1] + p12[1]) / 2)
        
        # 躯干长度
        torso_length = math.sqrt(
            (shoulder_center[0] - hip_center[0]) ** 2 + 
            (shoulder_center[1] - hip_center[1]) ** 2
        )
        
        return shoulder_center, hip_center, torso_length
    
    def normalize_keypoints(self, list_p, hip_center, torso_length):
        """归一化关键点坐标（以髋为中心，以躯干长度为单位）"""
        p = self.get_keypoints(list_p)
        normalized_points = []
        
        for point in p:
            # 中心化：减去髋中心
            centered_x = point[0] - hip_center[0]
            centered_y = point[1] - hip_center[1]
            
            # 归一化：除以躯干长度
            norm_x = centered_x / torso_length
            norm_y = centered_y / torso_length
            
            normalized_points.append((norm_x, norm_y))
        
        return normalized_points
    
    def draw_normalized_pose(self, img, normalized_points, scale=100):
        """在图像上绘制归一化后的姿态"""
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 定义骨架连接关系
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 身体和手臂
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # 腿部
        ]
        
        # 绘制骨架
        for start_idx, end_idx in skeleton:
            if start_idx < len(normalized_points) and end_idx < len(normalized_points):
                start = normalized_points[start_idx]
                end = normalized_points[end_idx]
                
                # 将归一化坐标转换为图像坐标
                start_x = int(center_x + start[0] * scale)
                start_y = int(center_y + start[1] * scale)
                end_x = int(center_x + end[0] * scale)
                end_y = int(center_y + end[1] * scale)
                
                cv2.line(img, (start_x, start_y), (end_x, end_y), 
                        (0, 255, 0), 2)
        
        # 绘制关节点
        for i, point in enumerate(normalized_points):
            x = int(center_x + point[0] * scale)
            y = int(center_y + point[1] * scale)
            cv2.circle(img, (x, y), 5, (0, 150, 255), -1)
            
            # 显示关节点编号（可选）
            # cv2.putText(img, str(i), (x+5, y-5), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def resize_to_same_height(self, img1, img2, target_height=300):
        """将两张图片调整到相同高度"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 使用较小的目标高度
        target_h = min(target_height, h1, h2)
        
        # 计算新的宽度（保持宽高比）
        scale1 = target_h / h1
        new_w1 = int(w1 * scale1)
        
        scale2 = target_h / h2
        new_w2 = int(w2 * scale2)
        
        # 调整图片大小
        img1_resized = cv2.resize(img1, (new_w1, target_h), interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, (new_w2, target_h), interpolation=cv2.INTER_AREA)
        
        return img1_resized, img2_resized
    
    def create_comparison_image(self, img1_resized, img2_resized, norm_img1, norm_img2):
        """创建对比图像，处理不同尺寸的问题"""
        # 获取各个图像的高度
        h1, w1 = img1_resized.shape[:2]
        h2, w2 = img2_resized.shape[:2]
        h_norm = norm_img1.shape[0]  # 归一化图像高度相同
        
        # 找出最大的宽度
        max_width = max(w1, w2)
        
        # 创建统一宽度的图像（用黑色填充）
        def pad_image_to_width(img, target_width):
            h, w = img.shape[:2]
            if w < target_width:
                # 计算左右填充
                left_pad = (target_width - w) // 2
                right_pad = target_width - w - left_pad
                img_padded = cv2.copyMakeBorder(img, 0, 0, left_pad, right_pad, 
                                               cv2.BORDER_CONSTANT, value=[0, 0, 0])
                return img_padded
            return img
        
        # 统一第一行宽度
        img1_padded = pad_image_to_width(img1_resized, max_width)
        img2_padded = pad_image_to_width(img2_resized, max_width)
        
        # 合并第一行
        row1 = np.hstack([img1_padded, img2_padded])
        
        # 统一第二行宽度（归一化图像）
        norm_img1_padded = pad_image_to_width(norm_img1, max_width)
        norm_img2_padded = pad_image_to_width(norm_img2, max_width)
        row2 = np.hstack([norm_img1_padded, norm_img2_padded])
        
        # 合并两行
        comparison_img = np.vstack([row1, row2])
        
        return comparison_img
    
    def compare_two_images(self, img1_path, img2_path):
        """对比两张图片的归一化结果"""
        start_time = time.time()
        
        print("=" * 50)
        print("开始姿态归一化对比")
        print("=" * 50)
        
        # 处理第一张图片
        print(f"处理图片1: {img1_path}")
        list_p1 = self.process_frame(img1_path)
        if list_p1 is None:
            print("❌ 图片1未检测到人物姿态")
            return None
            
        shoulder1, hip1, torso1 = self.calculate_torso_info(list_p1)
        norm_points1 = self.normalize_keypoints(list_p1, hip1, torso1)
        print(f"  躯干长度: {torso1:.1f}px")
        
        # 处理第二张图片
        print(f"处理图片2: {img2_path}")
        list_p2 = self.process_frame(img2_path)
        if list_p2 is None:
            print("❌ 图片2未检测到人物姿态")
            return None
            
        shoulder2, hip2, torso2 = self.calculate_torso_info(list_p2)
        norm_points2 = self.normalize_keypoints(list_p2, hip2, torso2)
        print(f"  躯干长度: {torso2:.1f}px")
        
        # 读取原始图片
        img1_original = cv2.imread(img1_path)
        img2_original = cv2.imread(img2_path)
        
        # 调整原始图片到相同高度
        img1_resized, img2_resized = self.resize_to_same_height(img1_original, img2_original, target_height=300)
        
        # 创建归一化显示图像（黑色背景）
        norm_img_size = 300  # 归一化图像大小
        norm_img1 = np.zeros((norm_img_size, norm_img_size, 3), dtype=np.uint8)
        norm_img2 = np.zeros((norm_img_size, norm_img_size, 3), dtype=np.uint8)
        
        # 在归一化图像上绘制姿态
        self.draw_normalized_pose(norm_img1, norm_points1, scale=80)
        self.draw_normalized_pose(norm_img2, norm_points2, scale=80)
        
        # 在归一化图像上添加标题
        cv2.putText(norm_img1, "归一化姿态1", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(norm_img1, f"躯干: {torso1:.0f}px", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(norm_img2, "归一化姿态2", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(norm_img2, f"躯干: {torso2:.0f}px", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 创建对比图像
        comparison_img = self.create_comparison_image(img1_resized, img2_resized, norm_img1, norm_img2)
        
        # 添加总体标题
        cv2.putText(comparison_img, "姿态归一化对比", 
                   (10, comparison_img.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison_img, "(髋中心为原点，躯干长度为单位)", 
                   (10, comparison_img.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 显示结果
        cv2.imshow("姿态归一化对比", comparison_img)
        
        # 计算耗时
        end_time = time.time()
        print(f"\n处理耗时: {end_time - start_time:.2f}秒")
        
        # 保存结果
        cv2.imwrite("normalization_comparison.jpg", comparison_img)
        print("对比结果已保存为: normalization_comparison.jpg")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 打印归一化数据对比
        print("\n归一化坐标对比 (单位: 躯干长度):")
        print(f"{'关节':<15} {'图像1':<20} {'图像2':<20} {'差异':<10}")
        print("-" * 70)
        
        key_joints = [5, 6, 11, 12, 13, 14, 15, 16]  # 肩膀、髋、膝、踝
        for idx in key_joints:
            if idx < len(norm_points1) and idx < len(norm_points2):
                p1 = norm_points1[idx]
                p2 = norm_points2[idx]
                distance = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                joint_name = self.dic_points.get(idx, f"关节{idx}")
                print(f"{joint_name:<15} ({p1[0]:.3f}, {p1[1]:.3f})   ({p2[0]:.3f}, {p2[1]:.3f})   {distance:.3f}")
        
        # 返回归一化数据供进一步分析
        return {
            'image1': {
                'original_torso': torso1,
                'normalized_points': norm_points1,
                'hip_center': hip1
            },
            'image2': {
                'original_torso': torso2,
                'normalized_points': norm_points2,
                'hip_center': hip2
            }
        }


# 使用示例
if __name__ == "__main__":
    # 创建归一化器
    normalizer = PoseNormalizer("./weights/yolo11n-pose.pt")
    
    # 对比两张图片（替换为你的实际图片路径）
    img1_path = "data/jpg/run_final3.png"
    img2_path = "data/jpg/run_start4.png"
    
    try:
        results = normalizer.compare_two_images(img1_path, img2_path)
        
        if results:
            print("\n✅ 对比完成！")
            print(f"图像1躯干长度: {results['image1']['original_torso']:.1f}px")
            print(f"图像2躯干长度: {results['image2']['original_torso']:.1f}px")
            print(f"躯干长度比例: {results['image1']['original_torso']/results['image2']['original_torso']:.2f}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查：")
        print("1. 图片路径是否正确")
        print("2. 图片中是否有人物")
        print("3. YOLO模型文件是否存在")
        import traceback
        traceback.print_exc()