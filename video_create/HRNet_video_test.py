import math
import os
import json
import torch
import numpy as np
import cv2
import sys
import time
from collections import deque
from HRNet_model import HighResolutionNet
import transforms
from utill import *
from time_utils import show_time
from ultralytics import YOLO
from Feature import *
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'  # 设置 matplotlib 中文字体

class VideoProcessor:
    # ==================== 可配置参数 ====================
    OUTPUT_VIDEO_SIZE = (1280, 720)          # 输出视频宽度、高度
    VIDEO_FRAME_SPEED = 24                   # 视频帧率
    SMOOTH_ALPHA = 0.7                       # 人物中心平滑指数（越大越跟随原始）
    YOLO_CONF_THRESHOLD = 0.7                # YOLO 检测人物置信度阈值
    NORMALIZE_TORSO_LENGTH = 100             # 归一化时躯干目标长度（像素）
    STEP_MIN_GAP = 0.2                       # 步频检测最小时间间隔（秒）

    def __init__(self, input_path):
        self.input_path = input_path                     # 输入视频路径
        self.video_name = os.path.splitext(
            os.path.basename(input_path))[0]            # 视频文件名（不含扩展名）
        self.output_dir = "result"                       # 输出根目录
        self.trajectory_tracker = KeypointTrajectoryTracker(
            num_keypoints=17, history_length=200,
            output_dir=os.path.join(self.output_dir, "track_img")
        )                                                # 轨迹跟踪器（保存最近200帧）

        # 模型相关（延迟初始化）
        self.device = None                               # 计算设备
        self.hrnet_model = None                          # HRNet 姿态估计模型
        self.yolo_model = None                           # YOLO 目标检测模型
        self.person_info = None                          # 关键点 json 信息（含flip_pairs）
        self.hrnet_transform = None                      # HRNet 图像预处理流水线

        # 帧处理状态
        self.smooth_center = None                        # 平滑后的人物中心坐标 (x,y)
        self.keypoint_prev = None                        # 上一帧的原始关键点列表
        self.normalized_prev = None                      # 上一帧的归一化关键点列表
        self.max_acc = []                                # 每帧的最大加速度（用于异常检测）
        self.vector_list = []                            # 每帧的位移向量 (帧数×17×2)
        self.step_state = 0                              # 步态状态：0初始，1左脚抬，2右脚抬
        self.step_count = 0                              # 累计步数
        self.step_freq = 0                               # 当前步频（Hz）
        self.last_step_time = 0.0                        # 最后一步发生的时间（秒）
        self.current_frame = 0                           # 已处理帧计数（从1开始）
        self.all_features = []                           # 存储所有帧的原始特征
        self.all_points = []                             # 存储所有帧的原始关键点 (x,y)
        self.all_normalized_points = []                  # 存储所有帧的归一化关键点

    @staticmethod
    def _progress_bar(current, total, bar_length=30, prefix="进度"):
        """显示自定义进度条，使用实心/空心方块字符"""
        percent = current / total
        filled_length = int(bar_length * percent)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        sys.stdout.write(f'\r{prefix}: |{bar}| {percent*100:.1f}% ({current}/{total})')
        sys.stdout.flush()
        if current == total:
            print()  # 完成时换行

    def init_models(self):
        """初始化 YOLO 和 HRNet 模型（只执行一次）"""
        if self.device is not None:  # 已经初始化过
            return True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            # 1. 加载 YOLO 模型
            self.yolo_model = YOLO('weights/yolo11n.pt')
            
            # 2. 加载 HRNet 模型
            self.hrnet_model = HighResolutionNet(base_channel=32)
            weights_path = "HRnet/pytorch/pose_coco/pose_hrnet_w32_384x288.pth"
            keypoint_json_path = "HRnet/person_keypoints.json"
            if not os.path.exists(weights_path) or not os.path.exists(keypoint_json_path):
                raise FileNotFoundError("HRNet weights or keypoint json not found")
            with open(keypoint_json_path, "r") as f:
                self.person_info = json.load(f)
            weights = torch.load(weights_path, map_location=self.device)
            weights = weights if "model" not in weights else weights["model"]
            self.hrnet_model.load_state_dict(weights)
            self.hrnet_model.to(self.device)
            self.hrnet_model.eval()
            
            # 3. 初始化 HRNet 图像预处理（输入尺寸 384x288）
            resize_hw = (288, 384)  # (H, W) 对应高度288，宽度384
            self.hrnet_transform = transforms.Compose([
                transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Model initialization failed: {e}")
            return False
        return True

    def detect_person(self, frame):
        """YOLO 检测人物，返回面积最大的人物边界框及置信度"""
        if self.yolo_model is None:
            return None, 0
        results = self.yolo_model(frame, verbose=False)
        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # YOLO 中 person 类别 ID = 0
                if int(box.cls[0]) != 0 or float(box.conf[0]) < self.YOLO_CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)  # 面积排序取最大
                persons.append(([float(x1), float(y1), float(x2), float(y2)], float(box.conf[0]), area))
        if not persons:
            return None, 0
        persons.sort(key=lambda x: x[2], reverse=True)  # 按面积降序
        return persons[0][0], persons[0][1]  # 返回最大人物的框和置信度

    def normalize_keypoints(self, p_pos):
        """
        关键点归一化：以髋部中心为原点，躯干长度缩放到指定数值
        返回：归一化关键点列表、尺度信息字典
        """
        if len(p_pos) < 17:
            return None, None
        # 左右肩中点
        sc = (p_pos[5][0] + p_pos[6][0]) / 2.0, (p_pos[5][1] + p_pos[6][1]) / 2.0
        # 左右髋中点
        hc = (p_pos[11][0] + p_pos[12][0]) / 2.0, (p_pos[11][1] + p_pos[12][1]) / 2.0
        torso_len = math.hypot(sc[0] - hc[0], sc[1] - hc[1])  # 躯干实际长度
        if torso_len < 1e-6:
            return None, None
        scale = self.NORMALIZE_TORSO_LENGTH / torso_len
        norm = [[(p[0] - hc[0]) * scale, (p[1] - hc[1]) * scale] for p in p_pos[:17]]
        return norm, {'scale': scale, 'torso_length': torso_len, 'center': hc}

    def predict_frame(self, frame):
        """
        单帧预测流程：
        1. YOLO 检测并平滑人物中心
        2. 调整 ROI（添加动态padding）
        3. HRNet 在 ROI 上预测关键点
        4. 坐标转换回原图
        返回：关键点列表 [ [x,y,score], ... ] 和置信度数组
        """
        if self.yolo_model is None or self.hrnet_model is None:
            if not self.init_models():
                return [], []
        bbox, conf = self.detect_person(frame)
        if bbox is None:
            return [], []

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        #中心点坐标
        current_center = ((x1 + x2) / 2, (y1 + y2) / 2)

        if self.smooth_center is None:
            self.smooth_center = current_center
        #用于平滑中心点，新平滑中心 = α × 当前真实中心 + (1 - α) × 旧平滑中心
        else:
            sx = self.SMOOTH_ALPHA * current_center[0] + (1 - self.SMOOTH_ALPHA) * self.smooth_center[0]
            sy = self.SMOOTH_ALPHA * current_center[1] + (1 - self.SMOOTH_ALPHA) * self.smooth_center[1]
            self.smooth_center = (sx, sy)

        # 根据平滑中心重新计算检测框（保持原框宽高）
        bbox_w, bbox_h = x2 - x1, y2 - y1
        x1 = max(0, min(self.smooth_center[0] - bbox_w / 2, w - 1))
        y1 = max(0, min(self.smooth_center[1] - bbox_h / 2, h - 1))
        x2 = max(0, min(x1 + bbox_w, w - 1))
        y2 = max(0, min(y1 + bbox_h, h - 1))

        # 动态 padding（人物大小的 15%，限制在 10~50 像素）
        padding = int(min(bbox_w, bbox_h) * 0.15)
        padding = max(10, min(padding, 50))
        roi_x1 = max(0, int(x1) - padding)
        roi_y1 = max(0, int(y1) - padding)
        roi_x2 = min(w, int(x2) + padding)
        roi_y2 = min(h, int(y2) + padding)

        # 绘制 YOLO 检测框
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Person: {conf:.2f}", (int(x1), max(20, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 裁剪 ROI
        person_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if person_roi.size == 0:
            return [], []

        # 转为 RGB 格式（HRNet 需要）
        if len(person_roi.shape) == 2:
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_GRAY2RGB)
        else:
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        roi_h, roi_w = person_rgb.shape[:2]

        # HRNet 前向推理
        img_tensor, target = self.hrnet_transform(
            person_rgb, {"box": [0, 0, roi_w - 1, roi_h - 1]}
        )
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        with torch.no_grad():
            outputs = self.hrnet_model(img_tensor.to(self.device))
            # 水平翻转测试（增强精度）
            flip_tensor = transforms.flip_images(img_tensor)
            flip_out = torch.squeeze(
                transforms.flip_back(self.hrnet_model(flip_tensor.to(self.device)),
                                     self.person_info["flip_pairs"]),
            )
            flip_out[..., 1:] = flip_out.clone()[..., 0:-1]  # 翻转后坐标修正？
            outputs = (outputs + flip_out) * 0.5

            keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
            keypoints = np.squeeze(keypoints)
            scores = np.squeeze(scores)
            if scores.ndim == 0:
                scores = np.array([scores])

            # 将 ROI 内坐标转换回原图坐标
            keypoints_list = []
            for kp, sc in zip(keypoints, scores):
                if hasattr(kp, '__iter__') and len(kp) >= 2:
                    ox = max(0, min(roi_x1 + float(kp[0]), w - 1))
                    oy = max(0, min(roi_y1 + float(kp[1]), h - 1))
                    keypoints_list.append([int(ox), int(oy), float(sc)])
                else:
                    keypoints_list.append([0, 0, 0.0])  # 无效点
        return [keypoints_list], scores

    def compute_step(self, p_pos, time_gap):
        """
        步频检测：比较两脚踝 y 坐标之差，结合时间间隔触发计步
        """
        if len(p_pos) < 17:
            return
        try:
            left_ankle_y = p_pos[15][1]   # 左踝
            right_ankle_y = p_pos[16][1]  # 右踝
        except (TypeError, IndexError):
            return
        delta = left_ankle_y - right_ankle_y  # 正 = 左脚高于右脚
        if abs(delta) > 10:  # 高度差阈值（像素）
            current_time = self.current_frame * time_gap
            if current_time - self.last_step_time >= self.STEP_MIN_GAP:
                new_state = 1 if delta > 0 else 2
                if new_state != self.step_state and self.step_state != 0:
                    self.step_count += 1
                    if self.step_count > 1:
                        gap = current_time - self.last_step_time
                        if gap > 0.1:
                            self.step_freq = round(1 / gap, 3)
                    self.last_step_time = current_time
                self.step_state = new_state

    def process_frame(self, frame, preview=True, normalize_for_storage=True):
        """
        处理单帧：预测关键点、校验、特征提取、更新状态、可视化
        返回：可视化帧、关键点、特征、归一化点、尺度信息
        """
        list_p, scores = self.predict_frame(frame)
        p_pos = get_keypoints(list_p)

        # 强校验：确保 p_pos 是 [[x,y], ...] 格式，长度>=17
        if p_pos and len(p_pos) >= 17:
            validated = []
            for pt in p_pos:
                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                    validated.append([float(pt[0]), float(pt[1])])
                else:
                    validated = []  # 遇到格式错误直接清空
                    break
            if len(validated) >= 17:
                p_pos = validated
            else:
                p_pos = []  # 视为无人帧

        if not p_pos or len(p_pos) < 17:
            # 无人帧：重置平滑状态，返回空白可视化
            self.smooth_center = None
            self.keypoint_prev = None
            self.normalized_prev = None
            vis = self._create_vis_frame(frame, None)
            if preview:
                cv2.imshow('YOLO Detection', vis)
                cv2.waitKey(1)
            return vis, [], [0]*50, None, None

        # 1. 归一化（用于存储和后继特征）
        norm_data = None
        scale_info = None
        if normalize_for_storage:
            norm_data, scale_info = self.normalize_keypoints(p_pos)

        # 2. 轨迹更新
        self.trajectory_tracker.update(p_pos)
        feature = Feature(p_pos)

        # 3. 向量计算（归一化关键点的帧间位移）
        if self.normalized_prev is None or norm_data is None:
            vec = np.zeros((17, 2), dtype=np.float32)
        else:
            vec = np.array(norm_data, dtype=np.float32) - np.array(self.normalized_prev, dtype=np.float32)
        self.vector_list.append(vec)
        #当前帧数据放入self.normalized_prev
        self.normalized_prev = norm_data

        # 4. 加速度计算（基于原始坐标）
        if self.keypoint_prev is None:
            acc_per_kp = []
        else:
            acc_per_kp = [acceleration(p_pos[j], self.keypoint_prev[j],
                                       1/self.VIDEO_FRAME_SPEED) for j in range(17)]
        self.keypoint_prev = [p.copy() for p in p_pos]
        if acc_per_kp:
            self.max_acc.append(max(acc_per_kp) / 1000.0)  # 单位转换为千分之一

        # 5. 步频检测
        self.compute_step(p_pos, 1/self.VIDEO_FRAME_SPEED)

        # 6. 可视化绘制
        self._draw_info(frame, p_pos)
        vis = self._create_vis_frame(frame, p_pos)
        if preview:
            window_title = f"Steps: {self.step_count}  Freq: {self.step_freq}Hz"
            cv2.setWindowTitle('YOLO Detection', window_title)
            cv2.imshow('YOLO Detection', vis)
            cv2.waitKey(1)

        # 7. 存储数据
        self.current_frame += 1
        feature_frame = feature.get_all_features() if len(p_pos) >= 17 else [0]*50
        self.all_features.append(feature_frame)
        self.all_points.append(p_pos)
        if norm_data is not None:
            self.all_normalized_points.append(np.array(norm_data, dtype=np.float32))
        else:
            self.all_normalized_points.append(np.zeros((17,2), dtype=np.float32))

        return vis, p_pos, feature_frame, norm_data, scale_info

    def _draw_info(self, frame, p_pos):
        """在帧上叠加步频、步数信息和关键点骨架"""
        try:
            draw_points = []
            for p in p_pos:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    draw_points.append([p[0], p[1], 1.0])
            if len(draw_points) >= 17:
                draw = Draw(frame, draw_points)
                draw.draw_select()
        except Exception as e:
            pass  # 绘制失败不影响主流程

    def _create_vis_frame(self, frame, p_pos):
        """
        创建输出可视化帧：以平滑中心裁剪固定尺寸（1280x720），未覆盖区域为黑色
        """
        out_w, out_h = self.OUTPUT_VIDEO_SIZE
        vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if self.smooth_center is not None:
            cx, cy = self.smooth_center
            src_x1 = max(0, int(cx - out_w / 2))
            src_y1 = max(0, int(cy - out_h / 2))
            src_x2 = min(frame.shape[1], src_x1 + out_w)
            src_y2 = min(frame.shape[0], src_y1 + out_h)
            dst_x1 = max(0, -int(cx - out_w / 2)) if cx - out_w/2 < 0 else 0
            dst_y1 = max(0, -int(cy - out_h / 2)) if cy - out_h/2 < 0 else 0
            dst_w = src_x2 - src_x1
            dst_h = src_y2 - src_y1
            if dst_w > 0 and dst_h > 0:
                vis[dst_y1:dst_y1+dst_h, dst_x1:dst_x1+dst_w] = frame[src_y1:src_y2, src_x1:src_x2]
        return vis

    def generate_video(self):
        """读取视频、逐帧处理、保存输出视频及特征文件"""
        cap = cv2.VideoCapture(self.input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
        if total_frames <= 0:
            print("Cannot get frame count")
            cap.release()
            return

        out_w, out_h = self.OUTPUT_VIDEO_SIZE
        os.makedirs(os.path.join(self.output_dir, "result_video", "video"), exist_ok=True)
        output_path = os.path.join(self.output_dir, "result_video", "video",
                                   f"yolo_hrnet-{self.video_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.VIDEO_FRAME_SPEED, (out_w, out_h))
        if not out.isOpened():
            print("VideoWriter failed")
            cap.release()
            return

        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_idx += 1
            try:
                processed, _, _, _, _ = self.process_frame(frame, preview=True)
                out.write(processed)
                self._progress_bar(frame_idx, total_frames)
            except Exception as e:
                print(f"\nError at frame {frame_idx}: {str(e)[:60]}")
                out.write(np.zeros((out_h, out_w, 3), dtype=np.uint8))  # 失败帧用黑色代替
                continue

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("\nVideo saved to:", output_path)

        # 保存特征文件
        os.makedirs(os.path.join(self.output_dir, "features"), exist_ok=True)
        if self.all_normalized_points:
            np.save(f'{self.output_dir}/features/{self.video_name}_normalized_points.npy',
                    np.stack(self.all_normalized_points, axis=0))
        np.save(f'{self.output_dir}/features/{self.video_name}_features.npy', np.array(self.all_features))
        np.save(f'{self.output_dir}/features/{self.video_name}_points.npy', np.array(self.all_points))
        if len(self.vector_list) > 0:
            vec_arr = np.stack(self.vector_list, axis=0)          # (帧数, 17, 2)
            vec_min = vec_arr.min(axis=(0,1), keepdims=True)
            vec_max = vec_arr.max(axis=(0,1), keepdims=True)
            vec_norm = (vec_arr - vec_min) / (vec_max - vec_min + 1e-8)  # Min-Max归一化
            np.save(f'{self.output_dir}/features/{self.video_name}_vector.npy', vec_norm)

        # 异常帧检测（基于加速度的DBSCAN聚类）
        if len(self.max_acc) > 0:
            eps = auto_eps(self.max_acc, 8)  # 自动估计eps参数
            print(f"Auto eps: {eps}")
            frames = list(range(2, self.current_frame + 1)) if self.current_frame > 1 else [1]
            point_acceleration(frames, self.max_acc, self.video_name,
                                use_dbscan=True, eps=eps, min_samples=8)

if __name__ == '__main__':
    input_path = 'video_origin/data_video/use/run_woman.mp4'
    processor = VideoProcessor(input_path)
    start = time.time()
    processor.generate_video()
    elapsed = show_time(start, time.time())
    print(f"Total time: {elapsed}")