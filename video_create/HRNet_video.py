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
rcParams['font.family'] = 'SimHei'
class VideoProcessor:
    VIDEO_FRAME_SPEED = 24
    YOLO_CONF_THRESHOLD = 0.6
    NORMALIZE_TORSO_LENGTH = 100
    STEP_MIN_GAP = 0.2
    OUTPUT_VIDEO_SIZE = (1280, 720)

    def __init__(self, input_path):
        self.input_path = input_path
        self.video_name = os.path.splitext(os.path.basename(input_path))[0]
        self.output_dir = "result"
        self.trajectory_tracker = KeypointTrajectoryTracker(
            num_keypoints=17, history_length=200,
            output_dir=os.path.join(self.output_dir, "track_img")
        )

        self.device = None
        self.hrnet_model = None
        self.yolo_model = None
        self.person_info = None
        self.hrnet_transform = None

        # 仅存储原始关键点和归一化关键点
        self.all_points = []               # 每帧原始关键点 (17,2)
        self.all_normalized_points = []    # 每帧归一化关键点 (17,2)
        self.all_scale_info = []           # 每帧缩放信息
        self.last_roi = None

        # 后处理时会填充的列表（初始为空）
        self.vector_list = []
        self.max_acc = []
        self.all_features = []

        # 步频状态
        self.step_state = 0
        self.step_count = 0
        self.step_freq = 0.0
        self.last_step_time = 0.0

        self.padding = 30 #扩展大小

    def init_models(self):
        if self.device is not None:
            return True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self.yolo_model = YOLO('weights/yolo11s.pt')
            self.hrnet_model = HighResolutionNet(base_channel=32)
            weights_path = "HRnet/pytorch/pose_coco/pose_hrnet_w32_256x192.pth"
            keypoint_json_path = "HRnet/person_keypoints.json"
            if not os.path.exists(weights_path) or not os.path.exists(keypoint_json_path):
                raise FileNotFoundError("HRNet weights / keypoint json not found")
            with open(keypoint_json_path, "r") as f:
                self.person_info = json.load(f)
            weights = torch.load(weights_path, map_location=self.device)
            weights = weights if "model" not in weights else weights["model"]
            self.hrnet_model.load_state_dict(weights)
            self.hrnet_model.to(self.device)
            self.hrnet_model.eval()
            resize_hw = (256, 192)
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
        if self.yolo_model is None:
            return None, 0
        results = self.yolo_model(frame, verbose=False)
        persons = []
        img_area = frame.shape[0] * frame.shape[1]  # 图像总像素数
        min_area = img_area * 0.003  # 最小面积阈值（图像面积的0.3%，可根据需要调整）

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id != 0 or conf < self.YOLO_CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)

                # 面积过滤：小于最小面积的直接跳过
                if area < min_area:
                    continue

                # 综合评分：置信度 × 面积（可根据需要调整权重）
                score = conf * area
                persons.append(([float(x1), float(y1), float(x2), float(y2)], conf, area, score))
        if not persons:
            return None, 0
        # 按综合评分降序排序，选择评分最高的
        persons.sort(key=lambda x: x[3], reverse=True)  # x[3] 是 score
        return persons[0][0], persons[0][1]  # 返回 (bbox, conf)

    def normalize_keypoints(self, p_pos):
        if len(p_pos) < 17:
            return None, 0, 0, (0, 0)
        #取中心点
        sc = (p_pos[5][0] + p_pos[6][0]) / 2.0, (p_pos[5][1] + p_pos[6][1]) / 2.0
        hc = (p_pos[11][0] + p_pos[12][0]) / 2.0, (p_pos[11][1] + p_pos[12][1]) / 2.0
        torso_len = math.hypot(sc[0] - hc[0], sc[1] - hc[1])
        if torso_len < 1e-6:
            return None, 0, 0, (0, 0)
        scale = self.NORMALIZE_TORSO_LENGTH / torso_len
        norm = [[(p[0] - hc[0]) * scale, (p[1] - hc[1]) * scale] for p in p_pos[:17]]
        return norm, scale, torso_len, hc

    def predict_frame(self, frame):
        if self.yolo_model is None or self.hrnet_model is None:
            if not self.init_models():
                return [[]], []
        bbox, conf = self.detect_person(frame)
        if bbox is None:
            return [[]], []

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  #画出框
        cv2.putText(frame, f"Person: {conf:.2f}", (int(x1), max(20, int(y1)-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        roi_x1 = max(0, int(x1) - self.padding)
        roi_y1 = max(0, int(y1) - self.padding)
        roi_x2 = min(w, int(x2) + self.padding)
        roi_y2 = min(h, int(y2) + self.padding)
        person_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if person_roi.size == 0:
            return [[]], []

        self.last_roi = (roi_x1, roi_y1, roi_x2, roi_y2)

        if len(person_roi.shape) == 2:
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_GRAY2RGB)
        else:
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

        roi_h, roi_w = person_rgb.shape[:2]
        img_tensor, target = self.hrnet_transform(person_rgb, {"box": [0, 0, roi_w-1, roi_h-1]})
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

        with torch.no_grad():
            outputs = self.hrnet_model(img_tensor.to(self.device))
            flip_tensor = transforms.flip_images(img_tensor)
            flip_out = torch.squeeze(
                transforms.flip_back(self.hrnet_model(flip_tensor.to(self.device)),
                                     self.person_info["flip_pairs"]),
            )
            flip_out[..., 1:] = flip_out.clone()[..., 0:-1]
            outputs = (outputs + flip_out) * 0.5  #翻转运算取平均

            keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
            keypoints = np.squeeze(keypoints)
            scores = np.squeeze(scores)
            if scores.ndim == 0:
                scores = np.array([scores])

            keypoints_list = []
            for kp, sc in zip(keypoints, scores):
                if hasattr(kp, '__iter__') and len(kp) >= 2:
                    ox = max(0, min(roi_x1 + float(kp[0]), w - 1))
                    oy = max(0, min(roi_y1 + float(kp[1]), h - 1))
                    keypoints_list.append([int(ox), int(oy), float(sc)])
                else:
                    keypoints_list.append([0, 0, 0.0])
        return [keypoints_list], scores

    def process_frame(self, frame, preview=True, normalize_for_storage=True):
        """处理单帧：只预测关键点、归一化、绘制、输出完整帧，不计算特征"""
        out_w, out_h = self.OUTPUT_VIDEO_SIZE
        try:
            list_p, _ = self.predict_frame(frame)
            p_pos = get_keypoints(list_p)

            if p_pos and len(p_pos) >= 17:
                validated = []
                for pt in p_pos:
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        validated.append([float(pt[0]), float(pt[1])])
                    else:
                        validated = []
                        break
                if len(validated) >= 17:
                    p_pos = validated
                else:
                    p_pos = []

            if not p_pos or len(p_pos) < 17:
                # 无人帧：显示当前帧图像（resize 到输出尺寸），并暂停片刻
                output_frame = cv2.resize(frame, (out_w, out_h))
                if preview:
                    cv2.setWindowTitle('YOLO Detection', "No person")
                    cv2.imshow('YOLO Detection', output_frame)
                    cv2.waitKey(500)  # 暂停 500ms，可调整
                self.all_points.append([])
                self.all_normalized_points.append(np.zeros((17, 2), dtype=np.float32))
                self.all_scale_info.append(None)
                return output_frame, [], None, None, None

            norm_data = None
            scale_info = None
            if normalize_for_storage:
                norm_data, scale, torso_len, center = self.normalize_keypoints(p_pos)
                if norm_data is not None:
                    scale_info = {'scale': scale, 'torso_length': torso_len, 'center': center}
                else:
                    norm_data = None

            self.trajectory_tracker.update(p_pos)
            # 绘制关键点及连线（直接在frame上绘制）
            draw_points = [[p[0], p[1], 1.0] for p in p_pos]
            draw = Draw(frame, [draw_points])
            draw.draw_select()

            # 将完整的绘制后帧 resize 到输出尺寸
            output_frame = cv2.resize(frame, (out_w, out_h))

            if preview:
                cv2.setWindowTitle('YOLO Detection', "Processing")
                cv2.imshow('YOLO Detection', output_frame)
                cv2.waitKey(1)

            self.all_points.append(p_pos)
            if norm_data is not None:
                self.all_normalized_points.append(np.array(norm_data, dtype=np.float32))
            else:
                self.all_normalized_points.append(np.zeros((17, 2), dtype=np.float32))
            self.all_scale_info.append(scale_info)

            return output_frame, p_pos, None, norm_data, scale_info

        except Exception as e:
            print(f"Error in process_frame: {str(e)[:80]}")
            output_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            self.all_points.append([])
            self.all_normalized_points.append(np.zeros((17,2), dtype=np.float32))
            self.all_scale_info.append(None)
            return output_frame, [], None, None, None

    def compute_and_save_features(self):
        """在所有帧处理完成后，统一计算特征、向量、加速度、步频并保存"""
        total = len(self.all_points)
        time_gap = 1.0 / self.VIDEO_FRAME_SPEED

        # 重置状态
        self.vector_list.clear()
        self.max_acc.clear()
        self.all_features.clear()
        keypoint_prev = None
        step_state = 0
        step_count = 0
        step_freq = 0.0
        last_step_time = 0.0

        # 向量计算需要下一帧手势数据，循环到倒数第二帧
        for i in range(total - 1):
            p_pos = self.all_points[i]
            norm_data = self.all_normalized_points[i]        # 当前帧归一化关键点
            norm_data_fut = self.all_normalized_points[i+1]  # 下一帧归一化关键点

            if not p_pos or len(p_pos) < 17:
                keypoint_prev = None
                self.all_features.append(np.zeros(50, dtype=np.float32))
                self.vector_list.append(np.zeros((17,2), dtype=np.float32))
                continue

            feature = Feature(norm_data) #获取手工特征
            feat = feature.get_all_features()
            self.all_features.append(np.array(feat, dtype=np.float32))

            if norm_data_fut is None or len(norm_data_fut) == 0 or norm_data is None or len(norm_data) == 0: #获取向量特征
                vec = self.vector_list[-1]
            else:
                vec = np.array(norm_data_fut, dtype=np.float32) - np.array(norm_data, dtype=np.float32)
            self.vector_list.append(vec)

            if keypoint_prev is None:
                acc_per_kp = []
            else:
                acc_per_kp = [acceleration(p_pos[j], keypoint_prev[j], time_gap) for j in range(17)]
            keypoint_prev = [p.copy() for p in p_pos]
            if acc_per_kp:
                self.max_acc.append(max(acc_per_kp) / 1000.0)

            if len(p_pos) >= 17:
                try:
                    left_ankle_y = p_pos[15][1]
                    right_ankle_y = p_pos[16][1]
                except:
                    continue
                delta = left_ankle_y - right_ankle_y
                if abs(delta) > 10:
                    current_time = (i+1) * time_gap
                    if current_time - last_step_time >= self.STEP_MIN_GAP:
                        new_state = 1 if delta > 0 else 2
                        if new_state != step_state and step_state != 0:
                            step_count += 1
                            if step_count > 1:
                                gap = current_time - last_step_time
                                if gap > 0.1:
                                    step_freq = round(1.0 / gap, 3)
                            last_step_time = current_time
                        step_state = new_state
        # 处理最后一帧
        if total > 0:
            last_idx = total - 1
            p_pos_last = self.all_points[last_idx]
            if p_pos_last and len(p_pos_last) >= 17:
                # 手工特征
                feature = Feature(p_pos_last)
                feat = feature.get_all_features()
                self.all_features.append(np.array(feat, dtype=np.float32))
            else:
                self.all_features.append(np.zeros(50, dtype=np.float32))

            if self.vector_list:
                self.vector_list.append(self.vector_list[-1].copy())  # 使用倒数第二帧的向量
            else:
                self.vector_list.append(np.zeros((17,2), dtype=np.float32))

        self.step_count = step_count
        self.step_freq = step_freq

        out_dir = os.path.join(self.output_dir, "features")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{self.video_name}_normalized_points.npy"),
                np.stack(self.all_normalized_points, axis=0))
        np.save(os.path.join(out_dir, f"{self.video_name}_features.npy"),
                np.array(self.all_features, dtype=np.float32))
        
        if len(self.vector_list) > 0:
            vec_arr = np.stack(self.vector_list, axis=0)
            vec_min = vec_arr.min(axis=(0,1), keepdims=True)
            vec_max = vec_arr.max(axis=(0,1), keepdims=True)
            vec_norm = (vec_arr - vec_min) / (vec_max - vec_min + 1e-8)
            np.save(os.path.join(out_dir, f"{self.video_name}_vector.npy"), vec_norm)

        if len(self.max_acc) > 0:
            eps = auto_eps(self.max_acc, 10)
            print(f"Auto eps: {eps}")
            frames = list(range(2, len(self.max_acc) + 2))
            point_acceleration(frames, self.max_acc, self.video_name,
                            use_dbscan=True, eps=eps, min_samples=10)
        print(f"所有特征已保存至 {out_dir}")

    def generate_video(self):
        cap = cv2.VideoCapture(self.input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                print(f"\nError at frame {frame_idx}: {str(e)[:80]}")
                out.write(np.zeros((out_h, out_w, 3), dtype=np.uint8))
                continue
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("\nVideo saved to:", output_path)
        self.compute_and_save_features()

    @staticmethod
    def _progress_bar(current, total, bar_length=30, prefix="进度"):
        percent = current / total
        filled = int(bar_length * percent)
        bar = '█' * filled + '░' * (bar_length - filled)
        sys.stdout.write(f'\r{prefix}: |{bar}| {percent*100:.1f}% ({current}/{total})')
        sys.stdout.flush()
        if current == total:
            print()

if __name__ == '__main__':
    input_path = 'D:/Dataset/sprint/Whole/run_6.mp4'
    processor = VideoProcessor(input_path)
    start = time.time()
    processor.generate_video()
    elapsed = show_time(start, time.time())
    print(f"Total time: {elapsed}")