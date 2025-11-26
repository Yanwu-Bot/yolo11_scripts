import numpy as np
import torch
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any

class YOLOKeypointToHeatMap(object):
    def __init__(
        self,
        heatmap_hw: Tuple[int, int] = (64, 48),  # 默认 (H, W) = (256//4, 192//4)，与原代码一致
        gaussian_sigma: int = 2,
        keypoints_weights: Optional[np.ndarray] = None,
        conf_threshold: float = 0.5,  # YOLO 关键点置信度阈值（替代原代码的 visible）
        downscale_factor: int = 4  # 坐标下采样因子（与原代码一致）
    ):
        """
        YOLO11 Pose 关键点转热力图类（基于你提供的 KeypointToHeatMap 逻辑适配）

        Args:
            heatmap_hw: 输出热力图尺寸 (H, W)
            gaussian_sigma: 高斯核标准差
            keypoints_weights: 各关键点权重（shape=(17,)），用于加权热力图
            conf_threshold: YOLO 关键点置信度阈值（低于此值视为不可见）
            downscale_factor: 图像坐标到热力图坐标的下采样因子
        """
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3  # 3σ 原则，覆盖 99.7% 能量
        self.conf_threshold = conf_threshold
        self.downscale_factor = downscale_factor

        # 关键点权重配置
        self.use_kps_weights = False if keypoints_weights is None else True
        self.kps_weights = keypoints_weights if keypoints_weights is not None else np.ones(17, dtype=np.float32)
        assert self.kps_weights.shape[0] == 17, "关键点权重必须为 17 个（COCO 标准）"

        # 预生成高斯核（与原代码逻辑一致，不提前归一化）
        self.kernel = self._generate_gaussian_kernel()

        # COCO 17 关键点名称（与 YOLO11 Pose 输出顺序一致）
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    def _generate_gaussian_kernel(self) -> np.ndarray:
        """生成高斯核（与原代码逻辑完全一致）"""
        kernel_size = 2 * self.kernel_radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2

        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[y, x] = np.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * self.sigma**2))
        
        return kernel

    def _process_single_person(
        self,
        kps: np.ndarray,  # 单个人体的关键点 (17, 3) -> (x, y, conf)
        img_hw: Tuple[int, int]  # 原始图像尺寸 (H, W)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """处理单个人体的关键点，生成热力图、关键点权重和综合热力图"""
        num_kps = kps.shape[0]
        assert num_kps == 17, f"关键点数量错误：预期 17 个，实际 {num_kps} 个"

        # 初始化热力图和关键点权重（替代原代码的 visible）
        heatmap = np.zeros((num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        kps_weights = np.ones((num_kps,), dtype=np.float32)

        # 1. 过滤低置信度关键点（用 YOLO 的 conf 替代原代码的 visible）
        for kp_id in range(num_kps):
            conf = kps[kp_id, 2]
            if conf < self.conf_threshold:
                kps_weights[kp_id] = 0.0  # 不可见关键点权重设为 0

        # 2. 图像坐标 -> 热力图坐标（下采样 + 四舍五入，与原代码一致）
        scale_h = self.heatmap_hw[0] / img_hw[0]
        scale_w = self.heatmap_hw[1] / img_hw[1]
        heatmap_kps_x = (kps[:, 0] * scale_w).astype(np.int32)  # x 坐标映射
        heatmap_kps_y = (kps[:, 1] * scale_h).astype(np.int32)  # y 坐标映射

        # 3. 生成每个关键点的热力图（与原代码逻辑完全一致）
        for kp_id in range(num_kps):
            if kps_weights[kp_id] < 0.5:  # 跳过不可见关键点
                continue

            x, y = heatmap_kps_x[kp_id], heatmap_kps_y[kp_id]
            ul = [x - self.kernel_radius, y - self.kernel_radius]  # 核的左上角坐标
            br = [x + self.kernel_radius, y + self.kernel_radius]  # 核的右下角坐标

            # 检查核与热力图是否有交集，无交集则跳过
            if (ul[0] > self.heatmap_hw[1] - 1 or
                ul[1] > self.heatmap_hw[0] - 1 or
                br[0] < 0 or
                br[1] < 0):
                kps_weights[kp_id] = 0.0
                continue

            # 计算高斯核的有效区域（核坐标系）
            g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
            g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])

            # 计算热力图的有效区域（热力图坐标系）
            img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
            img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

            # 将高斯核复制到热力图（与原代码一致，不叠加只覆盖）
            heatmap[kp_id][img_y[0]:img_y[1]+1, img_x[0]:img_x[1]+1] = \
                self.kernel[g_y[0]:g_y[1]+1, g_x[0]:g_x[1]+1]

        # 4. 应用关键点权重（与原代码一致）
        if self.use_kps_weights:
            kps_weights = np.multiply(kps_weights, self.kps_weights)
            # 权重应用到热力图（让重要关键点在综合图中更突出）
            for kp_id in range(num_kps):
                heatmap[kp_id] *= kps_weights[kp_id]

        # 5. 生成综合热力图（17个关键点热力图取最大值融合，突出所有有效关键点）
        combined_heatmap = np.max(heatmap, axis=0)
        # 归一化综合热力图（值范围 [0, 1]，便于显示）
        max_val = combined_heatmap.max()
        if max_val > 0:
            combined_heatmap /= max_val

        return heatmap, kps_weights, combined_heatmap

    def __call__(
        self,
        image_path: str,
        model_path: str = "yolo11pose.pt",
        visualize: bool = True,
        save_path: Optional[str] = None,
        force_cpu: bool = True  # 强制使用 CPU 避免设备冲突
    ) -> Tuple[np.ndarray, Dict[str, torch.Tensor], np.ndarray]:
        """
        主调用函数：加载 YOLO 模型，推理关键点，生成热力图

        Args:
            image_path: 输入图像路径
            model_path: YOLO11 Pose 模型路径
            visualize: 是否可视化热力图
            save_path: 对比图保存路径（None 不保存）
            force_cpu: 是否强制使用 CPU 运行（避免 GPU 张量转换错误）

        Returns:
            image: 原始图像（RGB 格式）
            target: 包含热力图和权重的字典，keys = ["heatmap", "kps_weights"]
            combined_heatmap: 综合热力图（shape=(H_heatmap, W_heatmap)）
        """
        # 1. 加载 YOLO11 Pose 模型（根据 force_cpu 决定运行设备）
        model = YOLO(model_path)
        device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

        # 2. 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像：{image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        img_hw = (img_h, img_w)

        # 3. YOLO 推理关键点（指定设备）
        results = model(
            image_rgb,
            conf=self.conf_threshold,
            iou=0.45,
            device=device,  # 明确指定运行设备
            verbose=False
        )

        # 4. 提取第一个人体的关键点（处理 GPU 张量）
        kps = None
        for r in results:
            if hasattr(r, "keypoints") and r.keypoints is not None:
                kp_data = r.keypoints.data
                # 兼容 YOLO 不同版本的输出格式（tensor/tuple）
                if isinstance(kp_data, tuple):
                    # tuple 格式：(x_tensor, y_tensor, conf_tensor) -> (num_persons, 17)
                    x_tensor, y_tensor, conf_tensor = kp_data
                    if x_tensor.shape[0] > 0:
                        # 先移到 CPU 再转 numpy
                        x = x_tensor[0].cpu().numpy() if x_tensor.is_cuda else x_tensor[0].numpy()
                        y = y_tensor[0].cpu().numpy() if y_tensor.is_cuda else y_tensor[0].numpy()
                        conf = conf_tensor[0].cpu().numpy() if conf_tensor.is_cuda else conf_tensor[0].numpy()
                        kps = np.stack([x, y, conf], axis=1)  # (17, 3)
                elif torch.is_tensor(kp_data):
                    # tensor 格式：(num_persons, 17, 3)
                    if kp_data.shape[0] > 0:
                        # 先移到 CPU 再转 numpy
                        kp_cpu = kp_data.cpu().numpy() if kp_data.is_cuda else kp_data.numpy()
                        kps = kp_cpu[0]  # 取第一个人体
                break

        if kps is None:
            raise ValueError("未检测到人体姿态")

        # 5. 生成热力图、关键点权重和综合热力图
        heatmap, kps_weights, combined_heatmap = self._process_single_person(kps, img_hw)

        # 6. 转换为 torch.Tensor（与原代码一致）
        target = {
            "heatmap": torch.as_tensor(heatmap, dtype=torch.float32),
            "kps_weights": torch.as_tensor(kps_weights, dtype=torch.float32)
        }

        # 7. 可视化（只显示原图+关键点 和 综合热力图 两张对比图）
        if visualize or save_path is not None:
            self._visualize_compare(image_rgb, combined_heatmap, kps, kps_weights, save_path)

        return image_rgb, target, combined_heatmap

    def _visualize_compare(
        self,
        image: np.ndarray,
        combined_heatmap: np.ndarray,
        kps: np.ndarray,
        kps_weights: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """可视化对比：左图=原始图像+关键点，右图=综合热力图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 左图：原始图像 + 关键点标注
        img_with_kps = image.copy()
        valid_kps_count = 0
        for kp_id in range(17):
            if kps_weights[kp_id] > 0.5:
                valid_kps_count += 1
                x, y = int(kps[kp_id, 0]), int(kps[kp_id, 1])
                # 绘制关键点（蓝色实心圆）
                cv2.circle(img_with_kps, (x, y), 6, (255, 0, 0), -1)
                # 绘制关键点名称（绿色文字）
                cv2.putText(img_with_kps, self.keypoint_names[kp_id][:3], 
                            (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        ax1.imshow(img_with_kps)
        ax1.set_title(f"Original Image\n(Valid Keypoints: {valid_kps_count}/17)", fontsize=14)
        ax1.axis("off")

        # 右图：综合热力图（所有关键点融合）
        im = ax2.imshow(combined_heatmap, cmap="jet", vmin=0, vmax=1)
        ax2.set_title("Combined Pose Heatmap", fontsize=14)
        ax2.axis("off")

        # 添加颜色条（显示热力值强度）
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Heatmap Intensity", fontsize=12)

        # 调整布局
        plt.tight_layout()

        # 保存图片（可选）
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"对比图已保存至：{save_path}")

        # 显示图片
        plt.show()


# ---------------------- 示例用法 ----------------------
if __name__ == "__main__":
    # 1. 初始化热力图生成器（可自定义参数）
    heatmap_generator = YOLOKeypointToHeatMap(
        heatmap_hw=(64, 48),  # 热力图尺寸（与原代码一致）
        gaussian_sigma=2,     # 高斯核标准差
        conf_threshold=0.6,   # 关键点置信度阈值
        # 可选：设置关键点权重（例如对四肢关键点加权）
        keypoints_weights=np.array([
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.5, 1.5, 1.5, 1.5,
            1.5, 1.5, 1.5, 1.5,
            1.5, 1.5, 1.5, 1.5
        ], dtype=np.float32)
    )

    # 2. 生成热力图（默认强制使用 CPU）
    try:
        image, target, combined_heatmap = heatmap_generator(
            image_path="data\img02.png",  # 替换为你的图像路径
            model_path="weights\yolo11n-pose.pt",
            visualize=True,
            save_path="pose_heatmap_compare.png",  # 保存对比图
            force_cpu=True
        )

        # 输出结果信息
        print(f"原始图像形状：{image.shape}")
        print(f"单个热力图形状：{target['heatmap'].shape}")  # (17, 64, 48)
        print(f"综合热力图形状：{combined_heatmap.shape}")  # (64, 48)
        print(f"有效关键点数量：{torch.sum(target['kps_weights'] > 0.5).item()}")
    except Exception as e:
        print(f"错误：{e}")