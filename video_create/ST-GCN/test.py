import os
import cv2
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
from HRNet_model import HighResolutionNet
import transforms
from ultralytics import YOLO

_device = None
_hrnet_model = None
_yolo_model = None
_person_info = None
_hrnet_transform = None
_models_initialized = False

def _init_models():
    global _device, _hrnet_model, _yolo_model, _person_info, _hrnet_transform, _models_initialized
    if _models_initialized:
        return
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {_device}")

    _yolo_model = YOLO('weights/yolo11n.pt')
    weights_path = "HRnet/pytorch/pose_coco/pose_hrnet_w32_256x192.pth"
    keypoint_json_path = "HRnet/person_keypoints.json"
    with open(keypoint_json_path, "r") as f:
        _person_info = json.load(f)

    _hrnet_model = HighResolutionNet(base_channel=32)
    weights = torch.load(weights_path, map_location=_device)
    weights = weights if "model" not in weights else weights["model"]
    _hrnet_model.load_state_dict(weights)
    _hrnet_model.to(_device)
    _hrnet_model.eval()

    resize_hw = (256, 192)
    _hrnet_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    _models_initialized = True
    print("Model initialization complete")

def _detect_person(frame):
    results = _yolo_model(frame, verbose=False)
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) >= 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    return [float(x1), float(y1), float(x2), float(y2)]
    return None

def _extract_single_frame_keypoints(frame):
    """提取单帧关键点（原始坐标）"""
    bbox = _detect_person(frame)
    if bbox is None:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    padding = 10
    roi_x1 = max(0, int(x1) - padding)
    roi_y1 = max(0, int(y1) - padding)
    roi_x2 = min(width, int(x2) + padding)
    roi_y2 = min(height, int(y2) + padding)
    person_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if person_roi.size == 0:
        return None
    if len(person_roi.shape) == 3 and person_roi.shape[2] == 3:
        person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
    else:
        person_rgb = person_roi
    roi_height, roi_width = person_rgb.shape[:2]
    img_tensor, target = _hrnet_transform(person_rgb, {"box": [0, 0, roi_width-1, roi_height-1]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    with torch.no_grad():
        outputs = _hrnet_model(img_tensor.to(_device))
        keypoints, _ = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
    result = []
    for kp in keypoints:
        if len(kp) >= 2:
            orig_x = roi_x1 + float(kp[0])
            orig_y = roi_y1 + float(kp[1])
            orig_x = max(0, min(orig_x, width-1))
            orig_y = max(0, min(orig_y, height-1))
            result.append([orig_x, orig_y])
        else:
            result.append([0, 0])
    return np.array(result, dtype=np.float32)

def _normalize_keypoints(p_pos, target_torso_length=100):
    if len(p_pos) < 17:
        return p_pos
    shoulder_center_x = (p_pos[5][0] + p_pos[6][0]) / 2
    shoulder_center_y = (p_pos[5][1] + p_pos[6][1]) / 2
    hip_center_x = (p_pos[11][0] + p_pos[12][0]) / 2
    hip_center_y = (p_pos[11][1] + p_pos[12][1]) / 2
    torso_length = math.hypot(shoulder_center_x - hip_center_x, shoulder_center_y - hip_center_y)
    if torso_length < 1e-6:
        return p_pos
    scale = target_torso_length / torso_length
    center_x, center_y = hip_center_x, hip_center_y
    normalized = []
    for i in range(17):
        norm_x = (p_pos[i][0] - center_x) * scale
        norm_y = (p_pos[i][1] - center_y) * scale
        normalized.append([norm_x, norm_y])
    return np.array(normalized, dtype=np.float32)

class COCOGraph:
    def __init__(self, hop_size=2):
        self.num_node = 17
        self.hop_size = hop_size
        self.get_edge()
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)
        self.get_adjacency()
    def get_edge(self):
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_base = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                         (11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]
        self.edge = self_link + neighbor_base
    def get_hop_distance(self, num_node, edge, hop_size):
        A = np.zeros((num_node, num_node))
        for i,j in edge:
            A[j,i]=1; A[i,j]=1
        hop_dis = np.zeros((num_node,num_node))+np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size+1)]
        arrive_mat = (np.stack(transfer_mat)>0)
        for d in range(hop_size,-1,-1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis
    def get_adjacency(self):
        valid_hop = range(0, self.hop_size+1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis==hop]=1
        normalize_adjacency = self.normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis==hop] = normalize_adjacency[self.hop_dis==hop]
        self.A = A
    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        Dn = np.zeros((A.shape[0], A.shape[0]))
        for i in range(A.shape[0]):
            if Dl[i]>0:
                Dn[i,i] = Dl[i]**(-1)
        return np.dot(A, Dn)

class SpatialGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels*s_kernel_size, 1)
    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class STGC_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.5):
        super().__init__()
        self.sgc = SpatialGraphConvolution(in_channels, out_channels, A_size[0])
        self.M = nn.Parameter(torch.ones(A_size))
        self.tgc = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, (t_kernel_size,1), (stride,1),
                      ((t_kernel_size-1)//2,0)),
            nn.BatchNorm2d(out_channels), nn.ReLU())
    def forward(self, x, A):
        return self.tgc(self.sgc(x, A * self.M))

class ContrastiveEncoder(nn.Module):
    def __init__(self, in_channels=2, t_kernel_size=9, hop_size=2, output_dim=128):
        super().__init__()
        graph = COCOGraph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()
        self.bn = nn.BatchNorm1d(in_channels*graph.num_node)
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size, dropout=0.1)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size, dropout=0.1)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size, dropout=0.1)
        self.projection = nn.Sequential(nn.Linear(64,128), nn.ReLU(), nn.Linear(128, output_dim))
    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0,3,1,2).contiguous().view(N, V*C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0,2,3,1).contiguous()
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        x = F.adaptive_avg_pool2d(x, (1,1)).view(N, -1)
        x = self.projection(x)
        return F.normalize(x, dim=1)

def compare_windows(kps1, kps2, model, device):
    """
    计算两个窗口关键点序列的特征相似度。
    Args:
        kps1: np.ndarray 形状 (W, 17, 2) 或 list of (17,2)
        kps2: 同上
        model: 已加载的 ContrastiveEncoder 模型（已置于 eval 模式）
        device: torch.device
    Returns:
        sim_cosine: 余弦相似度（float）
        score_percent: 映射到 0~100 的百分比得分（float）
    """
    import numpy as np
    import torch

    model.eval()
    # 确保为 numpy 数组
    arr1 = np.array(kps1, dtype=np.float32)  # (W,17,2)
    arr2 = np.array(kps2, dtype=np.float32)

    def get_feature(arr):
        # (W,17,2) -> (1,2,W,17)
        tensor = torch.FloatTensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor)
        return feat.cpu().numpy()[0]  # (128,)

    feat1 = get_feature(arr1)
    feat2 = get_feature(arr2)
    sim_cosine = np.dot(feat1, feat2)  # L2归一化后点积即余弦相似度
    score_percent = (sim_cosine + 1) / 2 * 100
    return sim_cosine, score_percent

if __name__ == '__main__':
    vid1 = 'video_origin/data_video/use/run_man.mp4'
    vid2 = 'video_origin/data_video/use/run_woman.mp4'
    model_weights = 'result/GCN/model/best.pth'
    compare_random_windows(vid1, vid2, model_weights, window_size=10,
                            output_path='result/GCN/comparison_result.png')