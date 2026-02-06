import cv2
import numpy as np
from collections import deque
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
from math import sqrt, acos, degrees
import math
import time
hist_ra= deque(maxlen=50)  # 右臂
hist_la = deque(maxlen=50) # 左臂
hist_rl= deque(maxlen=50)  # 右腿
hist_ll = deque(maxlen=50) # 左腿

#实时绘制图像
def draw_direct_plot(img,hist,value, pos=(400, 50), label="Knee"):  #250,120
    """
    直接在传入的img上绘图，无返回值
    """
    # 更新数据
    hist.append(value)
    if len(hist) < 2: return
    
    x, y, w, h = pos[0], pos[1], 200, 100
    
    # 图表位置和大小
    x, y = pos
    w, h = 250, 120
    
    # 创建半透明背景
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return
    
    overlay = roi.copy()
    cv2.rectangle(overlay, (0,0), (w,h), (30,30,30), -1)
    roi[:] = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)
    
    # 绘制数据
    data = list(hist)
    minv, maxv = min(data), max(data)
    if minv == maxv:
        minv, maxv = minv-1, maxv+1
    
    # 绘制点
    points = []
    for i, v in enumerate(data):
        px = x + 10 + int(i * (w-20) / (len(data)-1))
        py = y + h - 10 - int((v-minv) / (maxv-minv) * (h-20))
        py = max(y+10, min(y+h-10, py))
        points.append((px, py))
    
    # 绘制线
    for i in range(1, len(points)):
        cv2.line(img, points[i-1], points[i], (0,165,255), 2)
    
    # 高亮当前点
    if points:
        cv2.circle(img, points[-1], 5, (255,0,0), -1)
    
    # 显示文字
    cv2.putText(img, f"{label}:{value}", 
            (x+10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(img, f"Min:{minv}", (x+w-70, y+20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.putText(img, f"Max:{maxv}", (x+w-70, y+40), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

class Draw():
    def __init__(self, frame, list_p):
        self.frame = frame
        self.list_p = list_p
        self.p_pose = get_keypoints(self.list_p)
    
    def draw_head(self):
        if len(self.p_pose) < 5:
            return
        p_pos = self.p_pose
        p0 = p_pos[0]
        p1 = p_pos[1]
        p2 = p_pos[2]
        p3 = p_pos[3]
        p4 = p_pos[4]
        
        cv2.line(self.frame, p4, p2, (255, 0, 0), 2)
        cv2.line(self.frame, p2, p0, (255, 0, 0), 2)
        cv2.line(self.frame, p0, p1, (255, 0, 0), 2)
        cv2.line(self.frame, p1, p3, (255, 0, 0), 2)
    
    def draw_body(self):
        if len(self.p_pose) < 11:
            return
        p_pos = self.p_pose
        p5 = p_pos[5]
        p6 = p_pos[6]
        p7 = p_pos[7]
        p8 = p_pos[8]
        p9 = p_pos[9]
        p10 = p_pos[10]
        p0 = p_pos[0]
        
        p_m = (int((p5[0] + p6[0]) / 2), int((p5[1] + p6[1]) / 2))

        cv2.line(self.frame, p10, p8, (103, 216, 44), 2)
        cv2.line(self.frame, p8, p6, (103, 216, 44), 2)
        cv2.line(self.frame, p6, p5, (103, 216, 44), 2)
        cv2.line(self.frame, p5, p7, (103, 216, 44), 2)
        cv2.line(self.frame, p7, p9, (103, 216, 44), 2)
        cv2.line(self.frame, p0, p_m, (0, 0, 255), 3)

    def draw_leg(self):
        if len(self.p_pose) < 17:
            return
        p_pos = self.p_pose
        p5 = p_pos[5]
        p6 = p_pos[6]
        p11 = p_pos[11]
        p12 = p_pos[12]
        p13 = p_pos[13]
        p14 = p_pos[14]
        p15 = p_pos[15]
        p16 = p_pos[16]
        
        p_u = (int((p5[0] + p6[0]) / 2), int((p5[1] + p6[1]) / 2))
        p_d = (int((p12[0] + p11[0]) / 2), int((p12[1] + p11[1]) / 2))
        
        cv2.line(self.frame, p12, p14, (255, 0, 220), 2)
        cv2.line(self.frame, p14, p16, (255, 0, 220), 2)
        cv2.line(self.frame, p11, p13, (255, 0, 220), 2)
        cv2.line(self.frame, p13, p15, (255, 0, 220), 2)
        cv2.line(self.frame, p11, p12, (255, 0, 220), 2)
        cv2.line(self.frame, p_u, p_d, (255, 0, 116), 2)

    def draw_point(self):
        p_pos = self.p_pose
        for p in p_pos:
            circle_center = p
            radius = 5
            cv2.circle(self.frame, circle_center, radius, (0, 150, 255), thickness=-1, lineType=cv2.LINE_AA)
    
    def draw_select(self, d_h=True, d_b=True, d_l=True, d_p=True):
        if d_h:
            self.draw_head()
        if d_b:
            self.draw_body()
        if d_l:
            self.draw_leg()
        if d_p:
            self.draw_point()
    
#预测框
def predict():
    model = YOLO("./weights/yolo11s.pt")
    results = model.predict(
                            save=True,
                            conf=0.5 #置信度阈值
                            )
    
def get_keypoints(list_p):
    p_pos=[]
    for p in list_p[0]:
        x = p[0]
        x = int(x)
        y = p[1]
        y = int(y) 
        pos = (x,y)
        p_pos.append(pos)
    return p_pos

#角度显示
def angle_show(list_p,position,color,text,limb,img):  #显示位置，字体颜色，显示内容，显示肢体，投射图像
    p = get_keypoints(list_p)
    use_points = [p[limb[0]],p[limb[1]],p[limb[2]]]  #获取三个关键点，中间项为顶点
    angle = calculate_angle(use_points[0],use_points[1],use_points[2])
    angle = str(int(angle))
    cv2.putText(img,f"{text}:{angle}",position,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.6,thickness=1,color=color)  #显示角度
    return angle
    # 显示图像

#获取关键点坐标
def get_P_X_Y(img,marks):
    img_Heigt = img.shape[0]
    img_Width = img.shape[1]
    for point,pose in enumerate(marks):
        x_pos = int(pose.x*img_Width)
        y_pos = int(pose.y*img_Heigt)
        print(f"point:{point},x_pos:{x_pos},y_pos:{y_pos}")

#获取关键点
def get_keypoints(list_p):
    p_pos=[]
    for p in list_p[0]:
        x = p[0]
        x = int(x)
        y = p[1]
        y = int(y) 
        pos = (x,y)
        p_pos.append(pos)
    return p_pos
    
def create_swap_detector():
    """创建交换检测器"""
    previous_state = None
    
    def detector(a, b):
        nonlocal previous_state
        current_state = a > b
        
        if previous_state is None:
            previous_state = current_state
            return False
        
        if current_state != previous_state:
            previous_state = current_state
            return True
        return False
    return detector

#打分系统
def get_score(step_fre,Wf):
    if step_fre > 1.5 and step_fre < 2:
        score_F = 2
    elif step_fre < 1.5 or step_fre > 2:
        score_F = 1
    return score_F*Wf

def distance(p1_idx, p2_idx,p_pos):
    """计算两点之间的欧氏距离"""
    x1, y1 = p_pos[p1_idx][0], p_pos[p1_idx][1]
    x2, y2 = p_pos[p2_idx][0], p_pos[p2_idx][1]
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def limb_calculation(p_pos):
    # 计算各肢体长度
    upper_arm_l = distance(6, 8, p_pos)    # 左上臂 (肩-肘)
    lower_arm_l = distance(8, 10, p_pos)   # 左下臂 (肘-腕)
    upper_arm_r = distance(5, 7, p_pos)    # 右上臂 (注意：5是右肩，7是右肘)
    lower_arm_r = distance(7, 9, p_pos)    # 右下臂
    
    upper_leg_l = distance(12, 14, p_pos)  # 左大腿 (髋-膝)
    lower_leg_l = distance(14, 16, p_pos)  # 左小腿 (膝-踝)
    upper_leg_r = distance(11, 13, p_pos)  # 右大腿
    lower_leg_r = distance(13, 15, p_pos)  # 右小腿
    
    shoulder_w = distance(5, 6, p_pos)     # 肩宽 (左右肩之间)
    hip_w = distance(11, 12, p_pos)        # 髋宽 (左右髋之间)
    
    # 计算总长度
    left_arm_total = upper_arm_l + lower_arm_l
    right_arm_total = upper_arm_r + lower_arm_r
    left_leg_total = upper_leg_l + lower_leg_l
    right_leg_total = upper_leg_r + lower_leg_r
    
    # 计算比例（注意正确的括号）
    ratios = {
        # 腿臂比：腿长/臂长 (正常范围：1.0-1.3)
        "left_leg_arm": left_leg_total / left_arm_total,
        "right_leg_arm": right_leg_total / right_arm_total,
        
        # 小腿大腿比 (正常范围：0.8-1.0)
        "left_ul_leg": lower_leg_l / upper_leg_l,
        "right_ul_leg": lower_leg_r / upper_leg_r,
        
        # 前臂上臂比 (正常范围：0.8-1.0)
        "left_ul_arm": lower_arm_l / upper_arm_l,
        "right_ul_arm": lower_arm_r / upper_arm_r,
        
        # 肩宽与腿长比
        "left_shoulder_leg": shoulder_w / left_leg_total,
        "right_shoulder_leg": shoulder_w / right_leg_total,
        
        # 肩髋比 (男性：1.2-1.4，女性：0.9-1.1)
        "s_h": shoulder_w / hip_w,
        
        # 左右对称性 (应接近1.0)
        'arm_symmetry': min(left_arm_total, right_arm_total) / 
                        max(left_arm_total, right_arm_total),
        'leg_symmetry': min(left_leg_total, right_leg_total) / 
                        max(left_leg_total, right_leg_total),
        
        # 额外有用的比例
        'total_leg_to_height': (left_leg_total + right_leg_total) / 2,  # 需要身高数据归一化
        'total_arm_to_height': (left_arm_total + right_arm_total) / 2,
    }
    
    # 异常检测标志
    abnormal_flags = {
        # 腿臂比异常：腿应该比手臂长，所以<0.9或>1.4都是异常
        'leg_to_arm_abnormal': any([
            ratios['left_leg_arm'] > 1.4 or ratios['left_leg_arm'] < 0.9,
            ratios['right_leg_arm'] > 1.4 or ratios['right_leg_arm'] < 0.9
        ]),
        
        # 小腿大腿比异常
        'leg_abnormal': any([
            ratios['left_ul_leg'] > 1.0 or ratios['left_ul_leg'] < 0.75,
            ratios['right_ul_leg'] > 1.0 or ratios['right_ul_leg'] < 0.75
        ]),
        
        # 前臂上臂比异常
        'arm_abnormal': any([
            ratios['left_ul_arm'] > 1.0 or ratios['left_ul_arm'] < 0.75,
            ratios['right_ul_arm'] > 1.0 or ratios['right_ul_arm'] < 0.75
        ]),
        
        # 对称性异常
        'arm_symmetry_abnormal': ratios['arm_symmetry'] < 0.85,  # 宽松一点
        'leg_symmetry_abnormal': ratios['leg_symmetry'] < 0.85,
        
        # 肩髋比异常（注意：需要知道性别）
        's_h_abnormal': ratios['s_h'] < 0.9 or ratios['s_h'] > 1.5,  # 宽泛范围
        
        # 额外：左右腿臂比差异过大
        'leg_arm_balance_abnormal': abs(ratios['left_leg_arm'] - ratios['right_leg_arm']) > 0.2,
    }
    
    return ratios, abnormal_flags

def wrong_point(abnormal_flag,weight):
    point = 0 
    if abnormal_flag['leg_to_arm_abnormal']:
        point += weight[0]*1
    if abnormal_flag['leg_abnormal']:
        point += weight[1]*1
    if abnormal_flag['arm_abnormal']:
        point += weight[2]*1
    if abnormal_flag['arm_symmetry_abnormal']:
        point += weight[3]*1
    if abnormal_flag['leg_symmetry_abnormal']:
        point += weight[4]*1
    if abnormal_flag['s_h_abnormal']:
        point += weight[5]*1
    return point

# cv2.line(image,(100,200),(250,250),(255,0,0),2)#画线，起点，终点，颜色，粗细
# cv2.circle(image,(50,100),20,(0,0,255),2)#画圈，圆心，半径，颜色，粗细
def video_out(input_path,output_dir:str,pre:str):
    filehead = input_path.split('/')[-1]
    output_path = pre+filehead
    output_path = output_dir+output_path
    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path,fourcc,fps,(int(frame_size[0]),int(frame_size[1])))
    return out,output_path,cap

#调整图片大小
def sclae_img(img,percent):
    scale_percent = percent  # 百分比
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def calculate_angle(point_a, point_b, point_c):
    """
    计算三个点之间的角度（点B为顶点）
    :param point_a: 点A的坐标 (x, y)
    :param point_b: 点B的坐标 (x, y)
    :param point_c: 点C的坐标 (x, y)
    :return: 角度（度数）
    """
    # 创建向量BA和BC
    ba = (point_a[0] - point_b[0], point_a[1] - point_b[1])
    bc = (point_c[0] - point_b[0], point_c[1] - point_b[1])
    
    # 计算点积
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    
    # 计算向量模长
    magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    # 计算余弦值
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    
    # 防止浮点数精度问题导致的值超出[-1, 1]范围
    cos_angle = max(min(cos_angle, 1), -1)
    
    # 计算角度（弧度）并转换为度数
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

#根据位置差计算加速度
def acceleration(current_point, previous_point, time_gap):
    """
    计算两个连续帧之间关键点的加速度
    参考video.py中的加速度计算逻辑
    """
    # 计算位移
    dx = current_point[0] - previous_point[0]
    dy = current_point[1] - previous_point[1]
    displacement = math.sqrt(dx**2 + dy**2)
    
    # 计算速度（假设上一帧速度为瞬时速度）
    velocity = displacement / time_gap if time_gap > 0 else 0
    
    # 加速度计算
    acceleration = velocity / time_gap if time_gap > 0 else 0
    
    return acceleration