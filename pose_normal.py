import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
from utill import *
from time_utils import show_time 

'''
    p0=p_pos[0]#鼻子
    p1=p_pos[1]#左眼
    p2=p_pos[2]#右眼
    p3=p_pos[3]#左耳
    p4=p_pos[4]#右耳
    p5=p_pos[5]#左肩
    p6=p_pos[6]#右肩
    p7=p_pos[7]#左肘
    p8=p_pos[8]#右肘
    p9=p_pos[9]#左手
    p10=p_pos[10]#右手
    p11=p_pos[11]#左髋
    p12=p_pos[12]#右髋
    p13=p_pos[13]#左膝
    p14=p_pos[14]#右膝    
    p15=p_pos[15]#左脚
    p16=p_pos[16]#右脚
'''

# 加载YOLOv11n-pose模型
model = YOLO("./weights/yolo11n-pose.pt")
#图片路径
img_path = "data\img02.png"
#读取图片
img = cv2.imread(img_path,cv2.IMREAD_COLOR)
# 定义关键点索引与名称的对应关系
dic_points = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}

#输出对应点坐标

def draw_head(frame,list_p):
    p_pos = get_keypoints(list_p)
    #鼻子
    p0=p_pos[0]
    #左眼
    p1=p_pos[1]
    #右眼
    p2=p_pos[2]
    #左耳
    p3=p_pos[3]
    #右耳
    p4=p_pos[4]
    
    cv2.line(frame,p4,p2,(255,0,0),2)
    cv2.line(frame,p2,p0,(255,0,0),2)
    cv2.line(frame,p0,p1,(255,0,0),2)
    cv2.line(frame,p1,p3,(255,0,0),2)

def draw_body(frame,list_p):
    p_pos = get_keypoints(list_p)
    #左肩
    p5=p_pos[5]
    #右肩
    p6=p_pos[6]
    #左肘
    p7=p_pos[7]
    #右肘
    p8=p_pos[8]
    #左手
    p9=p_pos[9]
    #右手
    p10=p_pos[10]
    #鼻子
    p0=p_pos[0]
    p_m = (int((p5[0]+p6[0])/2),int((p5[1]+p6[1])/2))

    cv2.line(frame,p10,p8,(103,216,44),2)
    cv2.line(frame,p8,p6,(103,216,44),2)
    cv2.line(frame,p6,p5,(103,216,44),2)
    cv2.line(frame,p5,p7,(103,216,44),2)
    cv2.line(frame,p7,p9,(103,216,44),2)
    cv2.line(frame,p0,p_m,(0,0,255),3)

def draw_leg(frame,list_p):
    p_pos = get_keypoints(list_p)
    #左肩
    p5=p_pos[5]
    #右肩
    p6=p_pos[6]
    #左髋
    p11=p_pos[11]
    #右髋
    p12=p_pos[12]
    #左膝
    p13=p_pos[13]
    #右膝
    p14=p_pos[14]
    #左脚
    p15=p_pos[15]
    #右脚
    p16=p_pos[16]
    p_u = (int((p5[0]+p6[0])/2),int((p5[1]+p6[1])/2))
    p_d = (int((p12[0]+p11[0])/2),int((p12[1]+p11[1])/2))
    cv2.line(frame,p12,p14,(255,0,220),2)
    cv2.line(frame,p14,p16,(255,0,220),2)
    cv2.line(frame,p11,p13,(255,0,220),2)
    cv2.line(frame,p13,p15,(255,0,220),2)
    cv2.line(frame,p11,p12,(255,0,220),2)
    cv2.line(frame,p11,p12,(255,0,220),2)
    cv2.line(frame,p_u,p_d,(255,0,116),2)

def draw_point(p_pos):
    for p in p_pos:  
        circle_center = p
        # define the radius of the circle
        radius =5
        #  Draw a circle using the circle() Function
        cv2.circle(img, circle_center, radius, (0, 150, 255), thickness=-1, lineType=cv2.LINE_AA) 

def draw_select(frame,list_p,d_h=True,d_b=True,d_l=True,pre=True,d_p=True):
    if d_h:
        draw_head(frame,list_p)
    if d_b:
        draw_body(frame,list_p)
    if d_l:
        draw_leg(frame,list_p)
    if d_p:
        p_pos=get_keypoints(list_p)
        draw_point(p_pos)
    if pre:
        predict()

#物体预测
def predict():
    model = YOLO("./weights/yolo11s.pt")
    results = model.predict(source=img_path,
                            save=True,
                            conf=0.5 #置信度阈值
                            )

def process_frame(img_path):
    #输入模型获取预测结果
    model = YOLO("./weights/yolo11n-pose.pt")
    results = model(img_path)
    for result in results:
        keypoints = result.keypoints
        for p in keypoints:
            list_p = p.data.tolist()
            print(list_p)
            return list_p

def sclae_img(img,percent):
    scale_percent = percent  # 百分比
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def angle_show(list_p,position,color,text,limb,img):  #显示位置，字体颜色，显示内容，显示肢体，投射图像
    p = get_keypoints(list_p)
    use_points = [p[limb[0]],p[limb[1]],p[limb[2]]]  #获取三个关键点，中间项为顶点
    angle = calculate_angle(use_points[0],use_points[1],use_points[2])
    angle = str(int(angle))
    cv2.putText(img,f"{text}:{angle}",position,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.6,thickness=1,color=color)  #显示角度
    # 显示图像

def get_norm_points(list_p):
    p = get_keypoints(list_p)
    p_norm = []
    x1 = (p[5][0] + p[6][0])/2
    y1 = (p[5][1] + p[6][1])/2
    x2 = (p[12][0] + p[11][0])/2
    y2 = (p[12][1] + p[11][1])/2
    #求身体长度
    long = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    for i in range(17):
        x3 = (p[i][0] - x2)/long
        y3 = (p[i][1] - y2)/long
        p_norm.append([x3,y3])
    return p_norm,long

def visualize_comparison(imgpath1, imgpath2):
    """可视化两张图片的归一化对比（基于髋关节中点对齐）"""
    # 获取关键点
    list_p1 = process_frame(imgpath1)
    list_p2 = process_frame(imgpath2)
    
    if not list_p1 or not list_p2:
        print("错误：无法检测到关键点")
        return
    
    p_norm1, torso_length1 = get_norm_points(list_p1)
    p_norm2, torso_length2 = get_norm_points(list_p2)
    
    # 创建画布
    h, w = 600, 600
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 画布中心作为髋关节中点位置
    center_x, center_y = w//2, h//2
    scale = 120  # 缩放系数
    
    # 获取归一化后的髋关节中点坐标（应该是0,0）
    # 在get_norm_points中，髋关节中点已经是原点(0,0)
    
    # 绘制图1的姿态（绿色）
    # 连接线段
    connections = [
        # 头部
        (0, 1), (0, 2), (1, 3), (2, 4),  # 头部连接
        # 上肢
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 手臂
        # 躯干
        (5, 11), (6, 12), (11, 12),  # 躯干
        # 下肢
        (11, 13), (12, 14), (13, 15), (14, 16)  # 腿
    ]
    
    # 绘制图1的连接线（绿色）
    for conn in connections:
        if conn[0] < len(p_norm1) and conn[1] < len(p_norm1):
            x1 = int(center_x + p_norm1[conn[0]][0] * scale)
            y1 = int(center_y + p_norm1[conn[0]][1] * scale)
            x2 = int(center_x + p_norm1[conn[1]][0] * scale)
            y2 = int(center_y + p_norm1[conn[1]][1] * scale)
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 绘制图1的关键点（绿色圆点）
    for i in range(len(p_norm1)):
        x = int(center_x + p_norm1[i][0] * scale)
        y = int(center_y + p_norm1[i][1] * scale)
        cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)
    
    # 绘制图2的连接线（红色，半透明效果使用实线）
    for conn in connections:
        if conn[0] < len(p_norm2) and conn[1] < len(p_norm2):
            x1 = int(center_x + p_norm2[conn[0]][0] * scale)
            y1 = int(center_y + p_norm2[conn[0]][1] * scale)
            x2 = int(center_x + p_norm2[conn[1]][0] * scale)
            y2 = int(center_y + p_norm2[conn[1]][1] * scale)
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 绘制图2的关键点（红色圆点）
    for i in range(len(p_norm2)):
        x = int(center_x + p_norm2[i][0] * scale)
        y = int(center_y + p_norm2[i][1] * scale)
        cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
    
    # 绘制髋关节中点（黄色标记）
    cv2.circle(canvas, (center_x, center_y), 8, (0, 255, 255), -1)
    
    # 添加文字信息
    cv2.putText(canvas, "Image 1", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "Image 2", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(canvas, f"body1: {torso_length1:.1f}px", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(canvas, f"body2: {torso_length2:.1f}px", (10, 145), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # 添加说明
    cv2.putText(canvas, "Both normalized to same hip center (scale using torso length)", 
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 显示结果
    cv2.imshow("Pose Comparison - Hip Center Alignment", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_time = time.time()
    img_path1 = "data/jpg/run_final3.png"
    img_path2 = "data/jpg/run_start4.png"
    visualize_comparison(img_path1,img_path2)
    # resized = sclae_img(img,80)
    # cv2.imshow("Image Window", resized)
    current_time = time.time()
    print(show_time(start_time,current_time))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
