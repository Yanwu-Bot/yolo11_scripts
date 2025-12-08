import cv2
import numpy as np
from collections import deque
import ultralytics
from ultralytics import YOLO
from angle import *
import cv2
import numpy as np
from collections import deque
hist_ra= deque(maxlen=50)  # 右臂
hist_la = deque(maxlen=50) # 左臂
hist_rl= deque(maxlen=50)  # 右腿
hist_ll = deque(maxlen=50) # 左腿

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

#画出头部
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


#画出身体
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

#画出腿部
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

#画出关键点
def draw_point(p_pos,frame):
    for p in p_pos:  
        circle_center = p
        # define the radius of the circle
        radius =5
        #  Draw a circle using the circle() Function
        cv2.circle(frame, circle_center, radius, (0, 150, 255), thickness=-1, lineType=cv2.LINE_AA) 

#选择要显示内容
def draw_select(frame,list_p,d_h=True,d_b=True,d_l=True,pre=False,d_p=True):  #传入图像，关键点列表，显示头，显示身体，显示腿，显示关键点，预测
    if d_h:
        draw_head(frame,list_p)
    if d_b:
        draw_body(frame,list_p)
    if d_l:
        draw_leg(frame,list_p)
    if d_p:
        p_pos=get_keypoints(list_p)
        draw_point(p_pos,frame)
    if pre:
        predict(source=frame)

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

