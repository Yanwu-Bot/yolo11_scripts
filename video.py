#mpose 效果最好，n以上帧数会下降

import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
from angle import *


step=[0]
r_arm = [6,8,10]
l_arm = [5,7,9]
r_leg = [12,14,16]
l_leg = [11,13,15]

cTime=0
pTime=0
model = YOLO("./weights/yolo11l-pose.pt")

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

def draw_point(p_pos,frame):
    for p in p_pos:  
        circle_center = p
        # define the radius of the circle
        radius =5
        #  Draw a circle using the circle() Function
        cv2.circle(frame, circle_center, radius, (0, 150, 255), thickness=-1, lineType=cv2.LINE_AA) 

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

def predict():
    model = YOLO("./weights/yolo11s.pt")
    results = model.predict(
                            save=True,
                            conf=0.5 #置信度阈值
                            )

def angle_show(list_p,position,color,text,limb,img):  #显示位置，字体颜色，显示内容，显示肢体，投射图像
    p = get_keypoints(list_p)
    use_points = [p[limb[0]],p[limb[1]],p[limb[2]]]  #获取三个关键点，中间项为顶点
    angle = calculate_angle(use_points[0],use_points[1],use_points[2])
    angle = str(int(angle))
    cv2.putText(img,f"{text}:{angle}",position,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.6,thickness=1,color=color)  #显示角度
    # 显示图像

def get_P_X_Y(img,marks):
    img_Heigt = img.shape[0]
    img_Width = img.shape[1]
    for point,pose in enumerate(marks):
        x_pos = int(pose.x*img_Width)
        y_pos = int(pose.y*img_Heigt)
        print(f"point:{point},x_pos:{x_pos},y_pos:{y_pos}")

def change_detector(a,b):
    if a>b and step[-1] != 1:
        step.append(1)
        return True
    elif a<b and step[-1] != 2:
        step.append(2)
        return True

i = 0
def process_frame(img):
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    #输入模型获取预测结果
    results = model(img_RGB)
    global i
    for result in results:
        keypoints = result.keypoints
        
        for p in keypoints:
            list_p = p.data.tolist()
            print(list_p)
            
            # 显示角度
            angle_show(list_p, (10,20), (0,0,255), "RightArm", r_arm, img)
            angle_show(list_p, (10,40), (0,0,255), "LeftArm", l_arm, img)
            angle_show(list_p, (10,60), (0,0,255), "RightLeg", r_leg, img)
            angle_show(list_p, (10,80), (0,0,255), "LeftLeg", l_leg, img)
            
            # 获取关键点位置
            p_pos = get_keypoints(list_p)
            p15 = p_pos[15]
            p16 = p_pos[16]
            
            # 检测变化并计数
            if change_detector(p15, p16):
                i += 1
    cv2.putText(img, str(i), (10,100), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.6, thickness=2, color=(255,0,0))
    
    # 循环结束后再返回
    return img, list_p  # 注意：这里返回的是最后一个list_p

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


def generate_video(input_path):
    filehead = input_path.split('/')[-1]
    output_path = 'out-'+filehead
    output_path = './video/'+output_path
    print('视频开始处理',input_path)
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while(cap.isOpened()):
        success,frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数：',frame_count)

    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path,fourcc,fps,(int(frame_size[0]),int(frame_size[1])))

  
    try:
        while(cap.isOpened()):
            success,frame = cap.read()
            if not success:
                break

            try:
                frame,list_p = process_frame(frame)
                draw_select(frame,list_p)
  
                
            except:
                print('Finish')
                pass
                
            if success == True:
                out.write(frame)

    except:
        print('中断')
        pass
    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('Video saved',output_path)

input_path = 'data/run_woman.mp4'
generate_video(input_path)
print(step)