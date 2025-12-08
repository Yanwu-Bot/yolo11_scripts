#mpose 效果最好，n以上帧数会下降

import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
from angle import *
from utill import *
from collections import deque

data_buffer = deque(maxlen=50)
step=[0]
r_arm = [6,8,10]
l_arm = [5,7,9]
r_leg = [12,14,16]
l_leg = [11,13,15]

cTime=0
pTime=0
model = YOLO("./weights/yolo11l-pose.pt")

#检测腿部交替变化
def change_detector(a,b):
    if a>b and step[-1] != 1:
        step.append(1)
        return True
    elif a<b and step[-1] != 2:
        step.append(2)
        return True

i = 0
#帧处理函数，对每帧画面进行处理
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
            angle_ra = angle_show(list_p, (10,20), (0,0,255), "RightArm", r_arm, img)
            angle_la = angle_show(list_p, (10,40), (0,0,255), "LeftArm", l_arm, img)
            angle_rl = angle_show(list_p, (10,60), (0,0,255), "RightLeg", r_leg, img)
            angle_ll = angle_show(list_p, (10,80), (0,0,255), "LeftLeg", l_leg, img)
            draw_direct_plot(img,hist_ra ,int(angle_ra), pos=(img.shape[1]-280, 50), label="angle_ra")
            draw_direct_plot(img,hist_la ,int(angle_la), pos=(img.shape[1]-280, 300), label="angle_la")
            draw_direct_plot(img,hist_rl ,int(angle_rl), pos=(img.shape[1]-560, 50), label="angle_rl")
            draw_direct_plot(img,hist_ll ,int(angle_ll), pos=(img.shape[1]-560, 300), label="angle_ll")
            # 获取关键点位置
            p_pos = get_keypoints(list_p)
            p13 = p_pos[13]
            p14 = p_pos[14]
            p15 = p_pos[15]
            p16 = p_pos[16]
            angle1,angle2 = get_start_angle(img,p13,p14,p15,p16)
            #cv2.putText(img,f"RIGHT_START:{str(int(angle1))}",(10,120),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.6,thickness=1,color=(255,255,255))
            #scv2.putText(img,f"LEFT_START:{str(int(angle2))}",(10,140),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.6,thickness=1,color=(255,255,255))
            # 检测变化并计数
            if change_detector(p15, p16):
                i += 1
            show_start(img,angle1,angle2)     
    cv2.putText(img, str(i), (10,100), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.6, thickness=2, color=(255,0,0))
    
    # 循环结束后再返回
    return img, list_p  # 注意：这里返回的是最后一个list_p



#显示蹬起角度,不一定好用
def get_start_angle(img,p13,p14,p15,p16):
    img_x = img.shape[0]
    img_y = img.shape[1]
    y1 = p15[1]
    y2 = p16[1]
    y1 = int(y1)
    y2 = int(y2)
    angle1 = 180 - calculate_angle(p14, p16, (img_x,y2)) #右
    angle2 = 180 - calculate_angle(p13, p15, (img_x,y1)) #左
    return angle1,angle2

#显示起蹬角度 
def show_start(img,angle1,angle2):
    a = step[-1]
    #左脚先迈出
    if a == 1: 
       cv2.putText(img,f"LEFT_START:{str(int(angle2))}",(10,140),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.6,thickness=1,color=(255,255,255)) 
    elif a == 0 or a ==2:
        cv2.putText(img,f"RIGHT_START:{str(int(angle1))}",(10,120),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.6,thickness=1,color=(255,255,255))

#生成视频函数
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
                print('error')
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
#输入视频路径
input_path = 'data/run_woman.mp4'
generate_video(input_path)
print(step)

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
   