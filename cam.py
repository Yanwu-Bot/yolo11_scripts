import ultralytics
from ultralytics import YOLO
import cv2
import time
import numpy as np
from angle import *

cTime=0
pTime=0
# 加载YOLOv11n-pose模型
model = YOLO("./weights/yolo11n-pose.pt")

# 定义关键点索引与名称的对应关系
dic_points = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}

'''
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
'''
#输出对应点坐标
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
   
def draw_leg(frame,p_list):
    p10=p_pos[10]
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
    p_m = (int((p12[0]+p11[0])/2),int((p12[1]+p11[1])/2))
    cv2.line(frame,p12,p14,(255,0,220),2)
    cv2.line(frame,p14,p16,(255,0,220),2)
    cv2.line(frame,p11,p13,(255,0,220),2)
    cv2.line(frame,p13,p15,(255,0,220),2)
    cv2.line(frame,p11,p12,(255,0,220),2)

def draw_point(p_pos):
    for p in p_pos:  
        circle_center = p
        # define the radius of the circle
        radius =5
        #  Draw a circle using the circle() Function
        cv2.circle(frame, circle_center, radius, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA) 
# 打开默认摄像头

def process_frame(img):
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #输入模型获取预测结果
    results = model(img_RGB)
    for result in results:
        keypoints = result.keypoints
        for p in keypoints:
            list_p = p.data.tolist()
            print(list_p)
        
            return img,list_p

cap = cv2.VideoCapture(0)
cap.open(0)

while cap.isOpened():
    success,frame = cap.read()
    if not success:
        print("error")
        break

    frame,list_p = process_frame(frame)
    # define the center of circle
     
    p_pos=get_keypoints(list_p)
    use_points = [p_pos[6],p_pos[8],p_pos[10]]
    angle = calculate_angle(use_points[0],use_points[1],use_points[2])
    angle = str(int(angle))
    # draw_point(p_pos)
    draw_head(frame,list_p)
    draw_body(frame,list_p)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,f"FPS:{int(fps)}",(30,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
    cv2.putText(frame,f"RightArm:{angle}",(10,200),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.6,thickness=2,color=(255,0,0))  #显示角度
    cv2.imshow('my_window',frame)
    if cv2.waitKey(1) in [ord('q'),27]:
        break
cap.release()
cv2.destroyAllWindows()