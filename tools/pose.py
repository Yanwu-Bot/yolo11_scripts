import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
from utills.utill import *
from time_utils import show_time 

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

r_arm = [6,8,10]
l_arm = [5,7,9]
r_leg = [12,14,16]
l_leg = [11,13,15]

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
#输出对应点坐标

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

if __name__ == "__main__":
    start_time = time.time()
    list_p = process_frame(img_path)
    draw = Draw(img,list_p)
    draw.draw_select()
    resized = sclae_img(img,80)
    angle_show(list_p,(10,250),(0,0,255),"RightArm",r_arm,resized)
    angle_show(list_p,(10,150),(0,0,255),"LeftArm",l_arm,resized)
    angle_show(list_p,(10,200),(0,0,255),"RightLeg",r_leg,resized)
    angle_show(list_p,(10,100),(0,0,255),"LeftLeg",l_leg,resized)
    cv2.imshow("Image Window", resized)
    current_time = time.time()
    print(show_time(start_time,current_time))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
