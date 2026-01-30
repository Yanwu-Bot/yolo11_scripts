import cv2
import os
from ultralytics import solutions


def work(point_list,model_path,video_path,upangle=130,downangle=90):
    cap = cv2.VideoCapture(video_path) 

    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)) #获取视频宽，高，帧率
    out_path = "./runs/video/"+os.path.splitext(os.path.basename(video_path))[0]+".avi" #保存输出路径
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  

    gym = solutions.AIGym(
        model = model_path,    #yolo模型文件路径
        line_width=2,          #边界框线条宽度
        show=True,
        kpts=point_list,       #关键点列表,三个点构成角
        up_angle=upangle,      #向上姿势角度阈值
        down_angle=downangle   #向下姿势角度阈值
    )

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = gym(im0)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "data\pushup01.avi" #视频路径
    model_path = "weights/yolo11n-pose.pt"
    point_list = [5,7,9] 
    work(point_list,model_path,video_path)