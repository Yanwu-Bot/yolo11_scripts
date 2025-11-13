
from ultralytics import YOLO
 
if __name__ == '__main__':
 
    model = YOLO('yolo11s-pose.yaml')  # load a pretrained model (recommended for training)
    # Train the model
    model.train(data='yolov11/ultralytics/cfg/datasets/coco-pose_my.yaml', epochs=2, imgsz=640)

# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# # Train the model
# results = model.train(data="yolov11\\ultralytics\\cfg\\datasets\\coco-pose_my.yaml", epochs=1, imgsz=640)