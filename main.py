import ultralytics
from ultralytics import YOLO
from PIL import Image


model = YOLO("runs\detect\\train4\weights\\best.pt")
results = model("D:\work\Python\YOLOV11\data\\000092.jpg")
for result in results:
# 绘制检测结果
    annotated_image = result.plot() # 返回带标注的 NumPy 数组
    image = Image.fromarray(annotated_image[..., ::-1]) # 转换为 RGB 格式

# 显示结果
    result.show()