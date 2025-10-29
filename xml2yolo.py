'''xml to txt'''
# import os
# import glob
# import xml.etree.ElementTree as ET


# file_path = "D:/work/Python/YOLOV11/VOC2007/VOC2007/yolotxt"

# # 修改文件权限为可写
# os.chmod(file_path, 0o777)
 
# def get_classes(classes_path):
#     with open(classes_path, encoding='utf-8',errors='ignore') as f:
#         class_names = f.readlines()
#     class_names = [c.strip() for c in class_names]
#     print("Classes loaded:", class_names)
#     return class_names, len(class_names)
 
 
# def convert(size, box):
#     dw = 1.0 / size[0]
#     dh = 1.0 / size[1]
#     x = (box[0] + box[1]) / 2.0
#     y = (box[2] + box[3]) / 2.0
#     w = box[1] - box[0]
#     h = box[3] - box[2]
#     x = x * dw
#     w = w * dw
#     y = y * dh
#     h = h * dh
#     return (x, y, w, h)
 
 
# def convert_xml_to_yolo(xml_root_path, txt_save_path, classes_path):
#     print("XML root path:", xml_root_path)
#     print("TXT save path:", txt_save_path)
#     print("Classes path:", classes_path)
 
#     if not os.path.exists(txt_save_path):
#         os.makedirs(txt_save_path)
#     print("Directory created:", txt_save_path)
 
#     xml_paths = glob.glob(os.path.join(xml_root_path, '*.xml'))
#     print("XML files found:", xml_paths)
 
#     classes, _ = get_classes(classes_path)
 
#     for xml_id in xml_paths:
#         print("Processing file:", xml_id)
#         txt_id = os.path.join(txt_save_path, os.path.basename(xml_id)[:-4] + '.txt')
#         txt = open(txt_id, 'w')
#         xml = open(xml_id, encoding='utf-8')
#         tree = ET.parse(xml)
#         root = tree.getroot()
#         size = root.find('size')
#         w = int(size.find('width').text)
#         h = int(size.find('height').text)
#         for obj in root.iter('object'):
#             difficult = 0
#             if obj.find('difficult') is not None:
#                 difficult = obj.find('difficult').text
#             cls = obj.find('name').text
#             print("Class found:", cls)
#             if cls not in classes or int(difficult) == 1:
#                 continue
#             cls_id = classes.index(cls)
#             xmlbox = obj.find('bndbox')
#             b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('xmax').text)),
#                  int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('ymax').text)))
#             box = convert((w, h), b)
#             txt.write(str(cls_id) + ' ' + ' '.join([str(a) for a in box]) + '\n')
#         txt.close()
#         print("TXT file created:", txt_id)
# if __name__ == '__main__':
#     # 用户输入XML文件路径和TXT文件存放路径
#     xml_root_path = r"D:/work/Python/YOLOV11/VOC2007/VOC2007/Annotations"
#     txt_save_path = r"D:/work/Python/YOLOV11/VOC2007/VOC2007/yolotxt"
#     classes_path = r"D:/work/Python/YOLOV11/labels.txt"
#     convert_xml_to_yolo(xml_root_path, txt_save_path, classes_path)


'''生成train.txt test.txt'''
import os
import random
import argparse
parser = argparse.ArgumentParser()
#xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
parser.add_argument('--xml_path', default='D:/work/Python/YOLOV11/VOC2007/VOC2007/Annotations', type=str, help='input xml label path')
#数据集的划分，地址选择自己数据下的ImageSets/Main
parser.add_argument('--txt_path', default='D:\work\Python\YOLOV11\VOC2007\VOC2007\ImageSets\Main', type=str, help='output txt label path')
opt = parser.parse_args()
trainval_percent = 0.8
train_percent = 0.9
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)
num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)
file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')
for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()