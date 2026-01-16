import os
import random
import xml.etree.ElementTree as ET
from shutil import copyfile
from tqdm import tqdm

def xml_to_txt(xml_file, output_dir):
    """Converts a VOC XML annotation file to YOLO TXT format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_width = int(root.find(".//size/width").text)
    image_height = int(root.find(".//size/height").text)

    output_txt_path = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + ".txt")

    with open(output_txt_path, "w") as txt_file:
        for obj in root.findall(".//object"):
            class_name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2.0 / image_width
            y_center = (ymin + ymax) / 2.0 / image_height
            bbox_width = (xmax - xmin) / image_width
            bbox_height = (ymax - ymin) / image_height

            # Assuming classes are indexed starting from 0
            class_id = class_map.get(class_name, -1)
            if class_id == -1:
                continue

            txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

def prepare_datasets(voc_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    """Prepares YOLO datasets from VOC format."""
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("Train and validation ratios must sum to less than 1.0")

    annotations_dir = os.path.join(voc_dir, "Annotations")
    images_dir = os.path.join(voc_dir, "JPEGImages")

    images_output_dir = os.path.join(output_dir, "images")
    labels_output_dir = os.path.join(output_dir, "labels")

    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]

    # Shuffle and split the dataset
    random.shuffle(xml_files)
    train_count = int(len(xml_files) * train_ratio)
    val_count = int(len(xml_files) * val_ratio)
    test_count = len(xml_files) - train_count - val_count

    train_files = xml_files[:train_count]
    val_files = xml_files[train_count:train_count + val_count]
    test_files = xml_files[train_count + val_count:]

    dataset_splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    split_files = {}
    for split, files in dataset_splits.items():
        split_file_path = os.path.join(output_dir, f"{split}.txt")
        split_files[split] = open(split_file_path, "w")

        for xml_file in tqdm(files, desc=f"Processing {split} set"):
            xml_path = os.path.join(annotations_dir, xml_file)
            image_name = os.path.splitext(xml_file)[0] + ".jpg"
            image_path = os.path.join(images_dir, image_name)

            # Copy image to images directory
            output_image_path = os.path.join(images_output_dir, image_name)
            copyfile(image_path, output_image_path)

            # Convert XML to YOLO TXT and save to labels directory
            xml_to_txt(xml_path, labels_output_dir)

            # Write the image path to the corresponding split file
            split_files[split].write(f"{output_image_path}\n")

    # Close split files
    for split_file in split_files.values():
        split_file.close()

def generate_class_map(class_list):
    """Generates a class map from a list of class names."""
    return {class_name: idx for idx, class_name in enumerate(class_list)}

# Example usage
# 设置label
#class_list = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]  # Replace with your list of classes
class_list = ["dog","cat"]
class_map = generate_class_map(class_list)

# Set input and output directories
# 设置数据集位置和输出位置
voc_dir = r"D:\work\\Python\\YOLOV11\\VOCDC"
output_dir = r"D:\work\\Python\\YOLOV11\\VOCDC"

# Set dataset split ratios
# 设置训练集验证集比例
train_ratio = 0.7  # Adjust as needed
val_ratio = 0.2    # Adjust as needed

# Run the script
prepare_datasets(voc_dir, output_dir, train_ratio=train_ratio, val_ratio=val_ratio)
