import json

# 读取JSON文件
json_file_path = 'D:/Dataset/posedive1/posedive/annotations/pose_finediv_all.json'  # 请替换为您的实际文件路径

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有帧的关键点信息
all_keypoints = []

# 如果data是列表，直接遍历
if isinstance(data, list):
    for frame_data in data:
        joints = frame_data.get('joints', [])
        all_keypoints.append(joints)
# 如果data是字典，可能需要根据实际情况调整
elif isinstance(data, dict):
    # 假设字典的值是帧数据列表
    for key, value in data.items():
        if isinstance(value, list):
            for frame_data in value:
                joints = frame_data.get('joints', [])
                all_keypoints.append(joints)

# 保存到新文件
output_file = 'keypoints_output.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_keypoints, f, indent=2, ensure_ascii=False)

print(f"共提取 {len(all_keypoints)} 帧的关键点数据")
print(f"每帧关键点数量: {len(all_keypoints[0]) if all_keypoints else 0}")
print(f"数据已保存到: {output_file}")

# 可选：同时保存为更易读的格式（每行一个列表）
output_txt_file = 'result/keypoints_output.txt'
with open(output_txt_file, 'w', encoding='utf-8') as f:
    for i, keypoints in enumerate(all_keypoints):
        f.write(f"帧{i+1}: {keypoints}\n")

print(f"文本格式数据已保存到: {output_txt_file}")