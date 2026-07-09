import pickle

# 读取 pickle 文件
file_path = 'D:/Dataset/HP-MCoRe-main/Annotations/FineDiving_coarse_annotation.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 提取数据为嵌套列表
nested_list = []

for idx, (key, value) in enumerate(data.items(), start=1):
    start_frame = value.get('start_frame', 'N/A')
    end_frame = value.get('end_frame', 'N/A')
    dive_score = value.get('dive_score', 'N/A')
    nested_list.append([idx, start_frame, end_frame, dive_score])

# 直接打印嵌套列表（方便复制）
print("nested_list = [")
for row in nested_list:
    print(f"    {row},")
print("]")

# 同时保存到文件
output_file = 'result/diving_data.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("nested_list = [\n")
    for row in nested_list:
        f.write(f"    {row},\n")
    f.write("]\n")

print(f"\n嵌套列表已保存到: {output_file}")
print(f"共 {len(nested_list)} 条数据")