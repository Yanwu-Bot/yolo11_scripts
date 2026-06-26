import pickle

# 1. 打开文件：使用 'rb' 模式，表示以二进制方式读取
# 记得把 'your_file.pkl' 换成你的实际文件路径
file_path = 'D:/Dataset/HP-MCoRe-main/Annotations/FineDiving_coarse_annotation.pkl'

with open(file_path, 'rb') as f:
    # 2. 读取数据：pickle.load() 会帮你还原出里面的 Python 对象
    data = pickle.load(f)

# 3. 查看数据：现在 data 就是你之前看到的那个包含16个关键点的数据结构
print(data)