#读取npz数据
import numpy as np

a = np.load("D:/Dataset/sprint/result/video_point/run_2.npy")
print(a.files)

array1 = a['X']
array2 = a['y']

# 打印数组
print(array1)
print(array2)