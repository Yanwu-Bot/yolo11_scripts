#读取npz
import numpy as np

a = np.load("video_dataset/train.npz")
print(a.files)

array1 = a['X']
array2 = a['y']

# 打印数组
print(array1)
print(array2)