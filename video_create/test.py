import pickle

with open('D:/Dataset/sprint/result/diff_dataset/pair_dataset_run.pkl', 'rb') as f:
    data = pickle.load(f)

print("总样本数:", len(data))
print("前5个样本对的 (ref, que, score_diff):")
for item in data[:5]:
    print(item['ref_name'], "->", item['que_name'], "| diff:", round(item['score_diff'], 2))
    print("  fea_diff shape:", item['fea_diff'].shape,
          "point_diff shape:", item['point_diff'].shape,
          "vector_diff shape:", item['vector_diff'].shape)