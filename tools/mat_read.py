import scipy.io as sio
import pandas as pd
import os

def save_aqa7_list_to_file(mat_file_path, output_path=None, data_type='train'):
    """
    将 AQA-7 的 .mat 文件转换为表格文件（支持 txt、csv、xlsx）
    
    Args:
        mat_file_path: .mat 文件路径
        output_path: 输出文件路径（如果为 None，则自动生成）
        data_type: 'train' 或 'test'，用于自动生成文件名
    """
    # 加载数据
    data = sio.loadmat(mat_file_path)
    
    # 确定字段名
    if 'consolidated_train_list' in data:
        list_data = data['consolidated_train_list']
    elif 'consolidated_test_list' in data:
        list_data = data['consolidated_test_list']
    else:
        # 列出所有可用字段
        available = [k for k in data.keys() if not k.startswith('__')]
        print(f"可用字段: {available}")
        raise KeyError(f"未找到 consolidated_train_list 或 consolidated_test_list")
    
    # 转换为 DataFrame
    df = pd.DataFrame(list_data, columns=['action_class', 'sample_id', 'score'])
    df['action_class'] = df['action_class'].astype(int)
    df['sample_id'] = df['sample_id'].astype(int)
    
    # 添加动作名称列
    action_names = {
        1: 'diving',
        2: 'gymvault',
        3: 'skiing',
        4: 'snowboarding',
        5: 'sync_diving_3m',
        6: 'sync_diving_10m'
    }
    df['action_name'] = df['action_class'].map(action_names)
    
    # 自动生成输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
        output_path = f"{base_name}_list.txt"
    
    # 根据文件扩展名保存
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == '.csv':
        df.to_csv(output_path, index=False)
    elif ext == '.xlsx':
        df.to_excel(output_path, index=False)
    else:  # 默认 txt 格式（制表符分隔）
        df.to_csv(output_path, sep='\t', index=False)
    
    print(f"已保存到: {output_path}")
    print(f"总样本数: {len(df)}")
    print(f"分数范围: {df['score'].min():.2f} - {df['score'].max():.2f}")
    print(f"\n动作类别分布:")
    print(df['action_class'].value_counts().sort_index())
    
    return df

# 示例1：保存为 txt 文件
save_aqa7_list_to_file(
    mat_file_path='D:/Dataset/AQA-7/Split_4/split_4_train_list.mat',
    output_path='D:/Dataset/AQA-7/train_list.txt'
)

# 示例2：保存为 csv 文件
save_aqa7_list_to_file(
    mat_file_path='D:/Dataset/AQA-7/Split_4/split_4_test_list.mat',
    output_path='D:/Dataset/AQA-7/test_list.csv'
)
