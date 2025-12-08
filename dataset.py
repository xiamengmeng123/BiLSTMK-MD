import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
from parser_my import args
import math

class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx  # 特征数据
        self.y = yy  # 标签数据
        self.tranform = transform # 数据变换（如果有的话）,对数据进行缩放

    def __getitem__(self, index):
        x1 = self.x[index]  # 获取指定索引的特征数据        
        y1 = self.y[index]  # 获取指定索引的标签数据
        if self.tranform != None:
            return self.tranform(x1), y1 # 如果有数据变换，则应用变换
            # return x1, y1
        return x1, y1  # 否则，直接返回数据
 
    def __len__(self):
        return len(self.x) # 返回数据集的长度
        print(f"len(self.x) is : {len(self.x)}") 

##如果某个数据里有异常值的用这个
# def normalize_column(data, col, norm_params):
#     """归一化指定列并返回归一化参数"""
#     if col == 's1':
#         median = data[col].median()
#         iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
#         norm_params[col] = {'type': 'robust', 'median': median, 'iqr': iqr}
#         data.loc[:,col] = (data[col] - median) / iqr  #使用.loc进行赋值
#     else:
#         min_val = data[col].min()
#         max_val = data[col].max()
#         norm_params[col] = {'type': 'minmax', 'min': min_val, 'max': max_val}
#         data.loc[:,col] = (data[col] - min_val) / (max_val - min_val)

# *********
# HMX-ligand用的归一化，适用于数据分布范围大
# *********
# def normalize_column(data, col, norm_params):
#     """特征自适应归一化"""
#     # 计算特征统计量
#     min_val = data[col].min()
#     max_val = data[col].max()
#     median = data[col].median()
#     q1 = data[col].quantile(0.25)
#     q3 = data[col].quantile(0.75)
#     iqr = q3 - q1
    
#     # 计算特征范围
#     feature_range = max_val - min_val
    
#     # 根据特征范围选择归一化策略
#     if feature_range > 2000:  # 大范围特征
#         # 鲁棒归一化
#         # if iqr < 1e-8:
#         #     iqr = data[col].std() or 1.0
#         data.loc[:, col] = (data[col] - median) / iqr
#         norm_type = 'robust_large'
#         params = {'center': median, 'range': iqr}
        
#     elif feature_range < 20:  # 小范围特征
#         # 增强信号：中心化+适度放大
#         center = median
#         scale_factor = 5.0 / max(feature_range, 0.5)  # 放大因子限制在10倍内
#         data.loc[:, col] = (data[col] - center) * scale_factor
#         norm_type = 'enhanced_small'
#         params = {'center': center, 'scale': scale_factor}
        
#     else:  # 中等范围特征
#         # 标准零中心归一化
#         if iqr < 1e-8:
#             iqr = data[col].std() or 1.0
#         data.loc[:, col] = (data[col] - median) / iqr
#         norm_type = 'robust_medium'
#         params = {'center': median, 'range': iqr}
    
#     # 保存参数
#     norm_params[col] = {
#         'type': norm_type,
#         'original_min': min_val,
#         'original_max': max_val,
#         **params
#     }

## *********
## protein用的归一化，也适用大多数数据（包括CH4）
## *********
def normalize_column(data, col, norm_params):
    min_val = data[col].min()
    max_val = data[col].max()
    median = data[col].median()
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = max(q3 - q1, 1e-8)  # 防止除零
    
    # --- 新策略：统一使用分位数缩放 ---
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 计算缩放后的范围（保留原始分布形态）
    scale_range = upper_bound - lower_bound
    if scale_range < 1e-8:
        scale_range = max(np.std(data[col]), 1e-8)  # 备用标准差
    
    # 统一归一化公式
    data.loc[:, col] = (data[col] - median) / scale_range
    
    # 保存参数
    norm_params[col] = {
        'type': 'robust_scale',
        'original_min': min_val,
        'original_max': max_val,
        'center': median,
        'scale': scale_range,
        'iqr': iqr
    }

def getData(corpusFile, sequence_length, batchSize):
    # 读取数据并预处理
    stock_data = pd.read_csv(corpusFile)
    stock_data.drop('time', axis=1, inplace=True)
    stock_data = stock_data.loc[:, (stock_data != 0).any(axis=0)]

    # 按时间顺序分割训练集、验证集和测试集（比例为1:9:90）
    total_size = len(stock_data)
    train_size = int(total_size * 0.08)  # 8%
    val_size = int(total_size * 0.02)    # 2%
    
    # 划分数据集
    train_data = stock_data.iloc[:train_size]
    val_data = stock_data.iloc[train_size:train_size + val_size]
    test_data = stock_data.iloc[train_size + val_size:]
    
    # 保存数据集
    train_data.to_csv("data/ready_train.csv", index=False)
    val_data.to_csv("data/ready_val.csv", index=False)
    test_data.to_csv("data/ready_test.csv", index=False)

    # 合并训练集和验证集，创建训练+验证集
    train_val_data = pd.concat([train_data, val_data], axis=0)
    train_val_data.to_csv("data/ready_train_val.csv", index=False)
    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}, 训练+验证集大小: {len(train_val_data)}")

    # 对特征进行归一化（只在训练集计算参数）
    norm_params = {}
    for col in train_data.columns:
        if col in args.s_columns:
            normalize_column(train_data, col, norm_params)
            normalize_column(val_data, col, norm_params)  # 使用训练集的参数
            normalize_column(test_data, col, norm_params)  # 使用训练集的参数
    
    # 保存归一化后的数据集
    train_data.to_csv("data/ready_guiyi_train.csv", index=False)
    val_data.to_csv("data/ready_guiyi_val.csv", index=False)
    test_data.to_csv("data/ready_guiyi_test.csv", index=False)

    # 生成时间窗口数据
    def create_dataset(df, seq_length):
        X, Y = [], []
        for i in range(len(df) - seq_length):
            X.append(df.iloc[i:i+seq_length].values.astype(np.float32))
            Y.append(df.iloc[i+seq_length].values.astype(np.float32))
        return np.array(X), np.array(Y)

    # 分别在训练集、验证集和测试集上生成数据
    trainX, trainY = create_dataset(train_data, sequence_length)
    valX, valY = create_dataset(val_data, sequence_length)
    testX, testY = create_dataset(test_data, sequence_length)

    # 构建DataLoader
    train_loader = DataLoader(Mydataset(trainX, trainY), batch_size=batchSize, shuffle=False)
    val_loader = DataLoader(Mydataset(valX, valY), batch_size=batchSize, shuffle=False)
    test_loader = DataLoader(Mydataset(testX, testY), batch_size=batchSize, shuffle=False)

    return norm_params, train_loader, val_loader, test_loader


def getData_for_unknown(corpusFile, sequence_length, batchSize):
    # 读取数据并预处理
    stock_data = pd.read_csv(corpusFile)
    
    # 保存时间列信息
    time_column = stock_data['time'].copy()
    stock_data.drop('time', axis=1, inplace=True)
    stock_data = stock_data.loc[:, (stock_data != 0).any(axis=0)]
    
    # 获取最后一行的时间
    last_time = time_column.iloc[-1]
    
    # 计算需要预测的未来时间长度
    # 假设时间值是连续的，我们可以通过最后一行的时间值来估算
    total_time_needed = last_time / 0.10  # 需要预测到总时间的10%
    
    # 计算时间间隔（假设时间列是等间隔的）
    if len(time_column) > 1:
        time_interval = time_column.diff().dropna().mean()
    else:
        # 如果只有一个时间点，使用默认间隔
        time_interval = 1.0
    
    # 计算需要添加的未来时间步数
    current_total_time = last_time
    future_time_steps = 0
    while current_total_time < total_time_needed:
        current_total_time += time_interval
        future_time_steps += 1
    
    print(f"最后时间: {last_time}, 需要预测到总时间: {total_time_needed}")
    print(f"时间间隔: {time_interval}, 需要添加的未来时间步数: {future_time_steps}")
    
    # 创建未来的零值数据
    if future_time_steps > 0:
        # 创建与原始数据列结构相同的零值DataFrame
        future_data = pd.DataFrame(
            np.zeros((future_time_steps, len(stock_data.columns))),
            columns=stock_data.columns
        )
        
        # 将未来数据添加到原始数据后面
        extended_data = pd.concat([stock_data, future_data], axis=0, ignore_index=True)
        
        # 创建未来的时间序列
        future_times = []
        current_time = last_time
        for i in range(future_time_steps):
            current_time += time_interval
            future_times.append(current_time)
        
        # 将时间列添加回数据
        extended_times = pd.concat([time_column, pd.Series(future_times)], axis=0, ignore_index=True)
        extended_data['time'] = extended_times
    else:
        extended_data = stock_data.copy()
        extended_data['time'] = time_column
    
    # 保存扩展后的数据
    extended_data.to_csv("data/ready_unknown_extended.csv", index=False)
    
    # 按时间顺序分割训练集、验证集和测试集（比例为1:9:90）
    total_size = len(extended_data)
    train_size = int(total_size * 0.01)  # 1%
    val_size = int(total_size * 0.09)    # 9%
    
    # 划分数据集
    train_data = extended_data.iloc[:train_size].drop('time', axis=1)
    val_data = extended_data.iloc[train_size:train_size + val_size].drop('time', axis=1)
    test_data = extended_data.iloc[train_size + val_size:].drop('time', axis=1)
    
    # 保存数据集
    train_data.to_csv("data/ready_unknown_train.csv", index=False)
    val_data.to_csv("data/ready_unknown_val.csv", index=False)
    test_data.to_csv("data/ready_unknown_test.csv", index=False)
    
    # 合并训练集和验证集，创建训练+验证集
    train_val_data = pd.concat([train_data, val_data], axis=0)
    train_val_data.to_csv("data/ready_unknown_train_val.csv", index=False)
    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}, 测试集大小: {len(test_data)}")
    
    # 对特征进行归一化（只在训练集计算参数）
    norm_params = {}
    for col in train_data.columns:
        if col in args.s_columns:
            normalize_column(train_data, col, norm_params)
            normalize_column(val_data, col, norm_params)  # 使用训练集的参数
            normalize_column(test_data, col, norm_params)  # 使用训练集的参数
    
    # 保存归一化后的数据集
    train_data.to_csv("data/ready_unknown_guiyi_train.csv", index=False)
    val_data.to_csv("data/ready_unknown_guiyi_val.csv", index=False)
    test_data.to_csv("data/ready_unknown_guiyi_test.csv", index=False)
    
    # 生成时间窗口数据
    def create_dataset(df, seq_length):
        X, Y = [], []
        for i in range(len(df) - seq_length):
            X.append(df.iloc[i:i+seq_length].values.astype(np.float32))
            Y.append(df.iloc[i+seq_length].values.astype(np.float32))
        return np.array(X), np.array(Y)
    
    # 分别在训练集、验证集和测试集上生成数据
    trainX, trainY = create_dataset(train_data, sequence_length)
    valX, valY = create_dataset(val_data, sequence_length)
    testX, testY = create_dataset(test_data, sequence_length)
    
    # 构建DataLoader
    train_loader = DataLoader(Mydataset(trainX, trainY), batch_size=batchSize, shuffle=False)
    val_loader = DataLoader(Mydataset(valX, valY), batch_size=batchSize, shuffle=False)
    test_loader = DataLoader(Mydataset(testX, testY), batch_size=batchSize, shuffle=False)
    
    # 保存测试集的时间信息（用于后续分析）
    test_times = extended_data['time'].iloc[train_size + val_size + sequence_length:].reset_index(drop=True)
    test_times.to_csv("data/ready_unknown_test_times.csv", index=False)
    
    return norm_params, train_loader, val_loader, test_loader, test_times
