#yhrun -N 1 --gpus-per-node=1 -p v100 python train_lstm_kan.py
from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parser_my import args
from dataset import getData
import matplotlib.pyplot as plt


#设置目标损失阈值
target_loss = 0.01
#自定义损失函数，针对诸如数值分布存在异常高峰或低峰的预测，如s1
class WeightedLoss(nn.Module):
    def __init__(self, feature_weights):
        super().__init__()
        self.feature_weights = torch.tensor(feature_weights, dtype=torch.float32)

    def forward(self, pred, target):
        loss = (pred - target) ** 2
        # 按特征加权
        weighted_loss = loss * self.feature_weights.to(pred.device)
        return weighted_loss.mean()

class AdaptiveWeightedLoss(nn.Module):
    def __init__(self, feature_names, data_loader, 
                 high_mean_threshold=10000, high_mean_weight=0.00001,
                 low_mean_threshold=20, low_mean_weight=1000.0,
                 default_weight=1.0):
        """
        根据特征均值自动调整权重的损失函数
        
        参数:
            feature_names: 特征名称列表
            data_loader: 数据加载器，用于计算特征均值
            high_mean_threshold: 高均值阈值，均值绝对值大于此值的特征权重设为high_mean_weight
            high_mean_weight: 高均值特征的权重
            low_mean_threshold: 低均值阈值，均值绝对值小于此值的特征权重设为low_mean_weight
            low_mean_weight: 低均值特征的权重
            default_weight: 默认权重
        """
        super().__init__()
        
        # 计算每个特征的均值
        feature_means = self.calculate_feature_means(data_loader, len(feature_names))
        
        # 构建权重向量
        weights = []
        for i, name in enumerate(feature_names):
            mean_abs = abs(feature_means[i])
            if mean_abs > high_mean_threshold:
                weights.append(high_mean_weight)
                print(f"Feature '{name}' has high mean ({feature_means[i]:.2f}), weight set to {high_mean_weight}")
            elif mean_abs < low_mean_threshold:
                weights.append(low_mean_weight)
                print(f"Feature '{name}' has low mean ({feature_means[i]:.2f}), weight set to {low_mean_weight}")
            else:
                weights.append(default_weight)
                print(f"Feature '{name}' has moderate mean ({feature_means[i]:.2f}), weight set to {default_weight}")
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Final weights: {list(zip(feature_names, self.weights.tolist()))}")
    
    def calculate_feature_means(self, data_loader, num_features):
        """计算每个特征的均值"""
        # 初始化累加器
        sums = torch.zeros(num_features)
        counts = torch.zeros(num_features)
        
        # 遍历数据加载器计算均值
        for batch_x, batch_y in data_loader:
            # 计算每个特征的均值
            batch_sums = batch_y.sum(dim=0) if len(batch_y.shape) > 1 else batch_y.sum()
            batch_counts = torch.tensor(batch_y.shape[0])
            
            sums += batch_sums
            counts += batch_counts
        
        # 计算总体均值
        feature_means = sums / counts
        
        return feature_means.numpy()
    
    def forward(self, pred, target):
        # 计算每个特征的MSE
        mse_per_feature = (pred - target) ** 2
        
        # 应用权重
        weighted_mse = mse_per_feature * self.weights.to(pred.device)
        
        # 返回加权平均
        return weighted_mse.mean()
    
    
# def train():
#     # 创建lstm模型实例，并将其移动到指定的设备（CPU或GPU）
#     model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=args.input_size, dropout=args.dropout, batch_first=args.batch_first )
#     model.to(args.device)
#     norm_parmas, train_loader, test_loader = getData(args.corpusFile,args.sequence_length,args.batch_size) # 获取预处理后的数据
#     ## *********
#     ## protein适用的损失函数，适用于数据分布范围大
#     ## *********
#     # criterion = AdaptiveWeightedLoss(
#     #     feature_names=args.s_columns,
#     #     data_loader=train_loader,
#     #     high_mean_threshold=10000,  # 均值绝对值大于10000的特征
#     #     high_mean_weight=0.00001,   # 权重设为0.00001
#     #     low_mean_threshold=20,      # 均值绝对值小于20的特征
#     #     low_mean_weight=1000.0,       # 权重设为10.0
#     #     default_weight=1.0          # 其他特征权重设为1.0
#     # )

#     ## *********
#     ## HMX-ligand适用的损失函数，适用于数据分布范围大
#     ## *********
#     criterion = nn.HuberLoss()
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001
    
#     # 初始化列表用于记录每个epoch的总损失
#     epoch_losses = []
    
#     # 打开一个名为 'lstm_loss_log.txt' 的文本文件，用于记录每个epoch的损失值
#     with open('txt/lstm_loss_log.txt', 'w') as f:
#         for i in range(args.epochs):  # 循环遍历每个epoch
#             total_loss = 0  # 初始化当前epoch的总损失
#             for idx, (data, label) in enumerate(train_loader):
#                 if args.useGPU: # 如果使用GPU
#                     #print(f"data.shape is : {data.shape}") #(64,5,2)
#                     data1 = data.squeeze(1).cuda() # 删除data张量中的第一个维度，并将其移动到GPU
#                     #print(f"data1.shape is : {data1.shape}") #(64,5,2)
#                     pred = model(Variable(data1).cuda()) # 将data1封装成Variable并传入模型进行前向传播，得到预测值
#                     # print(pred.shape) 
#                     # pred = pred[1,:,:]  # 这里取pred的第二个维度的数据作为最终预测结果
#                     #label = label.unsqueeze(1).cuda()  # 将标签数据添加一个维度并移动到GPU (64,1)
#                     label = label.cuda()
#                     # print(label.shape)
#                 else:  # 如果使用CPU
#                     data1 = data.squeeze(1) # 删除data张量中的第一个维度
#                     pred = model(Variable(data1))   # 将data1封装成Variable并传入模型进行前向传播，得到预测值
#                     # pred = pred[1, :, :]  # 这里取pred的第二个维度的数据作为最终预测结果
#                     #label = label.unsqueeze(1) # 将标签数据添加一个维度
#                     label = label
#                 loss = criterion(pred, label) # 计算当前batch的损失值
#                 optimizer.zero_grad() # 清空优化器的梯度
#                 loss.backward()   # 反向传播，计算梯度
#                 optimizer.step() # 更新模型参数
#                 total_loss += loss.item()   # 累加当前batch的损失值到total_loss
            
#             # 记录每个epoch的总损失
#             epoch_losses.append(total_loss)
            
#             # 在终端输出第多少轮和对应的loss
#             print(f'Epoch {i+1}, Loss: {total_loss}')
            
#             # 将损失写入文件
#             f.write(f'Epoch {i+1}, Loss: {total_loss}\n')

#             # 检查是否达到目标损失值
#             if total_loss < target_loss:
#                 print(f'目标损失值 {target_loss} 达成，提前结束训练。')
#                 break  #提前结束训练
            
#             if i % 10 == 0: # 每10个epoch保存一次模型
#                 torch.save({'state_dict': model.state_dict()}, args.save_file) # 保存模型的状态字典到指定文件
#                 print('第%d epoch，保存模型' % i) # 打印当前epoch信息，表示模型已经保存
        
#         torch.save({'state_dict': model.state_dict()}, args.save_file) # 在训练结束后，保存最终模型

def train():
    # 创建lstm模型实例，并将其移动到指定的设备（CPU或GPU）
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=args.input_size, dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device)
    norm_parmas, train_loader, val_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size) # 获取预处理后的数据
    
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001
    
    # 初始化变量用于跟踪最佳模型和早停
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 20  # 早停耐心值，可以设置为参数
    
    # 初始化列表用于记录每个epoch的损失
    train_losses = []
    val_losses = []
    
    # 打开一个名为 'lstm_loss_log.txt' 的文本文件，用于记录每个epoch的损失值
    with open('txt/lstm_loss_log.txt', 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Best_Val_Loss,Patience\n")
        
        for i in range(args.epochs):  # 循环遍历每个epoch
            # 训练阶段
            model.train()
            train_loss = 0
            for idx, (data, label) in enumerate(train_loader):
                if args.useGPU: # 如果使用GPU
                    data1 = data.squeeze(1).cuda()
                    pred = model(Variable(data1).cuda())
                    label = label.cuda()
                else:  # 如果使用CPU
                    data1 = data.squeeze(1)
                    pred = model(Variable(data1))
                    label = label
                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for idx, (data, label) in enumerate(val_loader):
                    if args.useGPU: # 如果使用GPU
                        data1 = data.squeeze(1).cuda()
                        pred = model(Variable(data1).cuda())
                        label = label.cuda()
                    else:  # 如果使用CPU
                        data1 = data.squeeze(1)
                        pred = model(Variable(data1))
                        label = label
                    loss = criterion(pred, label)
                    val_loss += loss.item()
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # 检查是否为最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f'新的最佳模型，验证损失: {best_val_loss:.4f}')
                # 保存最佳模型
                torch.save({'state_dict': best_model_state}, args.save_file)
            else:
                patience_counter += 1
            
            # 在终端输出第多少轮和对应的loss
            print(f'Epoch {i+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Best Val: {best_val_loss:.4f}, Patience: {patience_counter}/{patience}')
            
            # 将损失写入文件
            f.write(f'{i+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{best_val_loss:.4f},{patience_counter}\n')

            # 早停检查
            if patience_counter >= patience:
                print(f'早停触发，连续 {patience} 个epoch验证损失没有改善。')
                break
        
        # 最终保存最佳模型
        torch.save({'state_dict': best_model_state}, args.save_file)
        print(f'训练完成，最佳验证损失: {best_val_loss:.4f}，模型已保存至 {args.save_file}')

train()
