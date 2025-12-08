
from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parser_my import args
from dataset import getData
import matplotlib.pyplot as plt


#设置目标损失阈值
target_val_loss = 0.01
#自定义损失函数，针对诸如数值分布存在异常高峰或低峰的预测，如s1
class WeightedLoss(nn.Module):
    def __init__(self, s1_index, weight=5.0):
        super().__init__()
        self.s1_idx = s1_index  # 假设s1是输出中的第0个特征
        self.weight = weight
    
    def forward(self, pred, target):
        loss = (pred - target)**2
        # 对s1的损失加权
        loss[:, self.s1_idx] *= self.weight  
        return loss.mean()

    
def train(optim_params=None):
    """支持参数动态注入的训练函数
    Args:
        optim_params (dict): 贝叶斯优化生成的参数字典，格式示例：
            {
                'hidden_size': 512, 
                'layers': 2,
                'dropout': 0.3,
                'lr': 0.001,
                'batch_size': 128,
                'sequence_length': 20
            }
    """
    if optim_params:
        print("\n[DEBUG] 正在应用优化参数:", optim_params)
        # 类型安全转换
        args.hidden_size = int(optim_params.get('hidden_size', args.hidden_size))
        args.layers = int(optim_params.get('layers', args.layers))
        args.dropout = float(optim_params.get('dropout', args.dropout))
        args.lr = float(optim_params.get('lr', args.lr))
        args.batch_size = int(optim_params.get('batch_size', args.batch_size))
        args.sequence_length = int(optim_params.get('sequence_length', args.sequence_length))

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
