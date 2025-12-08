from torch.autograd import Variable
import torch.nn as nn
import torch
import os

from LSTMModel import lstm
from parser_my import args
from dataset import getData

target_val_loss = 0.01  # 可选：达到这个就提前停


def train(optim_params=None, dataset=None):
    """
    统一版 train 函数：
    - 支持贝叶斯优化传入 optim_params
    - 支持根据 dataset 选择不同数据文件 / input_size / s_columns
    - 训练逻辑参考你“跑得更快”的那版：train+val、早停、保存最佳模型
    - 返回: {'val_loss': best_val_loss}
    """
    print("DEBUG: train() called", flush=True)

    # -------- 1. 备份原始 args，训练结束要恢复 --------
    original_params = {
        'hidden_size': args.hidden_size,
        'layers': args.layers,
        'dropout': args.dropout,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length,
        'input_size': args.input_size,
        'lambda_param': args.lambda_param,
        's_columns': args.s_columns,
        'corpusFile': args.corpusFile,
    }

    try:
        # -------- 2. 应用贝叶斯优化传进来的超参数（不动 input_size） --------
        if optim_params:
            print("\n[DEBUG] 正在应用优化参数:", optim_params)
            args.hidden_size = int(optim_params.get('hidden_size', args.hidden_size))
            args.layers = int(optim_params.get('layers', args.layers))
            args.dropout = float(optim_params.get('dropout', args.dropout))
            args.lr = float(optim_params.get('lr', args.lr))
            args.batch_size = int(optim_params.get('batch_size', args.batch_size))
            args.sequence_length = int(optim_params.get('sequence_length', args.sequence_length))

        # -------- 3. 根据 dataset 覆盖数据相关配置 --------
        dataset_configs = {
            'dataset1': {  # 二肽
                'corpus_file': 'data/bar/md0_new.csv',
                'lambda_param': 0,
                's_columns': ['s1', 's2'],
                'input_size': 2,
            },
            'dataset2': {  # HMX-NPBA
                'corpus_file': 'data/bar/md1_new.csv',
                'lambda_param': 1,
                's_columns': ['s0', 's1', 's2', 's4', 's6'],
                'input_size': 5,
            },
            'dataset3': {  # CH4
                'corpus_file': 'data/bar/md2_new.csv',
                'lambda_param': 2,
                's_columns': ['s5', 's7', 's8'],
                'input_size': 3,
            },
            'dataset4': {  # protein
                'corpus_file': 'data/bar/md3_new.csv',
                'lambda_param': 3,
                's_columns': ['s0', 's1', 's2', 's3', 's5'],
                'input_size': 5,
            },
        }

        if dataset and dataset in dataset_configs:
            cfg = dataset_configs[dataset]
            args.corpusFile = cfg['corpus_file']
            args.lambda_param = cfg['lambda_param']
            args.s_columns = cfg['s_columns']
            args.input_size = cfg['input_size']
            print(f"使用数据集 {dataset}: {args.corpusFile}, input_size={args.input_size}")
        else:
            # 不传 dataset 就用 parser_my 里的默认 corpusFile / input_size
            dataset = dataset or "default"
            print(f"使用默认数据集: {args.corpusFile}, input_size={args.input_size}")

        # -------- 4. 加载数据（和快版一样：train/val/test） --------
        norm_params, train_loader, val_loader, test_loader = getData(
            args.corpusFile,
            args.sequence_length,
            args.batch_size,
        )

        # -------- 5. 构建模型 & 损失 & 优化器 --------
        model = lstm(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            output_size=args.input_size,
            dropout=args.dropout,
            batch_first=args.batch_first,
        ).to(args.device)

        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # -------- 6. 训练循环（参考你“广泛用”的那版） --------
        best_val_loss = float('inf')
        best_model_state = None
        patience = 10
        patience_counter = 0

        # 为了加速 BO，这里可以限制最大 epoch 数
        max_epochs = min(args.epochs, 50)  # 你可以再改小一点，比如 30 / 20

        os.makedirs('txt', exist_ok=True)
        log_file = f'txt/lstm_opt_loss_log_{dataset}.txt'

        with open(log_file, 'w') as f:
            f.write("Epoch,Train_Loss,Val_Loss,Best_Val_Loss,Patience\n")

            for epoch in range(max_epochs):
                # ----- train -----
                model.train()
                train_loss = 0.0
                for data, label in train_loader:
                    data = data.squeeze(1).to(args.device)
                    label = label.to(args.device)

                    pred = model(data)
                    loss = criterion(pred, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # ----- val -----
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, label in val_loader:
                        data = data.squeeze(1).to(args.device)
                        label = label.to(args.device)
                        pred = model(data)
                        loss = criterion(pred, label)
                        val_loss += loss.item()

                avg_train_loss = train_loss / max(1, len(train_loader))
                avg_val_loss = val_loss / max(1, len(val_loader))

                # 更新最佳
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict()
                    patience_counter = 0
                    print(f"[{dataset}] Epoch {epoch+1}: 新的最佳模型，Val={best_val_loss:.4f}")
                    # 可选：保存一下
                    os.makedirs('models', exist_ok=True)
                    model_save_path = f'models/best_model_{dataset}.pt'
                    torch.save({'state_dict': best_model_state}, model_save_path)
                else:
                    patience_counter += 1

                # 打印 & 写日志
                msg = (f"[{dataset}] Epoch {epoch+1}, "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, "
                       f"Best Val: {best_val_loss:.4f}, "
                       f"Patience: {patience_counter}/{patience}")
                print(msg)
                f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},"
                        f"{best_val_loss:.4f},{patience_counter}\n")

                # 提前停止条件 1：早停
                if patience_counter >= patience:
                    print(f"[{dataset}] 早停触发，连续 {patience} 个 epoch 验证损失未改善")
                    break

                # 提前停止条件 2：达到目标损失
                if best_val_loss < target_val_loss:
                    print(f"[{dataset}] 达到目标验证损失 {target_val_loss}，提前结束")
                    break

        # -------- 7. 恢复 args & 返回给 BO 用的指标 --------
        for key, value in original_params.items():
            setattr(args, key, value)

        # 确保有值
        if best_model_state is None:
            best_val_loss = float('inf')

        print(f"[{dataset}] 训练完成，best_val_loss={best_val_loss:.4f}")
        return {'val_loss': best_val_loss}

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        # 出错也要恢复 args
        for key, value in original_params.items():
            setattr(args, key, value)
        return {'val_loss': float('inf')}









