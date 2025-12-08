from hyperopt import hp, fmin, tpe, Trials,space_eval
import numpy as np
import torch
from train_bayesion import train  # 导入修改后的训练函数
import json
import matplotlib.pyplot as plt
from bayesion_opt_random_for_fANOVA import _hybrid_optimization

def plot_convergence(trials):
    """绘制贝叶斯优化收敛曲线"""
    # 提取所有试验的损失值
    losses = [trial['result']['loss'] for trial in trials.trials]
    
    # 计算累积最小值（展示最佳损失变化）
    cumulative_min = np.minimum.accumulate(losses)
    
    # 创建画布
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(losses, 'bo-', label='Single experiment loss')
    plt.plot(cumulative_min, 'r--', linewidth=2, label='Best validation loss')
    
    # 添加标注
    plt.xlabel("Trials", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.title("Bayesian Optimization Convergence", fontsize=14)
    plt.legend()
    plt.grid(False)
    
    # 保存图片
    plt.savefig('figure/convergence_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


# space = {
#     'hidden_size': hp.choice('hidden_size', [512,1024]), 
#     'layers': hp.choice('layers', [1, 2, 3]),
#     'dropout': hp.choice('dropout', [0.1,0.2, 0.5]),  
#     'lr': hp.choice('lr', [0.01,0.001,0.0001]),  # 原默认0.0001
#     'batch_size': hp.choice('batch_size', [32, 64]),
#     'sequence_length': hp.choice('sequence_length', [2,5,8])  
# }

#缩小空间
space = {
    # 隐层单元数
    'hidden_size': hp.choice('hidden_size', [1024]),  
    
    # 层数
    'layers': hp.choice('layers', [1]),  
    
    # Dropout：提高下限并扩大范围
    'dropout': hp.uniform('dropout', 0.2, 0.6),  
    
    # 学习率：缩小范围并改用对数分布
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),  
    
    # Batch_size
    'batch_size': hp.choice('batch_size', [64]),  
    
    # 序列长度：限制最大长度
    'sequence_length': hp.choice('sequence_length', [2])  
}


def objective(params):
    # 类型转换保证安全
    params = {
        'hidden_size': int(params['hidden_size']),
        'layers': int(params['layers']),
        'dropout': float(params['dropout']),
        'lr': float(params['lr']),
        'batch_size': params['batch_size'],
        'sequence_length': int(params['sequence_length'])
    }

    
    # 执行训练并获取验证损失
    result = train(optim_params=params)
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {'loss': result['val_loss'], 'status': 'ok'}

def run_optimization():
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=_hybrid_optimization(
        max_evals=50,
        )
    )

        # 新增调用 ------------------
    plot_convergence(trials)  # 生成收敛曲线图
    
    
    # 转换参数类型
    best_params = space_eval(space, best)  # 将索引转换为实际值
    best_params = {
        k: int(v) if isinstance(v, np.integer) else 
           float(v) if isinstance(v, np.floating) else v 
        for k, v in best_params.items()
    }
    
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)  # 现在可正常序列化


    
    return best


if __name__ == "__main__":
    final_params = run_optimization()
    print("最终优化参数:", final_params)   
    
    # 用最优参数执行最终训练
    print("\n使用最优参数进行最终训练...")
    train(optim_params=final_params)


