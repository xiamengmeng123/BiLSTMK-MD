# yhrun -N 1 -p v100 --cpus-per-gpu=4 --gpus-per-node=1 python run_bayesion_fANOVA.py

import time
import numpy as np
import torch
from hyperopt import hp
# from hyperopt.pyll.base import scope
from bayesion_opt_random_for_fANOVA import TwoStageOptimization


# ===== 直接给定变量值 =====
n_samples = 80                   # 第一阶段每个数据集的采样数
n_evals = 50                   # 第二阶段每个优化器的评估次数
datasets = ['dataset1', 'dataset2', 'dataset3', 'dataset4']
top_k = 3                         # 保留的重要参数个数


print("=" * 60)
print("两阶段超参数优化系统")
print("=" * 60)
print(f"数据集: {datasets}")
print(f"第一阶段采样数: {n_samples}/数据集")
print(f"第二阶段评估数: {n_evals}/优化器")
print(f"保留重要参数数: {top_k}")
print("=" * 60)

# ===== 仅保留完整模式的搜索空间 =====
initial_space = {
    'hidden_size': hp.choice('hidden_size', [512, 1024]),
    'layers': hp.choice('layers', [1, 2]),
    'dropout': hp.uniform('dropout', 0.2, 0.6),
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),
    'batch_size': hp.choice('batch_size', [32, 64]),
    # 'sequence_length': scope.int(hp.quniform('sequence_length', 2, 20, 2)),
    'sequence_length': hp.choice('sequence_length', [2, 5, 8]),
}

# 创建优化器实例
print("\n初始化优化器...")

optimizer = TwoStageOptimization(
datasets=datasets,
initial_space=initial_space,
n_random_samples=n_samples
)

# ========== 第一阶段 ==========
print("\n" + "=" * 60)
print("第一阶段：随机采样与重要性分析")
print("=" * 60)

print("执行Sobol准随机采样...")
stage1_results = optimizer.stage1_random_sampling()
print(f"✓ 采样完成，共收集 {len(stage1_results)} 个样本")
# ========================== 不调用 optimizer.stage1_random_sampling() =====
# import pickle
# with open('results/stage1/random_sampling_results.pkl', 'rb') as f:
#     optimizer.stage1_results = pickle.load(f)


# # 如果后面代码还需要 stage1_results 这个“扁平 list”，可以自己展开一下：
# stage1_results = []
# for dataset, results in optimizer.stage1_results.items():
#     stage1_results.extend(results)

# print(f"✓ 载入 Stage1 结果，共 {len(stage1_results)} 个样本")
# ============================= 


print("\n执行fANOVA重要性分析...")
try:
    importance_scores = optimizer.fanova_analysis()
    print("✓ 重要性分析完成")

    # 显示重要性排名
    print("\n超参数重要性排名:")
    param_importance = []
    for param in ['hidden_size', 'layers', 'dropout', 'lr_log', 'batch_size', 'sequence_length']:
        if param in importance_scores:
            score = importance_scores[param].get('total', 0)
            param_importance.append((param.replace('_log', ''), score))

    param_importance.sort(key=lambda x: x[1], reverse=True)
    for i, (param, score) in enumerate(param_importance, 1):
        print(f"  {i}. {param:15s}: {score:.4f}")

except Exception as e:
    print(f"⚠ fANOVA分析失败: {e}")


print("\n生成重要性分析图表...")
try:
    optimizer.plot_importance_violin()
    print("✓ 图表已保存至 figures/stage1_importance_violin.png")
except Exception as e:
    print(f"⚠ 图表生成失败: {e}")

# ========== 空间缩减 ==========
print("\n缩小搜索空间...")
refined_space = optimizer.refine_search_space(top_k=top_k)
print(f"✓ 搜索空间已缩减至最重要的 {top_k} 个参数")

# ========== 第二阶段 ==========
print("\n" + "=" * 60)
print("第二阶段：多优化器效率对比")
print("=" * 60)

print("在缩小的空间内测试多个优化器...")
optimizer_comparison = optimizer.stage2_multi_optimizer_comparison(max_evals=n_evals)

print("\n优化器性能汇总:")
print("-" * 40)
for opt_name, results in optimizer_comparison.items():
    print(f"{opt_name:15s}: 最佳损失 = {results['best_loss']:.6f}")
    if 'convergence_speed' in results:
        speed = results['convergence_speed']
        print(
            f"{'':15s}  收敛速度: 90%={speed.get('90%', 'N/A')} iters, "
            f"95%={speed.get('95%', 'N/A')} iters"
        )

##=================
##加载pkl画图
# with open('results/stage2/optimizer_comparison.pkl', 'rb') as f:
#     optimizer.optimizer_results = pickle.load(f)

# #根据 optimizer.optimizer_results 构造一个 optimizer_comparison，方便后面打印汇总
# optimizer_comparison = {}
# for opt_name, res in optimizer.optimizer_results.items():
#     trials = res["trials"]
#     losses = [t["result"]["loss"] for t in trials]

#     optimizer_comparison[opt_name] = {
#         "best_loss": min(losses),
#     }

#     # 如果在保存时就已经算好了收敛速度，可以一并带上
#     if "convergence_speed" in res:
#         optimizer_comparison[opt_name]["convergence_speed"] = res["convergence_speed"]

# # 打印优化器性能汇总
# print("\n优化器性能汇总:")
# print("-" * 40)
# for opt_name, results in optimizer_comparison.items():
#     print(f"{opt_name:15s}: 最佳损失 = {results['best_loss']:.6f}")
#     if "convergence_speed" in results:
#         speed = results["convergence_speed"]
#         print(
#             f"{'':15s}  收敛速度: 90%={speed.get('90%', 'N/A')} iters, "
#             f"95%={speed.get('95%', 'N/A')} iters"
#         )
##=================

print(f"\n生成优化器对比图表...",flush=True)
try:
    optimizer.plot_optimizer_comparison_violin()
    print("✓ 图表已保存至 figures/stage2_optimizer_comparison.png")
except Exception as e:
    print(f"⚠ 图表生成失败: {e}")





# ========== 生成最终报告 ==========
print("\n" + "=" * 60)
print("生成最终报告")
print("=" * 60)

report = optimizer.generate_final_report()

print("\n优化总结:")
print("-" * 40)
print(f"总采样数: {report['stage1']['n_samples']}")
print(f"最重要的参数: {', '.join(report['stage1']['important_params'])}")

# 找出最佳优化器
best_optimizer = None
best_loss = float('inf')
for opt_name, perf in report['stage2']['optimizer_performance'].items():
    if perf['best_loss'] < best_loss:
        best_loss = perf['best_loss']
        best_optimizer = opt_name

print(f"最佳优化器: {best_optimizer} (损失: {best_loss:.6f})")

print("\n" + "=" * 60)
print("优化完成！")
print("=" * 60)

print("\n结果文件位置:")
print("  - 第一阶段结果: results/stage1/")
print("  - 第二阶段结果: results/stage2/")
print("  - 可视化图表: figures/")
print("  - 最终报告: results/final_optimization_report.json")

print(f"脚本全部跑完", flush=True)





