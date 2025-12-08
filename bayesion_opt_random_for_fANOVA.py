# # 两阶段超参数优化系统

# ## 系统概述

# #本系统实现了一个两阶段的超参数优化框架：

# ### 第一阶段：随机采样与重要性分析
# - 使用Sobol准随机序列进行均匀采样（80次/数据集）
# - 基于fANOVA进行超参数重要性分析
# - 生成方差贡献小提琴图，识别关键超参数

# ### 第二阶段：缩小空间与多优化器对比
# - 基于重要性分析缩小搜索空间
# - 比较多种优化器（TPE、Random、Hybrid等）
# - 生成优化器效率对比小提琴图

# ## 核心特性

# 1. **准随机采样**：使用Sobol序列实现更均匀的参数空间探索
# 2. **fANOVA分析**：量化各超参数对模型性能的贡献度
# 3. **空间缩减**：基于重要性自动缩小搜索空间
# 4. **多优化器对比**：评估不同优化策略的效率
# 5. **可视化分析**：生成详细的分析图表


from hyperopt import hp, fmin, tpe, rand, Trials, space_eval,STATUS_OK
from hyperopt.pyll.stochastic import sample as hp_sample
import numpy as np
# 兼容老库用的 np.float / np.int 等
if not hasattr(np, "float"):
    np.float = float
import torch
from train_bayesion_fANOVA import train  
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc  # 用于Sobol准随机采样
import pandas as pd
from fanova import fANOVA
import pickle
import os
from typing import Dict, List, Tuple
from hyperopt.pyll.base import scope
import time
from fanova import fANOVA
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TwoStageOptimization:
    """两阶段超参数优化框架"""
    
    def __init__(self, datasets: List[str], initial_space: Dict, n_random_samples: int = 80):
        """
        Args:
            datasets: 数据集列表
            initial_space: 初始超参空间
            n_random_samples: 每个数据集的随机采样数
        """
        self.datasets = datasets
        self.initial_space = initial_space
        self.n_random_samples = n_random_samples
        self.stage1_results = {}
        self.importance_scores = {}
        self.refined_space = None
        
        # 创建结果目录
        os.makedirs('results/stage1', exist_ok=True)
        os.makedirs('results/stage2', exist_ok=True)
        os.makedirs('figures', exist_ok=True)
    
    def sobol_sampling(self, n_samples: int) -> np.ndarray:
        """使用Sobol序列进行准随机采样

        Args:
            n_samples: 采样数量

        Returns:
            采样点列表，每个元素是一个 param dict
        """
        # 定义每个超参数的搜索配置
        param_config = {
            'hidden_size': {
                'type': 'choice',
                'values': [512, 1024],
            },
            'layers': {
                'type': 'choice',
                'values': [1, 2],
            },
            'dropout': {
                'type': 'uniform',
                'low': 0.2,
                'high': 0.6,
            },
            'lr': {
                'type': 'loguniform',
                'low': 1e-5,
                'high': 1e-2,
            },
            'batch_size': {
                'type': 'choice',
                'values': [32, 64],
            },
            'sequence_length': {
                'type': 'choice',
                'values': [2, 5, 8],
            },
        }

        param_names = list(param_config.keys())
        n_dims = len(param_names)

        # 创建Sobol采样器（[0, 1) 上的 quasi-random 点）
        sampler = qmc.Sobol(d=n_dims, scramble=True)
        samples = sampler.random(n=n_samples)

        scaled_samples = []

        for sample in samples:
            scaled_point = {}
            for i, param in enumerate(param_names):
                cfg = param_config[param]
                u = float(sample[i])  # [0, 1) 上的数

                if cfg['type'] == 'choice':
                    values = cfg['values']
                    idx = int(u * len(values))
                    # 理论上 u<1，但稳妥起见防一下边界
                    if idx == len(values):
                        idx -= 1
                    scaled_point[param] = values[idx]

                elif cfg['type'] == 'uniform':
                    low, high = cfg['low'], cfg['high']
                    scaled_point[param] = low + u * (high - low)

                elif cfg['type'] == 'loguniform':
                    low, high = cfg['low'], cfg['high']
                    scaled_point[param] = np.exp(
                        np.log(low) + u * (np.log(high) - np.log(low))
                    )
                else:
                    raise ValueError(f"Unknown param type: {cfg['type']}")

            scaled_samples.append(scaled_point)

        return scaled_samples

    
    def stage1_random_sampling(self):
        """第一阶段：随机/准随机采样用于重要性分析"""
        print("="*50)
        print("Stage 1: 随机采样与超参重要性分析")
        print("="*50)
        
        all_results = []
        
        for dataset in self.datasets:
            print(f"\n处理数据集: {dataset}")
            dataset_results = []
            
            # 使用Sobol准随机采样
            samples = self.sobol_sampling(self.n_random_samples)
            
            for i, params in enumerate(samples):
                
                print(f"  采样 {i+1}/{self.n_random_samples}: {params}")
                
                try:
                    # 修改train函数以接受dataset参数
                    print(f"train函数开始接收 ...。。。。。。。。", flush=True)
                    result = train(optim_params=params, dataset=dataset)
                    print(f"train函数接收成功 ...。。。。。。。。", flush=True)
                    
                    # 记录结果
                    dataset_results.append({
                        'params': params,
                        'val_loss': result['val_loss'],
                        'dataset': dataset
                    })
                    
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"    训练失败: {e}")
                    dataset_results.append({
                        'params': params,
                        'val_loss': float('inf'),
                        'dataset': dataset
                    })
            
            self.stage1_results[dataset] = dataset_results
            all_results.extend(dataset_results)
        
        # 保存第一阶段结果
        with open('results/stage1/random_sampling_results.pkl', 'wb') as f:
            pickle.dump(self.stage1_results, f)
        
        return all_results
    
    def fanova_analysis(self):
        """使用 fANOVA 进行超参数重要性分析（先归一化到 [0, 1] 再分析）"""
        print("\n执行fANOVA重要性分析...")

        # ===== 1. 收集 Stage1 数据 =====
        configs = []
        losses = []

        for dataset, results in self.stage1_results.items():
            for res in results:
                val = res.get("val_loss", float("inf"))
                if val == float("inf"):
                    continue
                cfg = res["params"]
                configs.append(cfg)
                losses.append(val)

        if not configs:
            raise ValueError("没有可用于 fANOVA 的有效样本（val_loss 全是 inf）")

        Y = np.array(losses, dtype=float)

        # ===== 2. 构造原始 X（还没归一化） =====
        # 顺序固定：hidden_size, layers, dropout, lr_log, batch_size, sequence_length
        X_raw = []
        for cfg in configs:
            X_raw.append([
                float(cfg["hidden_size"]),
                float(cfg["layers"]),
                float(cfg["dropout"]),
                float(np.log10(cfg["lr"])),
                float(cfg["batch_size"]),
                float(cfg["sequence_length"]),
            ])

        X_raw = np.array(X_raw, dtype=float)

        param_names = ["hidden_size", "layers", "dropout", "lr_log", "batch_size", "sequence_length"]

        # ===== 3. 按列归一化到 [0, 1] =====
        mins = X_raw.min(axis=0)
        maxs = X_raw.max(axis=0)
        ranges = maxs - mins

        # 避免除 0：如果某一维所有值都一样，就让 range=1，这一维的值全变成 0
        ranges[ranges == 0] = 1.0

        X = (X_raw - mins) / ranges
        X = np.clip(X, 0.0, 1.0)

        print("[DEBUG] 归一化前每维范围：")
        for i, name in enumerate(param_names):
            print(f"  {name}: min={X_raw[:, i].min()}, max={X_raw[:, i].max()}")

        print("[DEBUG] 归一化后每维范围：")
        for i, name in enumerate(param_names):
            print(f"  {name}: min={X[:, i].min()}, max={X[:, i].max()}")

        # ===== 4. 定义 0~1 的 ConfigSpace =====
        cs = ConfigurationSpace()
        hps = []

        for name in param_names:
            hp = UniformFloatHyperparameter(name, 0.0, 1.0)
            cs.add(hp)          # 如果 VSCode 画删除线也没关系，能跑就行
            hps.append(hp)

        # ===== 5. 调用 fANOVA =====
        f = fANOVA(X, Y, cs)

        importance_dict = {}

        # 单个超参的重要性
        for i, hp in enumerate(hps):
            imp = f.quantify_importance((i,))
            importance_dict[hp.name] = {
                "individual": imp[(i,)]['individual importance'],
                "total": imp[(i,)]['total importance'],
            }

        # 超参两两交互的重要性
        for i in range(len(hps)):
            for j in range(i + 1, len(hps)):
                imp = f.quantify_importance((i, j))
                pair_name = f"{hps[i].name}_{hps[j].name}"
                importance_dict[pair_name] = {
                    "interaction": imp[(i, j)]['individual importance']
                }

        self.importance_scores = importance_dict

        # ===== 6. 保存结果 =====
        os.makedirs("results/stage1", exist_ok=True)
        with open("results/stage1/importance_scores.json", "w") as f_out:
            json.dump(self.importance_scores, f_out, indent=2)

        return importance_dict



        # ===== 5. 调用 fANOVA =====
        f = fANOVA(X, Y, cs)

        importance_dict = {}

        # 单个超参的重要性
        for i, hp in enumerate(hps):
            imp = f.quantify_importance((i,))
            importance_dict[hp.name] = {
                "individual": imp[(i,)]['individual importance'],
                "total": imp[(i,)]['total importance'],
            }

        # 超参两两交互的重要性
        for i in range(len(hps)):
            for j in range(i + 1, len(hps)):
                imp = f.quantify_importance((i, j))
                pair_name = f"{hps[i].name}_{hps[j].name}"
                importance_dict[pair_name] = {
                    "interaction": imp[(i, j)]['individual importance']
                }

        self.importance_scores = importance_dict

        # ===== 6. 保存结果 =====
        os.makedirs("results/stage1", exist_ok=True)
        with open("results/stage1/importance_scores.json", "w") as f_out:
            json.dump(self.importance_scores, f_out, indent=2)

        return importance_dict


    
    def plot_importance_violin(self):
        """绘制方差贡献小提琴图"""
        print("\n绘制超参重要性小提琴图...")
        
        # 准备数据
        data_for_plot = []
        
        # 对每个数据集分别计算重要性分布
        for dataset, results in self.stage1_results.items():
            # 创建局部重要性样本（通过bootstrap）
            n_bootstrap = 50
            for _ in range(n_bootstrap):
                # 随机采样80%的数据
                sample_indices = np.random.choice(
                    len(results), 
                    size=int(0.8 * len(results)), 
                    replace=True
                )
                
                X_sample = []
                Y_sample = []
                
                for idx in sample_indices:
                    res = results[idx]
                    if res['val_loss'] != float('inf'):
                        X_sample.append([
                            res['params']['hidden_size'],
                            res['params']['layers'],
                            res['params']['dropout'],
                            np.log10(res['params']['lr']),
                            res['params']['batch_size'],
                            res['params']['sequence_length']
                        ])
                        Y_sample.append(res['val_loss'])
                
                if len(X_sample) > 10:  # 确保有足够的样本
                    X_sample = np.array(X_sample)
                    Y_sample = np.array(Y_sample)
                    
                    # 简化的重要性计算（基于方差分解）
                    for i, param in enumerate(['hidden_size', 'layers', 'dropout', 'lr', 'batch_size', 'sequence_length']):
                        # 计算该参数的方差贡献
                        unique_vals = np.unique(X_sample[:, i])
                        if len(unique_vals) > 1:
                            groups = [Y_sample[X_sample[:, i] == val] for val in unique_vals]
                            # 组间方差
                            between_var = np.var([np.mean(g) for g in groups if len(g) > 0])
                            # 总方差
                            total_var = np.var(Y_sample)
                            if total_var > 0:
                                variance_contribution = between_var / total_var
                                data_for_plot.append({
                                    'Parameter': param,
                                    'Variance Contribution': variance_contribution,
                                    'Dataset': dataset
                                })
        
        # 创建DataFrame
        df = pd.DataFrame(data_for_plot)
        
        # ========= 画图部分的修改从这里开始 =========
        # 1) 去掉灰色背景（白底）
        sns.set_style("white")
        
        # 2) 缩短两两小提琴之间的“视觉距离”：减小横向 figsize、加大 width
        plt.figure(figsize=(9, 6))   # 之前是 (12, 6)
        
        # 定义 6 个对比明显的颜色
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c",
                "#d62728", "#9467bd", "#8c564b"]
        
        ax = sns.violinplot(
            data=df, 
            x='Parameter', 
            y='Variance Contribution',
            inner='box',
            cut=0,
            palette=palette,   # 3) 六个小提琴不同颜色
            width=0.8          # 小提琴更宽，看起来间隔变小
        )
        
        # 设置白色背景（再保险一层）
        ax.set_facecolor("white")
        ax.figure.set_facecolor("white")
        
        # 添加散点
        sns.swarmplot(
            data=df,
            x='Parameter',
            y='Variance Contribution',
            color='black',
            alpha=0.3,
            size=2,
            ax=ax
        )
        
        plt.title('Hyperparameter Importance Analysis (Stage 1)', fontsize=16, fontweight='bold')
        plt.xlabel('Hyperparameters', fontsize=18)  # 4) 横坐标标题字体也放大一点
        plt.ylabel('Variance Contribution', fontsize=14)
        
        # 4) 横坐标刻度字体放大到原来的 1.5 倍（假设原来 ~12，这里用 18）
        plt.xticks(fontsize=18, rotation=45)
        
        # 轻一点的网格（只保留 y 轴，防止太花）
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/stage1_importance_violin.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("小提琴图已保存至 figures/stage1_importance_violin.png")

    
    def refine_search_space(self, top_k: int = 3):
        """基于重要性分析缩小搜索空间
        
        Args:
            top_k: 保留最重要的k个参数进行精细调优
        """
        print(f"\n基于重要性分析缩小搜索空间 (保留top {top_k} 参数)...")
        
        # 获取参数重要性排序
        param_importance = []
        for param in ['hidden_size', 'layers', 'dropout', 'lr_log', 'batch_size', 'sequence_length']:
            if param in self.importance_scores:
                param_importance.append((
                    param.replace('_log', ''),
                    self.importance_scores[param].get('total', 0)
                ))
        
        param_importance.sort(key=lambda x: x[1], reverse=True)
        important_params = [p[0] for p in param_importance[:top_k]]
        
        print(f"最重要的参数: {important_params}")
        
        # 创建缩小的搜索空间
        # 对重要参数使用更精细的范围，对其他参数固定为最佳值
        best_params = self._get_best_params()
        
        refined_space = {}
        for param in ['hidden_size', 'layers', 'dropout', 'lr', 'batch_size', 'sequence_length']:
            if param in important_params:
                # 重要参数：缩小范围但保持可调
                if param == 'hidden_size':
                    refined_space[param] = hp.choice(f'{param}_refined', [512, 768, 1024])
                elif param == 'layers':
                    refined_space[param] = hp.choice(f'{param}_refined', [1, 2])
                elif param == 'dropout':
                    refined_space[param] = hp.uniform(f'{param}_refined', 0.3, 0.5)
                elif param == 'lr':
                    refined_space[param] = hp.loguniform(f'{param}_refined', np.log(5e-5), np.log(5e-3))
                elif param == 'batch_size':
                    refined_space[param] = hp.choice(f'{param}_refined', [32, 48, 64])
                elif param == 'sequence_length':
                    refined_space[param] = hp.choice(f'{param}_refined', [3, 5, 7])
            else:
                # 非重要参数：固定为最佳值
                refined_space[param] = best_params[param]
        
        self.refined_space = refined_space
        return refined_space
    
    def _get_best_params(self) -> Dict:
        """获取第一阶段的最佳参数"""
        best_loss = float('inf')
        best_params = None
        
        for dataset, results in self.stage1_results.items():
            for res in results:
                if res['val_loss'] < best_loss:
                    best_loss = res['val_loss']
                    best_params = res['params']
        
        return best_params if best_params else self.initial_space
    
    def stage2_multi_optimizer_comparison(self, max_evals: int = 30):
        """第二阶段：多优化器对比"""
        print("="*50)
        print("Stage 2: 多优化器效率对比")
        print("="*50)
        
        optimizers = {
            'TPE': tpe.suggest,
            'Random': rand.suggest,
            'TPE-Adaptive': tpe.suggest,  # 将使用自适应采样
            'Hybrid': None  # 自定义混合策略
        }
        
        optimizer_results = {}
        
        for opt_name, opt_algo in optimizers.items():
            print(f"\n测试优化器: {opt_name}")
            
            if opt_name == 'Hybrid':
                # 混合策略：前50%用Random，后50%用TPE
                results = self._hybrid_optimization(max_evals)
            else:
                # 标准优化
                trials = Trials()
                
                def objective(params):
                    # 处理固定参数
                    full_params = {}
                    for key, value in params.items():
                        # 移除'_refined'后缀
                        clean_key = key.replace('_refined', '')
                        full_params[clean_key] = value
                    
                    # 添加固定参数
                    for key, value in self.refined_space.items():
                        if key not in full_params:
                            full_params[key] = value
                    
                    # 类型转换
                    full_params = {
                        'hidden_size': int(full_params['hidden_size']),
                        'layers': int(full_params['layers']),
                        'dropout': float(full_params['dropout']),
                        'lr': float(full_params['lr']),
                        'batch_size': int(full_params['batch_size']),
                        'sequence_length': int(full_params['sequence_length'])
                    }
                    
                    result = train(optim_params=full_params,dataset='dataset2')
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    return {'loss': result['val_loss'], 'status': 'ok'}
                
                # 执行优化
                best = fmin(
                    fn=objective,
                    space=self.refined_space,
                    algo=opt_algo,
                    max_evals=max_evals,
                    trials=trials
                )
                
                results = trials.trials
            
            optimizer_results[opt_name] = {
                'trials': results,
                'best_loss': min([t['result']['loss'] for t in results]),
                'convergence_speed': self._calculate_convergence_speed(results)
            }
        
        # 保存结果
        with open('results/stage2/optimizer_comparison.pkl', 'wb') as f:
            pickle.dump(optimizer_results, f)
        
        self.optimizer_results = optimizer_results
        return optimizer_results
    
    def _hybrid_optimization(self, max_evals: int) -> List:
        print(f"[Hybrid] Random warm-up + TPE (max_evals={max_evals})")

        trials = Trials()
        warmup_evals = max(3, max_evals // 3)

        def objective(params):
            full_params = {}
            for key, value in params.items():
                clean_key = key.replace('_refined', '')
                full_params[clean_key] = value

            for key, value in self.refined_space.items():
                clean_key = key.replace('_refined', '')
                if clean_key not in full_params and isinstance(value, (int, float)):
                    full_params[clean_key] = value

            full_params = {
                'hidden_size':     int(full_params['hidden_size']),
                'layers':          int(full_params['layers']),
                'dropout':         float(full_params['dropout']),
                'lr':              float(full_params['lr']),
                'batch_size':      int(full_params['batch_size']),
                'sequence_length': int(full_params['sequence_length']),
            }

            result = train(optim_params=full_params, dataset='dataset2')

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {'loss': result['val_loss'], 'status': STATUS_OK}

        # 1) Random warm-up
        print(f"[Hybrid] Random warm-up: {warmup_evals} evals")
        fmin(
            fn=objective,
            space=self.refined_space,
            algo=rand.suggest,
            max_evals=warmup_evals,
            trials=trials,
            show_progressbar=False,
            # 删掉 rstate
        )

        # 2) TPE phase
        print(f"[Hybrid] TPE phase: from {warmup_evals} to {max_evals} evals")
        fmin(
            fn=objective,
            space=self.refined_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            show_progressbar=False,
            # 这里也不要 rstate
        )

        return trials.trials

    
    # def _sample_from_space(self, space: Dict, method: str = 'random', history: List = None) -> Dict:
    #     """从空间中采样参数"""
    #     sampled = {}
        
    #     for param, hp_obj in space.items():
    #         if isinstance(hp_obj, (int, float)):
    #             # 固定值
    #             sampled[param] = hp_obj
    #         elif method == 'random':
    #             # 随机采样
    #             if 'choice' in str(type(hp_obj)):
    #                 sampled[param] = np.random.choice(hp_obj.pos_args[1])
    #             elif 'uniform' in str(type(hp_obj)):
    #                 sampled[param] = np.random.uniform(hp_obj.pos_args[1], hp_obj.pos_args[2])
    #             elif 'loguniform' in str(type(hp_obj)):
    #                 sampled[param] = np.exp(np.random.uniform(hp_obj.pos_args[1], hp_obj.pos_args[2]))
    #         else:
    #             # TPE采样（简化版）
    #             if history and len(history) > 5:
    #                 # 基于历史最佳结果
    #                 best_trials = sorted(history, key=lambda x: x['result']['loss'])[:5]
    #                 # 这里简化处理，实际TPE更复杂
    #                 sampled[param] = self._sample_from_space({param: hp_obj}, method='random')[param]
    #             else:
    #                 sampled[param] = self._sample_from_space({param: hp_obj}, method='random')[param]
        
    #     return sampled
    def _sample_from_space(self, space: Dict, method: str = 'random', history: List = None) -> Dict:
        """从空间中采样参数（简化版：TPE 分支暂时也用随机采样）"""
        sampled = {}

        for param, hp_obj in space.items():
            # 情况 1：固定标量
            if isinstance(hp_obj, (int, float)):
                sampled[param] = hp_obj
            else:
                # 情况 2：Hyperopt 的空间对象（choice/uniform/loguniform/...）
                # 不再手动判断类型，直接交给 hyperopt 的 sample
                try:
                    sampled[param] = hp_sample(hp_obj)
                except Exception as e:
                    print(f"[WARNING] 采样参数 {param} 时出错({e})，使用默认值 0 作为回退")
                    sampled[param] = 0

        return sampled
    
    def _evaluate_params(self, params: Dict) -> Dict:
        """评估参数"""
        # 类型转换
        params = {
            'hidden_size': int(params.get('hidden_size', 512)),
            'layers': int(params.get('layers', 1)),
            'dropout': float(params.get('dropout', 0.3)),
            'lr': float(params.get('lr', 0.001)),
            'batch_size': int(params.get('batch_size', 32)),
            'sequence_length': int(params.get('sequence_length', 5))
        }
        
        return train(optim_params=params,dataset='dataset2')
    
    def _calculate_convergence_speed(self, trials: List) -> Dict:
        """计算收敛速度指标"""
        losses = [t['result']['loss'] for t in trials]
        
        # 计算达到不同阈值所需的迭代次数
        best_loss = min(losses)
        thresholds = {
            '90%': best_loss * 1.1,
            '95%': best_loss * 1.05,
            '99%': best_loss * 1.01
        }
        
        convergence_iters = {}
        for name, threshold in thresholds.items():
            for i, loss in enumerate(losses):
                if loss <= threshold:
                    convergence_iters[name] = i + 1
                    break
            else:
                convergence_iters[name] = len(losses)
        
        return convergence_iters
    
    def plot_optimizer_comparison_violin(self):
        """绘制优化器对比小提琴图"""
        print("\n绘制优化器对比小提琴图...")
        
        # 准备数据
        data_for_plot = []
        
        for opt_name, results in self.optimizer_results.items():
            trials = results['trials']
            losses = [t['result']['loss'] for t in trials]
            
            # 计算探索效率指标
            n_trials = len(trials)
            best_loss = min(losses)
            
            # 为每个试验计算相对性能
            for i, loss in enumerate(losses):
                # 计算当前试验的探索空间占比
                exploration_ratio = (i + 1) / n_trials
                
                # 计算性能改进
                if i == 0:
                    improvement = 0
                else:
                    current_best = min(losses[:i+1])
                    prev_best = min(losses[:i])
                    improvement = (prev_best - current_best) / prev_best if prev_best > 0 else 0
                
                # 计算效率分数（结合探索和改进）
                efficiency_score = improvement / exploration_ratio if exploration_ratio > 0 else 0
                
                data_for_plot.append({
                    'Optimizer': opt_name,
                    'Exploration Ratio': exploration_ratio,
                    'Loss': loss,
                    'Efficiency': efficiency_score,
                    'Iteration': i + 1
                })
        
        df = pd.DataFrame(data_for_plot)
        
        # ========== 统一配色：为每个优化器指定一个颜色 ==========
        opt_names = list(self.optimizer_results.keys())
        # 这里用 tab10 调色板，够鲜明对比
        base_palette = sns.color_palette("tab10", n_colors=len(opt_names))
        color_map = {opt: base_palette[i] for i, opt in enumerate(opt_names)}
        
        # 创建两个子图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子图1：探索空间占比（按优化器着色）
        ax1 = axes[0]
        
        sns.violinplot(
            data=df,
            x='Optimizer',
            y='Exploration Ratio',
            ax=ax1,
            inner='box',
            cut=0,
            palette=color_map   # 使用我们自定义的颜色映射
        )
        
        ax1.set_title('Optimizer Efficiency Comparison (Stage 2)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Optimizer', fontsize=12)
        ax1.set_ylabel('Exploration Space Ratio', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 子图2：收敛曲线对比（与左图颜色保持一致）
        ax2 = axes[1]
        
        for opt_name in opt_names:
            opt_df = df[df['Optimizer'] == opt_name]
            cumulative_best = []
            current_best = float('inf')
            
            for loss in opt_df['Loss'].values:
                current_best = min(current_best, loss)
                cumulative_best.append(current_best)
            
            ax2.plot(
                range(1, len(cumulative_best) + 1),
                cumulative_best,
                label=opt_name,
                marker='o',
                markersize=3,
                alpha=0.8,
                color=color_map[opt_name]   # 和左边同色
            )
        
        ax2.set_title('Convergence Curves Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Best Validation Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/stage2_optimizer_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("优化器对比图已保存至 figures/stage2_optimizer_comparison.png")

    
    def generate_final_report(self):
        """生成最终报告"""
        print("\n生成最终优化报告...")
        
        report = {
            'stage1': {
                'n_samples': self.n_random_samples * len(self.datasets),
                'important_params': list(self.importance_scores.keys())[:3],
                'best_params_stage1': self._get_best_params()
            },
            'stage2': {
                'refined_space': str(self.refined_space),
                'optimizer_performance': {}
            }
        }
        
        for opt_name, results in self.optimizer_results.items():
            report['stage2']['optimizer_performance'][opt_name] = {
                'best_loss': results['best_loss'],
                'convergence_speed': results['convergence_speed']
            }
        
        # 保存报告
        with open('results/final_optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("报告已保存至 results/final_optimization_report.json")
        
        return report

