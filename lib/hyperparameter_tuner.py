#!/usr/bin/env python3
"""
自动调参系统 - 学习率调度与超参数搜索

实现智能超参数优化：
- 学习率调度（指数衰减、余弦退火）
- 超参数搜索（网格搜索、随机搜索、贝叶斯优化）
- 早停机制（Early Stopping）
- 超参数重要性分析

Phase 3.2 - 预计250行
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import json
from datetime import datetime
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class LearningRateScheduler:
    """
    学习率调度器

    支持多种调度策略：
    - constant: 常数学习率
    - exponential: 指数衰减
    - cosine: 余弦退火
    - step: 阶梯衰减
    """

    def __init__(self,
                 initial_lr: float = 0.001,
                 scheduler_type: str = "exponential",
                 decay_rate: float = 0.96,
                 decay_steps: int = 100,
                 min_lr: float = 1e-6):
        """
        初始化学习率调度器

        Args:
            initial_lr: 初始学习率
            scheduler_type: 调度器类型（constant/exponential/cosine/step）
            decay_rate: 衰减率
            decay_steps: 衰减步数
            min_lr: 最小学习率
        """
        self.initial_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr

        self.current_step = 0
        self.history: List[Tuple[int, float]] = []

    def get_lr(self, step: int = None) -> float:
        """
        获取当前学习率

        Args:
            step: 步数（None则使用内部计数）

        Returns:
            当前学习率
        """
        if step is None:
            step = self.current_step
            self.current_step += 1

        if self.scheduler_type == "constant":
            lr = self.initial_lr

        elif self.scheduler_type == "exponential":
            lr = self.initial_lr * (self.decay_rate ** (step / self.decay_steps))

        elif self.scheduler_type == "cosine":
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + np.cos(np.pi * step / self.decay_steps)
            )

        elif self.scheduler_type == "step":
            lr = self.initial_lr * (self.decay_rate ** (step // self.decay_steps))

        else:
            raise ValueError(f"未知的调度器类型: {self.scheduler_type}")

        # 确保不低于最小学习率
        lr = max(lr, self.min_lr)

        self.history.append((step, lr))
        return lr

    def reset(self):
        """重置调度器"""
        self.current_step = 0
        self.history = []


class HyperparameterSearch:
    """
    超参数搜索

    支持多种搜索策略：
    - grid: 网格搜索（穷举）
    - random: 随机搜索
    - bayesian: 贝叶斯优化（简化版）
    """

    def __init__(self,
                 search_space: Dict[str, Any],
                 search_type: str = "random",
                 n_trials: int = 50):
        """
        初始化超参数搜索

        Args:
            search_space: 搜索空间
                {
                    "learning_rate": (0.0001, 0.01),  # 连续范围
                    "batch_size": [32, 64, 128],       # 离散选择
                    "hidden_dims": [[64, 64], [128, 128]]
                }
            search_type: 搜索类型（grid/random/bayesian）
            n_trials: 试验次数
        """
        self.search_space = search_space
        self.search_type = search_type
        self.n_trials = n_trials

        self.trials: List[Dict[str, Any]] = []
        self.best_trial: Optional[Dict[str, Any]] = None

    def suggest(self, trial_id: int) -> Dict[str, Any]:
        """
        建议下一组超参数

        Args:
            trial_id: 试验ID

        Returns:
            超参数配置
        """
        if self.search_type == "grid":
            return self._grid_suggest(trial_id)
        elif self.search_type == "random":
            return self._random_suggest(trial_id)
        elif self.search_type == "bayesian":
            return self._bayesian_suggest(trial_id)
        else:
            raise ValueError(f"未知的搜索类型: {self.search_type}")

    def _grid_suggest(self, trial_id: int) -> Dict[str, Any]:
        """网格搜索"""
        # 生成所有可能的组合（简化版）
        all_configs = self._generate_grid_configs()

        if trial_id >= len(all_configs):
            logger.warning(f"试验ID {trial_id} 超出网格搜索范围，使用随机配置")
            return self._random_suggest(trial_id)

        return all_configs[trial_id]

    def _random_suggest(self, trial_id: int) -> Dict[str, Any]:
        """随机搜索"""
        config = {}

        for param_name, param_space in self.search_space.items():
            if isinstance(param_space, tuple):
                # 连续范围
                low, high = param_space
                value = np.random.uniform(low, high)
            elif isinstance(param_space, list):
                # 离散选择
                value = np.random.choice(param_space)
            else:
                raise ValueError(f"未知的参数空间类型: {type(param_space)}")

            config[param_name] = value

        return config

    def _bayesian_suggest(self, trial_id: int) -> Dict[str, Any]:
        """
        贝叶斯优化（简化版：基于历史结果的高斯采样）

        完整实现需要skopt或optuna库，这里提供简化版本
        """
        if len(self.trials) < 5:
            # 前几个试验使用随机搜索
            return self._random_suggest(trial_id)

        # 基于最佳试验进行局部搜索
        best_config = self.best_trial["config"]
        config = {}

        for param_name, param_space in self.search_space.items():
            if isinstance(param_space, tuple):
                # 连续范围：在最佳值附近采样
                low, high = param_space
                best_value = best_config[param_name]
                std = (high - low) * 0.1  # 标准差为范围的10%

                value = np.random.normal(best_value, std)
                value = np.clip(value, low, high)
            elif isinstance(param_space, list):
                # 离散选择：有80%概率选择最佳值
                if np.random.random() < 0.8:
                    value = best_config[param_name]
                else:
                    value = np.random.choice(param_space)
            else:
                raise ValueError(f"未知的参数空间类型: {type(param_space)}")

            config[param_name] = value

        return config

    def _generate_grid_configs(self) -> List[Dict[str, Any]]:
        """生成网格搜索的所有配置"""
        import itertools

        # 为每个参数生成候选值
        param_candidates = {}
        for param_name, param_space in self.search_space.items():
            if isinstance(param_space, tuple):
                # 连续范围：生成5个候选值
                low, high = param_space
                param_candidates[param_name] = np.linspace(low, high, 5).tolist()
            elif isinstance(param_space, list):
                # 离散选择：使用所有值
                param_candidates[param_name] = param_space
            else:
                raise ValueError(f"未知的参数空间类型: {type(param_space)}")

        # 生成所有组合
        param_names = list(param_candidates.keys())
        param_values = [param_candidates[name] for name in param_names]

        all_configs = []
        for combination in itertools.product(*param_values):
            config = dict(zip(param_names, combination))
            all_configs.append(config)

        return all_configs

    def record_trial(self, config: Dict[str, Any], score: float, metadata: Dict = None):
        """
        记录试验结果

        Args:
            config: 超参数配置
            score: 评分
            metadata: 额外元数据
        """
        trial = {
            "trial_id": len(self.trials),
            "config": deepcopy(config),
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.trials.append(trial)

        # 更新最佳试验
        if self.best_trial is None or score > self.best_trial["score"]:
            self.best_trial = trial
            logger.info(f"✅ 新最佳试验: {score:.4f}")

    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """获取最佳配置"""
        if self.best_trial is None:
            return None
        return self.best_trial["config"]

    def analyze_importance(self) -> Dict[str, float]:
        """
        分析超参数重要性

        使用简单的方差分析（ANOVA）思想

        Returns:
            参数重要性得分
        """
        if len(self.trials) < 10:
            logger.warning("⚠️ 试验数量不足，无法进行重要性分析")
            return {}

        importance = {}

        for param_name in self.search_space.keys():
            # 按参数值分组
            groups = {}
            for trial in self.trials:
                value = trial["config"][param_name]

                # 离散化连续值
                if isinstance(value, float):
                    value = round(value, 4)

                if value not in groups:
                    groups[value] = []
                groups[value].append(trial["score"])

            # 计算组间方差
            if len(groups) < 2:
                importance[param_name] = 0.0
                continue

            group_means = [np.mean(scores) for scores in groups.values()]
            overall_mean = np.mean([trial["score"] for trial in self.trials])

            # 组间平方和
            ss_between = sum(
                len(groups[value]) * (mean - overall_mean) ** 2
                for value, mean in zip(groups.keys(), group_means)
            )

            # 总平方和
            ss_total = sum(
                (trial["score"] - overall_mean) ** 2
                for trial in self.trials
            )

            # 重要性（解释方差比例）
            importance[param_name] = ss_between / ss_total if ss_total > 0 else 0.0

        # 归一化
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}

        return importance


class EarlyStopping:
    """
    早停机制

    当验证指标不再改善时停止训练
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = "max"):
        """
        初始化早停

        Args:
            patience: 容忍步数（多少步不改善就停止）
            min_delta: 最小改善量
            mode: 模式（max越大越好，min越小越好）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score = None
        self.counter = 0
        self.early_stop = False

        self.history: List[float] = []

    def check(self, score: float) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前评分

        Returns:
            True表示应该停止
        """
        self.history.append(score)

        if self.best_score is None:
            self.best_score = score
            return False

        # 检查是否改善
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        # 检查是否超过容忍度
        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"⏹️ 早停触发（{self.counter}步未改善）")
            return True

        return False

    def reset(self):
        """重置早停"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.history = []


class HyperparameterTuner:
    """
    超参数调优器（集成所有功能）

    完整的自动调参流程：
    1. 学习率调度
    2. 超参数搜索
    3. 早停机制
    4. 结果分析
    """

    def __init__(self,
                 search_space: Dict[str, Any],
                 search_type: str = "random",
                 n_trials: int = 50,
                 patience: int = 10):
        """
        初始化调优器

        Args:
            search_space: 搜索空间
            search_type: 搜索类型
            n_trials: 试验次数
            patience: 早停容忍步数
        """
        self.search = HyperparameterSearch(search_space, search_type, n_trials)
        self.early_stopping = EarlyStopping(patience=patience)
        self.results: List[Dict[str, Any]] = []

    def optimize(self,
                objective_fn: Callable[[Dict[str, Any]], float],
                save_dir: str = None) -> Dict[str, Any]:
        """
        执行超参数优化

        Args:
            objective_fn: 目标函数（接收超参数配置，返回评分）
            save_dir: 结果保存目录

        Returns:
            优化结果
        """
        logger.info(f"🔍 开始超参数搜索（{self.search.n_trials} 次试验）...")

        save_path = Path(save_dir) if save_dir else Path("./hyperparameter_tuning_results")
        save_path.mkdir(parents=True, exist_ok=True)

        for trial_id in range(self.search.n_trials):
            # 建议配置
            config = self.search.suggest(trial_id)

            logger.info(f"🎲 试验 {trial_id + 1}/{self.search.n_trials}")
            logger.info(f"   配置: {json.dumps(config, indent=2)}")

            try:
                # 评估配置
                score = objective_fn(config)

                # 记录结果
                self.search.record_trial(config, score)
                self.early_stopping.check(score)

                self.results.append({
                    "trial_id": trial_id,
                    "config": config,
                    "score": score
                })

                logger.info(f"   评分: {score:.4f}")

                # 保存中间结果
                self._save_intermediate_results(save_path, trial_id)

                # 检查早停
                if self.early_stopping.early_stop:
                    logger.info("⏹️ 早停触发，结束搜索")
                    break

            except Exception as e:
                logger.error(f"❌ 试验 {trial_id} 失败: {e}")
                continue

        # 分析结果
        return self._analyze_results(save_path)

    def _save_intermediate_results(self, save_path: Path, trial_id: int):
        """保存中间结果"""
        results_file = save_path / f"intermediate_results_trial_{trial_id}.json"

        with open(results_file, 'w') as f:
            json.dump({
                "trial_id": trial_id,
                "trials": self.search.trials,
                "best_trial": self.search.best_trial
            }, f, indent=2)

    def _analyze_results(self, save_path: Path) -> Dict[str, Any]:
        """分析优化结果"""
        # 获取最佳配置
        best_config = self.search.get_best_config()
        best_score = self.search.best_trial["score"] if self.search.best_trial else None

        # 分析参数重要性
        importance = self.search.analyze_importance()

        # 保存最终结果
        final_results = {
            "best_config": best_config,
            "best_score": best_score,
            "parameter_importance": importance,
            "all_trials": self.search.trials,
            "convergence_curve": [t["score"] for t in self.search.trials]
        }

        results_file = save_path / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"✅ 优化完成！")
        logger.info(f"   最佳评分: {best_score:.4f}")
        logger.info(f"   最佳配置: {json.dumps(best_config, indent=2)}")
        logger.info(f"   参数重要性: {json.dumps(importance, indent=2)}")
        logger.info(f"   结果已保存: {results_file}")

        return final_results


def main():
    """测试超参数调优器"""
    # 定义搜索空间
    search_space = {
        "learning_rate": (0.0001, 0.01),
        "batch_size": [32, 64, 128],
        "hidden_dims": [[64, 64], [128, 128], [256, 256]]
    }

    # 创建调优器
    tuner = HyperparameterTuner(
        search_space=search_space,
        search_type="random",
        n_trials=10,
        patience=5
    )

    # 定义目标函数（模拟）
    def objective_fn(config):
        # 模拟训练过程
        lr = config["learning_rate"]
        batch_size = config["batch_size"]

        # 模拟评分（越接近最优值越高）
        optimal_lr = 0.001
        optimal_batch = 64

        lr_score = 1.0 - abs(lr - optimal_lr) / optimal_lr
        batch_score = 1.0 - abs(batch_size - optimal_batch) / optimal_batch

        return (lr_score + batch_score) / 2 + np.random.normal(0, 0.05)

    # 执行优化
    results = tuner.optimize(objective_fn, save_dir="./tuning_test")

    print(f"\n📊 优化结果:")
    print(f"   最佳评分: {results['best_score']:.4f}")
    print(f"   参数重要性: {results['parameter_importance']}")


if __name__ == "__main__":
    main()
