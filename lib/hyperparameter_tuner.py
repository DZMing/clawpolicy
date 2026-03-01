#!/usr/bin/env python3
"""
Automatic parameter adjustment system - Learning rate scheduling and hyperparameter search

Implement intelligent hyperparameter optimization：
- Learning rate scheduling（exponential decay、cosine annealing）
- Hyperparameter search（grid search、random search、Bayesian optimization）
- Early stopping mechanism（Early Stopping）
- Hyperparameter importance analysis

Phase 3.2 - expected250OK
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class LearningRateScheduler:
    """
    Learning rate scheduler

    Support multiple scheduling strategies：
    - constant: constant learning rate
    - exponential: exponential decay
    - cosine: cosine annealing
    - step: step decay
    """

    def __init__(self,
                 initial_lr: float = 0.001,
                 scheduler_type: str = "exponential",
                 decay_rate: float = 0.96,
                 decay_steps: int = 100,
                 min_lr: float = 1e-6):
        """
        Initialize the learning rate scheduler

        Args:
            initial_lr: Initial learning rate
            scheduler_type: scheduler type（constant/exponential/cosine/step）
            decay_rate: decay rate
            decay_steps: Decay steps
            min_lr: minimum learning rate
        """
        self.initial_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr

        self.current_step = 0
        self.history: List[Tuple[int, float]] = []

    def get_lr(self, step: int | None = None) -> float:
        """
        Get the current learning rate

        Args:
            step: number of steps（Nonethen use the internal count）

        Returns:
            Current learning rate
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
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        # Make sure not to go below the minimum learning rate
        lr = max(lr, self.min_lr)

        self.history.append((step, lr))
        return lr

    def reset(self):
        """reset scheduler"""
        self.current_step = 0
        self.history = []


class HyperparameterSearch:
    """
    Hyperparameter search

    Supports multiple search strategies：
    - grid: grid search（exhaustive）
    - random: random search
    - bayesian: Bayesian optimization（Simplified version）
    """

    def __init__(self,
                 search_space: Dict[str, Any],
                 search_type: str = "random",
                 n_trials: int = 50):
        """
        Initialize hyperparameter search

        Args:
            search_space: search space
                {
                    "learning_rate": (0.0001, 0.01),  # continuous range
                    "batch_size": [32, 64, 128],       # discrete choice
                    "hidden_dims": [[64, 64], [128, 128]]
                }
            search_type: Search type（grid/random/bayesian）
            n_trials: number of trials
        """
        self.search_space = search_space
        self.search_type = search_type
        self.n_trials = n_trials

        self.trials: List[Dict[str, Any]] = []
        self.best_trial: Optional[Dict[str, Any]] = None

    def suggest(self, trial_id: int) -> Dict[str, Any]:
        """
        Suggest next set of hyperparameters

        Args:
            trial_id: testID

        Returns:
            Hyperparameter configuration
        """
        if self.search_type == "grid":
            return self._grid_suggest(trial_id)
        elif self.search_type == "random":
            return self._random_suggest(trial_id)
        elif self.search_type == "bayesian":
            return self._bayesian_suggest(trial_id)
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")

    def _grid_suggest(self, trial_id: int) -> Dict[str, Any]:
        """grid search"""
        # Generate all possible combinations（Simplified version）
        all_configs = self._generate_grid_configs()

        if trial_id >= len(all_configs):
            logger.warning(f"testID {trial_id} Grid search range exceeded，Use random configuration")
            return self._random_suggest(trial_id)

        return all_configs[trial_id]

    def _random_suggest(self, trial_id: int) -> Dict[str, Any]:
        """random search"""
        config = {}

        for param_name, param_space in self.search_space.items():
            if isinstance(param_space, tuple):
                # continuous range
                low, high = param_space
                value = np.random.uniform(low, high)
            elif isinstance(param_space, list):
                # discrete choice
                value = np.random.choice(param_space)
            else:
                raise ValueError(f"Unknown parameter space type: {type(param_space)}")

            config[param_name] = value

        return config

    def _bayesian_suggest(self, trial_id: int) -> Dict[str, Any]:
        """
        Bayesian optimization（Simplified version：Gaussian sampling based on historical results）

        Complete implementation needsskoptoroptunaLibrary，A simplified version is available here
        """
        if len(self.trials) < 5:
            # The first few experiments used random searches
            return self._random_suggest(trial_id)

        # Local search based on best trials
        if self.best_trial is None:
            return self._random_suggest(trial_id)

        best_config = self.best_trial["config"]
        config = {}

        for param_name, param_space in self.search_space.items():
            if isinstance(param_space, tuple):
                # continuous range：Sample around the optimal value
                low, high = param_space
                best_value = best_config[param_name]
                std = (high - low) * 0.1  # The standard deviation is the range10%

                value = np.random.normal(best_value, std)
                value = np.clip(value, low, high)
            elif isinstance(param_space, list):
                # discrete choice：have80%Probability selects the best value
                if np.random.random() < 0.8:
                    value = best_config[param_name]
                else:
                    value = np.random.choice(param_space)
            else:
                raise ValueError(f"Unknown parameter space type: {type(param_space)}")

            config[param_name] = value

        return config

    def _generate_grid_configs(self) -> List[Dict[str, Any]]:
        """Generate all configurations for grid search"""
        import itertools

        # Generate candidate values ​​for each parameter
        param_candidates = {}
        for param_name, param_space in self.search_space.items():
            if isinstance(param_space, tuple):
                # continuous range：generate5candidate values
                low, high = param_space
                param_candidates[param_name] = np.linspace(low, high, 5).tolist()
            elif isinstance(param_space, list):
                # discrete choice：use all values
                param_candidates[param_name] = param_space
            else:
                raise ValueError(f"Unknown parameter space type: {type(param_space)}")

        # Generate all combinations
        param_names = list(param_candidates.keys())
        param_values = [param_candidates[name] for name in param_names]

        all_configs = []
        for combination in itertools.product(*param_values):
            config = dict(zip(param_names, combination))
            all_configs.append(config)

        return all_configs

    def record_trial(
        self,
        config: Dict[str, Any],
        score: float,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """
        Record test results

        Args:
            config: Hyperparameter configuration
            score: score
            metadata: additional metadata
        """
        trial = {
            "trial_id": len(self.trials),
            "config": deepcopy(config),
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.trials.append(trial)

        # Update best experiments
        if self.best_trial is None or score > self.best_trial["score"]:
            self.best_trial = trial
            logger.info(f"✅ new best test: {score:.4f}")

    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get the best configuration"""
        if self.best_trial is None:
            return None
        return self.best_trial["config"]

    def analyze_importance(self) -> Dict[str, float]:
        """
        Analyze the importance of hyperparameters

        Use simple ANOVA（ANOVA）Thought

        Returns:
            Parameter importance score
        """
        if len(self.trials) < 10:
            logger.warning("⚠️ Not enough tests，Importance analysis cannot be performed")
            return {}

        importance: Dict[str, float] = {}

        for param_name in self.search_space.keys():
            # Group by parameter value
            groups: Dict[Any, List[float]] = {}
            for trial in self.trials:
                value = trial["config"][param_name]

                # Discretize continuous values
                if isinstance(value, float):
                    value = round(value, 4)

                if value not in groups:
                    groups[value] = []
                groups[value].append(trial["score"])

            # Calculate variance between groups
            if len(groups) < 2:
                importance[param_name] = 0.0
                continue

            group_means = [np.mean(scores) for scores in groups.values()]
            overall_mean = np.mean([trial["score"] for trial in self.trials])

            # sum of squares between groups
            ss_between = sum(
                len(groups[value]) * (mean - overall_mean) ** 2
                for value, mean in zip(groups.keys(), group_means)
            )

            # total sum of squares
            ss_total = sum(
                (trial["score"] - overall_mean) ** 2
                for trial in self.trials
            )

            # importance（proportion of explained variance）
            importance[param_name] = ss_between / ss_total if ss_total > 0 else 0.0

        # normalization
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}

        return importance


class EarlyStopping:
    """
    Early stopping mechanism

    Stop training when validation metrics no longer improve
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = "max"):
        """
        Initialization early stop

        Args:
            patience: Number of steps tolerated（How many steps does it take to stop without improvement?）
            min_delta: minimum amount of improvement
            mode: model（maxBigger is better，minThe smaller the better）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score: float | None = None
        self.counter = 0
        self.early_stop = False

        self.history: List[float] = []

    def check(self, score: float) -> bool:
        """
        Check if it should be stopped early

        Args:
            score: Current rating

        Returns:
            TrueIndicates it should stop
        """
        self.history.append(score)

        if self.best_score is None:
            self.best_score = score
            return False

        # Check if it improves
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        # Check if tolerance is exceeded
        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"⏹️ Early stop trigger（{self.counter}No improvement）")
            return True

        return False

    def reset(self):
        """Reset early stop"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.history = []


class HyperparameterTuner:
    """
    Hyperparameter tuner（Integrate all functions）

    Complete automatic parameter adjustment process：
    1. Learning rate scheduling
    2. Hyperparameter search
    3. Early stopping mechanism
    4. Result analysis
    """

    def __init__(self,
                 search_space: Dict[str, Any],
                 search_type: str = "random",
                 n_trials: int = 50,
                 patience: int = 10):
        """
        Initialize the tuner

        Args:
            search_space: search space
            search_type: Search type
            n_trials: number of trials
            patience: Early stopping tolerance steps
        """
        self.search = HyperparameterSearch(search_space, search_type, n_trials)
        self.early_stopping = EarlyStopping(patience=patience)
        self.results: List[Dict[str, Any]] = []

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        save_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization

        Args:
            objective_fn: objective function（Receive hyperparameter configuration，Return to rating）
            save_dir: Result saving directory

        Returns:
            Optimize results
        """
        logger.info(f"🔍 Start hyperparameter search（{self.search.n_trials} trials）...")

        save_path = Path(save_dir) if save_dir else Path("./hyperparameter_tuning_results")
        save_path.mkdir(parents=True, exist_ok=True)

        for trial_id in range(self.search.n_trials):
            # Recommended configuration
            config = self.search.suggest(trial_id)

            logger.info(f"🎲 test {trial_id + 1}/{self.search.n_trials}")
            logger.info(f"   Configuration: {json.dumps(config, indent=2)}")

            try:
                # Evaluate configuration
                score = objective_fn(config)

                # Record results
                self.search.record_trial(config, score)
                self.early_stopping.check(score)

                self.results.append({
                    "trial_id": trial_id,
                    "config": config,
                    "score": score
                })

                logger.info(f"   score: {score:.4f}")

                # Save intermediate results
                self._save_intermediate_results(save_path, trial_id)

                # Inspection stopped early
                if self.early_stopping.early_stop:
                    logger.info("⏹️ Early stop trigger，End search")
                    break

            except Exception as e:
                logger.error(f"❌ test {trial_id} fail: {e}")
                continue

        # Analyze results
        return self._analyze_results(save_path)

    def _save_intermediate_results(self, save_path: Path, trial_id: int):
        """Save intermediate results"""
        results_file = save_path / f"intermediate_results_trial_{trial_id}.json"

        with open(results_file, 'w') as f:
            json.dump({
                "trial_id": trial_id,
                "trials": self.search.trials,
                "best_trial": self.search.best_trial
            }, f, indent=2)

    def _analyze_results(self, save_path: Path) -> Dict[str, Any]:
        """Analyze optimization results"""
        # Get the best configuration
        best_config = self.search.get_best_config()
        best_score = self.search.best_trial["score"] if self.search.best_trial else None

        # Analyze parameter importance
        importance = self.search.analyze_importance()

        # Save final result
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

        logger.info("✅ Optimization completed！")
        logger.info(f"   best rating: {best_score:.4f}")
        logger.info(f"   Best configuration: {json.dumps(best_config, indent=2)}")
        logger.info(f"   Parameter importance: {json.dumps(importance, indent=2)}")
        logger.info(f"   Results saved: {results_file}")

        return final_results


def main():
    """Testing the hyperparameter tuner"""
    # Define the search space
    search_space = {
        "learning_rate": (0.0001, 0.01),
        "batch_size": [32, 64, 128],
        "hidden_dims": [[64, 64], [128, 128], [256, 256]]
    }

    # Create a tuner
    tuner = HyperparameterTuner(
        search_space=search_space,
        search_type="random",
        n_trials=10,
        patience=5
    )

    # Define objective function（simulation）
    def objective_fn(config):
        # Simulation training process
        lr = config["learning_rate"]
        batch_size = config["batch_size"]

        # mock scoring（The closer to the optimal value, the higher）
        optimal_lr = 0.001
        optimal_batch = 64

        lr_score = 1.0 - abs(lr - optimal_lr) / optimal_lr
        batch_score = 1.0 - abs(batch_size - optimal_batch) / optimal_batch

        return (lr_score + batch_score) / 2 + np.random.normal(0, 0.05)

    # Perform optimization
    results = tuner.optimize(objective_fn, save_dir="./tuning_test")

    print("\n📊 Optimize results:")
    print(f"   best rating: {results['best_score']:.4f}")
    print(f"   Parameter importance: {results['parameter_importance']}")


if __name__ == "__main__":
    main()
