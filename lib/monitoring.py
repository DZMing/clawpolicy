#!/usr/bin/env python3
"""
Monitoring panel - TensorBoardIntegration and training visualization

Realize complete training monitoring function：
- TensorBoardlogging
- Reward Curve Visualization
- loss curve tracking
- Hyperparameter comparison
- Model performance analysis

Phase 3.3 - expected180OK
"""

import numpy as np
from typing import Any, Dict, List, Tuple
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# try to importTensorBoard
try:
    from torch.utils import tensorboard as _tensorboard
    _SummaryWriterFactory: Any = _tensorboard.SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    _SummaryWriterFactory = None


class TrainingMonitor:
    """
    training monitor

    Function：
    - TensorBoardlogging
    - Metric tracking（award、loss、performance）
    - Checkpoint comparison
    - Training report generation
    """

    def __init__(self,
                 log_dir: str = "./logs/tensorboard",
                 experiment_name: str | None = None):
        """
        Initialize monitor

        Args:
            log_dir: Log directory
            experiment_name: Experiment name
        """
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment directory
        if experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        else:
            self.experiment_dir = self.log_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # initializationTensorBoard writer
        self.tensorboard_enabled = TENSORBOARD_AVAILABLE
        self.writer: Any | None = None
        if self.tensorboard_enabled:
            self.writer = _SummaryWriterFactory(log_dir=str(self.experiment_dir))
            logger.info(f"✅ TensorBoardEnabled: {self.experiment_dir}")
        else:
            logger.warning("⚠️ TensorBoardNot available，Use degraded mode（JSONlog）")

        # Indicator record
        self.metrics_history: Dict[str, List[Tuple[int, float]]] = {}
        self._closed = False

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Logging scalar metrics

        Args:
            tag: Indicator label（like"train/reward", "train/loss"）
            value: indicator value
            step: number of steps
        """
        if self.tensorboard_enabled and self.writer:
            self.writer.add_scalar(tag, value, step)

        # Record to memory simultaneously（for downgrade mode）
        if tag not in self.metrics_history:
            self.metrics_history[tag] = []
        self.metrics_history[tag].append((step, value))

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        Logging scalar metrics in batches

        Args:
            main_tag: main label
            tag_scalar_dict: Label-value dictionary
            step: number of steps
        """
        if self.tensorboard_enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

        # Record to memory
        for sub_tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{sub_tag}"
            if full_tag not in self.metrics_history:
                self.metrics_history[full_tag] = []
            self.metrics_history[full_tag].append((step, value))

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """
        Record histogram

        Args:
            tag: Label
            values: value array
            step: number of steps
        """
        if self.tensorboard_enabled and self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_hyperparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        """
        Record hyperparameters and final metrics

        Args:
            hparam_dict: Hyperparameter dictionary
            metric_dict: indicator dictionary
        """
        if self.tensorboard_enabled and self.writer:
            self.writer.add_hparams(hparam_dict, metric_dict)

    def log_training_step(self,
                         episode: int,
                         reward: float,
                         actor_loss: float,
                         critic_loss: float,
                         extra_metrics: Dict[str, float] | None = None):
        """
        Record training steps（Convenience method）

        Args:
            episode: Episodenumber
            reward: award
            actor_loss: Actorloss
            critic_loss: Criticloss
            extra_metrics: additional indicators
        """
        # Record key indicators
        self.log_scalar("train/reward", reward, episode)
        self.log_scalar("train/actor_loss", actor_loss, episode)
        self.log_scalar("train/critic_loss", critic_loss, episode)

        # Record additional metrics
        if extra_metrics:
            for key, value in extra_metrics.items():
                self.log_scalar(f"train/{key}", value, episode)

    def log_evaluation(self,
                      episode: int,
                      eval_reward: float,
                      eval_success_rate: float,
                      eval_metrics: Dict[str, float] | None = None):
        """
        Record evaluation metrics

        Args:
            episode: Episodenumber
            eval_reward: Evaluation rewards
            eval_success_rate: success rate
            eval_metrics: Additional evaluation metrics
        """
        self.log_scalar("eval/reward", eval_reward, episode)
        self.log_scalar("eval/success_rate", eval_success_rate, episode)

        if eval_metrics:
            for key, value in eval_metrics.items():
                self.log_scalar(f"eval/{key}", value, episode)

    def log_model_weights(self, tag: str, weights: np.ndarray, step: int):
        """
        Record model weight distribution

        Args:
            tag: Label（like"policy_net/layer1"）
            weights: weight array
            step: number of steps
        """
        self.log_histogram(tag, weights, step)

        # Record statistics
        self.log_scalars(
            f"{tag}/stats",
            {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights))
            },
            step
        )

    def save_metrics_to_json(self, filename: str | None = None) -> str:
        """
        Save the indicator toJSONdocument（downgrade mode）

        Args:
            filename: file name

        Returns:
            file path
        """
        if not filename:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.experiment_dir / filename

        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        logger.info(f"✅ Indicator saved: {filepath}")
        return str(filepath)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get indicator summary

        Returns:
            Indicator summary statistics
        """
        summary = {}

        for tag, history in self.metrics_history.items():
            if not history:
                continue

            steps, values = zip(*history)

            summary[tag] = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "latest": float(values[-1]),
                "first": float(values[0]),
                "improvement": float(values[-1] - values[0]) if len(values) > 1 else 0.0
            }

        return summary

    def plot_metrics(self, tags: List[str] | None = None, save_path: str | Path | None = None):
        """
        Draw indicator curve（downgrade mode）

        Args:
            tags: Indicator labels to plot（None=all）
            save_path: save path
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("⚠️ matplotlibNot available，Unable to draw")
            return

        if tags is None:
            tags = list(self.metrics_history.keys())

        # Create subgraph
        n_plots = len(tags)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Plot each indicator
        for i, tag in enumerate(tags):
            if tag not in self.metrics_history:
                continue

            steps, values = zip(*self.metrics_history[tag])

            axes[i].plot(steps, values, 'b-', linewidth=2)
            axes[i].set_title(tag)
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)

        # Hide redundant subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        plot_path = Path(save_path).expanduser() if save_path else self.experiment_dir / "metrics_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Chart saved: {plot_path}")

        plt.close()

    def compare_experiments(self,
                          experiment_dirs: List[str],
                          metric_tags: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments

        Args:
            experiment_dirs: Experiment directory list
            metric_tags: Metric labels to compare

        Returns:
            Compare results
        """
        comparison: Dict[str, Any] = {
            "experiments": [],
            "metrics": {}
        }

        for exp_dir in experiment_dirs:
            exp_path = Path(exp_dir)
            if not exp_path.exists():
                logger.warning(f"⚠️ Experiment directory does not exist: {exp_dir}")
                continue

            # Read indicator file
            metrics_file = exp_path / "metrics_summary.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    comparison["experiments"].append({
                        "name": exp_path.name,
                        "metrics": metrics
                    })

        # Calculate the best experiment for each metric
        for tag in metric_tags:
            best_exp = None
            best_value = None

            for exp in comparison["experiments"]:
                if tag in exp["metrics"]:
                    value = exp["metrics"][tag].get("latest", 0)

                    if best_value is None or value > best_value:
                        best_value = value
                        best_exp = exp["name"]

            comparison["metrics"][tag] = {
                "best_experiment": best_exp,
                "best_value": best_value
            }

        return comparison

    def close(self):
        """Close monitor"""
        if self._closed:
            return
        self._safe_close_writer()
        self._closed = True

        # Save final indicator
        summary = self.get_metrics_summary()
        summary_file = self.experiment_dir / "metrics_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("✅ Monitor is off，Indicator saved")

    def _safe_close_writer(self):
        """Safe shutdownwriter，Avoid thread exceptions"""
        if self.tensorboard_enabled and self.writer:
            try:
                self.writer.flush()
                self.writer.close()
                logger.info("✅ TensorBoard writerClosed")
            except Exception as e:
                logger.warning(f"⚠️ closureTensorBoard writerfail: {e}")

    def __del__(self):
        """Try to close when destructingwriter，Avoid background thread exceptions"""
        try:
            self.close()
        except Exception:
            # Avoid exceptions thrown during destruction
            pass


class MetricsAnalyzer:
    """
    Indicator Analyzer

    Analyze training metrics and provide insights
    """

    def __init__(self, metrics_history: Dict[str, List[Tuple[int, float]]]):
        """
        Initialize analyzer

        Args:
            metrics_history: Indicator history
        """
        self.metrics = metrics_history

    def detect_convergence(self,
                          metric_tag: str,
                          window: int = 10,
                          threshold: float = 0.01) -> Tuple[bool, int]:
        """
        Check whether the indicator converges

        Args:
            metric_tag: Indicator label
            window: sliding window size
            threshold: convergence threshold（variance）

        Returns:
            (Convergence or not, Convergence steps)
        """
        if metric_tag not in self.metrics:
            return False, -1

        steps, values = zip(*self.metrics[metric_tag])

        # Sliding window detection
        for i in range(window, len(values)):
            window_values = values[i-window:i]
            variance = np.var(window_values)

            if variance < threshold:
                return True, steps[i]

        return False, -1

    def detect_plateau(self,
                      metric_tag: str,
                      window: int = 20,
                      threshold: float = 0.001) -> Tuple[bool, int]:
        """
        Detect whether the indicator has entered the plateau period

        Args:
            metric_tag: Indicator label
            window: sliding window size
            threshold: plateau threshold（average rate of change）

        Returns:
            (Is it a plateau?, Start steps)
        """
        if metric_tag not in self.metrics:
            return False, -1

        steps, values = zip(*self.metrics[metric_tag])

        # Calculate rate of change
        for i in range(window, len(values)):
            recent_values = values[i-window:i]
            avg_change = np.mean(np.diff(recent_values))

            if abs(avg_change) < threshold:
                return True, steps[i-window]

        return False, -1

    def analyze_learning_curve(self, metric_tag: str) -> Dict[str, Any]:
        """
        Analyze the learning curve

        Args:
            metric_tag: Indicator label

        Returns:
            Analyze results
        """
        if metric_tag not in self.metrics:
            return {}

        steps, values = zip(*self.metrics[metric_tag])

        # Fit the learning curve（exponential saturation model）
        def model(x, a, b, c):
            return a * (1 - np.exp(-b * x)) + c

        try:
            from scipy.optimize import curve_fit

            # Normalization steps
            x_normalized = np.array(steps) / max(steps)
            y = np.array(values)

            # Initial parameter guessing
            p0 = [max(y) - min(y), 1.0, min(y)]

            # fitting
            popt, _ = curve_fit(model, x_normalized, y, p0=p0, maxfev=5000)

            # Calculate fit quality
            y_pred = model(x_normalized, *popt)
            r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

            # Predict asymptotic value
            asymptote = popt[0] + popt[2]

            return {
                "convergence_value": float(asymptote),
                "current_value": float(values[-1]),
                "progress_to_convergence": float(values[-1] / asymptote) if asymptote != 0 else 0,
                "r_squared": float(r_squared),
                "fit_params": {
                    "amplitude": float(popt[0]),
                    "rate": float(popt[1]),
                    "offset": float(popt[2])
                }
            }

        except (ImportError, RuntimeError):
            # scipyNot available or fitting failed，Return simple statistics
            return {
                "current_value": float(values[-1]),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "trend": "increasing" if values[-1] > values[0] else "decreasing"
            }


def main():
    """test monitor"""
    monitor = TrainingMonitor(experiment_name="test_monitor")

    print("✅ Monitor created")
    print(f"   TensorBoard: {'✅ enable' if monitor.tensorboard_enabled else '❌ Disable'}")

    # Simulation training process
    for episode in range(100):
        reward = 0.5 + 0.3 * (1 - np.exp(-episode / 50)) + np.random.normal(0, 0.02)
        actor_loss = 0.5 * np.exp(-episode / 30) + np.random.normal(0, 0.01)
        critic_loss = 0.3 * np.exp(-episode / 40) + np.random.normal(0, 0.01)

        monitor.log_training_step(
            episode=episode,
            reward=reward,
            actor_loss=actor_loss,
            critic_loss=critic_loss
        )

    # Save indicator
    monitor.save_metrics_to_json()

    # draw curve
    monitor.plot_metrics()

    # Get summary
    summary = monitor.get_metrics_summary()
    print("\n📊 Training summary:")
    print(f"   final reward: {summary['train/reward']['latest']:.4f}")
    print(f"   Reward improvements: {summary['train/reward']['improvement']:.4f}")
    print(f"   finalActorloss: {summary['train/actor_loss']['latest']:.4f}")

    # Analyze the learning curve
    analyzer = MetricsAnalyzer(monitor.metrics_history)
    curve_analysis = analyzer.analyze_learning_curve("train/reward")
    print("\n📈 Learning curve analysis:")
    print(f"   Convergence value prediction: {curve_analysis.get('convergence_value', 0):.4f}")
    print(f"   Current progress: {curve_analysis.get('progress_to_convergence', 0)*100:.1f}%")

    monitor.close()


if __name__ == "__main__":
    main()
