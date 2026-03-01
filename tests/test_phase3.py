#!/usr/bin/env python3
"""
Phase 3Advanced functional testing

Test distributed training、Automatic parameter adjustment、Monitoring panels and performance optimization
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile


class TestDistributedTrainer:
    """Distributed trainer testing"""

    def test_initialization(self):
        """Test initialization"""
        from lib.distributed_trainer import DistributedTrainer, DistributedTrainingConfig

        config = DistributedTrainingConfig()
        trainer = DistributedTrainer(config)

        assert trainer is not None
        assert trainer.config is not None

    def test_fallback_mode(self):
        """Test downgrade mode"""
        from lib.distributed_trainer import DistributedTrainer, DistributedTrainingConfig

        config = DistributedTrainingConfig()
        trainer = DistributedTrainer(config)

        # Distributed functions should detect dependencies
        assert hasattr(trainer, 'distributed_enabled')

        # There should be local trainer as downgrade
        assert hasattr(trainer, 'local_trainer') or trainer.distributed_enabled

    def test_train_sequential(self):
        """Test order training（downgrade mode）"""
        from lib.distributed_trainer import DistributedTrainer, DistributedTrainingConfig

        config = DistributedTrainingConfig()
        trainer = DistributedTrainer(config)

        project_configs = [
            {"project_id": "test_proj_1", "task_types": ["T1"]},
        ]

        # train1indivualepisode
        results = trainer.train_distributed(
            project_configs=project_configs,
            num_episodes_per_project=1,
            save_interval=1
        )

        assert results is not None
        assert "total_projects" in results
        assert results["total_projects"] == 1

    def test_fallback_when_runtime_unreachable(self):
        """Automatically downgrade when test dependencies exist but the distributed runtime is unreachable"""
        from lib.distributed_trainer import (
            DistributedTrainer,
            DistributedTrainingConfig,
            REDIS_AVAILABLE,
            CELERY_AVAILABLE,
        )

        if not (REDIS_AVAILABLE and CELERY_AVAILABLE):
            pytest.skip("Redis/Celery not installed in this environment")

        config = DistributedTrainingConfig(
            redis_host="127.0.0.1",
            redis_port=1,  # deliberately unreachable
            require_worker=True,
            connection_timeout=0.2,
        )
        trainer = DistributedTrainer(config)
        assert trainer.distributed_enabled is False


class TestHyperparameterTuner:
    """Hyperparameter tuner testing"""

    def test_scheduler_creation(self):
        """Test scheduler creation"""
        from lib.hyperparameter_tuner import LearningRateScheduler

        scheduler = LearningRateScheduler(
            initial_lr=0.001,
            scheduler_type="exponential"
        )

        assert scheduler is not None
        assert scheduler.initial_lr == 0.001

    def test_lr_schedule(self):
        """Test learning rate scheduling"""
        from lib.hyperparameter_tuner import LearningRateScheduler

        scheduler = LearningRateScheduler(
            initial_lr=0.001,
            scheduler_type="exponential",
            decay_rate=0.96,
            decay_steps=100
        )

        # Before test10step learning rate
        lrs = [scheduler.get_lr(i) for i in range(10)]

        # The learning rate should gradually decrease
        assert lrs[0] == 0.001  # Initial learning rate
        assert lrs[-1] < lrs[0]  # should drop

    def test_cosine_schedule(self):
        """Test cosine annealing schedule"""
        from lib.hyperparameter_tuner import LearningRateScheduler

        scheduler = LearningRateScheduler(
            initial_lr=0.001,
            scheduler_type="cosine",
            decay_steps=100
        )

        lr = scheduler.get_lr(50)
        assert lr > 0
        assert lr <= 0.001

    def test_hyperparameter_search(self):
        """Test hyperparameter search"""
        from lib.hyperparameter_tuner import HyperparameterSearch

        search_space = {
            "learning_rate": (0.0001, 0.01),
            "batch_size": [32, 64, 128]
        }

        search = HyperparameterSearch(search_space, search_type="random", n_trials=5)

        # Test recommended configuration
        config1 = search.suggest(0)
        config2 = search.suggest(1)

        assert config1 is not None
        assert config2 is not None
        assert "learning_rate" in config1
        assert "batch_size" in config1

        # The two configurations should be different（random）
        assert config1 != config2

    def test_record_trial(self):
        """test record test"""
        from lib.hyperparameter_tuner import HyperparameterSearch

        search_space = {
            "learning_rate": (0.0001, 0.01)
        }

        search = HyperparameterSearch(search_space, n_trials=5)

        # record test
        config = search.suggest(0)
        search.record_trial(config, score=0.85)

        assert len(search.trials) == 1
        assert search.best_trial is not None
        assert search.best_trial["score"] == 0.85

    def test_early_stopping(self):
        """Test early stopping mechanism"""
        from lib.hyperparameter_tuner import EarlyStopping

        early_stop = EarlyStopping(patience=3, min_delta=0.01)

        # Ratings improved in previous times
        assert not early_stop.check(0.5)
        assert not early_stop.check(0.55)
        assert not early_stop.check(0.60)

        # Subsequent ratings no longer improve
        assert not early_stop.check(0.60)  # counter=1
        assert not early_stop.check(0.60)  # counter=2
        assert early_stop.check(0.60)  # counter=3 -> Stop early


class TestMonitoring:
    """Monitoring panel testing"""

    def test_monitor_creation(self):
        """Test monitor creation"""
        from lib.monitoring import TrainingMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = TrainingMonitor(log_dir=tmpdir, experiment_name="test")

            assert monitor is not None
            assert monitor.experiment_dir.exists()
            monitor.close()

    def test_log_scalar(self):
        """test record scalar"""
        from lib.monitoring import TrainingMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = TrainingMonitor(log_dir=tmpdir)

            monitor.log_scalar("test/metric", 0.5, step=0)
            monitor.log_scalar("test/metric", 0.6, step=1)

            assert "test/metric" in monitor.metrics_history
            assert len(monitor.metrics_history["test/metric"]) == 2
            monitor.close()

    def test_log_training_step(self):
        """Test record training steps"""
        from lib.monitoring import TrainingMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = TrainingMonitor(log_dir=tmpdir)

            monitor.log_training_step(
                episode=0,
                reward=0.75,
                actor_loss=0.1,
                critic_loss=0.2
            )

            assert "train/reward" in monitor.metrics_history
            assert monitor.metrics_history["train/reward"][0][1] == 0.75
            monitor.close()

    def test_metrics_summary(self):
        """Summary of test indicators"""
        from lib.monitoring import TrainingMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = TrainingMonitor(log_dir=tmpdir)

            # Record some indicators
            for i in range(10):
                monitor.log_scalar("test/value", i * 0.1, step=i)

            summary = monitor.get_metrics_summary()

            assert "test/value" in summary
            assert summary["test/value"]["count"] == 10
            assert summary["test/value"]["mean"] > 0
            monitor.close()

    def test_save_metrics(self):
        """Test save metrics"""
        from lib.monitoring import TrainingMonitor
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = TrainingMonitor(log_dir=tmpdir)

            monitor.log_scalar("test/metric", 0.5, step=0)

            filepath = monitor.save_metrics_to_json("test_metrics.json")

            assert Path(filepath).exists()

            # Verify content
            with open(filepath, 'r') as f:
                data = json.load(f)

            assert "test/metric" in data
            monitor.close()


class TestPerformanceOptimizer:
    """Performance Optimizer Test"""

    def test_quantization(self):
        """Test model quantification"""
        from lib.performance_optimizer import ModelQuantization

        quantizer = ModelQuantization()

        # Create test weights
        weights = np.random.randn(128, 64).astype(np.float32)

        # Quantify
        quantized, params = quantizer.quantize_weights(weights, bits=8)

        assert quantized.dtype == np.int8
        assert "scale" in params
        assert "zero_point" in params

        # inverse quantization
        dequantized = quantizer.dequantize_weights(quantized, params)

        # The verification error is small
        error = np.mean(np.abs(weights - dequantized))
        assert error < 0.1  # The error should be less than0.1

    def test_inference_cache(self):
        """Testing the inference cache"""
        from lib.performance_optimizer import InferenceCache

        # Create a simulation model
        class DummyModel:
            def forward(self, state):
                return float(np.sum(state))

        model = DummyModel()
        cache = InferenceCache(model, cache_size=100)

        # first call（cache miss）
        state = np.array([1.0, 2.0, 3.0])
        result1 = cache.predict(state)

        # Second call to the same state（cache hit）
        result2 = cache.predict(state)

        assert result1 == result2

        # Check cache statistics（useLRUCachedcache_info）
        stats = cache.get_cache_stats()
        # first call：miss（Create cache），second call：hit
        # But since it is also counted when creating the cache for the first timemiss，So it should be1 miss, 1 hit
        assert stats["cache_misses"] >= 1  # at least oncemiss
        assert stats["cache_hits"] >= 1  # at least oncehit
        assert stats["total_requests"] == 2

    def test_cache_stats(self):
        """Test cache statistics"""
        from lib.performance_optimizer import InferenceCache

        class DummyModel:
            def forward(self, state):
                return float(np.sum(state))

        model = DummyModel()
        cache = InferenceCache(model, cache_size=100)

        # multiple predictions
        state = np.array([1.0, 2.0, 3.0])
        for _ in range(5):
            cache.predict(state)

        stats = cache.get_cache_stats()

        assert stats["total_requests"] == 5
        assert stats["hit_rate"] > 0  # There should be a cache hit

    def test_batch_inference(self):
        """Test batch inference"""
        from lib.performance_optimizer import BatchInference

        class DummyModel:
            def forward(self, state):
                return float(np.random.random())

        model = DummyModel()
        batch_infer = BatchInference(model, batch_size=5)

        # Add to2requests（less thanbatch_size）
        for _ in range(2):
            state = np.random.randn(10)
            request_id = batch_infer.predict(state, sync=False)
            assert isinstance(request_id, int)

        # The queue should have2requests
        assert batch_infer.get_queue_size() == 2

        # Refresh batch
        results = batch_infer.flush()
        assert len(results) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)

        # Queue should be empty
        assert batch_infer.get_queue_size() == 0

        # Test automatic batch triggering
        for _ in range(5):
            state = np.random.randn(10)
            batch_infer.predict(state, sync=False)

        # Queue should be empty（Automatically trigger batch inference and clear it after）
        assert batch_infer.get_queue_size() == 0


class TestIntegration:
    """Integration testing"""

    def test_phase3_integration(self):
        """testPhase 3integrated"""
        from lib.distributed_trainer import DistributedTrainer
        from lib.hyperparameter_tuner import HyperparameterTuner
        from lib.monitoring import TrainingMonitor
        from lib.performance_optimizer import PerformanceOptimizer

        # All modules should be importable
        assert DistributedTrainer is not None
        assert HyperparameterTuner is not None
        assert TrainingMonitor is not None
        assert PerformanceOptimizer is not None

    def test_full_pipeline(self):
        """Test the complete process"""
        from lib.monitoring import TrainingMonitor
        from lib.hyperparameter_tuner import LearningRateScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create monitor
            monitor = TrainingMonitor(log_dir=tmpdir, experiment_name="full_pipeline")

            # Create a learning rate scheduler
            scheduler = LearningRateScheduler(initial_lr=0.001, scheduler_type="exponential")

            # Simulation training loop
            for episode in range(10):
                scheduler.get_lr()
                reward = 0.5 + 0.1 * (episode / 10)

                monitor.log_training_step(
                    episode=episode,
                    reward=reward,
                    actor_loss=0.1 / (episode + 1),
                    critic_loss=0.05 / (episode + 1)
                )

            # Verify records
            summary = monitor.get_metrics_summary()
            assert "train/reward" in summary
            assert summary["train/reward"]["count"] == 10
            monitor.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
