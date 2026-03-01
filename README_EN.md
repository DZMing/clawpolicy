# OpenClaw Alignment System

> **Reinforcement Learning Driven Intelligent Alignment System**

**English** | **[简体中文](README.md)**

An Actor-Critic Reinforcement Learning framework that automatically learns user preferences and optimizes OpenClaw workflow decisions.

## 🎯 Core Features

- **Actor-Critic RL** - Complete RL algorithm implementation
- **Four-Dimensional Reward System** - Objective metrics + user behavior + explicit feedback + behavioral patterns
- **Neural Network Optimization** - PyTorch integration with intelligent fallback to NumPy
- **Distributed Training** - Redis + Celery support for multi-project parallel training
- **Auto Hyperparameter Tuning** - Grid/Random/Bayesian hyperparameter search
- **Real-time Monitoring** - TensorBoard integration with training visualization
- **Performance Optimization** - Model quantization, batch inference, LRU caching

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/412984588/openclaw-alignment.git
cd openclaw-alignment

# Install core dependencies (Phase 1-2)
pip install -r requirements.txt

# Install full dependencies (Phase 3, optional)
pip install -r requirements-full.txt
```

### Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific phase tests
python3 -m pytest tests/test_phase3.py -v
```

## 📖 System Architecture

### Phase 1: Core Features (2600 lines)

- **lib/reward.py** - Four-dimensional reward system
  - `RewardSignal`: Individual reward signal
  - `RewardCalculator`: Multi-dimensional reward calculation
  - Dynamic weight adjustment: Auto-adjust strategy on negative feedback

- **lib/environment.py** - OpenClaw interaction environment
  - `State`: State data class (17 dimensions)
  - `Action`: Action data class (10 dimensions)
  - `InteractionEnvironment`: Gym-style environment (reset/step)

- **lib/agent.py** - Actor-Critic agent
  - `PolicyNetwork`: Policy network (linear model)
  - `ValueNetwork`: Value network (linear model)
  - `AlignmentAgent`: Actor-Critic algorithm implementation

- **lib/learner.py** - RL learner
  - `RLLearner`: Online learning class
  - Maintain backward compatibility with PreferenceLearner

- **lib/integration.py** - OpenClaw integration
  - `RLAlignmentEngine`: Extend IntentAlignmentEngine
  - `on_task_start()`: Get recommendations when task starts
  - `on_task_complete()`: Update model when task completes

### Phase 2: Optimization Features (841 lines)

- **lib/nn_model.py** - Optional PyTorch neural networks
  - `MLPModel`: Multi-layer perceptron (128→128→output)
  - `PolicyNetworkPyTorch`: PyTorch policy network
  - `ValueNetworkPyTorch`: PyTorch value network
  - Intelligent fallback: Auto-use NumPy when PyTorch unavailable

- **lib/experience_replay.py** - Experience replay buffer
  - `Experience`: Single experience data class
  - `ExperienceReplay`: Experience replay buffer (capacity 10000)
  - Prioritized sampling (PER)

- **lib/trainer.py** - Complete training loop
  - `RLTrainer`: Trainer main class
  - Multi-episode training loop
  - Auto checkpoint save/load
  - Training statistics and visualization

### Phase 3: Advanced Features (2238 lines)

- **lib/distributed_trainer.py** - Distributed training support
  - `DistributedTrainer`: Redis + Celery distributed training
  - Intelligent fallback to single-machine mode
  - Multi-project parallel training
  - Training task status tracking

- **lib/hyperparameter_tuner.py** - Auto hyperparameter tuning
  - `LearningRateScheduler`: 4 learning rate scheduling strategies
  - `HyperparameterSearch`: Grid/Random/Bayesian search
  - `EarlyStopping`: Early stopping mechanism
  - Hyperparameter importance analysis

- **lib/monitoring.py** - Monitoring dashboard
  - `TrainingMonitor`: TensorBoard integration
  - Metric tracking and visualization
  - `MetricsAnalyzer`: Learning curve analysis
  - Convergence detection and plateau detection

- **lib/performance_optimizer.py** - Performance optimization
  - `BatchInference`: Batch inference
  - `ModelQuantization`: Model quantization (int8/int16)
  - `InferenceCache`: LRU inference cache
  - `JITOptimizer`: Numba JIT compilation optimization

## 📊 Test Coverage

- **Total Tests**: 56
- **Pass Rate**: 100%
- **Phase 1**: 33 tests ✅
- **Phase 2**: 3 tests ✅
- **Phase 3**: 20 tests ✅

## 🔧 Tech Stack

- **RL Algorithm**: Actor-Critic (A2C)
- **Deep Learning**: PyTorch (optional), NumPy
- **Distributed**: Redis + Celery (optional)
- **Monitoring**: TensorBoard (optional)
- **Testing**: pytest

## 📈 Performance Metrics

- **Training Convergence**: Average reward 0.82±0.07 (100 episodes)
- **Model Quantization**: 75% compression ratio (float32→int8)
- **Batch Inference**: 3x throughput improvement
- **Cache Hit Rate**: 80%+ (common states)

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

MIT License

---

**Version**: v3.0.0
**Last Updated**: 2026-03-01
**Total Lines of Code**: 5459
