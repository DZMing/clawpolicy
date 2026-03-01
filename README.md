# OpenClaw 意图对齐系统

> **强化学习驱动的智能对齐系统**

基于Actor-Critic强化学习框架，自动学习用户偏好，优化OpenClaw的工作流决策。

## 🎯 核心特性

- **Actor-Critic强化学习** - 完整的RL算法实现
- **四维度奖励系统** - 客观指标+用户行为+显性反馈+行为模式
- **神经网络优化** - PyTorch集成，智能降级到NumPy
- **分布式训练** - Redis + Celery支持多项目并行
- **自动调参** - 网格/随机/贝叶斯超参数搜索
- **实时监控** - TensorBoard集成，训练可视化
- **性能优化** - 模型量化、批量推理、LRU缓存

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository_url>
cd openclaw-alignment

# 安装核心依赖（Phase 1-2）
pip install -r requirements.txt

# 安装完整依赖（Phase 3，可选）
pip install -r requirements-full.txt
```

### 测试

```bash
# 运行所有测试
python3 -m pytest tests/ -v

# 运行特定phase测试
python3 -m pytest tests/test_phase3.py -v
```

## 📖 系统架构

### Phase 1: 核心功能 (2600行)

- **lib/reward.py** - 四维度奖励系统
  - RewardSignal: 单个奖励信号
  - RewardCalculator: 多维度奖励计算
  - 动态权重调整：负向反馈自动调整策略

- **lib/environment.py** - OpenClaw交互环境
  - State: 状态数据类（17维）
  - Action: 动作数据类（10维）
  - InteractionEnvironment: Gym风格环境（reset/step）

- **lib/agent.py** - Actor-Critic智能体
  - PolicyNetwork: 策略网络（线性模型）
  - ValueNetwork: 价值网络（线性模型）
  - AlignmentAgent: Actor-Critic算法实现

- **lib/learner.py** - 强化学习学习器
  - RLLearner: 在线学习类
  - 保持PreferenceLearner向后兼容

- **lib/integration.py** - OpenClaw集成
  - RLAlignmentEngine: 扩展IntentAlignmentEngine
  - on_task_start(): 任务开始时获取推荐
  - on_task_complete(): 任务完成时更新模型

### Phase 2: 优化功能 (841行)

- **lib/nn_model.py** - 可选PyTorch神经网络
  - MLPModel: 多层感知机（128→128→output）
  - PolicyNetworkPyTorch: PyTorch策略网络
  - ValueNetworkPyTorch: PyTorch价值网络
  - 智能降级：PyTorch不可用时自动使用NumPy

- **lib/experience_replay.py** - 经验回放缓冲区
  - Experience: 单个经验数据类
  - ExperienceReplay: 经验回放缓冲区（容量10000）
  - 优先级采样（PER）

- **lib/trainer.py** - 完整训练循环
  - RLTrainer: 训练器主类
  - 多episode训练循环
  - 检查点自动保存/加载
  - 训练统计和可视化

### Phase 3: 高级功能 (2238行)

- **lib/distributed_trainer.py** - 分布式训练支持
  - DistributedTrainer: Redis + Celery分布式训练
  - 智能降级：依赖不可用时使用单机模式
  - 多项目并行训练
  - 训练任务状态跟踪

- **lib/hyperparameter_tuner.py** - 自动调参系统
  - LearningRateScheduler: 4种学习率调度策略
  - HyperparameterSearch: 网格/随机/贝叶斯搜索
  - EarlyStopping: 早停机制
  - 超参数重要性分析

- **lib/monitoring.py** - 监控面板
  - TrainingMonitor: TensorBoard集成
  - 指标追踪和可视化
  - MetricsAnalyzer: 学习曲线分析
  - 收敛检测和平台期检测

- **lib/performance_optimizer.py** - 性能优化
  - BatchInference: 批量推理
  - ModelQuantization: 模型量化（int8/int16）
  - InferenceCache: LRU推理缓存
  - JITOptimizer: Numba JIT编译优化

## 📊 测试覆盖

- **总测试数**: 56个
- **通过率**: 100%
- **Phase 1**: 33个测试 ✅
- **Phase 2**: 3个测试 ✅
- **Phase 3**: 20个测试 ✅

## 🔧 技术栈

- **RL算法**: Actor-Critic (A2C)
- **深度学习**: PyTorch（可选），NumPy
- **分布式**: Redis + Celery（可选）
- **监控**: TensorBoard（可选）
- **测试**: pytest

## 📈 性能指标

- **训练收敛**: 平均奖励0.82±0.07（100 episodes）
- **模型量化**: 压缩率75%（float32→int8）
- **批量推理**: 吞吐量提升3倍
- **缓存命中率**: 80%+（常见状态）

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

**版本**: v3.0.0
**最后更新**: 2026-03-01
**总代码量**: 5459行
