#!/usr/bin/env python3
"""
Reward system unit testing
"""

import pytest
import numpy as np
from lib.reward import RewardCalculator, RewardSignal


class TestRewardSignal:
    """testRewardSignalkind"""

    def test_collect_normalize(self):
        """Test signal collection and normalization"""
        signal = RewardSignal(
            name="test",
            weight=0.5,
            collector=lambda ctx: ctx.get("value", 0),
            min_value=0.0,
            max_value=10.0
        )

        # Test the middle value
        result = signal.collect({"value": 5.0})
        assert result == 0.5

        # Test maximum
        result = signal.collect({"value": 10.0})
        assert result == 1.0

        # Test minimum
        result = signal.collect({"value": 0.0})
        assert result == 0.0

        # test out of range（Truncate）
        result = signal.collect({"value": 15.0})
        assert result == 1.0

        result = signal.collect({"value": -5.0})
        assert result == 0.0

    def test_history_tracking(self):
        """Test history"""
        signal = RewardSignal(
            name="test",
            weight=0.5,
            collector=lambda ctx: ctx.get("value", 0),
            min_value=0.0,
            max_value=10.0
        )

        signal.collect({"value": 3.0})
        signal.collect({"value": 7.0})

        assert len(signal.history) == 2
        assert signal.history[0] == 0.3
        assert signal.history[1] == 0.7

    def test_update_weight(self):
        """Test weight update"""
        signal = RewardSignal(
            name="test",
            weight=0.5,
            collector=lambda ctx: 0,
            min_value=0.0,
            max_value=1.0
        )

        signal.update_weight(0.1)
        assert signal.weight == 0.6

        signal.update_weight(-0.3)
        assert signal.weight == 0.3

        # test boundaries
        signal.update_weight(1.0)
        assert signal.weight == 1.0

        signal.update_weight(-2.0)
        assert signal.weight == 0.0


class TestRewardCalculator:
    """testRewardCalculatorkind"""

    def test_initialization(self):
        """Test initialization"""
        calc = RewardCalculator()

        # Check that all signals are initialized
        expected_signals = [
            "test_coverage", "code_quality", "bug_count", "task_time",
            "acceptance_rate", "adoption_rate", "rewrite_rate",
            "user_rating", "feedback_count",
            "agent_preference", "workflow_preference"
        ]

        for signal_name in expected_signals:
            assert signal_name in calc.signals

    def test_calculate_reward(self):
        """Test reward calculation"""
        calc = RewardCalculator()

        context = {
            "task_type": "T2",
            "task_result": {
                "agent": "claude",
                "workflow": "tdd",
                "duration": 300,
                "test_files_created": True,
                "committed": True
            },
            "test_result": {
                "coverage": 80.0,
                "passed": 10,
                "failed": 0
            },
            "user_feedback": {
                "accepted": True,
                "rating": 5
            },
            "metrics": {
                "complexity": 3,
                "duplication": 0.05,
                "lint_score": 0.9
            }
        }

        reward = calc.calculate_reward(context)

        # The reward should be in [0, 1] within range
        assert 0.0 <= reward <= 1.0

        # High-quality tasks should have higher rewards
        assert reward > 0.5

    def test_weight_adjustment(self):
        """Test weight adjustment after negative feedback"""
        calc = RewardCalculator()

        # Record initial weight
        initial_weights = {
            name: signal.weight
            for name, signal in calc.signals.items()
        }

        # Record negative feedback
        calc.record_feedback("rating", 2)

        # Check that the weights have been adjusted
        # Notice：due to normalization，We examine relative changes rather than absolute increases or decreases
        agent_auto_weights = ["agent_preference", "workflow_preference"]
        feedback_quality_weights = ["user_rating", "feedback_count", "code_quality", "test_coverage"]

        # Calculate average change
        agent_auto_change = np.mean([
            (calc.signals[name].weight - initial_weights[name]) / initial_weights[name]
            for name in agent_auto_weights
        ])
        feedback_quality_change = np.mean([
            (calc.signals[name].weight - initial_weights[name]) / initial_weights[name]
            for name in feedback_quality_weights
        ])

        # The average change in automation weights should be less than the feedback/quality weight
        assert agent_auto_change < feedback_quality_change

    def test_weight_normalization(self):
        """Test weight normalization"""
        calc = RewardCalculator(learning_phase="early")
        total_weight = sum(signal.weight for signal in calc.signals.values())
        assert total_weight == pytest.approx(1.0, rel=1e-3)

    def test_negative_signals_reduce_reward(self):
        """Testing the impact of negative signals on rewards"""
        calc = RewardCalculator()

        base_context = {
            "task_result": {"duration": 100, "errors": 0},
            "test_result": {"coverage": 90.0, "failed": 0},
            "user_feedback": {"accepted": True, "rating": 5},
            "metrics": {"complexity": 2, "duplication": 0.0, "lint_score": 1.0}
        }

        high_bug_context = {
            "task_result": {"duration": 1000, "errors": 5},
            "test_result": {"coverage": 90.0, "failed": 5},
            "user_feedback": {"accepted": True, "rating": 5},
            "metrics": {"complexity": 2, "duplication": 0.0, "lint_score": 1.0}
        }

        reward_base = calc.calculate_reward(base_context)
        reward_high_bug = calc.calculate_reward(high_bug_context)

        assert reward_high_bug < reward_base

    def test_learning_phase_transition(self):
        """Test learning phase transition"""
        calc = RewardCalculator(learning_phase="early")

        # The initial stage should beearly
        assert calc.learning_phase == "early"

        # simulation20tasks
        for i in range(20):
            context = {
                "task_result": {},
                "test_result": {},
                "user_feedback": {},
                "metrics": {}
            }
            calc.calculate_reward(context)

        # should be converted tomaturestage
        assert calc.learning_phase == "mature"

    def test_reward_stats(self):
        """Test reward statistics"""
        calc = RewardCalculator()

        # Generate some bonus data
        for i in range(10):
            context = {
                "task_result": {"duration": 300},
                "test_result": {"coverage": 50 + i * 5},
                "user_feedback": {"accepted": True},
                "metrics": {}
            }
            calc.calculate_reward(context)

        stats = calc.get_reward_stats()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["count"] == 10

    def test_signal_stats(self):
        """Test individual signal statistics"""
        calc = RewardCalculator()

        # generate some data
        for i in range(5):
            context = {
                "task_result": {},
                "test_result": {"coverage": 50 + i * 10},
                "user_feedback": {},
                "metrics": {}
            }
            calc.calculate_reward(context)

        stats = calc.get_signal_stats("test_coverage")

        assert stats["count"] == 5
        assert "mean" in stats
        assert "current_weight" in stats

    def test_save_load_state(self):
        """Test state saving and loading"""
        import tempfile
        import os

        calc1 = RewardCalculator()

        # generate some data
        for i in range(3):
            context = {
                "task_result": {"duration": 300},
                "test_result": {"coverage": 50 + i * 10},
                "user_feedback": {"rating": 4},
                "metrics": {}
            }
            calc1.calculate_reward(context)

        # save state
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            calc1.save_state(temp_path)

            # Create new calculator and load status
            calc2 = RewardCalculator()
            calc2.load_state(temp_path)

            # Check status restored
            assert calc2.task_count == calc1.task_count
            assert len(calc2.reward_history) == len(calc1.reward_history)
            assert calc2.learning_phase == calc1.learning_phase

        finally:
            os.unlink(temp_path)


class TestRewardCollectors:
    """Test individual data collection functions"""

    def test_collect_test_coverage(self):
        """Test test coverage collection"""
        calc = RewardCalculator()

        # There are test results
        context = {
            "test_result": {"coverage": 85.0}
        }
        assert calc._collect_test_coverage(context) == 85.0

        # No test results，But there is a test file created
        context = {
            "task_result": {"test_files_created": True}
        }
        assert calc._collect_test_coverage(context) == 50.0

    def test_collect_code_quality(self):
        """Test code quality collection"""
        calc = RewardCalculator()

        context = {
            "metrics": {
                "complexity": 3,
                "duplication": 0.1,
                "lint_score": 0.8
            }
        }

        quality = calc._collect_code_quality(context)

        # should be in [0, 10] within range
        assert 0.0 <= quality <= 10.0

        # low complexity、Low repetition、highlintShould get high quality points
        assert quality > 5.0

    def test_collect_bug_count(self):
        """testBugquantity collection"""
        calc = RewardCalculator()

        context = {
            "test_result": {"failed": 2},
            "task_result": {"errors": 1}
        }

        bugs = calc._collect_bug_count(context)
        assert bugs == 3

    def test_collect_task_time(self):
        """Test task time collection"""
        calc = RewardCalculator()

        context = {
            "task_result": {"duration": 600}
        }

        time = calc._collect_task_time(context)
        assert time == 600

    def test_collect_acceptance_rate(self):
        """Test acceptance rate collection"""
        calc = RewardCalculator()

        # accept
        context = {
            "user_feedback": {"accepted": True}
        }
        assert calc._collect_acceptance_rate(context) == 1.0

        # reject
        context = {
            "user_feedback": {"accepted": False}
        }
        assert calc._collect_acceptance_rate(context) == 0.0

        # There are modifications
        context = {
            "user_feedback": {"revisions": 3}
        }
        rate = calc._collect_acceptance_rate(context)
        assert rate < 1.0

    def test_collect_user_rating(self):
        """Test user rating collection"""
        calc = RewardCalculator()

        # Explicit scoring
        context = {
            "user_feedback": {"rating": 5}
        }
        assert calc._collect_user_rating(context) == 5

        # fromacceptanceinfer
        context = {
            "user_feedback": {"accepted": True}
        }
        assert calc._collect_user_rating(context) == 4.0

    def test_collect_agent_preference(self):
        """testAgentPreference collection"""
        calc = RewardCalculator()

        # T2Task usageClaude
        context = {
            "task_type": "T2",
            "task_result": {"agent": "claude"}
        }
        preference = calc._collect_agent_preference(context)
        assert preference > 0.5

        # T3Task usageCodex
        context = {
            "task_type": "T3",
            "task_result": {"agent": "codex"}
        }
        preference = calc._collect_agent_preference(context)
        assert preference > 0.5

    def test_collect_workflow_preference(self):
        """Test workflow preference collection"""
        calc = RewardCalculator()

        # TDDWorkflow
        context = {
            "task_result": {"workflow": "tdd"}
        }
        preference = calc._collect_workflow_preference(context)
        assert preference > 0.8

        # Standard workflow
        context = {
            "task_result": {"workflow": "standard"}
        }
        preference = calc._collect_workflow_preference(context)
        assert preference < 0.8


class StubHistoryProvider:
    """History provider for testing"""

    def __init__(self, agent_rate: float | None = None, workflow_rate: float | None = None):
        self.agent_rate = agent_rate
        self.workflow_rate = workflow_rate

    def get_agent_success_rate(self, task_type: str, agent: str):
        return self.agent_rate

    def get_workflow_success_rate(self, task_type: str, workflow: str):
        return self.workflow_rate


class TestRewardHistoryProvider:
    """Testing history preference provider access"""

    def test_history_provider_overrides_agent_preference(self):
        calc = RewardCalculator(history_provider=StubHistoryProvider(agent_rate=0.92))
        context = {"task_type": "T3", "task_result": {"agent": "codex"}}
        assert calc._collect_agent_preference(context) == pytest.approx(0.92)

    def test_history_provider_overrides_workflow_preference(self):
        calc = RewardCalculator(history_provider=StubHistoryProvider(workflow_rate=0.88))
        context = {"task_type": "T2", "task_result": {"workflow": "tdd"}}
        assert calc._collect_workflow_preference(context) == pytest.approx(0.88)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
