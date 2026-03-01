#!/usr/bin/env python3
"""
Reinforcement Learning Reward System - Four-dimensional reward calculation

Implement a complete reward mechanism，include：
- objective indicators（test coverage、Code quality、Bugquantity、Task time）
- user behavior signals（acceptance rate、Adoption rate、rewrite rate）
- explicit feedback（User ratings、Feedback times）
- behavior pattern（AgentPreference、Workflow preferences）

Support dynamic weight adjustment and negative feedback strategy adjustment
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Protocol
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class RewardSignal:
    """single reward signal"""
    name: str  # Signal name
    weight: float  # Current weight
    collector: Callable  # data collection function
    history: List[float] = field(default_factory=list)  # History
    min_value: float = 0.0  # minimum value（for normalization）
    max_value: float = 1.0  # maximum value（for normalization）

    def collect(self, context: Dict[str, Any]) -> float:
        """Collect and normalize signal values"""
        raw_value = self.collector(context)

        # normalized to [0, 1]
        if self.max_value > self.min_value:
            normalized = (raw_value - self.min_value) / (self.max_value - self.min_value)
        else:
            normalized = raw_value

        # limited to [0, 1] within range
        normalized = max(0.0, min(1.0, normalized))

        # record history
        self.history.append(normalized)

        return normalized

    def update_weight(self, delta: float) -> None:
        """Update weights"""
        self.weight = max(0.0, min(1.0, self.weight + delta))


class PreferenceHistoryProvider(Protocol):
    """Preference history provider interface"""

    def get_agent_success_rate(self, task_type: str, agent: str) -> Optional[float]:
        """returntask_typeDownagenthistorical success rate（0-1）"""

    def get_workflow_success_rate(self, task_type: str, workflow: str) -> Optional[float]:
        """returntask_typeDownworkflowhistorical success rate（0-1）"""


class RewardCalculator:
    """Multi-dimensional reward calculation engine

    Implement a four-dimensional reward system：
    1. objective indicators（40%）: test_coverage(15%), code_quality(10%), bug_count(10%), task_time(5%)
    2. user behavior（30%）: acceptance_rate(15%), adoption_rate(10%), rewrite_rate(5%)
    3. explicit feedback（20%）: user_rating(15%), feedback_count(5%)
    4. behavior pattern（10%）: agent_preference(5%), workflow_preference(5%)
    """

    def __init__(
        self,
        learning_phase: str = "early",
        history_provider: Optional[PreferenceHistoryProvider] = None,
    ):
        """
        Initialize reward calculator

        Args:
            learning_phase: learning stage ("early": forward10tasks, "mature": 20+tasks)
        """
        self.learning_phase = learning_phase
        self.history_provider = history_provider
        self.task_count = 0

        # Initialize reward signal
        self.signals: Dict[str, RewardSignal] = {}
        self._initialize_signals()

        # negative feedback count（for strategy adjustment）
        self.negative_feedback_count = 0

        # Reward history（for training）
        self.reward_history: List[float] = []

        # negative signal set（Invert after normalization）
        self.negative_signals = {"bug_count", "task_time", "rewrite_rate"}

    def _initialize_signals(self) -> None:
        """Initialize all reward signals"""

        # ===== priority1: objective indicators（40%） =====

        # test coverage（15%）
        self.signals["test_coverage"] = RewardSignal(
            name="test_coverage",
            weight=0.15,
            collector=lambda ctx: self._collect_test_coverage(ctx),
            min_value=0.0,
            max_value=100.0
        )

        # Code quality（10%）
        self.signals["code_quality"] = RewardSignal(
            name="code_quality",
            weight=0.10,
            collector=lambda ctx: self._collect_code_quality(ctx),
            min_value=0.0,
            max_value=10.0
        )

        # Bugquantity（10%）- negative reward
        self.signals["bug_count"] = RewardSignal(
            name="bug_count",
            weight=0.10,
            collector=lambda ctx: self._collect_bug_count(ctx),
            min_value=0.0,
            max_value=10.0  # hypothesis10indivualbugfor the worst case
        )

        # Task time（5%）- negative reward（The longer the time, the worse）
        self.signals["task_time"] = RewardSignal(
            name="task_time",
            weight=0.05,
            collector=lambda ctx: self._collect_task_time(ctx),
            min_value=0.0,
            max_value=3600.0  # 1hours as worst case scenario
        )

        # ===== priority2: user behavior signals（30%） =====

        # acceptance rate（15%）
        self.signals["acceptance_rate"] = RewardSignal(
            name="acceptance_rate",
            weight=0.15,
            collector=lambda ctx: self._collect_acceptance_rate(ctx),
            min_value=0.0,
            max_value=1.0
        )

        # Adoption rate（10%）
        self.signals["adoption_rate"] = RewardSignal(
            name="adoption_rate",
            weight=0.10,
            collector=lambda ctx: self._collect_adoption_rate(ctx),
            min_value=0.0,
            max_value=1.0
        )

        # rewrite rate（5%）- negative reward
        self.signals["rewrite_rate"] = RewardSignal(
            name="rewrite_rate",
            weight=0.05,
            collector=lambda ctx: self._collect_rewrite_rate(ctx),
            min_value=0.0,
            max_value=1.0
        )

        # ===== priority3: explicit feedback（20%） =====

        # User ratings（15%）
        self.signals["user_rating"] = RewardSignal(
            name="user_rating",
            weight=0.15,
            collector=lambda ctx: self._collect_user_rating(ctx),
            min_value=1.0,
            max_value=5.0
        )

        # Feedback times（5%）
        self.signals["feedback_count"] = RewardSignal(
            name="feedback_count",
            weight=0.05,
            collector=lambda ctx: self._collect_feedback_count(ctx),
            min_value=0.0,
            max_value=10.0  # hypothesis10The feedback is the best case
        )

        # ===== priority4: behavior pattern（10%） =====

        # AgentPreference（5%）
        self.signals["agent_preference"] = RewardSignal(
            name="agent_preference",
            weight=0.05,
            collector=lambda ctx: self._collect_agent_preference(ctx),
            min_value=0.0,
            max_value=1.0
        )

        # Workflow preferences（5%）
        self.signals["workflow_preference"] = RewardSignal(
            name="workflow_preference",
            weight=0.05,
            collector=lambda ctx: self._collect_workflow_preference(ctx),
            min_value=0.0,
            max_value=1.0
        )

        # Adjust weights according to learning stage
        self._adjust_weights_for_phase()

    def set_history_provider(self, provider: Optional[PreferenceHistoryProvider]) -> None:
        """Set history preference provider"""
        self.history_provider = provider

    def _adjust_weights_for_phase(self) -> None:
        """Adjust weights according to learning stage

        early stage（forward10tasks）：70%Instant rewards + 30%long term optimization
        mature stage（20+tasks）：30%Instant rewards + 70%long term optimization
        """
        if self.learning_phase == "early":
            # Early days：Pay more attention to immediate feedback（user behavior、explicit feedback）
            self.signals["acceptance_rate"].weight *= 1.5
            self.signals["adoption_rate"].weight *= 1.3
            self.signals["user_rating"].weight *= 1.5
        else:
            # Mature：Pay more attention to long-term optimization（objective indicators、behavior pattern）
            self.signals["test_coverage"].weight *= 1.5
            self.signals["code_quality"].weight *= 1.3
            self.signals["agent_preference"].weight *= 1.5
            self.signals["workflow_preference"].weight *= 1.3

        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """normalized weight，Make sure the sum is1"""
        total_weight = sum(s.weight for s in self.signals.values())
        if total_weight == 0:
            return
        for signal in self.signals.values():
            signal.weight /= total_weight

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate weighted total reward

        Args:
            context: task context，Include：
                - task_result: Task execution results
                - test_result: Test results（Optional）
                - user_feedback: User feedback（Optional）
                - metrics: Other indicators

        Returns:
            total reward value（normalized to [0, 1]）
        """
        total_reward = 0.0
        reward_breakdown = {}

        # Collect rewards for each signal
        for name, signal in self.signals.items():
            signal_value = signal.collect(context)
            if name in self.negative_signals:
                signal_value = 1.0 - signal_value
            weighted_value = signal_value * signal.weight

            total_reward += weighted_value
            reward_breakdown[name] = {
                "value": signal_value,
                "weight": signal.weight,
                "weighted": weighted_value
            }

        # record history
        self.reward_history.append(total_reward)
        self.task_count += 1

        # Check if the learning stage needs to be updated
        self._update_learning_phase()

        # Record reward details（for debugging）
        if "debug_info" not in context:
            context["debug_info"] = {}
        context["debug_info"]["reward_breakdown"] = reward_breakdown
        context["debug_info"]["total_reward"] = total_reward

        return max(0.0, min(1.0, total_reward))

    def record_feedback(self, feedback_type: str, value: Any) -> None:
        """
        Record user explicit feedback

        Args:
            feedback_type: feedback type ("rating", "comment", "correction")
            value: feedback value（score、Comment content, etc.）
        """
        if feedback_type == "rating":
            # User ratings（1-5）
            if isinstance(value, (int, float)) and 1 <= value <= 5:
                # Rating is dominantly positive/negative feedback
                if value <= 2:
                    self.negative_feedback_count += 1
                    self._adjust_weights_on_negative_feedback()
        elif feedback_type == "correction":
            # User correction（negative feedback）
            self.negative_feedback_count += 1
            self._adjust_weights_on_negative_feedback()

    def _adjust_weights_on_negative_feedback(self) -> None:
        """Adjust strategy weights when there is negative feedback

        When receiving negative feedback：
        1. Reduce automation weight（agent_preference, workflow_preference）
        2. Increase the weight of user feedback（user_rating, feedback_count）
        3. Improve code quality weight（code_quality, test_coverage）
        """
        # Reduce automation weight
        self.signals["agent_preference"].weight *= 0.8
        self.signals["workflow_preference"].weight *= 0.8

        # Increase user feedback and code quality weight
        self.signals["user_rating"].weight *= 1.3
        self.signals["feedback_count"].weight *= 1.2
        self.signals["code_quality"].weight *= 1.2
        self.signals["test_coverage"].weight *= 1.2

        # Renormalize the weights（Keep the sum as1）
        total_weight = sum(s.weight for s in self.signals.values())
        for signal in self.signals.values():
            signal.weight /= total_weight

    def _update_learning_phase(self) -> None:
        """Update learning stage"""
        if self.task_count >= 20 and self.learning_phase == "early":
            self.learning_phase = "mature"
            # Reweight
            self._adjust_weights_for_phase()

    # ===== data collection function =====

    def _collect_test_coverage(self, context: Dict[str, Any]) -> float:
        """Collect test coverage（0-100）"""
        test_result = context.get("test_result", {})
        coverage = test_result.get("coverage", 0.0)

        # If there are no test results，fromtask_resultinfer
        if coverage == 0:
            task_result = context.get("task_result", {})
            # Check if the test file was created
            if "test_files_created" in task_result:
                coverage = 50.0  # Suppose you create a test file = 50%cover
            if "tests_passed" in task_result:
                coverage = min(coverage + 30.0, 100.0)

        return coverage

    def _collect_code_quality(self, context: Dict[str, Any]) -> float:
        """Collect code quality scores（0-10）"""
        metrics = context.get("metrics", {})

        # Combining multiple quality indicators
        quality_score = 5.0  # Basic points

        # code complexity（The lower the better）
        complexity = metrics.get("complexity", 5)
        quality_score += (10 - complexity) * 0.3

        # code duplication rate（The lower the better）
        duplication = metrics.get("duplication", 0.1)
        quality_score += (1.0 - duplication) * 2.0

        # Code style check
        lint_score = metrics.get("lint_score", 0.8)
        quality_score += lint_score * 1.5

        return max(0.0, min(10.0, quality_score))

    def _collect_bug_count(self, context: Dict[str, Any]) -> float:
        """collectBugquantity（negative reward）"""
        test_result = context.get("test_result", {})
        failed_tests = test_result.get("failed", 0)

        # Also considertask_resultError in
        task_result = context.get("task_result", {})
        errors = task_result.get("errors", 0)

        total_bugs = failed_tests + errors

        # Convert to negative reward（bugthe more，The lower the reward）
        return total_bugs

    def _collect_task_time(self, context: Dict[str, Any]) -> float:
        """Collect task completion time（negative reward，Second）"""
        task_result = context.get("task_result", {})
        duration = task_result.get("duration", 0)

        return duration

    def _collect_acceptance_rate(self, context: Dict[str, Any]) -> float:
        """collection acceptance rate（0-1）"""
        user_feedback = context.get("user_feedback", {})

        # First check if there is an explicitacceptedField
        if "accepted" in user_feedback:
            return 1.0 if user_feedback["accepted"] else 0.0

        # if not clearaccepted，examinerevisionsto infer
        if "revisions" in user_feedback:
            # The more modifications，The lower the acceptance rate
            revisions = user_feedback["revisions"]
            return max(0.0, 1.0 - revisions * 0.2)

        # Accept by default
        return 1.0

    def _collect_adoption_rate(self, context: Dict[str, Any]) -> float:
        """Gather Adoption Rate（0-1）"""
        task_result = context.get("task_result", {})

        # Check if the generated code is used
        if "code_adoption" in task_result:
            return task_result["code_adoption"]

        # Check ifcommitGot it
        if "committed" in task_result:
            return 1.0 if task_result["committed"] else 0.0

        # Used by default
        return 1.0

    def _collect_rewrite_rate(self, context: Dict[str, Any]) -> float:
        """Collect rewrite rate（negative reward，0-1）"""
        user_feedback = context.get("user_feedback", {})

        # Check for extensive rewrites
        if "rewrite_percentage" in user_feedback:
            return user_feedback["rewrite_percentage"]

        # Check the number of modifications
        if "revisions" in user_feedback:
            revisions = user_feedback["revisions"]
            # hypothesis3More than one revision means extensive rewriting
            return min(1.0, revisions / 3.0)

        return 0.0

    def _collect_user_rating(self, context: Dict[str, Any]) -> float:
        """Collect user ratings（1-5）"""
        user_feedback = context.get("user_feedback", {})

        if "rating" in user_feedback:
            return user_feedback["rating"]

        # Without explicit scoring，fromacceptanceinfer
        acceptance = self._collect_acceptance_rate(context)
        if acceptance == 1.0:
            return 4.0  # accept = 4point
        else:
            return 2.0  # reject = 2point

    def _collect_feedback_count(self, context: Dict[str, Any]) -> float:
        """Number of feedback collected"""
        user_feedback = context.get("user_feedback", {})

        count = 0

        # positive feedback
        if user_feedback.get("positive_comments"):
            count += len(user_feedback["positive_comments"])

        # negative feedback
        if user_feedback.get("negative_comments"):
            count += len(user_feedback["negative_comments"])

        return count

    def _collect_agent_preference(self, context: Dict[str, Any]) -> float:
        """collectAgentSelect preferences（0-1）"""
        task_result = context.get("task_result", {})
        used_agent = task_result.get("agent", "claude")
        task_type = context.get("task_type", "T2")

        # Prioritize learner history
        if self.history_provider:
            history_score = self.history_provider.get_agent_success_rate(task_type, used_agent)
            if history_score is not None:
                return max(0.0, min(1.0, float(history_score)))

        # Fall back to default rules：Task type andAgentmatch

        # Simple matching rules
        preference_match = {
            ("T1", "claude"): 0.9,
            ("T2", "claude"): 0.8,
            ("T3", "codex"): 0.9,
            ("T4", "codex"): 0.95,
        }

        return preference_match.get((task_type, used_agent), 0.5)

    def _collect_workflow_preference(self, context: Dict[str, Any]) -> float:
        """Gather workflow preferences（0-1）"""
        task_result = context.get("task_result", {})
        workflow = task_result.get("workflow", "standard")
        task_type = context.get("task_type", "T2")

        # Prioritize learner history
        if self.history_provider:
            history_score = self.history_provider.get_workflow_success_rate(task_type, workflow)
            if history_score is not None:
                return max(0.0, min(1.0, float(history_score)))

        # Fall back to default rules：TDDWorkflow rewards are higher
        if workflow == "tdd":
            return 0.9
        elif workflow == "test_first":
            return 0.85
        else:
            return 0.6

    # ===== Tool method =====

    def get_reward_stats(self) -> Dict[str, Any]:
        """Get reward statistics"""
        if not self.reward_history:
            return {}

        rewards = np.array(self.reward_history)

        return {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "latest": float(rewards[-1]),
            "count": len(rewards)
        }

    def get_signal_stats(self, signal_name: str) -> Dict[str, Any]:
        """Get statistics for a single signal"""
        if signal_name not in self.signals:
            return {}

        signal = self.signals[signal_name]

        if not signal.history:
            return {}

        history = np.array(signal.history)

        return {
            "mean": float(np.mean(history)),
            "std": float(np.std(history)),
            "min": float(np.min(history)),
            "max": float(np.max(history)),
            "latest": float(history[-1]),
            "count": len(history),
            "current_weight": signal.weight
        }

    def save_state(self, path: str | Path) -> None:
        """Save reward calculator state"""
        state = {
            "learning_phase": self.learning_phase,
            "task_count": self.task_count,
            "negative_feedback_count": self.negative_feedback_count,
            "reward_history": self.reward_history,
            "signals": {
                name: {
                    "weight": signal.weight,
                    "history": signal.history
                }
                for name, signal in self.signals.items()
            }
        }

        state_path = Path(path).expanduser()
        state_path.parent.mkdir(parents=True, exist_ok=True)

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str | Path) -> None:
        """Loading reward calculator status"""
        state_path = Path(path).expanduser()

        if not state_path.exists():
            return

        with open(state_path, 'r') as f:
            state = json.load(f)

        self.learning_phase = state.get("learning_phase", "early")
        self.task_count = state.get("task_count", 0)
        self.negative_feedback_count = state.get("negative_feedback_count", 0)
        self.reward_history = state.get("reward_history", [])

        # Restore signal status
        for name, signal_state in state.get("signals", {}).items():
            if name in self.signals:
                self.signals[name].weight = signal_state.get("weight", self.signals[name].weight)
                self.signals[name].history = signal_state.get("history", [])


def main():
    """Test reward calculator"""
    # Create a reward calculator
    calculator = RewardCalculator(learning_phase="early")

    # Simulate task context
    context = {
        "task_type": "T2",
        "task_result": {
            "agent": "claude",
            "workflow": "tdd",
            "duration": 300,  # 5minute
            "test_files_created": True,
            "tests_passed": True,
            "committed": True,
            "code_adoption": 0.9
        },
        "test_result": {
            "coverage": 85.0,
            "passed": 10,
            "failed": 1
        },
        "user_feedback": {
            "accepted": True,
            "rating": 4,
            "positive_comments": ["very good", "helpful"],
            "revisions": 1
        },
        "metrics": {
            "complexity": 3,
            "duplication": 0.05,
            "lint_score": 0.9
        }
    }

    # Calculate rewards
    reward = calculator.calculate_reward(context)

    print(f"total reward: {reward:.3f}")
    print("\nReward details:")
    for name, detail in context["debug_info"]["reward_breakdown"].items():
        print(f"  {name}: {detail['value']:.3f} × {detail['weight']:.2f} = {detail['weighted']:.3f}")

    print("\nReward statistics:")
    stats = calculator.get_reward_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Simulate negative feedback
    print("\nRecord negative feedback...")
    calculator.record_feedback("rating", 2)

    print("adjusted weight:")
    for name, signal in calculator.signals.items():
        print(f"  {name}: {signal.weight:.3f}")


if __name__ == "__main__":
    main()
