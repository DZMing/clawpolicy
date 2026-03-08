#!/usr/bin/env python3
"""
Reinforcement learning interactive environment - ClawPolicy interactive environment

Define state space and action space，accomplish ClawPolicy interactive OpenAI Gymstyle interface：
- state: task context（task_type, tech_stack, user_moodwait）
- action: Agentchoose（agent_selection, automation_levelwait）
- reward: fromreward.RewardCalculatorcalculate
- done: Is the task completed?

support：
- reset(task_context): Reset environment to new task
- step(action): perform action → (next_state, reward, done, info)
- _encode_state(context): Encode context into state vector
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .contracts import (
    ACTION_HEAD_DIMS,
    ACTION_VECTOR_DIM,
    AGENT_ORDER,
    AUTOMATION_ORDER,
    CONFIRM_ORDER,
    STATE_DIMENSIONS,
    STATE_VECTOR_DIM,
    STYLE_ORDER,
    TECH_STACK_ORDER,
    USER_MOOD_ORDER,
)
from .reward import RewardCalculator


class TaskType(Enum):
    """Task type enum"""
    T1 = "T1"  # lightweight：<20OK
    T2 = "T2"  # medium：20-200OK
    T3 = "T3"  # Severe：200+OK
    T4 = "T4"  # Danger：core system


class AgentType(Enum):
    """Agenttype enum"""
    CLAUDE = "claude"
    CODEX = "codex"
    GEMINI = "gemini"


class AutomationLevel(Enum):
    """Automation level enumeration"""
    LOW = "low"  # Requires frequent confirmation
    MEDIUM = "medium"  # Partially automated
    HIGH = "high"  # Highly automated


class CommunicationStyle(Enum):
    """communication style enum"""
    BRIEF = "brief"  # concise
    DETAILED = "detailed"  # detailed
    INTERACTIVE = "interactive"  # interactive


@dataclass
class State:
    """Status data class"""
    # Task type（one-hotcoding）
    task_type: np.ndarray  # [4] (T1, T2, T3, T4)

    # technology stack（one-hotcoding）
    tech_stack: np.ndarray  # [N] Based on the number of supported technologies

    # User mood（one-hotcoding）
    user_mood: np.ndarray  # [3] (focused, relaxed, stressed)

    # time（normalization0-1）
    time_of_day: float  # 0-1 (0=midnight, 0.5=noon, 1=midnight)

    # Recent performance（normalization0-1）
    recent_performance: float  # 0-1

    # historyAgentuse（statistics）
    agent_usage_history: Dict[str, int]  # agentname -> Number of uses

    def to_vector(self) -> np.ndarray:
        """Convert state to vector"""
        # Combine all features
        vectors: List[np.ndarray] = [
            self.task_type.flatten(),
            self.tech_stack.flatten(),
            self.user_mood.flatten(),
            np.array([self.time_of_day], dtype=float),
            np.array([self.recent_performance], dtype=float),
        ]

        return np.concatenate(vectors)

    def __repr__(self) -> str:
        return (f"State(task_type={self.task_type}, "
                f"tech_stack={self.tech_stack}, "
                f"user_mood={self.user_mood}, "
                f"time={self.time_of_day:.2f}, "
                f"perf={self.recent_performance:.2f})")


@dataclass
class Action:
    """action data class"""
    agent_selection: AgentType  # Which one to chooseAgent
    automation_level: AutomationLevel  # Automation level
    communication_style: CommunicationStyle  # communication style
    confirmation_needed: bool  # Do you need to confirm

    def to_vector(self, agent_map: Dict[AgentType, int],
                  automation_map: Dict[AutomationLevel, int],
                  style_map: Dict[CommunicationStyle, int],
                  confirm_map: Dict[bool, int]) -> np.ndarray:
        """Convert action toone-hotencoding vector"""
        vector_size = (
            len(agent_map) +
            len(automation_map) +
            len(style_map) +
            len(confirm_map)  # confirmation_needed
        )

        vector = np.zeros(vector_size)

        # Agentchoose（one-hot）
        vector[agent_map[self.agent_selection]] = 1

        # Automation level（one-hot）
        offset = len(agent_map)
        vector[offset + automation_map[self.automation_level]] = 1

        # communication style（one-hot）
        offset += len(automation_map)
        vector[offset + style_map[self.communication_style]] = 1

        # Confirmation sign
        offset += len(style_map)
        vector[offset + confirm_map[self.confirmation_needed]] = 1.0

        return vector

    def __repr__(self) -> str:
        return (f"Action(agent={self.agent_selection.value}, "
                f"automation={self.automation_level.value}, "
                f"style={self.communication_style.value}, "
                f"confirm={self.confirmation_needed})")


class InteractionEnvironment:
    """
    ClawPolicy interactive environment

    accomplishOpenAI Gymstyle environment interface：
    - observation_space: state space dimensions
    - action_space: action space dimensions
    - reset(): Reset environment
    - step(): perform action
    """

    # State space definition
    STATE_DIM = STATE_DIMENSIONS.copy()
    TOTAL_STATE_DIM = STATE_VECTOR_DIM  # 17dimension

    # action space definition
    ACTION_DIM = {
        "agent_selection": ACTION_HEAD_DIMS["agent"],
        "automation_level": ACTION_HEAD_DIMS["automation"],
        "communication_style": ACTION_HEAD_DIMS["style"],
        "confirmation_needed": ACTION_HEAD_DIMS["confirm"],
    }
    TOTAL_ACTION_DIM = ACTION_VECTOR_DIM  # 11dimension

    # Supported technology stack
    SUPPORTED_TECH = {name: idx for idx, name in enumerate(TECH_STACK_ORDER)}

    # Agentmapping
    AGENT_MAP = {AgentType(name): idx for idx, name in enumerate(AGENT_ORDER)}

    # Automation level mapping
    AUTOMATION_MAP = {AutomationLevel(name): idx for idx, name in enumerate(AUTOMATION_ORDER)}

    # communication style mapping
    STYLE_MAP = {CommunicationStyle(name): idx for idx, name in enumerate(STYLE_ORDER)}

    # Confirm flag mapping
    CONFIRM_MAP = {value: idx for idx, value in enumerate(CONFIRM_ORDER)}

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize environment

        Args:
            config_path: Model configuration path（Used to load historical data）
        """
        self.reward_calculator = RewardCalculator()

        # Current status
        self.current_state: Optional[State] = None
        self.current_task_context: Optional[Dict[str, Any]] = None

        # historical statistics
        self.episode_rewards: List[float] = []
        self.episode_count = 0
        self.agent_usage_history: Dict[str, int] = {
            "claude": 0,
            "codex": 0,
            "gemini": 0
        }

        # Recent performance（Initial is0.5Means medium）
        self.recent_performance: float = 0.5

        # Try loading history
        if config_path:
            self._load_history(config_path)

    def reset(self, task_context: Dict[str, Any]) -> State:
        """
        Reset environment to new task

        Args:
            task_context: task context，Include：
                - task_type: Task type（"T1", "T2", "T3", "T4"）
                - tech_stack: Technology stack used（list）
                - user_mood: User mood（"focused", "relaxed", "stressed"，Optional）
                - time_of_day: current time（0-24Hour，Optional）

        Returns:
            initial state
        """
        # Parse task type
        task_type_str = str(task_context.get("task_type", "T2")).upper()
        if task_type_str not in TaskType.__members__:
            task_type_str = "T2"
        task_type = TaskType[task_type_str]
        task_context_normalized = dict(task_context)
        task_context_normalized["task_type"] = task_type_str

        # Analyze technology stack
        tech_stacks = task_context.get("tech_stack", ["python"])
        tech_stack_vector = self._encode_tech_stack(tech_stacks)

        # Analyze user mood（Default isfocused）
        user_mood_str = task_context.get("user_mood", "focused")
        user_mood_vector = self._encode_user_mood(user_mood_str)

        # parsing time（Default is noon）
        time_of_day = task_context.get("time_of_day", 12.0)
        try:
            time_of_day = float(time_of_day)
        except (TypeError, ValueError):
            time_of_day = 12.0
        time_of_day = max(0.0, min(24.0, time_of_day))
        time_normalized = time_of_day / 24.0
        task_context_normalized["time_of_day"] = time_of_day

        self.current_task_context = task_context_normalized

        # Create status
        self.current_state = State(
            task_type=self._encode_task_type(task_type),
            tech_stack=tech_stack_vector,
            user_mood=user_mood_vector,
            time_of_day=time_normalized,
            recent_performance=self.recent_performance,
            agent_usage_history=self.agent_usage_history.copy()
        )

        return self.current_state

    def step(self, action: Action, task_result: Dict[str, Any]) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        perform action

        Args:
            action: action taken
            task_result: Task execution results，Include：
                - duration: Task time（Second）
                - test_result: Test results
                - user_feedback: User feedback
                - metrics: Other indicators

        Returns:
            (next_state, reward, done, info)
        """
        if self.current_state is None or self.current_task_context is None:
            raise ValueError("Environment not initialized, call reset() before step().")

        # 1. RecordAgentuse
        agent_name = action.agent_selection.value
        self.agent_usage_history[agent_name] += 1

        # 2. Prepare context for reward calculation
        reward_context = {
            "task_type": self.current_task_context.get("task_type", "T2"),
            "task_result": task_result,
            "test_result": task_result.get("test_result", {}),
            "user_feedback": task_result.get("user_feedback", {}),
            "metrics": task_result.get("metrics", {})
        }

        # 3. Calculate rewards
        reward = self.reward_calculator.calculate_reward(reward_context)

        # 4. Update recent performance（moving average）
        self.recent_performance = 0.7 * self.recent_performance + 0.3 * reward

        # 5. Prepare for next state（keep current context，But update performance）
        next_state = State(
            task_type=self.current_state.task_type.copy(),
            tech_stack=self.current_state.tech_stack.copy(),
            user_mood=self.current_state.user_mood.copy(),
            time_of_day=self.current_state.time_of_day,
            recent_performance=self.recent_performance,
            agent_usage_history=self.agent_usage_history.copy()
        )

        self.current_state = next_state

        # 6. examineepisodeIs it over?
        done = task_result.get("completed", True)
        if done:
            self.episode_count += 1
            self.episode_rewards.append(float(reward))

        # 7. Prepareinfo
        info = {
            "action_taken": str(action),
            "agent_used": agent_name,
            "reward_breakdown": reward_context.get("debug_info", {}).get("reward_breakdown", {}),
            "task_duration": task_result.get("duration", 0)
        }

        return next_state, reward, done, info

    def _encode_task_type(self, task_type: TaskType) -> np.ndarray:
        """Encode the task type asone-hotvector"""
        vector = np.zeros(self.STATE_DIM["task_type"])
        index = list(TaskType).index(task_type)
        vector[index] = 1.0
        return vector

    def _encode_tech_stack(self, tech_stacks: List[str]) -> np.ndarray:
        """Codify the technology stack asone-hotvector（Support multiple tags）"""
        vector = np.zeros(self.STATE_DIM["tech_stack"])

        for tech in tech_stacks:
            tech_lower = tech.lower()
            # Check for direct matches
            if tech_lower in self.SUPPORTED_TECH:
                vector[self.SUPPORTED_TECH[tech_lower]] = 1.0
            # Check for partial matches
            else:
                for supported_tech, idx in self.SUPPORTED_TECH.items():
                    if supported_tech in tech_lower:
                        vector[idx] = 1.0

        # If no technology matches，defaultpython
        if not vector.any():
            vector[self.SUPPORTED_TECH["python"]] = 1.0

        return vector

    def _encode_user_mood(self, user_mood: str) -> np.ndarray:
        """Encode user mood asone-hotvector"""
        moods = list(USER_MOOD_ORDER)
        vector = np.zeros(len(moods))

        if user_mood in moods:
            index = moods.index(user_mood)
            vector[index] = 1.0
        else:
            # defaultfocused
            vector[0] = 1.0

        return vector

    def get_action_space_size(self) -> int:
        """Get action space size"""
        return self.TOTAL_ACTION_DIM

    def get_state_space_size(self) -> int:
        """Get the state space size"""
        return self.TOTAL_STATE_DIM

    def save_history(self, path: str | Path) -> None:
        """Save history"""
        history = {
            "episode_count": self.episode_count,
            "agent_usage_history": self.agent_usage_history,
            "recent_performance": self.recent_performance,
            "reward_history": self.episode_rewards
        }

        history_path = Path(path).expanduser()
        history_path.parent.mkdir(parents=True, exist_ok=True)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    def _load_history(self, path: str | Path) -> None:
        """Load history"""
        history_path = Path(path).expanduser()

        if not history_path.exists():
            return

        with open(history_path, 'r') as f:
            history = json.load(f)

        self.episode_count = history.get("episode_count", 0)
        self.agent_usage_history = history.get("agent_usage_history", self.agent_usage_history)
        self.recent_performance = history.get("recent_performance", 0.5)
        self.episode_rewards = history.get("reward_history", [])


def main():
    """Test interactive environment"""
    # Create environment
    env = InteractionEnvironment()

    # Simulate task context
    task_context = {
        "task_type": "T2",
        "tech_stack": ["python", "fastapi"],
        "user_mood": "focused",
        "time_of_day": 14.0  # afternoon2point
    }

    # Reset environment
    state = env.reset(task_context)
    print(f"initial state: {state}")
    print(f"state vector: {state.to_vector()}")
    print(f"State space size: {env.get_state_space_size()}")
    print(f"action space size: {env.get_action_space_size()}")

    # simulate action
    action = Action(
        agent_selection=AgentType.CLAUDE,
        automation_level=AutomationLevel.MEDIUM,
        communication_style=CommunicationStyle.DETAILED,
        confirmation_needed=True
    )

    print(f"\ntake action: {action}")

    # Simulation task results
    task_result = {
        "duration": 300,  # 5minute
        "completed": True,
        "test_result": {
            "coverage": 85.0,
            "passed": 10,
            "failed": 1
        },
        "user_feedback": {
            "accepted": True,
            "rating": 4
        },
        "metrics": {
            "complexity": 3,
            "duplication": 0.05,
            "lint_score": 0.9
        }
    }

    # Execution steps
    next_state, reward, done, info = env.step(action, task_result)

    print(f"\nnext state: {next_state}")
    print(f"award: {reward:.3f}")
    print(f"Finish: {done}")
    print(f"Info: {info}")

    # save history
    env.save_history("/tmp/clawpolicy_env_history.json")
    print("\nHistory saved to /tmp/clawpolicy_env_history.json")


if __name__ == "__main__":
    main()
