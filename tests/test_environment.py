#!/usr/bin/env python3
"""
Interactive environment unit testing
"""

import pytest
import numpy as np
from lib.environment import (
    InteractionEnvironment, State, Action,
    AgentType, AutomationLevel, CommunicationStyle
)


class TestState:
    """testStatekind"""

    def test_state_creation(self):
        """Test state creation"""
        state = State(
            task_type=np.array([1, 0, 0, 0]),
            tech_stack=np.array([1, 0, 0, 0, 1, 0, 0, 0]),
            user_mood=np.array([1, 0, 0]),
            time_of_day=0.5,
            recent_performance=0.7,
            agent_usage_history={"claude": 5, "codex": 2}
        )

        assert state.task_type.shape == (4,)
        assert state.tech_stack.shape == (8,)
        assert state.user_mood.shape == (3,)
        assert state.time_of_day == 0.5
        assert state.recent_performance == 0.7

    def test_to_vector(self):
        """Test state vector transitions"""
        state = State(
            task_type=np.array([1, 0, 0, 0]),
            tech_stack=np.array([1, 0, 0, 0, 0, 0, 0, 0]),
            user_mood=np.array([0, 1, 0]),
            time_of_day=0.5,
            recent_performance=0.8,
            agent_usage_history={}
        )

        vector = state.to_vector()

        # 4 + 8 + 3 + 1 + 1 = 17
        assert vector.shape == (17,)
        assert vector[0] == 1.0  # task_type[0]
        assert vector[4] == 1.0  # tech_stack[0]
        assert vector[13] == 1.0  # user_mood[1]


class TestAction:
    """testActionkind"""

    def test_action_creation(self):
        """Test action creation"""
        action = Action(
            agent_selection=AgentType.CLAUDE,
            automation_level=AutomationLevel.HIGH,
            communication_style=CommunicationStyle.BRIEF,
            confirmation_needed=False
        )

        assert action.agent_selection == AgentType.CLAUDE
        assert action.automation_level == AutomationLevel.HIGH
        assert action.communication_style == CommunicationStyle.BRIEF
        assert not action.confirmation_needed

    def test_to_vector(self):
        """Test action vector conversion"""
        action = Action(
            agent_selection=AgentType.GEMINI,
            automation_level=AutomationLevel.MEDIUM,
            communication_style=CommunicationStyle.INTERACTIVE,
            confirmation_needed=True
        )

        vector = action.to_vector(
            InteractionEnvironment.AGENT_MAP,
            InteractionEnvironment.AUTOMATION_MAP,
            InteractionEnvironment.STYLE_MAP,
            InteractionEnvironment.CONFIRM_MAP
        )

        # 3 + 3 + 3 + 2 = 11
        assert vector.shape == (11,)
        assert vector[2] == 1.0  # gemini
        assert vector[4] == 1.0  # medium
        assert vector[8] == 1.0  # interactive
        assert vector[10] == 1.0  # confirmation (True)


class TestInteractionEnvironment:
    """testInteractionEnvironmentkind"""

    def test_initialization(self):
        """Test environment initialization"""
        env = InteractionEnvironment()

        assert env.get_state_space_size() == 17
        assert env.get_action_space_size() == 11
        assert env.episode_count == 0
        assert env.recent_performance == 0.5

    def test_reset(self):
        """Test environment reset"""
        env = InteractionEnvironment()

        task_context = {
            "task_type": "T2",
            "tech_stack": ["python", "fastapi"],
            "user_mood": "focused",
            "time_of_day": 14.0
        }

        state = env.reset(task_context)

        assert state is not None
        assert state.task_type[1] == 1.0  # T2
        assert state.time_of_day == pytest.approx(14.0 / 24.0)

    def test_reset_with_default_values(self):
        """Test reset using default values"""
        env = InteractionEnvironment()

        task_context = {
            "task_type": "T1"
        }

        state = env.reset(task_context)

        # Default value check
        assert state.user_mood[0] == 1.0  # focused
        assert state.time_of_day == pytest.approx(12.0 / 24.0)  # noon

    def test_reset_invalid_task_type_fallback_to_t2(self):
        """Test is illegaltask_typeAutomatically fall back toT2"""
        env = InteractionEnvironment()
        state = env.reset({"task_type": "INVALID", "tech_stack": ["python"]})
        assert state.task_type[1] == 1.0  # T2
        assert env.current_task_context["task_type"] == "T2"

    def test_reset_clamps_time_of_day_into_valid_range(self):
        """testtime_of_dayrestricted to[0, 24]Then normalize"""
        env = InteractionEnvironment()
        state_early = env.reset({"task_type": "T2", "time_of_day": -3})
        state_late = env.reset({"task_type": "T2", "time_of_day": 30})
        assert state_early.time_of_day == 0.0
        assert state_late.time_of_day == 1.0

    def test_reset_invalid_time_of_day_uses_default_noon(self):
        """Test is illegaltime_of_dayEnter to fall back to the default noon"""
        env = InteractionEnvironment()
        state = env.reset({"task_type": "T2", "time_of_day": "not-a-number"})
        assert state.time_of_day == pytest.approx(0.5)

    def test_step(self):
        """Test execution steps"""
        env = InteractionEnvironment()

        # Reset environment
        task_context = {"task_type": "T2", "tech_stack": ["python"]}
        env.reset(task_context)

        # Create action
        action = Action(
            agent_selection=AgentType.CLAUDE,
            automation_level=AutomationLevel.MEDIUM,
            communication_style=CommunicationStyle.DETAILED,
            confirmation_needed=True
        )

        # Simulation task results
        task_result = {
            "duration": 300,
            "completed": True,
            "test_result": {"coverage": 80.0, "passed": 10, "failed": 0},
            "user_feedback": {"accepted": True, "rating": 5},
            "metrics": {"complexity": 2}
        }

        # Execution steps
        next_state, reward, done, info = env.step(action, task_result)

        # Check return value
        assert next_state is not None
        assert 0.0 <= reward <= 1.0
        assert done
        assert "action_taken" in info
        assert "reward_breakdown" in info

        # examineAgentUsage history updated
        assert env.agent_usage_history["claude"] == 1

    def test_step_without_reset_raises_value_error(self):
        """Not testedresetdirectstepwill throw an explicit error"""
        env = InteractionEnvironment()
        action = Action(
            agent_selection=AgentType.CLAUDE,
            automation_level=AutomationLevel.MEDIUM,
            communication_style=CommunicationStyle.DETAILED,
            confirmation_needed=True
        )
        with pytest.raises(ValueError, match="reset"):
            env.step(action, {"completed": True})

    def test_step_done_updates_episode_stats(self):
        """Update after test completionepisodestatistics"""
        env = InteractionEnvironment()
        env.reset({"task_type": "T2", "tech_stack": ["python"]})
        action = Action(
            agent_selection=AgentType.CLAUDE,
            automation_level=AutomationLevel.MEDIUM,
            communication_style=CommunicationStyle.DETAILED,
            confirmation_needed=True
        )
        _, reward, done, _ = env.step(
            action,
            {
                "duration": 100,
                "completed": True,
                "test_result": {},
                "user_feedback": {},
                "metrics": {},
            },
        )
        assert done is True
        assert env.episode_count == 1
        assert env.episode_rewards[-1] == pytest.approx(reward)

    def test_multi_episode(self):
        """Test a lotepisoderun"""
        env = InteractionEnvironment()

        for episode in range(3):
            # reset
            task_context = {
                "task_type": f"T{(episode % 4) + 1}",
                "tech_stack": ["python"]
            }
            env.reset(task_context)

            # perform action
            action = Action(
                agent_selection=AgentType.CODEX,
                automation_level=AutomationLevel.HIGH,
                communication_style=CommunicationStyle.BRIEF,
                confirmation_needed=False
            )

            task_result = {
                "duration": 200,
                "completed": True,
                "test_result": {},
                "user_feedback": {},
                "metrics": {}
            }

            env.step(action, task_result)

        # examinecodexused3Second-rate
        assert env.agent_usage_history["codex"] == 3

    def test_performance_update(self):
        """Test performance updates"""
        env = InteractionEnvironment()

        # initial performance
        assert env.recent_performance == 0.5

        # Reset and perform high reward tasks
        env.reset({"task_type": "T2", "tech_stack": ["python"]})

        action = Action(
            agent_selection=AgentType.CLAUDE,
            automation_level=AutomationLevel.MEDIUM,
            communication_style=CommunicationStyle.DETAILED,
            confirmation_needed=True
        )

        task_result = {
            "duration": 100,
            "completed": True,
            "test_result": {"coverage": 100.0, "passed": 10, "failed": 0},
            "user_feedback": {"accepted": True, "rating": 5},
            "metrics": {"complexity": 1, "duplication": 0.0, "lint_score": 1.0}
        }

        env.step(action, task_result)

        # Performance should improve
        assert env.recent_performance > 0.5

    def test_save_load_history(self):
        """Test history saving and loading"""
        import tempfile
        import os

        env1 = InteractionEnvironment()

        # run aepisode
        env1.reset({"task_type": "T2", "tech_stack": ["python"]})

        action = Action(
            agent_selection=AgentType.GEMINI,
            automation_level=AutomationLevel.LOW,
            communication_style=CommunicationStyle.INTERACTIVE,
            confirmation_needed=True
        )

        task_result = {
            "duration": 400,
            "completed": True,
            "test_result": {},
            "user_feedback": {},
            "metrics": {}
        }

        env1.step(action, task_result)

        # save history
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            env1.save_history(temp_path)

            # Create new environment and load history
            env2 = InteractionEnvironment(config_path=temp_path)

            # Check history restored
            assert env2.agent_usage_history["gemini"] == 1

        finally:
            os.unlink(temp_path)

    def test_tech_stack_encoding(self):
        """Test technology stack coding"""
        env = InteractionEnvironment()

        # single technology
        state = env.reset({"task_type": "T2", "tech_stack": ["python"]})
        assert state.tech_stack[4] == 1.0  # python index

        # multiple technologies
        state = env.reset({"task_type": "T2", "tech_stack": ["python", "react"]})
        assert state.tech_stack[4] == 1.0  # python
        assert state.tech_stack[0] == 1.0  # react

        # unknown technology（should default topython）
        state = env.reset({"task_type": "T2", "tech_stack": ["unknown_tech"]})
        assert state.tech_stack[4] == 1.0  # python as default

    def test_task_type_encoding(self):
        """Test task type encoding"""
        env = InteractionEnvironment()

        for i, task_type in enumerate(["T1", "T2", "T3", "T4"], 1):
            state = env.reset({"task_type": task_type, "tech_stack": ["python"]})
            assert state.task_type[i-1] == 1.0
            assert np.sum(state.task_type) == 1.0

    def test_user_mood_encoding(self):
        """Test user mood coding"""
        env = InteractionEnvironment()

        for mood in ["focused", "relaxed", "stressed"]:
            state = env.reset({
                "task_type": "T2",
                "tech_stack": ["python"],
                "user_mood": mood
            })

            assert np.sum(state.user_mood) == 1.0


class TestEnvironmentIntegration:
    """Integration testing"""

    def test_full_episode(self):
        """Test completeepisode"""
        env = InteractionEnvironment()

        # 1. Reset
        task_context = {
            "task_type": "T3",
            "tech_stack": ["react", "typescript"],
            "user_mood": "stressed",
            "time_of_day": 16.0
        }

        state = env.reset(task_context)
        assert state is not None

        # 2. Step
        action = Action(
            agent_selection=AgentType.CODEX,
            automation_level=AutomationLevel.HIGH,
            communication_style=CommunicationStyle.BRIEF,
            confirmation_needed=False
        )

        task_result = {
            "duration": 600,  # 10minute
            "completed": True,
            "test_result": {
                "coverage": 75.0,
                "passed": 8,
                "failed": 2
            },
            "user_feedback": {
                "accepted": True,
                "rating": 4,
                "revisions": 1
            },
            "metrics": {
                "complexity": 4,
                "duplication": 0.08,
                "lint_score": 0.85
            }
        }

        next_state, reward, done, info = env.step(action, task_result)

        # 3. verify
        assert 0.0 <= reward <= 1.0
        assert done
        assert info["agent_used"] == "codex"
        assert "reward_breakdown" in info

        # 4. Check status updated
        assert next_state.recent_performance >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
