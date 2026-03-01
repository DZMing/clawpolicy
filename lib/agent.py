#!/usr/bin/env python3
"""
reinforcement learning agent - Actor-Criticaccomplish

accomplishActor-Criticalgorithm，include：
- PolicyNetwork: policy network（Output action probability distribution）
- ValueNetwork: value network（Estimated status value）
- AlignmentAgent: Actor-Criticagent

Phase 1: pureNumPyaccomplish（linear model）
Phase 2: OptionalPyTorchaccomplish（neural network）
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from .contracts import (
    ACTION_HEAD_DIMS,
    ACTION_VECTOR_DIM,
    AGENT_ORDER,
    AUTOMATION_ORDER,
    CONFIRM_ORDER,
    STYLE_ORDER,
)
from .environment import State, Action, AgentType, AutomationLevel, CommunicationStyle


@dataclass
class Trajectory:
    """Trajectory data class"""
    states: List[np.ndarray]  # status sequence
    actions: List[np.ndarray]  # action sequence（index vector）
    rewards: List[float]  # reward sequence
    dones: List[bool]  # Done sign
    next_states: List[np.ndarray]  # next state sequence

    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:
        return f"Trajectory(length={len(self)}, total_reward={sum(self.rewards):.2f})"


class PolicyNetwork:
    """
    policy network - Output action probability distribution

    Phase 1: linear model（logits = state @ weights + bias）
    Phase 2: Optional neural network（PyTorch）
    """

    def __init__(self, state_dim: int, action_dim: int = ACTION_VECTOR_DIM, hidden_dim: int = 64):
        """
        Initialize policy network

        Args:
            state_dim: status dimension
            action_dim: action dimension
            hidden_dim: Hidden layer dimensions（Phase 2use）
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Bull action space
        self.head_dims = ACTION_HEAD_DIMS.copy()

        # Phase 1: Linear model parameters（long）
        self.weights = {
            name: np.random.randn(state_dim, dim) * 0.01
            for name, dim in self.head_dims.items()
        }
        self.bias = {
            name: np.zeros(dim)
            for name, dim in self.head_dims.items()
        }

    def forward(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """forward propagation：Count each headlogits"""
        return {
            name: state @ self.weights[name] + self.bias[name]
            for name in self.head_dims
        }

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Softmaxactivation function"""
        # numerical stability：Subtract the maximum value
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def get_action_probs(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Get action probability distribution（long）"""
        logits = self.forward(state)
        return {name: self.softmax(head_logits) for name, head_logits in logits.items()}

    def sample_action(self, state: np.ndarray, explore: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Sampling action

        Args:
            state: Current status
            explore: Whether to explore（epsilon-greedy）

        Returns:
            (action_indices, action_probs)
        """
        action_probs = self.get_action_probs(state)

        if not explore:
            # reasoning mode：greedy choice，Avoid random jitter in recommended actions
            action_indices = np.array([
                int(np.argmax(action_probs["agent"])),
                int(np.argmax(action_probs["automation"])),
                int(np.argmax(action_probs["style"])),
                int(np.argmax(action_probs["confirm"]))
            ], dtype=int)
        elif np.random.random() < 0.1:  # 10% epsilon-greedy
            # Random exploration（Each head is independently randomized）
            action_indices = np.array([
                np.random.randint(self.head_dims["agent"]),
                np.random.randint(self.head_dims["automation"]),
                np.random.randint(self.head_dims["style"]),
                np.random.randint(self.head_dims["confirm"])
            ], dtype=int)
        else:
            # Sampling by probability（Each head is sampled independently）
            action_indices = np.array([
                np.random.choice(self.head_dims["agent"], p=action_probs["agent"]),
                np.random.choice(self.head_dims["automation"], p=action_probs["automation"]),
                np.random.choice(self.head_dims["style"], p=action_probs["style"]),
                np.random.choice(self.head_dims["confirm"], p=action_probs["confirm"])
            ], dtype=int)

        return action_indices, action_probs

    def update(self, state: np.ndarray, action_indices: np.ndarray, advantage: float,
               learning_rate: float = 0.01) -> float:
        """
        Update policy network（REINFORCEalgorithm）

        Args:
            state: Current status
            action_indices: Action index executed（long）
            advantage: advantage function A(s,a) = Q(s,a) - V(s)
            learning_rate: learning rate

        Returns:
            loss value
        """
        action_probs = self.get_action_probs(state)

        total_loss = 0.0
        head_order = ["agent", "automation", "style", "confirm"]

        for head_idx, head_name in enumerate(head_order):
            probs = action_probs[head_name]
            action_idx = int(action_indices[head_idx])
            log_prob = np.log(probs[action_idx] + 1e-10)

            # Policy gradient: -log π(a|s) * A
            loss = -log_prob * advantage
            total_loss += loss

            one_hot = np.zeros_like(probs)
            one_hot[action_idx] = 1.0
            grad_logits = (one_hot - probs) * advantage

            # gradient ascent（Equivalent to loss reduction）
            self.weights[head_name] += learning_rate * np.outer(state, grad_logits)
            self.bias[head_name] += learning_rate * grad_logits

        return float(total_loss)

    def save(self, path: str | Path) -> None:
        """Save model parameters"""
        params = {
            "weights": {name: w.tolist() for name, w in self.weights.items()},
            "bias": {name: b.tolist() for name, b in self.bias.items()},
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "head_dims": self.head_dims
        }

        model_path = Path(path).expanduser()
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, 'w') as f:
            json.dump(params, f)

    def load(self, path: str | Path) -> None:
        """Load model parameters"""
        model_path = Path(path).expanduser()

        if not model_path.exists():
            return

        with open(model_path, 'r') as f:
            params = json.load(f)

        self.weights = {name: np.array(w) for name, w in params["weights"].items()}
        self.bias = {name: np.array(b) for name, b in params["bias"].items()}
        self.state_dim = params["state_dim"]
        self.action_dim = params["action_dim"]
        self.head_dims = params.get("head_dims", self.head_dims)


class ValueNetwork:
    """
    value network - Estimated status value V(s)

    Phase 1: linear model
    Phase 2: Optional neural network（PyTorch）
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        """
        Initialize the value network

        Args:
            state_dim: status dimension
            hidden_dim: Hidden layer dimensions（Phase 2use）
        """
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Linear model parameters
        self.weights = np.random.randn(state_dim) * 0.01
        self.bias = 0.0

    def forward(self, state: np.ndarray) -> float:
        """forward propagation：Calculate status value"""
        return float(state @ self.weights + self.bias)

    def update(self, state: np.ndarray, target_value: float,
               learning_rate: float = 0.01) -> float:
        """
        Update value network（MSEloss）

        Args:
            state: Current status
            target_value: target value（actual return）
            learning_rate: learning rate

        Returns:
            loss value
        """
        # Calculate current value
        current_value = self.forward(state)

        # Calculate losses
        loss = (current_value - target_value) ** 2

        # Calculate gradient
        grad_w = 2 * (current_value - target_value) * state
        grad_b = 2 * (current_value - target_value)

        # Update weights
        self.weights -= learning_rate * grad_w
        self.bias -= learning_rate * grad_b

        return loss

    def save(self, path: str | Path) -> None:
        """Save model parameters"""
        params = {
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "state_dim": self.state_dim
        }

        model_path = Path(path).expanduser()
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, 'w') as f:
            json.dump(params, f)

    def load(self, path: str | Path) -> None:
        """Load model parameters"""
        model_path = Path(path).expanduser()

        if not model_path.exists():
            return

        with open(model_path, 'r') as f:
            params = json.load(f)

        self.weights = np.array(params["weights"])
        self.bias = params["bias"]
        self.state_dim = params["state_dim"]


class AlignmentAgent:
    """
    Actor-Criticagent

    Combining strategy network and value network，accomplishActor-Criticalgorithm：
    - Actor: policy network，Select action
    - Critic: value network，Evaluate state value

    Algorithm process：
    1. Select an action using the current strategy
    2. perform action，Get rewards and next status
    3. Compute advantage function A = R + γV(s') - V(s)
    4. renewActor：-log π(a|s) * A
    5. renewCritic：(V(s) - R)²
    """

    def __init__(self, state_dim: int, action_dim: int,
                 gamma: float = 0.99,
                 actor_lr: float = 0.01,
                 critic_lr: float = 0.01):
        """
        Initialize the agent

        Args:
            state_dim: status dimension
            action_dim: action dimension
            gamma: discount factor
            actor_lr: Actorlearning rate
            critic_lr: Criticlearning rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Initialize the network
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)

        # learning rate
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # training statistics
        self.episode_count = 0
        self.total_steps = 0

    def select_action(self, state: State, explore: bool = True) -> Action:
        """
        Select action

        Args:
            state: Current status
            explore: Whether to explore

        Returns:
            selected action
        """
        # Convert state to vector
        state_vector = state.to_vector()

        # Sampling action index
        action_indices, action_probs = self.policy_net.sample_action(state_vector, explore)

        # Convert action index toActionobject
        action = self.decode_action_indices(action_indices)

        return action

    def encode_action_indices(self, action: Action) -> np.ndarray:
        """WillActionencoded as index vector"""
        agent_idx = AGENT_ORDER.index(action.agent_selection.value)
        automation_idx = AUTOMATION_ORDER.index(action.automation_level.value)
        style_idx = STYLE_ORDER.index(action.communication_style.value)
        confirm_idx = CONFIRM_ORDER.index(action.confirmation_needed)

        return np.array([agent_idx, automation_idx, style_idx, confirm_idx], dtype=int)

    def decode_action_indices(self, action_indices: np.ndarray) -> Action:
        """Decode action index asActionobject"""
        indices = [int(x) for x in action_indices]
        if len(indices) != 4:
            raise ValueError(f"action_indices must contain 4 values, got {len(indices)}")
        agent_idx, automation_idx, style_idx, confirm_idx = indices

        if not (0 <= agent_idx < len(AGENT_ORDER)):
            raise ValueError(f"agent index out of range: {agent_idx}")
        if not (0 <= automation_idx < len(AUTOMATION_ORDER)):
            raise ValueError(f"automation index out of range: {automation_idx}")
        if not (0 <= style_idx < len(STYLE_ORDER)):
            raise ValueError(f"style index out of range: {style_idx}")
        if not (0 <= confirm_idx < len(CONFIRM_ORDER)):
            raise ValueError(f"confirm index out of range: {confirm_idx}")

        agent_type = AgentType(AGENT_ORDER[agent_idx])
        automation = AutomationLevel(AUTOMATION_ORDER[automation_idx])
        style = CommunicationStyle(STYLE_ORDER[style_idx])
        confirm = CONFIRM_ORDER[confirm_idx]

        return Action(
            agent_selection=agent_type,
            automation_level=automation,
            communication_style=style,
            confirmation_needed=confirm
        )

    def update_policy(self, trajectory: Trajectory) -> Dict[str, float]:
        """
        update strategy（Actor-Criticalgorithm）

        Args:
            trajectory: complete trajectory

        Returns:
            Loss statistics
        """
        if len(trajectory) == 0:
            return {}

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        # Gradually update
        for i in range(len(trajectory)):
            state = trajectory.states[i]
            action_indices = trajectory.actions[i]
            reward = trajectory.rewards[i]
            next_state = trajectory.next_states[i]
            done = trajectory.dones[i]

            # Calculate target value
            if done:
                target_value = reward
            else:
                target_value = reward + self.gamma * self.value_net.forward(next_state)

            # Compute advantage function
            current_value = self.value_net.forward(state)
            advantage = target_value - current_value

            # renewActor
            actor_loss = self.policy_net.update(state, action_indices, advantage, self.actor_lr)
            total_actor_loss += actor_loss

            # renewCritic
            critic_loss = self.value_net.update(state, target_value, self.critic_lr)
            total_critic_loss += critic_loss

        # Update statistics
        self.episode_count += 1
        self.total_steps += len(trajectory)

        return {
            "actor_loss": total_actor_loss / len(trajectory),
            "critic_loss": total_critic_loss / len(trajectory),
            "episode_length": len(trajectory),
            "total_return": sum(trajectory.rewards)
        }

    def _compute_returns(self, rewards: List[float], dones: List[bool]) -> List[float]:
        """
        Calculate discounted returns

        Args:
            rewards: reward sequence
            dones: Complete flag sequence

        Returns:
            discount return sequence
        """
        returns: List[float] = []
        running_return = 0.0

        # Calculate from back to front
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running_return = reward
            else:
                running_return = reward + self.gamma * running_return

            returns.insert(0, running_return)

        return returns

    def save_model(self, path: str) -> None:
        """Save model"""
        model_dir = Path(path).expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)

        # save policy network
        self.policy_net.save(str(model_dir / "policy_network.json"))

        # Save value network
        self.value_net.save(str(model_dir / "value_network.json"))

        # Save metadata
        metadata = {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "gamma": self.gamma,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr
        }

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_model(self, path: str) -> None:
        """Load model"""
        model_dir = Path(path).expanduser()

        # Load policy network
        self.policy_net.load(str(model_dir / "policy_network.json"))

        # Load value network
        self.value_net.load(str(model_dir / "value_network.json"))

        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.episode_count = metadata.get("episode_count", 0)
            self.total_steps = metadata.get("total_steps", 0)


def main():
    """test agent"""
    from .environment import InteractionEnvironment

    # Create environments and agents
    env = InteractionEnvironment()
    agent = AlignmentAgent(
        state_dim=env.get_state_space_size(),
        action_dim=env.get_action_space_size()
    )

    print("Agent has been created")
    print(f"state space: {env.get_state_space_size()}")
    print(f"action space: {env.get_action_space_size()}")

    # Simulation training
    for episode in range(3):
        print(f"\n=== Episode {episode + 1} ===")

        # Reset environment
        task_context = {
            "task_type": "T2",
            "tech_stack": ["python"],
            "user_mood": "focused"
        }

        state = env.reset(task_context)

        # Collect tracks
        trajectory = Trajectory([], [], [], [], [])

        for step in range(5):  # eachepisodemost5step
            # Select action
            action = agent.select_action(state, explore=True)

            # Simulation task results
            task_result = {
                "duration": 200 + step * 50,
                "completed": step == 4,  # Last step completed
                "test_result": {"coverage": 70 + step * 5},
                "user_feedback": {"accepted": True},
                "metrics": {}
            }

            # Execution steps
            next_state, reward, done, info = env.step(action, task_result)

            # record track
            trajectory.states.append(state.to_vector())
            trajectory.actions.append(agent.encode_action_indices(action))
            trajectory.rewards.append(reward)
            trajectory.dones.append(done)
            trajectory.next_states.append(next_state.to_vector())

            state = next_state

            print(f"  Step {step + 1}: reward={reward:.3f}, done={done}")

            if done:
                break

        # update strategy
        stats = agent.update_policy(trajectory)
        print(f"  total return: {stats['total_return']:.3f}")
        print(f"  Actorloss: {stats['actor_loss']:.4f}")
        print(f"  Criticloss: {stats['critic_loss']:.4f}")

    # Save model
    agent.save_model("/tmp/openclaw_rl_agent")
    print("\nModel saved to /tmp/openclaw_rl_agent")


if __name__ == "__main__":
    main()
