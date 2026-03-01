#!/usr/bin/env python3
"""
neural network model - OptionalPyTorchaccomplish

supplyPyTorchversion of the policy network and value network，ifPyTorchIf unavailable, downgrade toNumPyVersion
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple

from .agent import PolicyNetwork as NumpyPolicyNetwork
from .agent import ValueNetwork as NumpyValueNetwork

# try to importPyTorch
torch: Any = None
nn: Any = None
F: Any = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


_MLP_BASE: Any = nn.Module if TORCH_AVAILABLE else object


class MLPModel(_MLP_BASE):
    """Multilayer perceptron model"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorchNot available，Please useNumPyVersion")

        super().__init__()

        layers: List[Any] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        return self.network(x)


class PolicyNetworkPyTorch:
    """PyTorchpolicy network（long）"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] | None = None):
        """
        initializationPyTorchpolicy network

        Args:
            state_dim: status dimension
            action_dim: action dimension（should be11）
            hidden_dims: Hidden layer dimension list
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorchNot available，Please useNumPyVersion")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [128, 128]

        self.head_dims = {
            "agent": 3,
            "automation": 3,
            "style": 3,
            "confirm": 2
        }

        # shared backbone network
        layers: List[Any] = []
        prev_dim = state_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # multi-head output layer
        self.heads = nn.ModuleDict({
            name: nn.Linear(prev_dim, dim)
            for name, dim in self.head_dims.items()
        })

        # optimizer
        self.optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.heads.parameters()),
            lr=0.001
        )

    def _forward_logits(self, state_tensor: Any) -> Dict[str, Any]:
        """forward propagation，Get each headlogits"""
        features = self.backbone(state_tensor)
        return {name: head(features) for name, head in self.heads.items()}

    def get_action_probs(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Get action probability distribution（long）"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            logits = self._forward_logits(state_tensor)
            probs = {name: F.softmax(head_logits, dim=-1).numpy() for name, head_logits in logits.items()}
            return probs

    def sample_action(self, state: np.ndarray, explore: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Sampling action"""
        action_probs = self.get_action_probs(state)

        if explore and np.random.random() < 0.1:
            action_indices = np.array([
                np.random.randint(self.head_dims["agent"]),
                np.random.randint(self.head_dims["automation"]),
                np.random.randint(self.head_dims["style"]),
                np.random.randint(self.head_dims["confirm"])
            ], dtype=int)
        else:
            action_indices = np.array([
                np.random.choice(self.head_dims["agent"], p=action_probs["agent"]),
                np.random.choice(self.head_dims["automation"], p=action_probs["automation"]),
                np.random.choice(self.head_dims["style"], p=action_probs["style"]),
                np.random.choice(self.head_dims["confirm"], p=action_probs["confirm"])
            ], dtype=int)

        return action_indices, action_probs

    def update(self, state: np.ndarray, action_indices: np.ndarray, advantage: float) -> float:
        """Update policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self._forward_logits(state_tensor)

        # Calculate losses（Long accumulation）
        loss: Any = torch.tensor(0.0)
        head_order = ["agent", "automation", "style", "confirm"]
        for head_idx, head_name in enumerate(head_order):
            log_prob = F.log_softmax(logits[head_name], dim=-1)
            loss = loss - log_prob[0, int(action_indices[head_idx])] * advantage

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save(self, path: str) -> None:
        """Save model"""
        torch.save({
            "backbone": self.backbone.state_dict(),
            "heads": self.heads.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """Load model"""
        state = torch.load(path)
        self.backbone.load_state_dict(state["backbone"])
        self.heads.load_state_dict(state["heads"])


class ValueNetworkPyTorch:
    """PyTorchvalue network"""

    def __init__(self, state_dim: int, hidden_dims: List[int] | None = None):
        """
        initializationPyTorchvalue network

        Args:
            state_dim: status dimension
            hidden_dims: Hidden layer dimension list
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorchNot available，Please useNumPyVersion")

        self.state_dim = state_dim
        self.hidden_dims = hidden_dims or [128, 128]

        # createMLPModel（output1dimension）
        self.model = MLPModel(state_dim, self.hidden_dims, 1)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, state: np.ndarray) -> float:
        """forward propagation"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            value = self.model(state_tensor)
            return value.item()

    def update(self, state: np.ndarray, target_value: float) -> float:
        """Update value network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        value = self.model(state_tensor)

        # Calculate losses（MSE）
        target_tensor = torch.FloatTensor([target_value])
        loss = F.mse_loss(value, target_tensor)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path: str) -> None:
        """Save model"""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model"""
        self.model.load_state_dict(torch.load(path))


def create_policy_network(
    state_dim: int,
    action_dim: int,
    use_pytorch: bool = True,
) -> Any:
    """
    Create a policy network（automatic selectionPyTorchorNumPyVersion）

    Args:
        state_dim: status dimension
        action_dim: action dimension
        use_pytorch: Have you tried usingPyTorch

    Returns:
        Policy Network Example
    """
    if use_pytorch and TORCH_AVAILABLE:
        return PolicyNetworkPyTorch(state_dim, action_dim)
    else:
        return NumpyPolicyNetwork(state_dim, action_dim)


def create_value_network(state_dim: int, use_pytorch: bool = True) -> Any:
    """
    Create a value network（automatic selectionPyTorchorNumPyVersion）

    Args:
        state_dim: status dimension
        use_pytorch: Have you tried usingPyTorch

    Returns:
        Value Network Example
    """
    if use_pytorch and TORCH_AVAILABLE:
        return ValueNetworkPyTorch(state_dim)
    else:
        return NumpyValueNetwork(state_dim)


def main():
    """Test neural network models"""
    state_dim = 17
    action_dim = 11

    print(f"PyTorchAvailable: {TORCH_AVAILABLE}")

    # Create network
    policy_net = create_policy_network(state_dim, action_dim)
    value_net = create_value_network(state_dim)

    print(f"✅ policy network: {type(policy_net).__name__}")
    print(f"✅ value network: {type(value_net).__name__}")

    # Test forward propagation
    state = np.random.randn(state_dim)

    action_probs = policy_net.get_action_probs(state)
    value = value_net.forward(state)

    print(f"action probability: {action_probs}")
    print(f"status value: {value:.3f}")


if __name__ == "__main__":
    main()
