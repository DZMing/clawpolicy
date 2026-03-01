#!/usr/bin/env python3
"""
Experience replay buffer - Improve sample efficiency

Storage and sampling experience（state, action, reward, next_state, done），
Support priority sampling and batch sampling
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import random


@dataclass
class Experience:
    """single experience"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0  # priority（for priority sampling）

    def __repr__(self) -> str:
        return f"Experience(reward={self.reward:.3f}, done={self.done})"


class ExperienceReplay:
    """
    Experience replay buffer

    Function：
    - Store experience
    - random sampling
    - priority sampling（Optional）
    - Batch sampling
    """

    def __init__(self, capacity: int = 10000, use_prioritized: bool = False):
        """
        Initialize experience playback buffer

        Args:
            capacity: Buffer capacity
            use_prioritized: Whether to use priority sampling
        """
        self.capacity = capacity
        self.use_prioritized = use_prioritized

        # Buffer for storing experience
        self.buffer: List[Experience] = []
        self.position = 0

        # Priority sampling related
        self.priorities = np.zeros(capacity)
        self.max_priority = 1.0
        self.alpha = 0.6  # priority index
        self.beta = 0.4   # importance sampling index
        self.epsilon = 1e-6  # Avoid zero priority

    def add(self, experience: Experience) -> None:
        """
        Add experience to buffer

        Args:
            experience: Experience to add
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            idx = len(self.buffer) - 1
        else:
            # Overwrite the oldest experience（circular buffer）
            idx = self.position
            self.buffer[idx] = experience

        if self.use_prioritized:
            self.priorities[idx] = self.max_priority

        self.position = (self.position + 1) % self.capacity

        # Update maximum priority
        if self.use_prioritized:
            self.max_priority = max(self.max_priority, experience.priority)

    def sample(self, batch_size: int = 32) -> List[Experience]:
        """
        Randomly sample a batch of experiences

        Args:
            batch_size: batch size

        Returns:
            experience batch
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if self.use_prioritized:
            return self._prioritized_sample(batch_size)
        else:
            return random.sample(self.buffer, batch_size)

    def _prioritized_sample(self, batch_size: int) -> List[Experience]:
        """
        priority sampling

        Sampling based on priority，The higher the priority, the greater the probability of being sampled.
        """
        if len(self.buffer) == 0:
            return []

        # Calculate sampling probability
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        total = probs.sum()
        if total == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= total

        # Sampling index
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), p=probs)

        # Calculate importance weight
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return [self.buffer[idx] for idx in indices]

    def get_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get batch data（for training）

        Args:
            batch_size: batch size

        Returns:
            (states, actions, rewards, next_states, dones)
        """
        experiences = self.sample(batch_size)

        if not experiences:
            return (
                np.zeros((0, 17)),  # states
                np.zeros((0, 4), dtype=int),  # actions (indices)
                np.zeros(0),      # rewards
                np.zeros((0, 17)), # next_states
                np.zeros(0)       # dones
            )

        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])

        return states, actions, rewards, next_states, dones

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update experience priority

        Args:
            indices: experience index
            priorities: new priority
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self.buffer):
                self.priorities[idx] = priority + self.epsilon
                self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int = 100) -> bool:
        """
        Check if buffer is ready for sampling

        Args:
            min_size: Minimum size requirements

        Returns:
            Is it possible to sample
        """
        return len(self.buffer) >= min_size

    def clear(self) -> None:
        """Clear buffer"""
        self.buffer.clear()
        self.priorities = np.zeros(self.capacity)
        self.max_priority = 1.0
        self.position = 0


def main():
    """Test experience replay"""
    # Create experience replay buffer
    replay = ExperienceReplay(capacity=1000, use_prioritized=True)

    print(f"✅ Experience replay buffer created（capacity：{replay.capacity}）")

    # Add some simulation experience
    for i in range(10):
        state = np.random.randn(17)
        action = np.random.randint(0, 3, size=4)
        reward = np.random.randn()
        next_state = np.random.randn(17)
        done = i % 5 == 0

        exp = Experience(state, action, reward, next_state, done, priority=abs(reward))
        replay.add(exp)

    print(f"✅ Added {len(replay)} experience")

    # Sampling batch
    states, actions, rewards, next_states, dones = replay.get_batch(batch_size=4)

    print(f"✅ Sampling batch: states={states.shape}, actions={actions.shape}")
    print(f"   award: {rewards}")
    print(f"   Finish: {dones}")

    # Check if you are ready
    print(f"✅ Prepare for sampling: {replay.is_ready(min_size=5)}")


if __name__ == "__main__":
    main()
