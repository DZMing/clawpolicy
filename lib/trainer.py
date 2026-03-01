#!/usr/bin/env python3
"""
reinforcement learning trainer - Complete training cycle

Implement a complete training process：
- manyepisodetrain
- checkpoint save/load
- Training statistics and visualization
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from .agent import AlignmentAgent, Trajectory
from .environment import InteractionEnvironment
from .experience_replay import ExperienceReplay, Experience


class RLTrainer:
    """
    reinforcement learning trainer

    Function：
    - Complete training cycle（Multipleepisode）
    - checkpoint save/load
    - Training statistics and visualization
    """

    def __init__(self, model_dir: str | Path | None = None,
                 use_experience_replay: bool = True,
                 replay_capacity: int = 10000):
        """
        Initialize the trainer

        Args:
            model_dir: Model save directory
            use_experience_replay: Whether to use experience playback
            replay_capacity: Experience playback capacity
        """
        self.model_dir = Path(model_dir).expanduser() if model_dir else Path("./models/rl")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize environment
        self.env = InteractionEnvironment()

        # Initialize the agent
        self.agent = AlignmentAgent(
            state_dim=self.env.get_state_space_size(),
            action_dim=self.env.get_action_space_size()
        )

        # Initialize experience playback
        self.use_experience_replay = use_experience_replay
        self.replay_buffer = ExperienceReplay(capacity=replay_capacity) if use_experience_replay else None

        # training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_losses: List[Dict[str, float]] = []

        # currentepisode
        self.current_episode = 0

    def train(self, num_episodes: int = 100,
              max_steps_per_episode: int = 100,
              save_interval: int = 10) -> Dict[str, Any]:
        """
        Train the agent

        Args:
            num_episodes: trainepisodequantity
            max_steps_per_episode: eachepisodeMaximum number of steps
            save_interval: save interval

        Returns:
            training statistics
        """
        print(f"🚀 Start training（{num_episodes} episodes）...")

        for episode in range(num_episodes):
            self.current_episode = episode + 1

            # run aepisode
            episode_reward, episode_length, episode_loss = self._run_episode(max_steps_per_episode)

            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.training_losses.append(episode_loss)

            # Printing progress
            if episode % 10 == 0 or episode == num_episodes - 1:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"award: {episode_reward:.3f} | "
                      f"average: {avg_reward:.3f} | "
                      f"number of steps: {episode_length}")

            # Save regularly
            if episode % save_interval == 0 and episode > 0:
                self.save_checkpoint(f"checkpoint_episode_{episode}")

        print("✅ Training completed！")

        # Save final model
        self.save_checkpoint("final")

        return self.get_training_stats()

    def _run_episode(self, max_steps: int) -> Tuple[float, int, Dict[str, float]]:
        """
        run aepisode

        Args:
            max_steps: Maximum number of steps

        Returns:
            (total reward, number of steps, Loss statistics)
        """
        # Random task context
        task_context = self._generate_random_task_context()

        # Reset environment
        state = self.env.reset(task_context)

        trajectory = Trajectory([], [], [], [], [])
        total_reward = 0.0

        for step in range(max_steps):
            # Select action
            action = self.agent.select_action(state, explore=True)

            # Simulation task results
            task_result = self._simulate_task_result()

            # Execution steps
            next_state, reward, done, info = self.env.step(action, task_result)

            action_indices = self.agent.encode_action_indices(action)

            # record track
            trajectory.states.append(state.to_vector())
            trajectory.actions.append(action_indices)
            trajectory.rewards.append(reward)
            trajectory.dones.append(done)
            trajectory.next_states.append(next_state.to_vector())

            # Add to experience replay
            if self.replay_buffer:
                exp = Experience(
                    state=state.to_vector(),
                    action=action_indices,
                    reward=reward,
                    next_state=next_state.to_vector(),
                    done=done,
                    priority=abs(reward)
                )
                self.replay_buffer.add(exp)

            total_reward += reward
            state = next_state

            if done:
                break

        # update strategy
        loss_stats = self.agent.update_policy(trajectory)

        # If you use experience replay and have enough experience，Make additional training updates
        if self.replay_buffer and self.replay_buffer.is_ready(min_size=32):
            additional_loss = self._train_from_replay()
            # Combined loss statistics
            if additional_loss:
                for key, value in additional_loss.items():
                    loss_stats[key] = loss_stats.get(key, 0) + value

        return total_reward, len(trajectory), loss_stats

    def _train_from_replay(self, num_updates: int = 4) -> Optional[Dict[str, float]]:
        """
        Train from experience replays

        Args:
            num_updates: Update times

        Returns:
            Loss statistics
        """
        if not self.replay_buffer:
            return None

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(num_updates):
            states, actions, rewards, next_states, dones = self.replay_buffer.get_batch(batch_size=32)

            if len(states) == 0:
                continue

            # Using Experience Replay Data Update Strategies
            for i in range(len(states)):
                state = states[i]
                action_indices = actions[i]
                reward = rewards[i]
                next_state = next_states[i]
                done = dones[i]

                # Calculate target value
                if done:
                    target_value = reward
                else:
                    target_value = reward + self.agent.gamma * self.agent.value_net.forward(next_state)

                # Computational Advantage
                current_value = self.agent.value_net.forward(state)
                advantage = target_value - current_value

                # renewActorandCritic
                actor_loss = self.agent.policy_net.update(state, action_indices, advantage)
                critic_loss = self.agent.value_net.update(state, target_value)

                total_actor_loss += actor_loss
                total_critic_loss += critic_loss

        return {
            "actor_loss": total_actor_loss / (num_updates * 32),
            "critic_loss": total_critic_loss / (num_updates * 32)
        }

    def _generate_random_task_context(self) -> Dict[str, Any]:
        """Generate random task context"""
        task_types = ["T1", "T2", "T3", "T4"]
        tech_stacks = [["python"], ["javascript"], ["python", "fastapi"], ["react", "typescript"]]
        moods = ["focused", "relaxed", "stressed"]

        # usePythonofrandom.choiceinstead ofnumpy
        import random
        return {
            "task_type": random.choice(task_types),
            "tech_stack": random.choice(tech_stacks),
            "user_mood": random.choice(moods),
            "time_of_day": float(np.random.uniform(0, 24))
        }

    def _simulate_task_result(self) -> Dict[str, Any]:
        """Simulation task results"""
        # Randomly generate results
        coverage = np.random.uniform(50, 100)
        passed = int(np.random.uniform(5, 15))
        failed = int(np.random.uniform(0, 3))

        return {
            "duration": np.random.uniform(100, 600),
            "completed": True,
            "test_result": {
                "coverage": coverage,
                "passed": passed,
                "failed": failed
            },
            "user_feedback": {
                "accepted": np.random.random() > 0.2,
                "rating": np.random.randint(3, 6)
            },
            "metrics": {
                "complexity": np.random.uniform(1, 7),
                "duplication": np.random.uniform(0, 0.2),
                "lint_score": np.random.uniform(0.6, 1.0)
            }
        }

    def save_checkpoint(self, name: str) -> None:
        """
        Save checkpoint

        Args:
            name: Checkpoint name
        """
        checkpoint_dir = self.model_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save agent
        self.agent.save_model(str(checkpoint_dir))

        # save statistics
        stats = {
            "episode": self.current_episode,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses[-100:]  # recent100indivual
        }

        with open(checkpoint_dir / "training_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✅ Checkpoint saved: {checkpoint_dir}")

    def load_checkpoint(self, name: str) -> None:
        """
        Load checkpoint

        Args:
            name: Checkpoint name
        """
        checkpoint_dir = self.model_dir / name

        # Load the agent
        self.agent.load_model(str(checkpoint_dir))

        # Load statistics
        stats_path = checkpoint_dir / "training_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)

            self.current_episode = stats["episode"]
            self.episode_rewards = stats["episode_rewards"]
            self.episode_lengths = stats["episode_lengths"]
            self.training_losses = stats["training_losses"]

        print(f"✅ Checkpoint loaded: {checkpoint_dir}")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.episode_rewards:
            return {}

        return {
            "total_episodes": len(self.episode_rewards),
            "average_reward": float(np.mean(self.episode_rewards)),
            "reward_std": float(np.std(self.episode_rewards)),
            "max_reward": float(np.max(self.episode_rewards)),
            "min_reward": float(np.min(self.episode_rewards)),
            "average_length": float(np.mean(self.episode_lengths)),
            "latest_reward": float(self.episode_rewards[-1]),
            "improvement": float(np.mean(self.episode_rewards[-10:]) - np.mean(self.episode_rewards[:10])) if len(self.episode_rewards) >= 20 else 0.0
        }


def main():
    """Test trainer"""
    trainer = RLTrainer(model_dir="/tmp/rl_trainer_test", use_experience_replay=True)

    print("✅ Trainer has been created")

    # train a fewepisode
    stats = trainer.train(num_episodes=5, max_steps_per_episode=10, save_interval=2)

    print("\n📊 training statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
