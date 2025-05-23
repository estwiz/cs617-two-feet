import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from common.base_agent import BaseAgent

def get_device() -> torch.device:
    """
    Get the device to use for training.

    Returns:
        torch.device: The device to use (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """
    Perform soft update of target network parameters.

    Args:
        target: Target network
        source: Source network
        tau: Soft update coefficient
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """
    Perform hard update of target network parameters.

    Args:
        target: Target network
        source: Source network
    """
    target.load_state_dict(source.state_dict()) 


def plot_training_curves(rewards: List[float], episode_lengths: List[int], save_dir: str) -> None:
    """Create plots of training metrics."""
    episodes = np.arange(len(episode_lengths))
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot reward vs total steps
    ax1.plot(episodes, rewards, "b-", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Training Progress")
    ax1.grid(True)

    # Plot episode length vs episodes
    ax2.plot(episodes, episode_lengths, "r-", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length")
    ax2.set_title("Episode Length Over Time")
    ax2.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_policy(
    agent: BaseAgent,
    env: gym.Env,
    num_episodes: int = 100,
    save_dir: Optional[str] = None
) -> Dict[str, Union[float, List[float], List[int]]]:
    """
    Evaluate a trained policy over multiple episodes and calculate reward statistics.

    Args:
        agent: The trained agent
        env: The environment to evaluate in
        num_episodes: Number of episodes to run
        save_dir: Directory to save evaluation results

    Returns:
        dict: Dictionary containing evaluation statistics
    """
    all_rewards: List[float] = []
    all_lengths: List[int] = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += float(reward)
            episode_length += 1

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}")

    # Calculate statistics
    mean_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))
    min_reward = float(np.min(all_rewards))
    max_reward = float(np.max(all_rewards))

    mean_length = float(np.mean(all_lengths))
    std_length = float(np.std(all_lengths))
    min_length = int(np.min(all_lengths))
    max_length = int(np.max(all_lengths))

    if save_dir is not None:
        # Create evaluation subdirectory
        eval_dir = os.path.join(save_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot reward distribution
        ax1.hist(all_rewards, bins=20, alpha=0.75, color='blue')
        ax1.axvline(mean_reward, color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean: {mean_reward:.2f} ± {std_reward:.2f}')
        ax1.set_title('Reward Distribution')
        ax1.set_xlabel('Episode Reward')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot episode length distribution
        ax2.hist(all_lengths, bins=20, alpha=0.75, color='green')
        ax2.axvline(mean_length, color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean: {mean_length:.2f} ± {std_length:.2f}')
        ax2.set_title('Episode Length Distribution')
        ax2.set_xlabel('Episode Length')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "distributions.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Number of episodes: {num_episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print(f"Mean episode length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"Min episode length: {min_length}")
    print(f"Max episode length: {max_length}")

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'mean_length': mean_length,
        'std_length': std_length,
        'min_length': min_length,
        'max_length': max_length,
        'all_rewards': all_rewards,
        'all_lengths': all_lengths
    }


class EvalWrapper(gym.Wrapper):
    """Wrapper that uses evaluation mode for actions."""

    def __init__(self, env: gym.Env, agent: BaseAgent):
        super().__init__(env)
        self.agent = agent
        self.last_observation: Optional[np.ndarray] = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Use evaluation mode for actions
        if self.last_observation is None:
            raise ValueError("Environment must be reset before stepping")
        eval_action = self.agent.select_action(self.last_observation, evaluate=True)
        next_obs, reward, terminated, truncated, info = self.env.step(eval_action)
        self.last_observation = next_obs
        return next_obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.env.reset(**kwargs)
        self.last_observation = state
        return state, info