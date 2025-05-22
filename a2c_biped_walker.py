import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import argparse
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from utils import get_device

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")


# Create a wrapper that uses evaluation mode for actions
class EvalWrapper(gym.Wrapper):
    def __init__(self, env, agent):
        super().__init__(env)
        self.agent = agent
        self.last_observation = None

    def step(self, action):
        # Use evaluation mode for actions
        eval_action = self.agent.select_action(self.last_observation, evaluate=True)
        next_obs, reward, terminated, truncated, info = self.env.step(eval_action)
        self.last_observation = next_obs
        return next_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.last_observation = state
        return state, info


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(ActorCritic, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.shared(state)

        # Actor outputs
        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)

        # Critic output
        value = self.critic(x)

        return mean, log_std, value

    def sample(self, state):
        mean, log_std, value = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action * self.max_action, log_prob, value


class A2C:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
    ):
        self.actor_critic = ActorCritic(
            state_dim, action_dim, max_action=max_action
        ).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.max_action = max_action
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            mean, _, _ = self.actor_critic(state)
            action = torch.tanh(mean) * self.max_action
            return action.cpu().data.numpy().flatten()
        action, _, _ = self.actor_critic.sample(state)
        return action.cpu().data.numpy().flatten()

    def compute_returns_and_advantages(self, rewards, values, dones, next_value):
        returns = []
        advantages = []
        R = next_value

        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            advantage = R - v
            returns.insert(0, R)
            advantages.insert(0, advantage)

        # Convert to float32 tensors explicitly (mps does not support float64)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(self, states, actions, rewards, dones, next_state):
        # Convert lists to numpy arrays before creating tensors for better performance
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)

        # Get current values and log probs
        _, log_probs, values = self.actor_critic.sample(states)

        # Get next value
        with torch.no_grad():
            next_state = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
            _, _, next_value = self.actor_critic(next_state)
            next_value = next_value.squeeze()

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(
            rewards,
            values.squeeze().detach().cpu().numpy(),
            dones,
            next_value.detach().cpu().numpy(),
        )

        # Compute losses
        policy_loss = -(log_probs * advantages.unsqueeze(1)).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)
        entropy_loss = -self.entropy_coef * log_probs.mean()

        total_loss = policy_loss + value_loss + entropy_loss

        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
        }

    def save(self, directory, name):
        # Save regular checkpoint
        if name.startswith("step_"):
            torch.save(
                self.actor_critic.state_dict(),
                f"{directory}/a2c_actor_critic_{name}.pth",
            )
        # Save best model
        elif name.startswith("best_"):
            step = name.split("_")[1]
            torch.save(
                self.actor_critic.state_dict(),
                f"{directory}/a2c_actor_critic_best_step_{step}.pth",
            )

    def load(self, directory, name):
        self.actor_critic.load_state_dict(
            torch.load(f"{directory}/a2c_actor_critic_{name}.pth")
        )


def evaluate_policy(agent, env, num_episodes=100, save_dir=None):
    """
    Evaluate a trained A2C policy over multiple episodes and calculate reward statistics.

    Args:
        agent: The trained A2C agent
        env: The environment to evaluate in
        num_episodes: Number of episodes to run
        save_dir: Directory to save evaluation results

    Returns:
        dict: Dictionary containing evaluation statistics
    """
    all_rewards = []
    all_lengths = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}"
            )

    # Calculate statistics
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    min_reward = np.min(all_rewards)
    max_reward = np.max(all_rewards)

    mean_length = np.mean(all_lengths)
    std_length = np.std(all_lengths)
    min_length = np.min(all_lengths)
    max_length = np.max(all_lengths)

    # Create evaluation subdirectory
    eval_dir = os.path.join(save_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot reward distribution
    ax1.hist(all_rewards, bins=20, alpha=0.75, color="blue")
    ax1.axvline(
        mean_reward,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_reward:.2f} ± {std_reward:.2f}",
    )
    ax1.set_title("Reward Distribution")
    ax1.set_xlabel("Episode Reward")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot episode length distribution
    ax2.hist(all_lengths, bins=20, alpha=0.75, color="green")
    ax2.axvline(
        mean_length,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_length:.2f} ± {std_length:.2f}",
    )
    ax2.set_title("Episode Length Distribution")
    ax2.set_xlabel("Episode Length")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{eval_dir}/distributions.png", dpi=300, bbox_inches="tight")
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
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "mean_length": mean_length,
        "std_length": std_length,
        "min_length": min_length,
        "max_length": max_length,
        "all_rewards": all_rewards,
        "all_lengths": all_lengths,
    }


def plot_training_curves(rewards, episode_lengths, save_dir):
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
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v3", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_timesteps", default=1_000_000, type=int)
    parser.add_argument(
        "--update_freq", default=2048, type=int
    )  # Number of steps between updates
    parser.add_argument("--save_freq", default=50000, type=int)
    parser.add_argument(
        "--eval_freq",
        default=5000,
        type=int,
        help="Number of steps between evaluations during training",
    )
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    parser.add_argument("--model_path", type=str, help="Path to the model to evaluate")
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation",
    )
    args = parser.parse_args()

    # Set up environment
    env = gym.make(args.env, render_mode="rgb_array")

    # Set seeds
    env.reset(seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = get_device()
    print(f"Using device: {device}")

    # Initialize agent
    agent = A2C(state_dim, action_dim, max_action, device)

    if args.evaluate:
        if args.model_path is None:
            raise ValueError("Model path must be provided for evaluation mode")

        # Get model directory and step number
        model_dir = os.path.dirname(args.model_path)
        step_name = os.path.basename(args.model_path).replace(".pth", "").split("_")[-1]
        model_identifier = f"step_{step_name}"

        # Load the trained model
        agent.load(directory=model_dir, name=model_identifier)
        print(f"Loaded model from {args.model_path}")

        # Run evaluation
        evaluate_policy(agent, env, num_episodes=args.eval_episodes, save_dir=model_dir)
        env.close()
        return

    # Create save directory
    save_dir = f"results/a2c_{args.env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # Set up video recording if enabled
    if args.save_video:
        # Create evaluation environment for video recording
        eval_env = EvalWrapper(env, agent)
        env = RecordVideo(
            eval_env, f"{save_dir}/videos", episode_trigger=lambda x: x % 50 == 0
        )

    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Storage for update
    states = []
    actions = []
    rewards = []
    dones = []

    # Lists to store metrics for plotting
    episode_rewards = []
    episode_lengths = []

    # Evaluation metrics
    eval_episodes = 5
    best_eval_reward = -float("inf")

    # Start timing the total training
    total_training_start = time.time()
    episode_start = time.time()

    for t in range(args.max_timesteps):
        episode_timesteps += 1

        # Select action
        action = agent.select_action(state)

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store data
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))

        state = next_state
        episode_reward += reward

        # Update agent
        if len(states) >= args.update_freq or done:
            if done:
                next_value = 0
            else:
                next_state_tensor = (
                    torch.FloatTensor(next_state).to(device).unsqueeze(0)
                )
                with torch.no_grad():
                    _, _, next_value = agent.actor_critic(next_state_tensor)
                    next_value = next_value.squeeze().cpu().numpy()

            # Update agent
            update_info = agent.update(states, actions, rewards, dones, next_state)

            # Clear storage
            states.clear()
            actions.clear()
            rewards.clear()
            dones.clear()

        if done:
            # Calculate episode time
            episode_time = time.time() - episode_start

            print(
                f"Total steps: {t+1:7d} | "
                f"Episode num: {episode_num+1:4d} | "
                f"Episode steps: {episode_timesteps:4d} | "
                f"Reward: {episode_reward:8.3f} | "
                f"Time: {episode_time:6.2f}s"
            )

            # Store metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_timesteps)

            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_start = time.time()

        # Evaluate agent
        if args.eval_freq != 0 and (t + 1) % args.eval_freq == 0:
            eval_rewards = []
            for _ in range(eval_episodes):
                eval_state, _ = env.reset()
                eval_episode_reward = 0
                eval_done = False

                while not eval_done:
                    eval_action = agent.select_action(eval_state, evaluate=True)
                    eval_state, eval_reward, eval_terminated, eval_truncated, _ = (
                        env.step(eval_action)
                    )
                    eval_done = eval_terminated or eval_truncated
                    eval_episode_reward += eval_reward

                eval_rewards.append(eval_episode_reward)

            mean_eval_reward = np.mean(eval_rewards)
            print(f"\nEvaluation at step {t+1}:")
            print(f"Mean evaluation reward: {mean_eval_reward:.3f}")
            print(f"Evaluation reward std: {np.std(eval_rewards):.3f}\n")

            if mean_eval_reward > best_eval_reward:
                best_eval_reward = mean_eval_reward
                agent.save(save_dir, f"best_{t+1}")

        # Save model
        if (t + 1) % args.save_freq == 0:
            agent.save(save_dir, f"step_{t+1}")

    # Calculate and print total training time
    total_training_time = time.time() - total_training_start
    print(f"\nTotal training time: {total_training_time:.2f} seconds")

    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, save_dir)

    env.close()


if __name__ == "__main__":
    main()
