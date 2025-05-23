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
from common.base_agent import BaseAgent, plot_training_curves, evaluate_policy, EvalWrapper
from common.actor import Actor
from common.critic import ValueCritic

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")


class A2C(BaseAgent):
    """Advantage Actor-Critic algorithm."""
    
    def __init__(self, state_dim, action_dim, max_action, device, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        super().__init__(state_dim, action_dim, max_action, device)
        
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.critic = ValueCritic(state_dim).to(device)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=lr)

        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
            return action.cpu().data.numpy().flatten()
        action, _ = self.actor.sample(state)
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

    def train(self, states, actions, rewards, dones, next_state):
        # Convert lists to numpy arrays before creating tensors for better performance
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)

        # Get current values and log probs
        _, log_probs = self.actor.sample(states)
        values = self.critic(states)

        # Get next value
        with torch.no_grad():
            next_state = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
            next_value = self.critic(next_state)
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
        torch.save(self.actor.state_dict(), f"{directory}/a2c_actor_{name}.pth")
        torch.save(self.critic.state_dict(), f"{directory}/a2c_critic_{name}.pth")

    def load(self, directory, name):
        self.actor.load_state_dict(torch.load(f"{directory}/a2c_actor_{name}.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/a2c_critic_{name}.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v3", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_timesteps", default=1_000_000, type=int)
    parser.add_argument("--update_freq", default=2048, type=int)  # Number of steps between updates
    parser.add_argument("--save_freq", default=50000, type=int)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    parser.add_argument("--model_path", type=str, help="Path to the model to evaluate")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Number of episodes for evaluation")
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
        step_name = os.path.basename(args.model_path).replace(".pth", "").split('_')[-1]
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
        # eval_env = EvalWrapper(env, agent)
        env = RecordVideo(env, f"{save_dir}/videos", episode_trigger=lambda x: x % 50 == 0)

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
                next_state_tensor = torch.FloatTensor(next_state).to(device).unsqueeze(0)
                with torch.no_grad():
                    next_value = agent.critic(next_state_tensor)
                    next_value = next_value.squeeze().cpu().numpy()

            # Update agent
            update_info = agent.train(states, actions, rewards, dones, next_state)

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
