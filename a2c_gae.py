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
from common import Actor, Critic
from common.utils import (
    get_device,
    setup_environment,
    get_env_info,
    setup_save_directory,
    setup_video_recording,
    print_episode_info,
    evaluate_policy,
    plot_training_curves
)
import time


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_init=0.0):
        super().__init__()
        self.activation = torch.tanh

        self.affine_layers = nn.ModuleList([
            nn.Linear(state_dim, 128),
            nn.Linear(128, 128)
        ])

        self.action_mean = nn.Linear(128, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, state):
        x = state
        for layer in self.affine_layers:
            x = self.activation(layer(x))
        mean = self.action_mean(x)
        std = torch.exp(self.action_log_std).expand_as(mean)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.rsample()
        action_tanh = torch.tanh(action)
        log_prob = dist.log_prob(action) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        return action_tanh, log_prob.sum(1, keepdim=True)


class Value(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.activation = torch.tanh

        self.affine_layers = nn.ModuleList([
            nn.Linear(state_dim, 128),
            nn.Linear(128, 128)
        ])

        self.value_head = nn.Linear(128, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, state):
        x = state
        for layer in self.affine_layers:
            x = self.activation(layer(x))
        return self.value_head(x).squeeze(-1)


class A2C:
    def __init__(self, state_dim, action_dim, max_action, device, learning_rate=3e-4):
        self.actor = Policy(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Value(state_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.max_action = max_action
        self.device = device

    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.9):
        advantages = []
        returns = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages * 0
        return advantages, returns

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state)
            return torch.tanh(mean).detach().cpu().numpy().flatten()
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def train(self, states, actions, rewards, next_states, dones, gamma=0.99, gae_lambda=0.9, entropy_coef=0.01):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        values = self.critic(states).squeeze(-1)
        if values.ndim == 0:
            values = values.unsqueeze(0)
        next_values = self.critic(next_states).squeeze(-1)
        if next_values.ndim == 0:
            next_values = next_values.unsqueeze(0)

        advantages, returns = self.compute_gae(
            rewards.cpu().numpy(),
            values.detach().cpu().numpy(),
            next_values.detach().cpu().numpy(),
            dones.cpu().numpy(),
            gamma,
            gae_lambda
        )

        advantages = torch.clamp(advantages, -10, 10)

        _, log_prob = self.actor.sample(states)
        actor_loss = -(log_prob * advantages.unsqueeze(1)).mean()

        _, std = self.actor(states)
        entropy = 0.5 * (torch.log(2 * np.pi * std.pow(2)) + 1).sum(1).mean()
        actor_loss -= entropy_coef * entropy

        critic_loss = F.mse_loss(values, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), f'{directory}/a2c_actor_{name}.pth')
        torch.save(self.critic.state_dict(), f'{directory}/a2c_critic_{name}.pth')

    def load(self, directory, name):
        self.actor.load_state_dict(torch.load(f'{directory}/a2c_actor_{name}.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/a2c_critic_{name}.pth'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v3", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_timesteps", default=1_000_000, type=int)
    parser.add_argument("--n_steps", default=8, type=int)
    parser.add_argument("--save_freq", default=50000, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--entropy_coef", default=0.01, type=float)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--gae_lambda", default=0.9, type=float)
    args = parser.parse_args()

    # Use utility functions for setup
    save_dir = setup_save_directory("a2c_gae", args.env)
    env = setup_environment(args.env, args.seed, "rgb_array")
    if args.save_video:
        env = setup_video_recording(env, save_dir)

    # Get environment info using utility function
    state_dim, action_dim, max_action = get_env_info(env)
    device = get_device()
    
    agent = A2C(state_dim, action_dim, max_action, device, learning_rate=args.learning_rate)

    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_start_time = time.time()

    # Lists to store training metrics
    episode_rewards = []
    episode_lengths = []

    states, actions, rewards, next_states, dones = [], [], [], [], []

    for t in range(args.max_timesteps):
        episode_timesteps += 1
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(float(done))

        state = next_state
        episode_reward += reward

        if len(states) >= args.n_steps or done:
            train_info = agent.train(
                states, actions, rewards, next_states, dones,
                entropy_coef=args.entropy_coef,
                gae_lambda=args.gae_lambda
            )
            states, actions, rewards, next_states, dones = [], [], [], [], []

        if done:
            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_timesteps)
            
            # Print episode info using utility function
            episode_time = time.time() - episode_start_time
            print_episode_info(t, episode_num, episode_timesteps, episode_reward, episode_time)
            
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_start_time = time.time()

        if (t + 1) % args.save_freq == 0:
            agent.save(save_dir, f"step_{t+1}")

        # Reduce evaluation frequency and number of episodes
        if t % args.eval_freq == 0:
            evaluate_policy(agent, env, num_episodes=5, save_dir=save_dir)  # Reduced from 10 to 5 episodes

    # Plot training curves at the end
    plot_training_curves(episode_rewards, episode_lengths, save_dir)

    # Final evaluation with more episodes
    evaluate_policy(agent, env, num_episodes=100, save_dir=save_dir)

    env.close()


if __name__ == "__main__":
    main()
