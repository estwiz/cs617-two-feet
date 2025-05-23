
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns

from common.utils import (
    EvalWrapper, plot_training_curves, get_device,
    setup_environment, get_env_info, setup_save_directory, setup_video_recording,
    print_episode_info, run_evaluation
)
from common.base_agent import BaseAgent
from common.replay_buffer import ReplayBuffer
from common.actor import Actor
from common.critic import Critic

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")


class SAC(BaseAgent):
    """Soft Actor-Critic algorithm."""
    
    def __init__(self, state_dim, action_dim, max_action, device):
        super().__init__(state_dim, action_dim, max_action, device)
        
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Automatically tune temperature
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
            return action.cpu().data.numpy().flatten()
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, gamma=0.99, tau=0.005):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        alpha = self.log_alpha.exp()

        # Update critic
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_pi
            target_q = reward + (1 - done) * gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        action_new, log_pi = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (alpha * log_pi - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), f"{directory}/sac_actor_{name}.pth")
        torch.save(self.critic.state_dict(), f"{directory}/sac_critic_{name}.pth")

    def load(self, directory, name):
        self.actor.load_state_dict(torch.load(f"{directory}/sac_actor_{name}.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/sac_critic_{name}.pth"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v3", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_timesteps", default=1_000_000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--save_freq", default=50000, type=int)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    parser.add_argument("--model_path", type=str, help="Path to the model to evaluate")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Number of episodes for evaluation")
    return parser.parse_args()

def train_sac(agent, env, args, save_dir):
    """Main training loop for SAC."""
    replay_buffer = ReplayBuffer(1_000_000)
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0
    episode_rewards = []
    episode_lengths = []
    
    total_training_start = time.time()
    episode_start = time.time()

    for t in range(args.max_timesteps):
        episode_steps += 1

        # Select and perform action
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store data in replay buffer
        replay_buffer.push(state, action, reward, next_state, float(done)) 

        state = next_state
        episode_reward += reward

        # Train agent
        if len(replay_buffer) > args.batch_size:
            agent.train(replay_buffer, args.batch_size)

        if done:
            # Calculate episode time
            episode_time = time.time() - episode_start

            # Print episode info
            print_episode_info(t+1, episode_num, episode_steps, episode_reward, episode_time)

            # Track metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)

            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
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

def main():
    args = parse_args()

    # Set up environment
    env = setup_environment(args.env, args.seed)
    state_dim, action_dim, max_action = get_env_info(env)

    # Initialize agent
    device = get_device()
    print(f"Using device: {device}")
    agent = SAC(state_dim, action_dim, max_action, device)

    if args.evaluate:
        run_evaluation(agent, env, args)
        return

    # Set up save directory and video recording
    save_dir = setup_save_directory("sac", args.env)
    if args.save_video:
        env = setup_video_recording(env, save_dir)

    # Run training
    train_sac(agent, env, args, save_dir)
    env.close()

if __name__ == "__main__":
    main()
