import os
import numpy as np
import torch
import optuna
from optuna.trial import Trial
import json
from datetime import datetime
from a2c_gae import A2C, Policy, Value
import gymnasium as gym
from typing import Dict, Any

from common.utils import evaluate_policy


def create_env(env_name: str, seed: int) -> gym.Env:
    """Create and return a gym environment."""
    env = gym.make(env_name, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def objective(trial: Trial) -> float:
    """Optuna objective function for hyperparameter optimization."""
    # Define hyperparameter search space
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'n_steps': trial.suggest_int('n_steps', 8, 32),
        'entropy_coef': trial.suggest_float('entropy_coef', 1e-5, 5e-3, log=True),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 256),
        'log_std_init': trial.suggest_float('log_std_init', -1.0, 1.0),
    }

    # Environment setup
    env_name = "BipedalWalker-v3"
    seed = 0
    max_timesteps = 3_000_000
    eval_interval = 50_000
    n_eval_episodes = 5

    env = create_env(env_name, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize agent with trial parameters
    agent = A2C(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        learning_rate=params['learning_rate']
    )

    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    eval_rewards = []

    states, actions, rewards, next_states, dones = [], [], [], [], []

    for t in range(max_timesteps):
        print(f"Steps taken in current trial: {t}/{max_timesteps}", end="\r")
        episode_timesteps += 1

        # Select action
        action = agent.select_action(state)

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(float(done))

        state = next_state
        episode_reward += reward

        # Update policy every n_steps or when episode ends
        if len(states) >= params['n_steps'] or done:
            train_info = agent.train(
                states, actions, rewards, next_states, dones,
                entropy_coef=params['entropy_coef'],
                gae_lambda=params['gae_lambda']
            )
            states, actions, rewards, next_states, dones = [], [], [], [], []

        if done:
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluation
        if (t + 1) % eval_interval == 0:
            eval_results = evaluate_policy(agent, env, n_eval_episodes, verbose=False)
            eval_reward = eval_results["mean_reward"]
            # eval_reward = evaluate_agent(agent, env, n_eval_episodes)
            eval_rewards.append(eval_reward)
            trial.report(eval_reward, t + 1)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.TrialPruned()

    env.close()

    # Return the best evaluation reward
    return max(eval_rewards)


def evaluate_agent(agent: A2C, env: gym.Env, n_episodes: int) -> float:
    """Evaluate the agent's performance over multiple episodes."""
    eval_rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            # Convert action to numpy array if it's a tensor
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            # Ensure action is a 1D array
            action = np.array(action).flatten()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        eval_rewards.append(episode_reward)

    return np.mean(eval_rewards)


def save_optimization_results(study: optuna.Study, save_dir: str):
    """Save optimization results and best parameters."""
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)

    # Save best parameters
    best_params = study.best_params
    with open(os.path.join(save_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)

    # Save optimization history
    history = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            history.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params
            })

    with open(os.path.join(save_dir, 'optimization_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # Save study statistics
    stats = {
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }

    with open(os.path.join(save_dir, 'optimization_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)


def main():
    # Create study directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"results/optuna_optimization_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Create and run the study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        ),
        sampler=optuna.samplers.TPESampler(seed=0)
    )

    # Run optimization
    study.optimize(objective, n_trials=10, timeout=8 * 3600)  # 50 trials or 8 hour timeout

    # Save results
    save_optimization_results(study, save_dir)

    # Print results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()