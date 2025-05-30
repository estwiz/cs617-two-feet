import optuna
import torch
import numpy as np
import gymnasium as gym
from sac import SAC
from common.utils import (
    setup_environment,
    get_env_info,
    get_device,
)
from common.replay_buffer import ReplayBuffer
import os
import json
from datetime import datetime
from optuna.trial import Trial


def evaluate_agent(agent: SAC, env: gym.Env, n_episodes: int = 5) -> float:
    """Evaluate the agent's performance over multiple episodes."""
    eval_rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        eval_rewards.append(episode_reward)

    return float(np.mean(eval_rewards))


def save_optimization_results(study: optuna.Study, save_dir: str) -> None:
    """Save optimization results and best parameters."""
    os.makedirs(save_dir, exist_ok=True)

    # Save best parameters
    best_params = study.best_params
    with open(os.path.join(save_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    # Save history
    history = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            history.append(
                {"number": trial.number, "value": trial.value, "params": trial.params}
            )
    with open(os.path.join(save_dir, "optimization_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Save study stats
    stats = {
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "n_complete_trials": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "n_pruned_trials": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
    }
    with open(os.path.join(save_dir, "optimization_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)


def objective(trial: Trial) -> float:
    """Optuna objective function for hyperparameter optimization."""
    # hyperparameter search space
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "batch_size": trial.suggest_int("batch_size", 64, 512, step=64),
        "tau": trial.suggest_float("tau", 0.001, 0.01, log=True),
        "train_freq": trial.suggest_int("train_freq", 64, 128, step=8),
        "gradient_steps": trial.suggest_int("gradient_steps", 16, 52, step=4),
    }

    # Environment setup
    max_timesteps = 400_000
    eval_interval = 2_000
    n_eval_episodes = 5
    env = setup_environment(env_name="BipedalWalker-v3", seed=0)
    state_dim, action_dim, max_action = get_env_info(env)

    # Initialize agent
    device = get_device()
    agent = SAC(state_dim, action_dim, max_action, device)
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=params["lr"])
    agent.critic_optimizer = torch.optim.Adam(
        agent.critic.parameters(), lr=params["lr"]
    )
    agent.alpha_optimizer = torch.optim.Adam([agent.log_alpha], lr=params["lr"])

    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    learning_start = 10_000
    eval_rewards = []
    replay_buffer = ReplayBuffer(300_000)

    for t in range(max_timesteps):
        print(f"Steps taken in current trial: {t}/{max_timesteps}", end="\r")

        # train agent
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, float(done))
        if len(replay_buffer) > learning_start and t % params["train_freq"] == 0:
            for _ in range(params["gradient_steps"]):
                agent.train(
                    replay_buffer, params["batch_size"], params["gamma"], params["tau"]
                )

        state = next_state
        episode_reward += reward
        episode_timesteps += 1

        if done:
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0

        # Evaluate and report every eval_interval timesteps
        if (t + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(agent, env, n_eval_episodes)
            eval_rewards.append(eval_reward)
            trial.report(float(eval_reward), t + 1)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(max(eval_rewards)) if eval_rewards else float(-np.inf)


def main():
    # Results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/sac_optimization_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Run optimization
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, interval_steps=1
        ),
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(
        objective, n_trials=32, timeout=8 * 3600
    )
    save_optimization_results(study, save_dir)

    # print results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
