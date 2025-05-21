# BipedalWalker SAC Implementation

This repository contains an implementation of Soft Actor-Critic (SAC) for training the BipedalWalker-v3 environment from OpenAI Gymnasium.

## Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train the agent with default parameters:
```bash
python sac_biped_walker.py
```

### Command Line Arguments

- `--env`: Environment name (default: "BipedalWalker-v3")
- `--seed`: Random seed (default: 0)
- `--max_timesteps`: Maximum number of training timesteps (default: 1,000,000)
- `--batch_size`: Batch size for training (default: 256)
- `--save_freq`: How often to save model checkpoints (default: 50000 steps)
- `--eval_freq`: How often to evaluate the agent (default: 5000 steps)
- `--save_video`: Flag to enable video recording of episodes (records every 50th episode)

Example with custom parameters:
```bash
python sac_biped_walker.py --max_timesteps 2000000 --save_video
```

## Features

- Automatic temperature tuning
- Model checkpointing
- Video recording of training episodes
- Efficient replay buffer implementation
- Support for both training and evaluation modes

## Output

The training results will be saved in the `results` directory with the following structure:
```
results/
    sac_BipedalWalker-v3_YYYYMMDD_HHMMSS/
        sac_actor_step_XXXXX.pth
        sac_critic_step_XXXXX.pth
        videos/  (if --save_video is enabled)
```

## Training Tips

- The agent typically requires around 1-2 million steps to achieve good performance
- GPU training is automatically enabled if available
- Early episodes may have low rewards as the agent learns to balance
- The SAC algorithm is quite stable and should show steady improvement over time 
README.md
2 KB