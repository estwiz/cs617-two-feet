import torch

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