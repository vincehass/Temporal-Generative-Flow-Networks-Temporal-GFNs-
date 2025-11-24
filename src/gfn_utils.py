import torch

def trajectory_balance_loss(
    log_pf_traj: torch.Tensor,   # Sum of log P_F forward
    log_pb_traj: torch.Tensor,   # Sum of log P_B backward (often uniform constant)
    log_reward: torch.Tensor,    # Log R(tau)
    log_z: torch.nn.Parameter,   # Learnable partition function
    entropy: torch.Tensor,       # Sum of entropies along trajectory
    lambda_entropy: float
):
    """
    Equation 4 + Entropy Regularization
    """
    # TB Loss
    diff = log_z + log_pf_traj - log_pb_traj - log_reward
    loss_tb = diff.pow(2).mean()
    
    # Total Loss
    loss = loss_tb - lambda_entropy * entropy.mean()
    
    return loss

