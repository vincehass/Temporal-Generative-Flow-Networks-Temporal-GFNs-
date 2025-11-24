from dataclasses import dataclass

@dataclass
class Config:
    # Data
    sequence_length: int = 512  # Context window (T)
    prediction_horizon: int = 64 # T'
    
    # GFN
    start_k: int = 10
    max_k: int = 128
    hidden_dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    
    # Training
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    n_trajectories: int = 1000 # Per epoch
    
    # Adaptive Quantization (Eq 1)
    lambda_adapt: float = 0.1
    epsilon_threshold: float = 0.02
    delta_window: int = 5      # For rolling average of reward
    warmup_epochs: int = 10
    
    # Reward & Loss (Eq 3 & 4)
    beta_reward: float = 10.0
    lambda_entropy: float = 0.01

