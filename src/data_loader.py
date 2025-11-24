import torch
import numpy as np

def get_sine_data(batch_size, seq_len, prediction_horizon=64):
    """
    Generate synthetic sine wave data for testing.
    
    Args:
        batch_size: Number of samples in the batch
        seq_len: Length of context window
        prediction_horizon: Number of future steps to predict
    
    Returns:
        context: [batch_size, seq_len] - Historical context
        target: [batch_size, prediction_horizon] - Ground truth future values
    """
    t = torch.linspace(0, 100, seq_len + prediction_horizon)
    data = torch.sin(t).unsqueeze(0).repeat(batch_size, 1)
    # Add some noise to make it more realistic
    data += torch.randn_like(data) * 0.1
    return data[:, :seq_len], data[:, seq_len:]

def get_mixed_sine_data(batch_size, seq_len, prediction_horizon=64):
    """
    Generate mixed frequency sine waves for more complex patterns.
    
    Args:
        batch_size: Number of samples in the batch
        seq_len: Length of context window
        prediction_horizon: Number of future steps to predict
    
    Returns:
        context: [batch_size, seq_len] - Historical context
        target: [batch_size, prediction_horizon] - Ground truth future values
    """
    total_len = seq_len + prediction_horizon
    data = []
    
    for _ in range(batch_size):
        t = torch.linspace(0, 100, total_len)
        # Mix of different frequencies
        freq1 = np.random.uniform(0.5, 2.0)
        freq2 = np.random.uniform(0.1, 0.5)
        signal = torch.sin(freq1 * t) + 0.5 * torch.sin(freq2 * t)
        signal += torch.randn_like(signal) * 0.05
        data.append(signal)
    
    data = torch.stack(data)
    return data[:, :seq_len], data[:, seq_len:]

def normalize_data(data, v_min=-1, v_max=1):
    """
    Normalize data to [v_min, v_max] range.
    
    Args:
        data: Input tensor
        v_min: Minimum value
        v_max: Maximum value
    
    Returns:
        Normalized data
    """
    data_min = data.min()
    data_max = data.max()
    
    if data_max - data_min < 1e-8:
        return torch.zeros_like(data)
    
    normalized = (data - data_min) / (data_max - data_min)
    normalized = normalized * (v_max - v_min) + v_min
    return normalized

