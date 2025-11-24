#!/usr/bin/env python3
"""
Quick test script to verify the installation and basic functionality.
"""

import torch
import sys
from src.config import Config
from src.env import TemporalForecastingEnv, TimeSeriesState
from src.model import TemporalPolicy
from src.gfn_utils import trajectory_balance_loss
from src.data_loader import get_sine_data, normalize_data

def test_imports():
    """Test that all modules can be imported."""
    print("✓ All modules imported successfully")

def test_config():
    """Test configuration."""
    cfg = Config()
    assert cfg.start_k == 10
    assert cfg.max_k == 128
    print("✓ Configuration loaded successfully")

def test_environment():
    """Test environment creation and basic operations."""
    context = torch.randn(4, 512)
    env = TemporalForecastingEnv(context, horizon=64, k_bins=10)
    
    # Test state
    state = TimeSeriesState(context)
    actions = torch.randint(0, 10, (4,))
    new_state = env.step(state, actions)
    
    assert new_state.tensor.shape == context.shape
    print("✓ Environment working correctly")

def test_model():
    """Test model creation and forward pass."""
    cfg = Config()
    model = TemporalPolicy(cfg)
    
    x = torch.randn(4, cfg.sequence_length)
    logits = model(x)
    
    assert logits.shape == (4, cfg.start_k)
    
    # Test resize
    model.resize_output_layer(20)
    logits = model(x)
    assert logits.shape == (4, 20)
    
    print("✓ Model working correctly")

def test_loss():
    """Test trajectory balance loss."""
    batch_size = 4
    log_pf = torch.randn(batch_size)
    log_pb = torch.tensor(-2.3026)  # log(1/10)
    log_reward = torch.randn(batch_size)
    log_z = torch.nn.Parameter(torch.zeros(1))
    entropy = torch.rand(batch_size)
    
    loss = trajectory_balance_loss(log_pf, log_pb, log_reward, log_z, entropy, 0.01)
    assert loss.item() >= 0
    
    print("✓ Loss computation working correctly")

def test_data_loader():
    """Test data loading."""
    context, target = get_sine_data(4, 512, 64)
    assert context.shape == (4, 512)
    assert target.shape == (4, 64)
    
    # Test normalization
    normalized = normalize_data(context, -1, 1)
    assert normalized.min() >= -1.0 - 1e-5
    assert normalized.max() <= 1.0 + 1e-5
    
    print("✓ Data loader working correctly")

def test_full_forward_pass():
    """Test a complete forward pass through the system."""
    cfg = Config()
    cfg.batch_size = 4
    cfg.sequence_length = 512
    cfg.prediction_horizon = 64
    
    device = torch.device('cpu')
    
    # Initialize
    policy = TemporalPolicy(cfg).to(device)
    log_z = torch.nn.Parameter(torch.zeros(1, device=device))
    
    # Get data
    context, target = get_sine_data(cfg.batch_size, cfg.sequence_length, cfg.prediction_horizon)
    context = normalize_data(context, -1, 1).to(device)
    target = normalize_data(target, -1, 1).to(device)
    
    # Environment
    env = TemporalForecastingEnv(
        torch.zeros(cfg.batch_size, cfg.sequence_length), 
        cfg.prediction_horizon, 
        cfg.start_k
    )
    
    # Forward pass
    state = context
    log_pf_sum = torch.zeros(cfg.batch_size, device=device)
    entropy_sum = torch.zeros(cfg.batch_size, device=device)
    generated_values = []
    
    for t in range(cfg.prediction_horizon):
        logits = policy(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        
        log_pf_sum += dist.log_prob(actions)
        entropy_sum += dist.entropy()
        
        ts_state = TimeSeriesState(state)
        ts_state = env.step(ts_state, actions)
        state = ts_state.tensor
        
        current_vals = env.bin_centers.to(device)[actions]
        generated_values.append(current_vals)
    
    generated_seq = torch.stack(generated_values, dim=1)
    
    # Compute loss
    import math
    log_pb_sum = -math.log(cfg.start_k) * cfg.prediction_horizon
    mse = torch.mean((generated_seq - target)**2, dim=1)
    reward = torch.exp(-cfg.beta_reward * mse)
    log_reward = torch.log(reward + 1e-8)
    
    loss = trajectory_balance_loss(
        log_pf_sum, 
        log_pb_sum, 
        log_reward, 
        log_z, 
        entropy_sum, 
        cfg.lambda_entropy
    )
    
    # Backward pass
    loss.backward()
    
    assert loss.item() >= 0
    print("✓ Full forward and backward pass working correctly")

def main():
    print("\n" + "="*60)
    print("Testing Temporal GFN Installation")
    print("="*60 + "\n")
    
    try:
        test_imports()
        test_config()
        test_environment()
        test_model()
        test_loss()
        test_data_loader()
        test_full_forward_pass()
        
        print("\n" + "="*60)
        print("✓ All tests passed! Installation is working correctly.")
        print("="*60 + "\n")
        print("You can now run training with: python main.py")
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

