#!/usr/bin/env python3
"""
Demo script showing how to use Temporal GFN for forecasting.
This script demonstrates:
1. Loading/training a model
2. Making predictions
3. Visualizing results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.config import Config
from src.env import TemporalForecastingEnv, TimeSeriesState
from src.model import TemporalPolicy
from src.data_loader import get_sine_data, normalize_data

def train_quick_model():
    """Train a quick model for demonstration."""
    print("Training a quick model (30 epochs)...")
    import torch.optim as optim
    import math
    from src.gfn_utils import trajectory_balance_loss
    
    cfg = Config()
    cfg.epochs = 30
    cfg.batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy = TemporalPolicy(cfg).to(device)
    log_z = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = optim.Adam(list(policy.parameters()) + [log_z], lr=cfg.lr)
    
    current_k = cfg.start_k
    env = TemporalForecastingEnv(
        torch.zeros(cfg.batch_size, cfg.sequence_length), 
        cfg.prediction_horizon, 
        current_k
    )
    
    for epoch in range(cfg.epochs):
        context, target = get_sine_data(cfg.batch_size, cfg.sequence_length, cfg.prediction_horizon)
        context = normalize_data(context, -1, 1).to(device)
        target = normalize_data(target, -1, 1).to(device)
        
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
        
        log_pb_sum = -math.log(current_k) * cfg.prediction_horizon
        mse = torch.mean((generated_seq - target)**2, dim=1)
        reward = torch.exp(-cfg.beta_reward * mse)
        log_reward = torch.log(reward + 1e-8)
        
        loss = trajectory_balance_loss(
            log_pf_sum, log_pb_sum, log_reward, log_z, 
            entropy_sum, cfg.lambda_entropy
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss={loss.item():.4f} | Reward={reward.mean().item():.4f}")
    
    print("✓ Training complete!\n")
    return policy, cfg, device

def generate_forecast(policy, context, cfg, device, num_samples=10):
    """Generate multiple forecast trajectories."""
    policy.eval()
    
    env = TemporalForecastingEnv(
        torch.zeros(1, cfg.sequence_length), 
        cfg.prediction_horizon, 
        cfg.start_k
    )
    
    all_forecasts = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            state = context.to(device)
            generated_values = []
            
            for t in range(cfg.prediction_horizon):
                logits = policy(state)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                
                ts_state = TimeSeriesState(state)
                ts_state = env.step(ts_state, actions)
                state = ts_state.tensor
                
                current_vals = env.bin_centers.to(device)[actions]
                generated_values.append(current_vals.cpu())
            
            forecast = torch.stack(generated_values, dim=1)
            all_forecasts.append(forecast)
    
    return torch.cat(all_forecasts, dim=0)  # [num_samples, horizon]

def visualize_results(context, target, forecasts):
    """Visualize the forecasting results."""
    plt.figure(figsize=(12, 6))
    
    # Context window
    context_x = np.arange(len(context))
    plt.plot(context_x, context, 'b-', linewidth=2, label='Context (Historical)')
    
    # Ground truth
    target_x = np.arange(len(context), len(context) + len(target))
    plt.plot(target_x, target, 'g-', linewidth=2, label='Ground Truth')
    
    # Forecasts
    for i, forecast in enumerate(forecasts):
        if i == 0:
            plt.plot(target_x, forecast, 'r-', alpha=0.3, linewidth=1, label='GFN Samples')
        else:
            plt.plot(target_x, forecast, 'r-', alpha=0.3, linewidth=1)
    
    # Mean forecast
    mean_forecast = forecasts.mean(dim=0)
    plt.plot(target_x, mean_forecast, 'r-', linewidth=2, label='GFN Mean')
    
    # Uncertainty bounds (std)
    std_forecast = forecasts.std(dim=0)
    plt.fill_between(target_x, 
                     mean_forecast - std_forecast, 
                     mean_forecast + std_forecast, 
                     color='r', alpha=0.2, label='±1 std')
    
    plt.axvline(x=len(context), color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Temporal GFN: Probabilistic Time Series Forecasting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('temporal_gfn_demo.png', dpi=300)
    print("✓ Plot saved to 'temporal_gfn_demo.png'")
    plt.show()

def calculate_metrics(target, forecasts):
    """Calculate forecasting metrics."""
    mean_forecast = forecasts.mean(dim=0)
    
    # MSE
    mse = torch.mean((mean_forecast - target)**2).item()
    
    # MAE
    mae = torch.mean(torch.abs(mean_forecast - target)).item()
    
    # Coverage (percentage of true values within ±1 std)
    std_forecast = forecasts.std(dim=0)
    lower = mean_forecast - std_forecast
    upper = mean_forecast + std_forecast
    coverage = ((target >= lower) & (target <= upper)).float().mean().item()
    
    return mse, mae, coverage

def main():
    print("="*60)
    print("Temporal GFN Demo: Probabilistic Time Series Forecasting")
    print("="*60 + "\n")
    
    # Step 1: Train or load model
    print("Step 1: Training model...")
    policy, cfg, device = train_quick_model()
    
    # Step 2: Generate test data
    print("Step 2: Generating test data...")
    context, target = get_sine_data(1, cfg.sequence_length, cfg.prediction_horizon)
    context = normalize_data(context, -1, 1)
    target = normalize_data(target, -1, 1)
    print(f"Context shape: {context.shape}, Target shape: {target.shape}\n")
    
    # Step 3: Generate forecasts
    print("Step 3: Generating forecasts (20 samples)...")
    forecasts = generate_forecast(policy, context, cfg, device, num_samples=20)
    print(f"Forecasts shape: {forecasts.shape}\n")
    
    # Step 4: Calculate metrics
    print("Step 4: Calculating metrics...")
    mse, mae, coverage = calculate_metrics(target[0], forecasts)
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Coverage (±1σ): {coverage*100:.2f}%\n")
    
    # Step 5: Visualize
    print("Step 5: Visualizing results...")
    visualize_results(
        context[0].cpu().numpy(), 
        target[0].cpu().numpy(), 
        forecasts.cpu()
    )
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)

if __name__ == "__main__":
    main()

