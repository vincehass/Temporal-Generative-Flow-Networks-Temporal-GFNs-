import torch
import torch.optim as optim
import numpy as np
import argparse
import math
from src.config import Config
from src.env import TemporalForecastingEnv, TimeSeriesState
from src.model import TemporalPolicy
from src.gfn_utils import trajectory_balance_loss
from src.data_loader import get_sine_data, normalize_data

def parse_args():
    parser = argparse.ArgumentParser(description='Train Temporal GFN for time series forecasting')
    parser.add_argument('--start_k', type=int, default=10, help='Initial number of quantization bins')
    parser.add_argument('--max_k', type=int, default=128, help='Maximum number of quantization bins')
    parser.add_argument('--lambda_adapt', type=float, default=0.1, help='Adaptation sensitivity parameter')
    parser.add_argument('--epsilon', type=float, default=0.02, help='Reward improvement threshold')
    parser.add_argument('--beta', type=float, default=10.0, help='Reward temperature parameter')
    parser.add_argument('--entropy_reg', type=float, default=0.01, help='Entropy regularization weight')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs before adaptive quantization')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup configuration
    cfg = Config()
    cfg.start_k = args.start_k
    cfg.max_k = args.max_k
    cfg.lambda_adapt = args.lambda_adapt
    cfg.epsilon_threshold = args.epsilon
    cfg.beta_reward = args.beta
    cfg.lambda_entropy = args.entropy_reg
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.warmup_epochs = args.warmup_epochs
    
    # Device setup
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Configuration: {cfg}")
    
    # 1. Initialize
    policy = TemporalPolicy(cfg).to(device)
    log_z = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = optim.Adam(list(policy.parameters()) + [log_z], lr=cfg.lr)
    
    # History for adaptation
    reward_history = []
    entropy_history = []
    k_history = []
    
    current_k = cfg.start_k
    env = TemporalForecastingEnv(
        torch.zeros(cfg.batch_size, cfg.sequence_length), 
        cfg.prediction_horizon, 
        current_k
    )

    print(f"\n{'='*60}")
    print(f"Starting training with K={current_k}...")
    print(f"{'='*60}\n")

    # 2. Training Loop (Algo 1)
    for epoch in range(cfg.epochs):
        
        # --- A. Adaptive Quantization Update (Algo 1 lines 3-10) ---
        if epoch >= cfg.warmup_epochs and len(reward_history) >= cfg.delta_window * 2:
            avg_reward = np.mean(reward_history[-cfg.delta_window:])
            prev_avg = np.mean(reward_history[-(cfg.delta_window*2):-cfg.delta_window])
            delta_R = avg_reward - prev_avg
            
            # Avg Entropy calculation
            avg_entropy = np.mean(entropy_history[-cfg.delta_window:]) if len(entropy_history) >= cfg.delta_window else 0.5
            
            # Equation 1
            imp_signal = max(0, cfg.epsilon_threshold - delta_R) / cfg.epsilon_threshold
            conf_signal = 1 - avg_entropy
            eta = 1 + cfg.lambda_adapt * (imp_signal + conf_signal)
            
            # Equation 2
            new_k = min(cfg.max_k, int(current_k * eta))
            
            if new_k > current_k:
                print(f"\n[Epoch {epoch}] Adapting: K {current_k} -> {new_k}")
                print(f"  ΔR={delta_R:.4f}, H={avg_entropy:.4f}, η={eta:.4f}")
                policy.resize_output_layer(new_k)
                policy.to(device)
                optimizer = optim.Adam(list(policy.parameters()) + [log_z], lr=cfg.lr) # Re-init opt for new params
                env.update_k(new_k)
                current_k = new_k

        k_history.append(current_k)

        # --- B. Trajectory Sampling (Algo 1 lines 12-24) ---
        context, target = get_sine_data(cfg.batch_size, cfg.sequence_length, cfg.prediction_horizon)
        
        # Normalize data to [-1, 1] range
        context = normalize_data(context, -1, 1)
        target = normalize_data(target, -1, 1)
        
        context = context.to(device)
        target = target.to(device)
        
        # Initialize Trajectory
        state = context
        log_pf_sum = torch.zeros(cfg.batch_size, device=device)
        entropy_sum = torch.zeros(cfg.batch_size, device=device)
        generated_values = []

        # Forward Unroll
        for t in range(cfg.prediction_horizon):
            logits = policy(state)
            probs = torch.softmax(logits, dim=-1)
            
            # Categorical Sampling (Hard action for state update)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample() # Indices [Batch]
            
            # Accumulate Flow Probs
            log_pf_sum += dist.log_prob(actions)
            entropy_sum += dist.entropy()
            
            # State Update (Hard)
            # In env.step we map index -> value
            # We create a dummy TimeSeriesState wrapper for consistency
            ts_state = TimeSeriesState(state)
            ts_state = env.step(ts_state, actions)
            state = ts_state.tensor
            
            # Soft Sample (for Reward/Analysis - theoretical requirement)
            # Note: For standard TB, we use hard actions to traverse.
            # If we needed gradients *through* the values into the reward, 
            # we would use: values = sum(probs * bin_centers).
            # Here we just record the physical value of the hard action for MSE.
            # (Matches standard GFN logic where Reward is on the discrete object)
            
            current_vals = env.bin_centers.to(device)[actions]
            generated_values.append(current_vals)

        generated_seq = torch.stack(generated_values, dim=1) # [Batch, Horizon]

        # --- C. Loss Computation (Algo 1 lines 25-30) ---
        
        # 1. Backward Probability (Uniform P_B)
        # log P_B = sum(log(1/K)) for t steps
        log_pb_sum = -math.log(current_k) * cfg.prediction_horizon
        
        # 2. Reward (Eq 3)
        # Exp(-beta * MSE)
        mse = torch.mean((generated_seq - target)**2, dim=1)
        reward = torch.exp(-cfg.beta_reward * mse)
        log_reward = torch.log(reward + 1e-8)
        
        # 3. TB Loss
        loss = trajectory_balance_loss(
            log_pf_sum, 
            log_pb_sum, 
            log_reward, 
            log_z, 
            entropy_sum, 
            cfg.lambda_entropy
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        mean_r = reward.mean().item()
        mean_mse = mse.mean().item()
        mean_entropy = (entropy_sum / cfg.prediction_horizon).mean().item()
        
        reward_history.append(mean_r)
        entropy_history.append(mean_entropy)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | K={current_k:3d} | Loss={loss.item():.4f} | "
                  f"Reward={mean_r:.4f} | MSE={mean_mse:.4f} | Entropy={mean_entropy:.4f} | "
                  f"log(Z)={log_z.item():.4f}")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Final K: {current_k}")
    print(f"Final Reward: {reward_history[-1]:.4f}")
    print(f"Final MSE: {mean_mse:.4f}")
    print(f"{'='*60}\n")
    
    # Save model
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'log_z': log_z,
        'config': cfg,
        'final_k': current_k,
        'reward_history': reward_history,
        'k_history': k_history,
    }, 'temporal_gfn_model.pt')
    print("Model saved to 'temporal_gfn_model.pt'")

if __name__ == "__main__":
    main()

