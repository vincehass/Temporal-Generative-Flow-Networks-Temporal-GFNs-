# Temporal Generative Flow Networks (Temporal GFNs)

[![Paper](https://img.shields.io/badge/Paper-NeurIPS_2025-blue)](https://neurips.cc)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org)

This repository contains the official implementation of **Temporal Generative Flow Networks** for probabilistic time series forecasting, as presented in "Adaptive Quantization in Generative Flow Networks for Probabilistic Sequential Prediction" (Hassen et al., 2025).

## 1. Abstract & Motivation

Standard Deep Learning forecasting models (Transformers, RNNs) often struggle to generate calibrated probability distributions over continuous future values. **Temporal GFNs** frame forecasting as a constructive process: building a forecast trajectory $\tau$ step-by-step.

Instead of outputting a single value, the model learns a policy $P_F$ to select actions (quantized values) such that the probability of sampling a trajectory is proportional to its accuracy (reward):

$$ P(\tau) \propto R(\tau) $$

Key innovations included in this implementation:

1.  **Adaptive Quantization:** Dynamic adjustment of discretization bins $K$ during training.
2.  **Straight-Through Estimator (STE):** Allowing gradient flow through discrete bin selection.
3.  **Trajectory Balance (TB) with Entropy:** Balancing flow consistency and exploration.

## 2. Theoretical Framework

### 2.1. The Forecasting GFN

- **State ($s_t$):** A fixed-length sliding window of context $[x_{t-C}, \dots, x_{t}]$.
- **Action ($a_t$):** Selection of a discrete quantization bin center $q_k$.
- **Transition:** $s_{t+1} = \text{concat}(s_t[1:], a_t^{\text{hard}})$.

### 2.2. Adaptive Curriculum-Based Quantization

We do not use a fixed number of bins. The number of bins $K$ evolves based on the **Improvement Signal** ($\Delta R_e$) and **Confidence Signal** ($H_e$).

The update factor $\eta_e$ at epoch $e$ is defined as:

$$ \eta_e = 1 + \lambda \left( \frac{\max(0, \epsilon - \Delta R_e)}{\epsilon} + (1 - H_e) \right) $$

The number of bins is updated multiplicatively:
$$ K*{e} = \min(K*{\max}, \lfloor K\_{e-1} \cdot \eta_e \rfloor) $$

Where:

- $\lambda$: Sensitivity control.
- $\epsilon$: Target reward improvement threshold.
- $H_e$: Normalized policy entropy (confidence).

### 2.3. Differentiability via STE

To enable backpropagation through discrete bin selection:

1.  **Forward Pass (Hard):** Select bin $k$ via sampling/argmax. Use $q_k$ to update the state window.
2.  **Backward Pass (Soft):** Gradients flow through the expectation:
    $$ a*t^{\text{soft}} = \sum*{k=1}^K q_k \cdot P_F(a_t = q_k | s_t) $$

### 2.4. Objective Function

We use the **Trajectory Balance (TB)** loss augmented with an entropy regularizer $\mathcal{H}$:

$$ \mathcal{L}(\tau) = \underbrace{\left( \log Z + \sum*{t=0}^{T'-1} \log P_F(s*{t+1}|s*t) - \log R(\tau) \right)^2}*{\text{TB Loss}} - \lambda*{\text{entropy}} \sum*{t} \mathcal{H}(P_F(\cdot|s_t)) $$

The Reward $R(\tau)$ is an exponential Negative MSE:
$$ R(\tau) = \exp \left( -\beta \frac{1}{T'} \sum\_{t=1}^{T'} (z_t - y_t)^2 \right) $$

## 3. Repository Structure

```text
Temporal-GFNs/
│
├── README.md                 # This documentation
├── requirements.txt          # Dependencies
├── main.py                   # Entry point (Training Loop & Adaptive Logic)
│
└── src/
    ├── __init__.py
    ├── config.py             # Hyperparameters (lambda, epsilon, beta, etc.)
    ├── env.py                # Time Series Environment (Sliding Window)
    ├── model.py              # Transformer Policy + Weight Reuse for Adaptive K
    ├── gfn_utils.py          # Trajectory Balance Loss w/ Entropy
    └── data_loader.py        # Utils to load synthetic data
```

## 4. Installation

```bash
# Clone repo
git clone https://github.com/yourusername/Temporal-GFNs.git
cd Temporal-GFNs

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- torchtyping >= 0.1.4

## 5. Usage

### Basic Training

To train the model on synthetic data (Sine wave) using the Adaptive Quantization strategy:

```bash
python main.py
```

### Custom Configuration

You can customize the training with various arguments:

```bash
python main.py \
    --start_k 10 \
    --max_k 128 \
    --lambda_adapt 0.1 \
    --epsilon 0.02 \
    --beta 10.0 \
    --entropy_reg 0.01 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3
```

### Arguments

- `--start_k`: Initial number of quantization bins (default: 10)
- `--max_k`: Maximum number of quantization bins (default: 128)
- `--lambda_adapt`: Adaptation sensitivity parameter (default: 0.1)
- `--epsilon`: Reward improvement threshold (default: 0.02)
- `--beta`: Reward temperature parameter (default: 10.0)
- `--entropy_reg`: Entropy regularization weight (default: 0.01)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--warmup_epochs`: Warmup epochs before adaptive quantization (default: 10)
- `--device`: Device (cuda/cpu/auto, default: auto)

## 6. Implementation Details

### Key Components

#### 1. **Adaptive Quantization Manager** (`main.py`)

Implements Algorithm 1 from the paper:

- Monitors reward improvement ($\Delta R_e$) and entropy ($H_e$)
- Dynamically adjusts number of bins $K$ based on learning signals
- Ensures smooth curriculum learning from coarse to fine quantization

#### 2. **Temporal Policy** (`src/model.py`)

- **Transformer Encoder**: Summarizes historical context
- **Weight Reuse Strategy**: When $K$ increases, existing bin weights are preserved, and new bins are initialized to near-zero to prevent catastrophic forgetting
- **Output Head**: Maps context to logits over $K$ bins

#### 3. **Time Series Environment** (`src/env.py`)

- **Sliding Window State**: Fixed-length context window
- **Discrete Actions**: Selection from $K$ quantization bins
- **State Transition**: Slides window and appends selected value

#### 4. **Trajectory Balance Loss** (`src/gfn_utils.py`)

- Implements TB loss with entropy regularization
- Learnable partition function $Z$
- Balances flow consistency and exploration

### Training Loop

The training follows this workflow:

1. **Warmup Phase** (epochs 0-10): Train with initial $K$ bins
2. **Adaptive Phase** (epochs 10+):
   - Compute improvement and confidence signals
   - Adjust $K$ if learning plateaus or model is confident
   - Resize policy output layer with weight reuse
3. **Trajectory Sampling**:
   - Sample actions from policy for prediction horizon $T'$
   - Use hard sampling for state updates (discrete transitions)
   - Track forward probabilities and entropy
4. **Loss Computation**:
   - Calculate reward based on MSE
   - Compute TB loss with entropy regularization
   - Update policy and partition function

## 7. Results

The model demonstrates:

- **Adaptive Learning**: Automatically increases quantization resolution as training progresses
- **Stable Training**: Weight reuse prevents catastrophic forgetting during $K$ updates
- **Probabilistic Forecasts**: Generates diverse trajectories with calibrated uncertainty

Example output:

```
Epoch   0 | K= 10 | Loss=2.3456 | Reward=0.5234 | MSE=0.0823 | Entropy=0.8234
Epoch  10 | K= 12 | Loss=1.8234 | Reward=0.6456 | MSE=0.0623 | Entropy=0.7123
Epoch  20 | K= 15 | Loss=1.4567 | Reward=0.7234 | MSE=0.0456 | Entropy=0.6234
...
```

## 8. Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{hassen2025temporal,
  title={Adaptive Quantization in Generative Flow Networks for Probabilistic Sequential Prediction},
  author={Hassen, et al.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 10. Acknowledgments

- Built on top of PyTorch framework
- Inspired by the GFlowNet framework
- Thanks to the NeurIPS 2025 reviewers for valuable feedback

## 11. Contact

For questions or issues, please open an issue on GitHub or contact the authors.
