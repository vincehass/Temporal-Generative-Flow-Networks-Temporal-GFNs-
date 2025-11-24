# 🚀 Quick Start Guide

## Your Repository is Ready!

Everything has been implemented, tested, and verified. Here's how to use it:

---

## ⚡ 3-Step Quick Start

### Step 1: Activate Virtual Environment

```bash
cd /Users/nhassen/Documents/AIML/NeurIPS25_logistics/repo/Temporal-Generative-Flow-Networks-Temporal-GFNs-
source venv/bin/activate
```

### Step 2: Verify Installation

```bash
python test_installation.py
```

Expected output:

```
✓ All tests passed! Installation is working correctly.
```

### Step 3: Run Training

```bash
# Quick test (20 epochs)
python main.py --epochs 20 --batch_size 8

# OR full training (100 epochs)
python main.py
```

---

## 📊 Demo with Visualization

Generate forecasts and see beautiful plots:

```bash
python demo.py
```

This will:

- Train a model (30 epochs)
- Generate 20 forecast samples
- Calculate metrics (MSE, MAE, Coverage)
- Create visualization: `temporal_gfn_demo.png`

---

## 🎛️ Command Line Options

### Basic Configuration

```bash
python main.py --epochs 100 --batch_size 32 --lr 1e-3
```

### Adaptive Quantization

```bash
python main.py \
    --start_k 10 \          # Initial bins
    --max_k 128 \           # Maximum bins
    --lambda_adapt 0.1 \    # Adaptation sensitivity
    --epsilon 0.02          # Improvement threshold
```

### Reward & Loss

```bash
python main.py \
    --beta 10.0 \           # Reward temperature
    --entropy_reg 0.01      # Entropy weight
```

### Hardware

```bash
python main.py --device cuda   # Force GPU
python main.py --device cpu    # Force CPU
```

---

## 📁 What Was Implemented

### Core Files Created

```
✅ main.py                   - Training script (Algorithm 1)
✅ demo.py                   - Demo with visualization
✅ test_installation.py      - Installation verification
✅ requirements.txt          - Dependencies (all installed)
✅ README.md                 - Comprehensive documentation
✅ SETUP.md                  - Setup & troubleshooting
✅ PROJECT_SUMMARY.md        - Implementation summary
✅ .gitignore               - Git ignore patterns
```

### Source Code (`src/`)

```
✅ config.py                 - Hyperparameters
✅ env.py                    - Time series environment
✅ model.py                  - Transformer policy + weight reuse
✅ gfn_utils.py              - Trajectory balance loss
✅ data_loader.py            - Data loading utilities
```

### Virtual Environment

```
✅ venv/                     - Virtual environment
  ├── ✅ PyTorch 2.9.1
  ├── ✅ NumPy 2.3.5
  ├── ✅ Pandas 2.3.3
  ├── ✅ Matplotlib 3.10.7
  └── ✅ torchtyping 0.1.5
```

---

## 🎯 Key Features

### 1. Adaptive Quantization ✓

```python
# Automatically adjusts K during training
[Epoch 15] Adapting: K 10 -> 12
  ΔR=0.0150, H=0.6234, η=1.0523
```

### 2. Weight Reuse ✓

```python
# Preserves learned weights when K increases
Policy resized from 10 to 20 bins.
```

### 3. Trajectory Balance Loss ✓

```python
# TB loss + entropy regularization
loss = tb_loss - λ_entropy * H(π)
```

### 4. Probabilistic Forecasting ✓

```python
# Generates multiple diverse trajectories
forecasts = generate_forecast(policy, context, num_samples=20)
```

---

## 📈 Expected Training Output

```
Using device: cpu
Configuration: Config(sequence_length=512, prediction_horizon=64, ...)

============================================================
Starting training with K=10...
============================================================

Epoch   0 | K= 10 | Loss=2.3456 | Reward=0.5234 | MSE=0.0823 | ...
Epoch  10 | K= 12 | Loss=1.8234 | Reward=0.6456 | MSE=0.0623 | ...

[Epoch 15] Adapting: K 12 -> 15
  ΔR=0.0150, H=0.6234, η=1.0523

Epoch  20 | K= 15 | Loss=1.4567 | Reward=0.7234 | MSE=0.0456 | ...
...

============================================================
Training completed!
Final K: 15
Final Reward: 0.8234
Final MSE: 0.0234
============================================================

Model saved to 'temporal_gfn_model.pt'
```

---

## 🔧 Customization Examples

### Use Your Own Data

Edit `src/data_loader.py`:

```python
def get_my_data(batch_size, seq_len, prediction_horizon):
    # Load your data
    context = load_historical_data()
    target = load_future_data()
    return context, target
```

Then in `main.py`:

```python
from src.data_loader import get_my_data
# Replace get_sine_data with get_my_data
context, target = get_my_data(cfg.batch_size, ...)
```

### Adjust Model Size

Edit `src/config.py`:

```python
@dataclass
class Config:
    hidden_dim: int = 128      # Increase capacity
    n_layers: int = 4          # Deeper network
    n_heads: int = 8           # More attention
```

### Change Adaptive Strategy

Edit `src/config.py`:

```python
@dataclass
class Config:
    lambda_adapt: float = 0.2        # More aggressive
    epsilon_threshold: float = 0.01  # Stricter
    warmup_epochs: int = 20          # Longer warmup
```

---

## 📚 Documentation

| File                 | Description                        |
| -------------------- | ---------------------------------- |
| `README.md`          | Complete documentation with theory |
| `SETUP.md`           | Detailed setup & troubleshooting   |
| `PROJECT_SUMMARY.md` | Implementation overview            |
| `QUICK_START.md`     | This file!                         |

---

## 🐛 Troubleshooting

### Problem: ModuleNotFoundError

**Solution**: Activate virtual environment

```bash
source venv/bin/activate
```

### Problem: Loss not decreasing

**Solution**: Train longer or adjust hyperparameters

```bash
python main.py --epochs 200 --lr 5e-4 --beta 20.0
```

### Problem: Slow training

**Solution**: Use GPU or reduce batch/sequence size

```bash
python main.py --device cuda --batch_size 16
```

See `SETUP.md` for more troubleshooting.

---

## 📊 Evaluation Metrics

After training, the model saves metrics:

```python
# Load saved model
checkpoint = torch.load('temporal_gfn_model.pt')
reward_history = checkpoint['reward_history']
k_history = checkpoint['k_history']

# Plot learning curves
import matplotlib.pyplot as plt
plt.plot(reward_history)
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.show()
```

---

## 🎓 Paper Implementation

This code implements:

- ✅ **Algorithm 1**: Adaptive training loop
- ✅ **Equation 1**: Adaptation factor η
- ✅ **Equation 2**: K update rule
- ✅ **Equation 3**: Reward function
- ✅ **Equation 4**: TB loss
- ✅ **Section 3.2.2**: Weight reuse strategy
- ✅ **Section 3.2.1**: Transformer encoder

---

## 🎉 You're All Set!

Everything is installed, tested, and ready to use.

### Recommended Next Steps:

1. ✅ **Run test**: `python test_installation.py`
2. ✅ **Quick train**: `python main.py --epochs 20`
3. ✅ **Run demo**: `python demo.py`
4. 📊 **Analyze results**: Check plots and metrics
5. 🔧 **Customize**: Adapt for your data/task
6. 📝 **Read docs**: `README.md` for theory details

---

## 💡 Tips

- Start with small epochs (20-30) to test
- Monitor K adaptations - they show learning progress
- Use GPU for faster training: `--device cuda`
- Check `temporal_gfn_demo.png` for visual results
- Adjust `beta` if rewards are too low/high

---

## 📞 Support

- **Questions?** Check `SETUP.md` troubleshooting
- **Issues?** Review error messages carefully
- **Customization?** See code comments for guidance

---

**Happy Forecasting! 🚀**

_Temporal GFN implementation ready for research and production use._
