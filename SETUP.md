# Setup Guide for Temporal GFN

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Temporal-GFNs.git
cd Temporal-GFNs
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python test_installation.py
```

You should see:

```
============================================================
Testing Temporal GFN Installation
============================================================

✓ All modules imported successfully
✓ Configuration loaded successfully
✓ Environment working correctly
✓ Model working correctly
✓ Loss computation working correctly
✓ Data loader working correctly
✓ Full forward and backward pass working correctly

============================================================
✓ All tests passed! Installation is working correctly.
============================================================
```

### 5. Run Training

#### Quick Test (20 epochs)

```bash
python main.py --epochs 20 --batch_size 8
```

#### Full Training (Default Configuration)

```bash
python main.py
```

#### Custom Configuration

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

## Command Line Arguments

| Argument          | Description                                | Default |
| ----------------- | ------------------------------------------ | ------- |
| `--start_k`       | Initial number of quantization bins        | 10      |
| `--max_k`         | Maximum number of quantization bins        | 128     |
| `--lambda_adapt`  | Adaptation sensitivity parameter           | 0.1     |
| `--epsilon`       | Reward improvement threshold               | 0.02    |
| `--beta`          | Reward temperature parameter               | 10.0    |
| `--entropy_reg`   | Entropy regularization weight              | 0.01    |
| `--epochs`        | Number of training epochs                  | 100     |
| `--batch_size`    | Batch size                                 | 32      |
| `--lr`            | Learning rate                              | 1e-3    |
| `--warmup_epochs` | Warmup epochs before adaptive quantization | 10      |
| `--device`        | Device (cuda/cpu/auto)                     | auto    |

## Understanding the Output

During training, you'll see output like:

```
Epoch   0 | K= 10 | Loss=2.3456 | Reward=0.5234 | MSE=0.0823 | Entropy=0.8234 | log(Z)=-0.0001
Epoch  10 | K= 12 | Loss=1.8234 | Reward=0.6456 | MSE=0.0623 | Entropy=0.7123 | log(Z)=-0.0045
[Epoch 15] Adapting: K 12 -> 15
  ΔR=0.0150, H=0.6234, η=1.0523
```

### Output Metrics

- **K**: Current number of quantization bins
- **Loss**: Total trajectory balance loss (lower is better)
- **Reward**: Average reward (higher is better, range [0, 1])
- **MSE**: Mean squared error between prediction and target (lower is better)
- **Entropy**: Average policy entropy (measures exploration)
- **log(Z)**: Learned partition function

### Adaptive Quantization Events

When you see:

```
[Epoch 15] Adapting: K 12 -> 15
  ΔR=0.0150, H=0.6234, η=1.0523
```

This means:

- The model detected a learning plateau (low ΔR) or high confidence (low H)
- The number of bins is increasing from 12 to 15
- η is the adaptation factor calculated from Equation 1 in the paper

## Troubleshooting

### Issue: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Make sure you activated the virtual environment:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size:

```bash
python main.py --batch_size 8
```

Or use CPU:

```bash
python main.py --device cpu
```

### Issue: Slow Training

**Problem**: Training is very slow

**Solution**:

1. Use GPU if available (it will be auto-detected)
2. Reduce sequence length or prediction horizon in `src/config.py`
3. Use fewer transformer layers or heads in `src/config.py`

### Issue: Loss Not Decreasing

**Problem**: Loss stays high or increases

**Solution**:

1. Increase number of epochs (default 100 might not be enough)
2. Adjust learning rate: `--lr 5e-4` or `--lr 2e-3`
3. Adjust beta parameter: `--beta 5.0` (lower) or `--beta 20.0` (higher)
4. Increase warmup epochs: `--warmup_epochs 20`

## Next Steps

After successful training:

1. **Check the saved model**: `temporal_gfn_model.pt`
2. **Analyze results**: Load the model and examine reward history
3. **Customize data**: Modify `src/data_loader.py` to use your own time series data
4. **Tune hyperparameters**: Adjust configuration based on your specific task

## Advanced Usage

### Using Custom Data

To use your own time series data, modify `src/data_loader.py`:

```python
def get_custom_data(batch_size, seq_len, prediction_horizon=64):
    # Load your data here
    # Return: (context, target)
    # context: [batch_size, seq_len]
    # target: [batch_size, prediction_horizon]
    pass
```

Then update `main.py` to use your custom data loader.

### Monitoring Training

You can add logging to track training progress:

```python
import wandb  # or tensorboard

# In main.py, add logging
wandb.log({
    'epoch': epoch,
    'k': current_k,
    'loss': loss.item(),
    'reward': mean_r,
    'mse': mean_mse,
    'entropy': mean_entropy
})
```

## GPU Support

The code automatically detects and uses GPU if available:

```bash
# Will use GPU if available
python main.py

# Force CPU
python main.py --device cpu

# Force GPU
python main.py --device cuda
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{hassen2025temporal,
  title={Adaptive Quantization in Generative Flow Networks for Probabilistic Sequential Prediction},
  author={Hassen, et al.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
