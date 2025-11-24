import torch
from torchtyping import TensorType

class TimeSeriesState:
    """
    Represents the sliding window state s_t.
    """
    def __init__(self, tensor: TensorType["batch_shape", "window_size"]):
        self.tensor = tensor
        self.batch_shape = tensor.shape[:-1]

    def extend(self, value: TensorType["batch_shape", 1]):
        # Slide window: remove first, append new value
        new_window = torch.cat([self.tensor[:, 1:], value], dim=1)
        return TimeSeriesState(new_window)

class TemporalForecastingEnv:
    def __init__(self, context_window, horizon, k_bins, v_min=-1, v_max=1):
        # We define action space as Discrete(K)
        # Note: In torchgfn, 'n_actions' usually implies the graph structure.
        # Here we just treat it as a step-by-step generation.
        self.window_size = context_window.shape[-1]
        self.horizon = horizon
        self.k_bins = k_bins
        self.v_min = v_min
        self.v_max = v_max
        self.bin_centers = self._compute_bin_centers(k_bins)
        
        # Initial states (batch of contexts)
        self.context = context_window 
        
    def _compute_bin_centers(self, k):
        # Eq in Section 3.2.1
        step = (self.v_max - self.v_min) / (k - 1) if k > 1 else 0
        return torch.linspace(self.v_min, self.v_max, k)

    def update_k(self, new_k):
        """Called by the Adaptive Manager in main loop"""
        self.k_bins = new_k
        self.bin_centers = self._compute_bin_centers(new_k)

    def step(self, states: TimeSeriesState, actions: torch.Tensor) -> TimeSeriesState:
        """
        actions: Indices [0, K-1]
        """
        # Map discrete action index -> continuous value (Hard Sample logic for state update)
        # In a real GFN graph, we usually return a 'new_state'.
        
        # Get bin centers for the current K
        centers = self.bin_centers.to(states.tensor.device)
        
        # actions are indices, we pick the values
        selected_values = centers[actions].unsqueeze(-1)
        
        return states.extend(selected_values)
    
    def is_sink_state(self, states: TimeSeriesState) -> torch.Tensor:
        # Since we just simulate fixed horizon T', we handle termination in the loop
        # or via a step counter attached to the state if strictly following torchgfn.
        # For simplicity in this tutorial, we assume fixed horizon unrolling.
        return torch.zeros(states.batch_shape, dtype=torch.bool)

