import torch
import torch.nn as nn
import math

class TemporalPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_dim
        
        # Encoder (Section 3.2.1: Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=config.n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # Input projection (Window -> d_model)
        self.input_proj = nn.Linear(config.sequence_length, self.d_model)
        
        # Output Head (Logits over K bins)
        self.output_head = nn.Linear(self.d_model, config.start_k)
        
    def forward(self, x):
        # x: [Batch, Window]
        h = self.input_proj(x) # [Batch, Hidden]
        # Transformer expects [Seq, Batch, Dim] usually, but here we treat window as feature
        # Or if treating time explicitly: [Window, Batch, 1] -> Transformer.
        # For simplicity based on paper description "summarizes relevant historical context":
        h = h.unsqueeze(1) # Fake seq len 1
        h = self.transformer(h)
        h = h.squeeze(1)
        
        logits = self.output_head(h)
        return logits

    def resize_output_layer(self, new_k):
        """
        Implements Weight Reuse (Section 3.2.2)
        Preserve existing weights, initialize new ones to near-zero.
        """
        old_k = self.output_head.out_features
        if new_k == old_k:
            return

        old_weight = self.output_head.weight.data
        old_bias = self.output_head.bias.data

        # Create new layer
        new_layer = nn.Linear(self.d_model, new_k)
        
        # Copy old weights into the "rescaled" positions or first indices?
        # The paper says: "weights... corresponding to pre-existing bins are preserved."
        # Since bins are uniformly distributed, bin i in K is roughly bin i*ratio in K_new.
        # However, purely appending is safer for implementation simplicity unless interpolation is used.
        # Simple strategy: Copy first old_k weights, init rest to zero.
        
        with torch.no_grad():
            new_layer.weight[:old_k] = old_weight
            new_layer.bias[:old_k] = old_bias
            
            # Initialize new bins to near-zero (epsilon noise)
            new_layer.weight[old_k:].normal_(0, 0.001)
            new_layer.bias[old_k:].zero_()
        
        self.output_head = new_layer
        print(f"Policy resized from {old_k} to {new_k} bins.")

