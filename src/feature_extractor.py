import torch
import torch.nn as nn
from .layers import LiquidCell, FeatureAttention, TemporalAttention, ImplicitPhysicsHead, MultiScaleConv

class LiquidTransformerFeatureExtractor(nn.Module):
    """Transformer and liquid-cell feature extractor used by the full model."""

    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.feature_attention = FeatureAttention(input_dim)
        self.embedding = nn.Linear(input_dim, d_model)
        self.temporal_attention = TemporalAttention(d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.multiscale_conv = MultiScaleConv(d_model)
        self.lnn_layer = LiquidCell(d_model, d_model)
        self.smoothing_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=15, padding=7, groups=d_model)
        self.physics_head = ImplicitPhysicsHead(d_model * 2)
        self.out_dim = d_model * 2

    def _encode_windows(self, x):
        context = x.mean(dim=1)
        attn_weights = self.feature_attention.fc(context)
        x = x * (1.0 + attn_weights.unsqueeze(1))
        x = self.embedding(x)
        x = self.temporal_attention(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.multiscale_conv(x)
        return x

    def _summarize_window(self, output_h, alphas):
        output_h = torch.stack(output_h, dim=1)
        output_h = output_h.permute(0, 2, 1)
        output_h = self.smoothing_conv(output_h)
        output_h = output_h.permute(0, 2, 1)
        output_h = output_h[:, -1, :]
        alphas_stack = torch.stack(alphas, dim=1)
        mean_alpha = torch.mean(alphas_stack, dim=1)
        features = torch.cat([output_h, mean_alpha], dim=1)
        physics_out = None
        physics_out = self.physics_head(features)
        return (features, physics_out)

    def _forward_independent_windows(self, x, init_state=None):
        batch, seq, dim = x.shape
        h = torch.zeros(batch, dim, device=x.device)
        lnn_outputs = []
        alphas = []
        for t in range(seq):
            h, alpha = self.lnn_layer(x[:, t, :], h)
            lnn_outputs.append(h)
            alphas.append(alpha)
        features, physics_out = self._summarize_window(lnn_outputs, alphas)
        return (features, physics_out, None)

    def _forward_stateful_windows(self, x, init_state=None, detach_state=True):
        return self._forward_independent_windows(x, init_state=None)

    def forward(self, x, init_state=None, return_state=False, stateful_windows=False, detach_state=True):
        x = self._encode_windows(x)
        features, physics_out, final_state = self._forward_independent_windows(x, init_state=None)
        if return_state:
            return (features, physics_out, final_state)
        return (features, physics_out)
