import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class LiquidCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Cell.
    Introduces input-dependent time constants for dynamic adaptability.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau_net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.update_net = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h_prev, attention_weights=None):
        alpha = self.tau_net(x)
        if attention_weights is not None:
            if attention_weights.shape[-1] == alpha.shape[-1]:
                alpha = alpha * attention_weights
        candidate = torch.tanh(self.update_net(torch.cat([x, h_prev], dim=1)))
        h_new = (1 - alpha) * h_prev + alpha * candidate
        return (h_new, alpha)

class FeatureAttention(nn.Module):
    """
    Feature Attention Mechanism to automatically weigh important features.
    Helps the model focus on features most relevant to degradation.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, input_dim // 2, bias=False), nn.ReLU(), nn.Linear(input_dim // 2, input_dim, bias=False), nn.Sigmoid())

    def forward(self, x):
        context = x.mean(dim=1)
        weights = self.fc(context)
        return x * (1.0 + weights.unsqueeze(1))

class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism.
    Identifies and highlights critical time steps while suppressing noisy ones.
    Operates after embedding to use richer context.
    """

    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(d_model, d_model // 4, kernel_size=3, padding=1), nn.ReLU(), nn.Conv1d(d_model // 4, 1, kernel_size=1), nn.Sigmoid())
        self.boost_strength = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        scores = self.conv(x_perm)
        scores = scores.permute(0, 2, 1)
        return x * (1.0 + self.boost_strength * scores)

class ImplicitPhysicsHead(nn.Module):
    """
    Implicit Physics-Informed Head.
    Predicts degradation rate/trend to enforce monotonicity constraint.
    """

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))

    def forward(self, x):
        return self.net(x)

class MultiScaleConv(nn.Module):
    """
    Multi-scale temporal convolution to capture degradation patterns
    at different time scales simultaneously.
    """

    def __init__(self, d_model):
        super().__init__()
        self.conv_short = nn.Conv1d(d_model, d_model // 3, kernel_size=3, padding=1, groups=1)
        self.conv_mid = nn.Conv1d(d_model, d_model // 3, kernel_size=7, padding=3, groups=1)
        self.conv_long = nn.Conv1d(d_model, d_model - 2 * (d_model // 3), kernel_size=15, padding=7, groups=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        s = F.relu(self.conv_short(x_perm))
        m = F.relu(self.conv_mid(x_perm))
        l = F.relu(self.conv_long(x_perm))
        out = torch.cat([s, m, l], dim=1).permute(0, 2, 1)
        return self.norm(out + x)
