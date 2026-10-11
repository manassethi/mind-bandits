"""Top-k sparse autoencoder for pooled transformer activations."""
import torch
from torch import nn


class TopKSAE(nn.Module):
    def __init__(self, input_dimensions, latent_dimensions, top_k, center, scale):
        super().__init__()
        if not 0 < top_k <= latent_dimensions:
            raise ValueError('top_k must be between 1 and latent dimensions')
        self.top_k = top_k
        self.register_buffer('center', torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.encoder = nn.Linear(input_dimensions, latent_dimensions)
        self.decoder = nn.Linear(latent_dimensions, input_dimensions, bias=False)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, inputs):
        positive = torch.relu(self.encoder((inputs-self.center)/self.scale))
        values, indices = positive.topk(self.top_k, dim=-1)
        return torch.zeros_like(positive).scatter(-1, indices, values)

    def forward(self, inputs):
        codes = self.encode(inputs)
        return self.decoder(codes)*self.scale+self.center, codes


def preprocess_states(states, mode="none"):
    """Apply the checkpoint's per-article preprocessing before centering."""
    import numpy as np
    if mode == "none":
        return states
    if mode == "unit":
        return states / np.maximum(np.linalg.norm(states, axis=1, keepdims=True), 1e-12)
    raise ValueError(f"Unknown state normalization: {mode}")
