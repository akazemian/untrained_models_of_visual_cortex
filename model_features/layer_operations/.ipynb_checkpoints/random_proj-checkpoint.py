import pickle
import os
from torch import nn
import numpy as np
import torch



class RandomProjection(nn.Module):
    def __init__(self, *, out_channels: int, seed: int = 0, allow_expansion: bool = True) -> None:
        super().__init__()
        torch.manual_seed(seed)  # Seed for reproducibility in PyTorch
        self.out_channels = out_channels
        self.expand = allow_expansion
        self.projections = nn.ParameterDict()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Applies a random orthonormal projection to the features."""
        features = features.flatten(start_dim=1)
        in_channels = features.shape[-1]
        key = str(in_channels)

        if in_channels <= self.out_channels and not self.expand:
            return features

        if key not in self.projections:
            projection = self._compute_projection(in_channels=in_channels)
        # Register the projection as a parameter so it gets moved to the GPU
        self.projections[key] = nn.Parameter(projection, requires_grad=False)

        return features @ self.projections[key]

    def _compute_projection(self, *, in_channels: int) -> torch.Tensor:
        # Generate a random matrix and perform QR decomposition in PyTorch
        # The result will be on the same device as the module
        random_matrix = torch.randn((in_channels, self.out_channels), dtype=torch.float32)
        q, r = torch.linalg.qr(random_matrix)
        return q


class RandomProjection(nn.Module):
    def __init__(self, *, out_channels: int, seed: int = 0, allow_expansion: bool = True) -> None:
        super().__init__()
        torch.manual_seed(seed)  # Seed for reproducibility in PyTorch
        self.out_channels = out_channels
        self.expand = allow_expansion
        self.projections = nn.ModuleDict()  # Changed from ParameterDict to ModuleDict

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Applies a random orthonormal projection to the features."""
        features = features.flatten(start_dim=1)
        in_channels = features.shape[-1]
        key = str(in_channels)

        if in_channels <= self.out_channels and not self.expand:
            return features

        if key not in self.projections:
            projection = self._compute_projection(in_channels=in_channels)
            # Use register_buffer to add the projection without making it a parameter
            self.register_buffer(f'proj_{key}', projection, persistent=False)

        return features @ getattr(self, f'proj_{key}')

    def _compute_projection(self, *, in_channels: int) -> torch.Tensor:
        random_matrix = torch.randn((in_channels, self.out_channels))
        q, _ = torch.linalg.qr(random_matrix)
        return q

    def clear_projections(self):
        """Clear projections to free memory."""
        for name in list(self._buffers.keys()):
            if name.startswith('proj_'):
                delattr(self, name)



