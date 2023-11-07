import pickle
import os
from torch import nn
import numpy as np
import torch



# class RandomProjection(nn.Module):
#     def __init__(
#         self, *, out_channels: int, seed: int = 0, allow_expansion: bool = True
#     ) -> None:
#         self.seed = seed
#         self.out_channels = out_channels
#         self.expand = allow_expansion
#         self.projections = {}
#         super().__init__()

#     def __call__(self, features: torch.Tensor) -> torch.Tensor:
#         """Applies a random orthonormal projection to the features."""
#         features = features.flatten(start_dim=1)
#         in_channels = features.shape[-1]
        
#         if in_channels <= self.out_channels:
#             if not self.expand:
#                 return features

#         if in_channels not in self.projections:
#             self.projections[in_channels] = self._compute_projection(
#                 in_channels=in_channels
#             )

#         projection = torch.from_numpy(self.projections[in_channels])
#         #projection = self.projections[in_channels]
#         return self._project(features=features, projection=projection)

#     def _project(
#         self, *, features: torch.Tensor, projection: torch.Tensor
#     ) -> torch.Tensor:
#         features = features.cpu().numpy()
#         projection = projection.cpu().numpy()
#         return features @ projection


#     def _compute_projection(self, *, in_channels: int) -> np.ndarray:
#         rng = np.random.default_rng(seed=self.seed)
#         projection, _ = torch.linalg.qr(
#             torch.from_numpy(
#                 rng.standard_normal(
#                     size=(in_channels, self.out_channels), dtype=np.float32
#                 )
#             )
#         )
#         return projection.cpu().numpy()



import torch
import torch.nn as nn

class RandomProjection(nn.Module):
    def __init__(self, *, out_channels: int, seed: int = 0, allow_expansion: bool = True) -> None:
        super().__init__()
        torch.manual_seed(seed)  # Seed for reproducibility in PyTorch
        self.out_channels = out_channels
        self.expand = allow_expansion
        self.projections = nn.ParameterDict()
        self.device= 'cuda'

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
        random_matrix = torch.randn((in_channels, self.out_channels), dtype=torch.float32, device=self.device)
        q, r = torch.linalg.qr(random_matrix)
        return q.to(self.device)


class RandomProjection(nn.Module):
    def __init__(self, *, out_channels: int, seed: int = 0, allow_expansion: bool = True) -> None:
        super().__init__()
        torch.manual_seed(seed)  # Seed for reproducibility in PyTorch
        self.out_channels = out_channels
        self.expand = allow_expansion
        self.projections = nn.ModuleDict()  # Changed from ParameterDict to ModuleDict
        self.device = 'cpu'

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Applies a random orthonormal projection to the features."""
        features = features.flatten(start_dim=1).to(self.device)
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
        random_matrix = torch.randn((in_channels, self.out_channels), device=self.device)
        q, _ = torch.linalg.qr(random_matrix)
        return q.to(self.device)

    def clear_projections(self):
        """Clear projections to free memory."""
        for name in list(self._buffers.keys()):
            if name.startswith('proj_'):
                delattr(self, name)



