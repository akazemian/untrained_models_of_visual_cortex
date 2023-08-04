import pickle
import os
from torch import nn
import numpy as np
import torch



class RandomProjection(nn.Module):
    def __init__(
        self, *, out_channels: int, seed: int = 0, allow_expansion: bool = True
    ) -> None:
        self.seed = seed
        self.out_channels = out_channels
        self.expand = allow_expansion
        self.projections = {}
        super().__init__()

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Applies a random orthonormal projection to the features."""
        features = features.flatten(start_dim=1)
        in_channels = features.shape[-1]
        if in_channels <= self.out_channels:
            if not self.expand:
                return features

        if in_channels not in self.projections:
            self.projections[in_channels] = self._compute_projection(
                in_channels=in_channels
            )

        projection = torch.from_numpy(self.projections[in_channels])
        #projection = self.projections[in_channels]
        return self._project(features=features, projection=projection)

    def _project(
        self, *, features: torch.Tensor, projection: torch.Tensor
    ) -> torch.Tensor:
        features = features.cpu().numpy()
        projection = projection.cpu().numpy()
        return features @ projection


    def _compute_projection(self, *, in_channels: int) -> np.ndarray:
        rng = np.random.default_rng(seed=self.seed)
        projection, _ = torch.linalg.qr(
            torch.from_numpy(
                rng.standard_normal(
                    size=(in_channels, self.out_channels), dtype=np.float32
                )
            )
        )
        return projection.cpu().numpy()