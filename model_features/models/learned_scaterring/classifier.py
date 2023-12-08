import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from model_features.models.learned_scaterring.utils import *


class Classifier(nn.Module):
    def __init__(self, input_type: TensorType, nb_classes, avg_ker_size=1, avgpool=False,
                 bias=True, batch_norm="affine"):
        """
        :param input_type:
        :param nb_classes:
        :param avg_ker_size:
        :param avgpool:
        :param bias: whether to include a bias in the linear layer
        :param batch_norm: any of "affine", "std", "none".
        """
        super().__init__()

        self.input_type = input_type
        self.num_input_channels = (1 + self.input_type.complex) * self.input_type.num_channels
        spatial_shape = np.array(self.input_type.spatial_shape)

        if batch_norm not in ["affine", "std", "none"]:
            raise ValueError(f"Unknown batch norm parameter: {batch_norm}")
        self.has_batch_norm = batch_norm != "none"
        if self.has_batch_norm:
            self.bn = nn.BatchNorm2d(self.num_input_channels, affine=batch_norm == "affine")

        self.avg_ker_size = avg_ker_size
        self.avgpool = avgpool
        if self.avgpool:
            spatial_shape = np.array([1, 1])
        elif self.avg_ker_size > 1:
            spatial_shape = spatial_shape - avg_ker_size + 1

        num_features = self.num_input_channels * np.prod(spatial_shape)
        self.classifier = nn.Linear(num_features, nb_classes, bias=bias)

        self.output_type = TensorType(num_channels=nb_classes, spatial_shape=None, complex=False)

    def forward(self, x):
        if self.input_type.complex:
            x = torch.view_as_real(x).permute(0, 1, -1, 2, 3).flatten(1, 2)

        if self.has_batch_norm:
            x = self.bn(x)

        if self.avgpool:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        elif self.avg_ker_size > 1:
            x = F.avg_pool2d(x, self.avg_ker_size, stride=1)

        x = x.reshape((x.shape[0], -1))
        output = self.classifier(x)
        return output
