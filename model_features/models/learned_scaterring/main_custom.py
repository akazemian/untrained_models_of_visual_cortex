import torch
import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import os
import shutil
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from model_features.models.learned_scaterring.scattering import Scattering2D, ScatNonLinearity, ScatNonLinearityAndSkip, Realifier, Complexifier, MaxPooool
from model_features.models.learned_scaterring.projection import ComplexConv2d
from model_features.models.learned_scaterring.classifier import Classifier
from model_features.models.learned_scaterring.standardization import Standardization, Normalization

from model_features.models.learned_scaterring.utils import *

def build_layers(input_type: SplitTensorType, modules, psi_modules=[], i: int = 0, num_channels: List[int] = []):
    """ Builds a list of layers from a list of modules.
    :param input_type:
    :param modules: list of strings describing the layers
    :poaram psi_modules: list of strings describing the layers applied after the high-frequency filters
    :param i: index of the current block
    :param num_channels: number of output channels for each block
    :return: Sequential module
    """
    builder = Builder(input_type)

    def branching_kwargs(**submodules):
        """ Builds kwargs for Branching. Expects a dict of module_name -> architecture list of strings. """
        kwargs = {}
        for name, arch in submodules.items():
            kwargs[f"{name}_module_class"] = build_layers
            kwargs[f"{name}_module_kwargs"] = dict(modules=arch, i=i)
        return kwargs

    for module in modules:
        if module == "Fw":
            kwargs = dict(
                scales_per_octave=1, L=8, full_angles=False,
            )
            two_blocks_per_scale_after_block = len(num_channels) % 2
            if i >= two_blocks_per_scale_after_block:
                kwargs.update(factorize_filters=True, i=(i - two_blocks_per_scale_after_block) % 2)

            builder.add_layer(Scattering2D, kwargs)

            kwargs = dict(phi=[], psi=psi_modules)
            builder.add_layer(Branching, branching_kwargs(**kwargs))

        elif module == "R":
            builder.add_batched(Realifier)

        elif module == "C":
            builder.add_batched(Complexifier)

        elif module in ["mod", "rho"]:
            kwargs = dict(
                non_linearity="mod", bias=None, gain=None, learned_params=False,
            )
            if module == "mod":
                builder.add_layer(ScatNonLinearity, kwargs)
            elif module == "rho":
                builder.add_layer(ScatNonLinearityAndSkip, kwargs)
                builder.add_layer(Branching, branching_kwargs(linear=[], non_linear=[]))  # Possibility to add different modules to the linear/non_linear part

        elif module == "Std":
            builder.add_batched(Standardization, dict(remove_mean=True))

        elif module == "MP":
            builder.add_layer(MaxPooool)
        
        elif module in ["P", "Pr", "Pc"]:
            out_channels = {0: num_channels[i]}
            # Determine type of weights (default is type of input).
            complex_weights = dict(P=None, Pr=True, Pc=False)[module]

            kwargs = dict(
                complex_weights=complex_weights, out_channels=out_channels,
            )

            builder.add_diagonal(ComplexConv2d, kwargs)

        elif module == "N":
            builder.add_diagonal(Normalization)
            
        elif module == "id":
            builder.add_layer(Identity)
            
        else:
            assert False

    return builder.module()



def load_model():
    skip = False  # Whether to include a skip-connection over the complex modulus or not
    if skip:
        arch = ["Fw", "rho", "Std", "P", "N"]
        psi_arch = []
    else:
        arch = ["Fw", "Std", "MP", "P"]
        psi_arch = ["mod"]
    
    dataset = "imagenet"

    num_channels_in = 3
    if dataset == "imagenet":
        num_channels = [32, 1024, 10000] 
        spatial_size = 224
        num_classes = 1000
    elif dataset == "cifar":
        num_channels = [64, 128, 256, 512, 512, 512, 512, 512]
        spatial_size = 32
        num_classes = 10
    num_blocks = len(num_channels)

    input_type = TensorType(num_channels=num_channels_in, spatial_shape=(spatial_size, spatial_size), complex=False)
    builder = Builder(input_type)

    builder.add_layer(ToSplitTensor, dict(groups={(0): num_channels_in}))
    for i in range(num_blocks):
        builder.add_layer(build_layers, dict(modules=arch, psi_modules=psi_arch, i=i, num_channels=num_channels))
    builder.add_layer(ToTensor)

#     builder.add_layer(
#         Classifier, dict(
#             nb_classes=num_classes, avg_ker_size=1, avgpool=False, bias=True, batch_norm="affine",
#         ),
#     )

    model = builder.module()

    return model



if __name__ == '__main__':
    model = load_model()

