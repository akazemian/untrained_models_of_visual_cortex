import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from model_features.models.learned_scaterring.utils import *


class ComplexConv2d(nn.Module):
    """ Conv2D class which also works for complex input, and can initialize its weight to a unitary operator. """
    def __init__(self, input_type: TensorType, complex_weights, out_channels):
        """
        :param input_type:
        :param complex_weights: whether the weights will be complex or real (None defaults to type of input)
        :param out_channels: int or id/"id", or identity convolution
        :param kernel_size:
        """
        super().__init__()

        self.input_type = input_type
        self.in_channels = self.input_type.num_channels
        self.complex_weights = complex_weights if complex_weights is not None else self.input_type.complex
        self.is_identity = out_channels in [id, "id"]

        num_input_channels = self.in_channels

        self.out_channels = num_input_channels if self.is_identity else out_channels
        self.output_type = TensorType(self.out_channels, self.input_type.spatial_shape,
                                      complex=self.input_type.complex or self.complex_weights)

        if self.is_identity:
            pass
        elif self.out_channels == 0:  # PyTorch doesn't cleanly handle 0-sized tensors...
            self.register_buffer("param", torch.empty((0, num_input_channels, 1, 1)))
        else:
            shape = (out_channels, num_input_channels, 1, 1)
            if self.complex_weights:
                param = unitary_init(shape)
                param = torch.view_as_real(param)
            else:
                param = nn.Conv2d(in_channels=num_input_channels, out_channels=out_channels,
                                  kernel_size=1).weight.data

            self.param = nn.Parameter(param)

    def extra_repr(self) -> str:
        if self.is_identity:
            extra = "is_identity=True"
        else:
            extra = f"out_channels={type_to_str(self.output_type)}, " \
               f"complex_weights={self.complex_weights}"
        return f"in_channels={type_to_str(self.input_type)}, {extra}"

    def forward(self, x):
        if self.is_identity:
            return x
        elif self.out_channels > 0:
            return conv2d(x, self.param, self.output_type.complex)
        else:
            return x.new_empty((x.shape[0], 0) + x.shape[2:])


def conv2d(x, w, complex):
    """ Real or complex convolution between x (B, C, M, N, [2]) and w (K, C, H, W, [2]), handles type problems.
    x and w can be real, complex, or real with an additional trailing dimension of size 2.
    A complex convolution causes the view or cast of both x and w as complex tensors.
    Returns a real or complex tensor of size (B, K, M', N'). """
    def real_to_complex(z):
        if z.is_complex():
            return z
        elif z.ndim == 5:
            # return torch.view_as_complex(z)  # View
            return torch.complex(z[..., 0], z[..., 1])  # Temporary copy instead of view...
        elif z.ndim == 4:
            return z.type(torch.complex64)  # Cast
        else:
            assert False

    if w.shape[0] == 0:  # Stupid special case because pytorch can't handle zero-sized convolutions.
        y = x[:, 0:0]  # (B, 0, M, N), this assumes that x is the right type
        if complex:
            y = real_to_complex(y)

    else:
        if complex:
            x = real_to_complex(x)
            w = real_to_complex(w)
            conv_fn = complex_conv2d
        else:
            conv_fn = F.conv2d
        y = conv_fn(x, w)

    return y


def complex_to_real_channels(x, separate_real_imag=False):
    """ C complex channels to C*2 real channels (or 2*C if separate_real_imag). """
    if separate_real_imag:
        permute = (0, 4, 1, 2, 3)
    else:
        permute = (0, 1, 4, 2, 3)
    return channel_reshape(torch.view_as_real(x).permute(*permute), (-1,))


def real_to_complex_channels(x, separate_real_imag=False):
    """ Inverse of complex_as_real_channels: C*2 real channels (or 2*C if separate_real_imag) to C complex channels. """
    if separate_real_imag:
        channel_shape = (2, -1)
        permute = (0, 2, 3, 4, 1)
    else:
        channel_shape = (-1, 2)
        permute = (0, 1, 3, 4, 2)
    return torch.view_as_complex(channel_reshape(x, channel_shape).permute(*permute).contiguous())


def complex_conv2d(x, w):
    """ (B, C, M, N) complex and (K, C, H, W) complex to (B, K, M', N') complex """
    x = complex_to_real_channels(x, separate_real_imag=True)  # (B, 2C, M, N)
    w = torch.cat([torch.cat([w.real, -w.imag], dim=1), torch.cat([w.imag, w.real], dim=1)], dim=0)  # (2K, 2C, H, W)
    y = F.conv2d(x, w)  # (B, 2K, M', N')
    return real_to_complex_channels(y, separate_real_imag=True)  # (B, K, M', N') complex


def unitary_init(shape):
    N = shape[0]
    C = np.array(shape[1:]).prod()

    rand_matrix = torch.randn(N, C, dtype=torch.complex64)
    svd = torch.linalg.svd(rand_matrix, full_matrices=False)

    if N > C:
        return svd[0].reshape(shape)
    else:
        return svd[2].reshape(shape)
