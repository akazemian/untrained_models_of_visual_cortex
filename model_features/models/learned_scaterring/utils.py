import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import *


def kwargs_to_str(kwargs):
    """ Returns a string of the form '(kw1=val1, kw2=val2)'. """
    if len(kwargs) == 0:
        return ""
    else:
        return "(" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")"


def ceil_div(a: int, b: int) -> int:
    """ Return ceil(a / b). """
    return a // b + (a % b > 0)


def to_tuple(x):
    if hasattr(x, "__iter__"):
        return tuple(x)
    else:
        return (x,)


def shapes(x, names=None):
    """ Traverse x as nested lists/dictionaries of arrays/tensors and return the shapes of those.
    If names is provided, it should have the same structure as x and give the names of axes of tensors.
    """
    if isinstance(x, list):
        if names is None:
            return [shapes(v) for v in x]
        else:
            assert isinstance(names, list) and len(x) == len(names)
            return [shapes(x[i], names[i]) for i in range(len(x))]
    elif isinstance(x, dict):
        if names is None:
            return {k: shapes(v) for k, v in x.items()}
        else:
            assert isinstance(names, dict) and x.keys() == names.keys()
            return {k: shapes(x[k], names[k]) for k in x.keys()}
    elif x is None:
        assert names is None
        return None
    else:
        if names is None:
            return tuple(x.shape)
        else:
            assert len(names) == len(x.shape)
            return ", ".join(f"{names[i]}: {x.shape[i]}" for i in range(len(x.shape)))


def transpose_view(x, axes, names=None):
    """ Performs a general transpose of x. Returns transposed tensor and new names.
    :param x: numpy array to transpose
    :param names: names of each axis, in the order of x
    :param transpose: iterable of integers corresponding to the new order of axes.
    None correspond to new unsqueezed axes, and axes not present are squeezed (should have a size of 1).
    :return: the transposed array, and names of axes in the new order (if provided)
    """
    # First transpose: put not present at the top, remove Nones.
    not_present = tuple(i for i in range(x.ndim) if i not in axes)
    assert all(x.shape[i] == 1 for i in not_present)
    transpose = not_present + tuple(i for i in axes if i is not None)
    # Then view: indexing and unsqueezes.
    view = (0,) * len(not_present) + tuple(None if i is None else slice(None) for i in axes)

    x = x.transpose(transpose)[view]

    if names is None:
        return x
    else:
        # Handle names: new axes are replaced by dots.
        names = [r"\cdot" if axes[i] is None else names[axes[i]] for i in range(len(axes))]
        return x, names


def channel_reshape(x, channel_shape):
    """ (B, *, H, W) to (B, custom, H, W) """
    return x.reshape((x.shape[0],) + channel_shape + x.shape[-2:])


def optimized_cat(tensors, dim):
    """ Avoids creating a new tensor for lists of length 1. """
    if len(tensors) > 1:
        return torch.cat(tensors, dim)
    else:
        return tensors[0]


class ModuleDict(nn.Module):
    """ Better ModuleDict than Pytorch which supports non-string keys. """
    def __init__(self, modules):
        super().__init__()
        self.dict = nn.ModuleDict()
        for key, module in modules.items():
            self[key] = module

    def __getitem__(self, item):
        return self.dict[str(item)]

    def __setitem__(self, key, value):
        self.dict[str(key)] = value

    def __repr__(self):
        return self.dict.__repr__()

    def __iter__(self):
        return self.dict.__iter__()

    def __len__(self):
        return self.dict.__len__()

    def keys(self):
        return self.dict.keys()

    def values(self):
        return self.dict.values()

    def items(self):
        return self.dict.items()


class Indexer:
    """ Dummy class used to construct slices with the : syntax. """
    def __getitem__(self, item):
        return item


idx = Indexer()


"""
Here is how we handle separation across orders and the like.

We distinguish between "standard" tensors (torch.Tensor, whose shape is described by a TensorType)
and "split" tensors (SplitTensor, whose shapes is described by a SplitTensorType).
A split tensor is a tensor whose channels are split in different groups. Each group is identified by a key.

Now a module may take as input either a Tensor or a SplitTensor. Its __init__ method will take as an argument a
TensorType or SplitTensorType describing the shape of its input. It will then indicate its output by storing as
attribute the corresponding TensorType or SplitTensorType.
"""


class TensorType:
    """ Type of a 2D tensor. """
    def __init__(self, num_channels, spatial_shape, complex):
        self.num_channels = num_channels
        self.spatial_shape = spatial_shape
        self.complex = complex
        self.dtype = torch.complex64 if self.complex else torch.float32

    def __repr__(self):
        return f"TensorType(num_channels={self.num_channels}, spatial_shape={self.spatial_shape}, complex={self.complex})"


def complex_to_str(complex):
    return "C" if complex else "R"


def type_to_str(type: TensorType):
    return f"{type.num_channels}{complex_to_str(type.complex)}"


Tensor = torch.Tensor


class ChannelSlicer(nn.Module):
    """ Simple module which extracts a slice of channels. """
    def __init__(self, input_type: TensorType, channel_slice):
        super().__init__()

        self.input_type = input_type

        # Normalize the input slice: no more None, and no negative numbers.
        def normalize(i, default, pos):
            if i is None:
                i = default
            if pos and i < 0:
                i += self.input_type.num_channels
            return i

        start = normalize(channel_slice.start, 0, pos=True)
        stop = normalize(channel_slice.stop, self.input_type.num_channels, pos=True)
        step = normalize(channel_slice.step, 1, pos=False)
        assert step > 0  # Negative step not implemented.
        self.slice = slice(start, stop, step)

        output_channels = ceil_div(self.slice.stop - self.slice.start, self.slice.step)
        self.output_type = TensorType(output_channels, self.input_type.spatial_shape, self.input_type.complex)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, channel_slice={self.slice}"

    def forward(self, x: Tensor) -> Tensor:
        return x[:, self.slice]

    def equivalent_proj(self, device):
        full = torch.zeros(self.out_channels, self.in_channels, device=device)
        idx = lambda *args: torch.arange(*args, dtype=torch.int64, device=device)
        full[idx(self.out_channels), idx(self.slice.start, self.slice.stop, self.slice.step)] = 1
        return full


class Sequential(nn.Module):
    """ Sequential module which specifies its output type. """
    def __init__(self, layers):
        super().__init__()
        self.module = nn.Sequential(*layers)
        self.output_type = layers[-1].output_type

    def __repr__(self):
        return self.module.__repr__()

    def __getitem__(self, item):
        return self.module[item]

    def forward(self, x):
        return self.module(x)


class SplitTensorType:
    """ Type of a 2D split tensor. """
    def __init__(self, groups, spatial_shape, complex):
        self.groups = groups
        self.keys = list(sorted(self.groups.keys()))
        self.num_channels = sum(self.groups.values())  # Total number of input channels
        self.spatial_shape = spatial_shape
        self.complex = complex
        self.dtype = torch.complex64 if self.complex else torch.float32

    def tensor_type(self):
        """ Cast to TensorType, forgetting about group information. """
        return TensorType(self.num_channels, self.spatial_shape, self.complex)

    def __repr__(self):
        return f"SplitTensorType(groups={self.groups}, spatial_shape={self.spatial_shape}, complex={self.complex})"


class SplitTensor:
    """ Tensor whose channels are naturally split in groups.
    Allows optimized viewing as a full, split diagonally or split triangularly tensor. """
    def __init__(self, x, groups=None):
        """
        :param x: full tensor (in which case groups is specified), or ordered dict group_key -> tensor
        :param groups: ordered dict, group_key -> number of channels in the group
        """
        if groups is not None:
            assert isinstance(x, Tensor)
            if sum(groups.values()) != x.shape[1]:
                raise ValueError(f"Got groups {groups} for tensor of shape {x.shape}")
            self.full = x
            self.split = None
            self.num_channels = groups
        else:
            self.full = None
            self.split = x
            self.num_channels = {key: x_key.shape[1] for key, x_key in self.split.items()}

        self.keys = list(self.num_channels.keys())

        self.start = {}
        self.end = {}
        num_channels = 0
        for key in self.keys:
            prev = num_channels
            num_channels += self.num_channels[key]
            self.start[key] = prev
            self.end[key] = num_channels

    def full_view(self) -> torch.Tensor:
        if self.full is None:
            self.full = optimized_cat([self.split[key] for key in self.keys], dim=1)
        return self.full

    def diag_view(self) -> Dict[Any, torch.Tensor]:
        if self.split is None:
            self.split = {key: self.full[:, self.start[key]:self.end[key]] for key in self.keys}
        return self.split

    def triang_view(self) -> Dict[Any, torch.Tensor]:
        x = self.full_view()
        return {key: x[:, :self.end[key]] for key in self.keys}


class Identity(nn.Module):
    """ Identity module which specifies its output type. """
    def __init__(self, input_type: Union[TensorType, SplitTensorType]):
        super().__init__()
        self.input_type = input_type
        self.output_type = self.input_type

    def forward(self, x):
        return x


class ToSplitTensor(nn.Module):
    """ Splits an input Tensor into a SplitTensor.
    Also works with dictionaries, applies the same groups to each tensor. """
    def __init__(self, input_type: Union[TensorType, Dict[str, TensorType]], groups):
        super().__init__()
        self.input_type = input_type
        if isinstance(self.input_type, dict):
            self.output_type = {key: SplitTensorType(groups, input_type.spatial_shape, input_type.complex)
                                for key, input_type in self.input_type.items()}
        else:
            self.output_type = SplitTensorType(groups, self.input_type.spatial_shape, self.input_type.complex)

    def forward(self, x: Union[Tensor, Dict[str, Tensor]]) -> Union[SplitTensor, Dict[str, SplitTensor]]:
        if isinstance(x, dict):
            return {key: SplitTensor(tensor, groups=self.output_type[key].groups) for key, tensor in x.items()}
        else:
            return SplitTensor(x, groups=self.output_type.groups)


class BatchedModule(nn.Module):
    """ Interfaces a module which treats independently each channel to work with split tensors.
    More precisely, the module is a function from (B, C, H, W) to (B, CC', H, W) or
    (B, CC', H, W) to (B, C, H, W) (or a dict of those). """
    def __init__(self, input_type: SplitTensorType, module_class, module_kwargs=None):
        # Could be implemented with ToTensor and ToSplitTensor...
        super().__init__()

        self.input_type = input_type

        self.to_tensor = ToTensor(self.input_type)
        total_input_channels = self.to_tensor.output_type.num_channels

        if module_kwargs is None:
            module_kwargs = {}
        self.module = module_class(self.to_tensor.output_type, **module_kwargs)
        cat_output_type = self.module.output_type

        def get_to_split_tensor(cat_output_subtype):
            """ Computes appropriate dimension factor and returns corresponding ToSplitTensor. """
            if cat_output_subtype.num_channels >= total_input_channels:
                # C to (C, C')
                assert cat_output_subtype.num_channels % total_input_channels == 0
                dimension_factor = cat_output_subtype.num_channels // total_input_channels
                groups = {key: num_channels * dimension_factor for key, num_channels in self.input_type.groups.items()}
            else:
                # (C, C') to C
                dimension_factor = total_input_channels // cat_output_subtype.num_channels
                assert all(num_channels % dimension_factor == 0 for num_channels in self.input_type.groups.values())
                groups = {key: num_channels // dimension_factor for key, num_channels in self.input_type.groups.items()}
            return ToSplitTensor(cat_output_subtype, groups)

        to_split_tensors = {}
        if isinstance(cat_output_type, dict):
            for sub_key, cat_output_subtype in cat_output_type.items():
                to_split_tensors[sub_key] = get_to_split_tensor(cat_output_subtype)

            self.to_split_tensor = ModuleDict(to_split_tensors)
            self.output_type = {key: self.to_split_tensor[key].output_type for key in self.to_split_tensor.keys()}
        else:
            self.to_split_tensor = get_to_split_tensor(cat_output_type)
            self.output_type = self.to_split_tensor.output_type

    def __repr__(self):
        return f"BatchedModule({self.module})"

    def forward(self, x: SplitTensor) -> Union[SplitTensor, Dict[str, SplitTensor]]:
        y = self.module(self.to_tensor(x))
        if isinstance(y, dict):
            return {key: self.to_split_tensor[key](y[key]) for key in y}
        else:
            return self.to_split_tensor(y)


class DiagonalModule(nn.Module):
    """ Applies a module independently to each group. """
    def __init__(self, input_type: SplitTensorType, module_class, module_kwargs=None):
        """
        :param input_type: type of the split tensor input
        :param module_class: class of the submodules, which have standard tensors as input and output
        :param module_kwargs: arguments to pass to the module_class, in addition to the input type description
        One can pass a list or a dictionary to set per-module arguments.
        """
        super().__init__()

        self.input_type = input_type
        self.keys = self.input_type.keys

        if module_kwargs is None:
            module_kwargs = {}

        def get_module_kwargs(i, key):
            def handle_value(value):
                if isinstance(value, list):
                    return value[i]
                elif isinstance(value, dict):
                    return value[key]
                else:
                    return value

            return {name: handle_value(value) for name, value in module_kwargs.items()}

        self.submodules = ModuleDict({key: module_class(
            input_type=TensorType(self.input_type.groups[key], self.input_type.spatial_shape, self.input_type.complex),
            **get_module_kwargs(i, key)) for i, key in enumerate(self.keys)})

        one_output_type = self.submodules[self.keys[0]].output_type
        for key in self.keys:
            output_type = self.submodules[key].output_type
            assert output_type.spatial_shape == one_output_type.spatial_shape and \
                   output_type.complex == one_output_type.complex
        self.output_type = SplitTensorType(
            groups={key: self.submodules[key].output_type.num_channels for key in self.keys},
            spatial_shape=one_output_type.spatial_shape, complex=one_output_type.complex,
        )

    def __repr__(self):
        return f"DiagonalModule({self.submodules})"

    def forward(self, x: SplitTensor) -> SplitTensor:
        """ Applies each submodule to its corresponding group. """
        x_diag = x.diag_view()
        return SplitTensor({key: self.submodules[key](x_diag[key]) for key in self.keys})


class Merger(nn.Module):
    """ Merge different groups together. """
    def __init__(self, input_type: Dict[str, SplitTensorType]):
        super().__init__()

        self.input_type = input_type
        self.keys = list(sorted(set().union(*(set(input_desc.keys) for input_desc in self.input_type.values()))))
        desc_0 = list(self.input_type.values())[0]
        assert all(desc.spatial_shape == desc_0.spatial_shape for desc in self.input_type.values())
        complex = any(desc.complex for desc in self.input_type.values())

        output_channels = {key: 0 for key in self.keys}
        for input_desc in self.input_type.values():
            for key, num_channels in input_desc.groups.items():
                output_channels[key] += num_channels
        self.output_type = SplitTensorType(output_channels, desc_0.spatial_shape, complex)

    def forward(self, tensors: Dict[str, SplitTensor]) -> SplitTensor:
        y = {key: [] for key in self.keys}
        for tensor in tensors.values():
            for key, x in tensor.diag_view().items():
                y[key].append(x)

        # Build the output as a full tensor, avoids concatenation further down the line.
        y = optimized_cat(sum(y.values(), start=[]), dim=1)
        return SplitTensor(y, groups=self.output_type.groups)


class Branching(nn.Module):
    """ Implements a branching path. Expects a dict of split tensors, applies a submodule to each one
    and merges the result. """
    def __init__(self, input_type: Dict[str, SplitTensorType], **kwargs):
        """
        :param input_type: dictionary of SplitTensorTypes, for each branch of the path
        :param kwargs: of the form `key`_module_class and `key`_module_kwargs
        """
        super().__init__()

        self.input_type = input_type

        submodules = {}
        for key, input_type in self.input_type.items():
            module_class = kwargs.pop(f"{key}_module_class", Identity)
            module_kwargs = kwargs.pop(f"{key}_module_kwargs", {})
            submodules[key] = module_class(input_type, **module_kwargs)
        self.submodules = ModuleDict(submodules)
        assert len(kwargs) == 0

        self.merger = Merger({key: self.submodules[key].output_type for key in self.submodules.keys()})
        self.output_type = self.merger.output_type

    def __repr__(self):
        repr = f"Branching("
        for key, submodule in self.submodules.items():
            if not isinstance(submodule, Identity):
                repr = f"{repr}\n  ({key}): {submodule}"
        if '\n' in repr:
            repr = f"{repr}\n"
        return f"{repr})"

    def forward(self, tensors: Dict[str, SplitTensor]) -> SplitTensor:
        return self.merger({key: self.submodules[key](tensor) for key, tensor in tensors.items()})


class ToTensor(nn.Module):
    def __init__(self, input_type: SplitTensorType):
        super().__init__()
        self.input_type = input_type
        self.output_type = TensorType(sum(self.input_type.groups.values()),
                                      self.input_type.spatial_shape, self.input_type.complex)

    def forward(self, x: SplitTensor) -> Tensor:
        return x.full_view()


class Builder:
    """ Class for building a sequential module. """
    def __init__(self, input_type):
        self.input_type = input_type
        self.layers = []

    def add_layer(self, module_class, module_kwargs=None):
        if module_kwargs is None:
            module_kwargs = {}
        layer = module_class(self.input_type, **module_kwargs)
        self.input_type = layer.output_type
        self.layers.append(layer)
        return layer

    def add_batched(self, module_class, module_kwargs=None):
        return self.add_layer(BatchedModule, dict(module_class=module_class, module_kwargs=module_kwargs))

    def add_diagonal(self, module_class, module_kwargs=None):
        return self.add_layer(DiagonalModule, dict(module_class=module_class, module_kwargs=module_kwargs))

    def module(self):
        if len(self.layers) == 0:
            return Identity(self.input_type)
        elif len(self.layers) == 1:
            return self.layers[0]
        else:
            return Sequential(self.layers)


def dummy_input_tensor(input_type: Union[TensorType, SplitTensorType]) -> Union[Tensor, SplitTensor]:
    """ Returns a dummy CPU Tensor or SplitTensor with the correct shape and dtype. """
    # Dummy can't be zero-sized because some ops (notably fft) do not work with zero-sized tensors...
    x = torch.empty((1, input_type.num_channels) + input_type.spatial_shape,
                    dtype=input_type.dtype)
    if isinstance(input_type, SplitTensorType):
        return SplitTensor(x, input_type.groups)
    else:
        return x


def infer_type(y: Union[Tensor, SplitTensor, Dict[str, Tensor], Dict[str, SplitTensor]]) \
        -> Union[TensorType, SplitTensorType, Dict[str, TensorType], Dict[str, SplitTensorType]]:
    """ Returns the type of the input tensor, split tensor or dict of those. """
    if isinstance(y, dict):
        return {key: infer_type(group) for key, group in y.items()}
    elif isinstance(y, SplitTensor):
        output_type = infer_type(y.full_view())
        return SplitTensorType(groups=y.num_channels, spatial_shape=output_type.spatial_shape,
                               complex=output_type.complex)
    elif isinstance(y, Tensor):
        return TensorType(num_channels=y.shape[1], spatial_shape=y.shape[2:], complex=torch.is_complex(y))


def infer_output_type(module, input_type: Union[TensorType, SplitTensorType]):
    """ Infers the output type of the given module. """
    x = dummy_input_tensor(input_type)
    y = module(x)
    return infer_type(y)
