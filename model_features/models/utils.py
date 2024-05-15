from typing import Optional, Union
from alexnet import Alexnet
from expansion import Expansion5L
from fully_connected import FullyConnected5L
from vit import CustomViT

# Constants
ALEXNET_CONV_LAYERS = {1: 64, 2: 192, 3: 384, 4: 256, 5: 256}
ALEXNET_LAYER_NUMS = {1: 2, 2: 5, 3: 7, 4: 9, 5: 12}
VALID_ALEXNET_LAYERS = [1, 2, 3, 4, 5, 'best']
LAYER_1_RANDOM_FILTERS = 3000
VIT_BLOCK_NUM = 11

def iden_generator(*args) -> str:
    """Generates a model identifier by concatenating arbitrary arguments with underscores."""
    return '_'.join(str(arg) for arg in args if arg!='')

def load_identifier(model_name: str, dataset: str, features: Optional[int] = None, layers: Optional[Union[int, str]] = None) -> str:
    """Generates a model identifier based on the model configuration."""
    if model_name in ['expansion', 'fully_connected', 'expansion_linear', 'fully_random']:
        if model_name == 'fully_connected':
            features *= 36
        return iden_generator(model_name, f'features={features}', f'layers={layers}', f'dataset={dataset}')

    elif 'alexnet' in model_name:
        if layers not in VALID_ALEXNET_LAYERS:
            raise ValueError('Invalid layers argument for AlexNet.')
        if layers == 'best':
            return iden_generator(model_name, f'dataset={dataset}')
        else:
            return iden_generator(model_name, f'conv{layers}_layers={layers}', f'features={ALEXNET_CONV_LAYERS[layers]}', f'dataset={dataset}')

    elif 'vit' in model_name:
        return iden_generator(model_name, f'features={features}', f'dataset={dataset}')

    else:
        raise ValueError('Unknown model name provided.')

def load_full_identifier(model_name: str, features: int, layers: int, dataset: str, 
                         components: Optional[int] = None, non_linearity: str = 'relu', init_type: str = 'kaiming_uniform') -> str:
    """Constructs a full identifier for a model, including optional specifications."""
    identifier = load_identifier(model_name, dataset, features, layers)
    extension = (f'principal_components={components}' if components else '') + \
                      (f'{non_linearity}' if non_linearity != 'relu' else '') + \
                      (f'{init_type}' if init_type != 'kaiming_uniform' else '')
    return iden_generator(identifier, extension)

def load_model(model_name: str, features: Optional[int] = None, layers: Optional[int] = None) -> object:
    """Instantiates and builds a model based on the given model name and specifications."""
    match model_name:
        case 'expansion':
            return Expansion5L(filters_5=features).build()
        case 'expansion_linear':
            return Expansion5L(filters_5=features, non_linearity='none').build()
        case 'fully_connected':
            return FullyConnected5L(features_5=features).build()
        case 'alexnet':
            if layers not in ALEXNET_LAYER_NUMS:
                raise ValueError("Invalid layer number for Alexnet model.")
            return Alexnet(features_layer=ALEXNET_LAYER_NUMS[layers]).build()
        case 'vit':
            return CustomViT(out_features=features, block=VIT_BLOCK_NUM).build()
        case 'fully_random':
            return Expansion5L(filters_1=LAYER_1_RANDOM_FILTERS, filters_5=features).build()
        case _:
            raise ValueError(f"Unknown model name: {model_name}")