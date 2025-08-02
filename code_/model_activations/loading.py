import os
from typing import Union
from config import PREDS_PATH
from pathlib import Path

# Constants
# ALEXNET_CONV_LAYERS = {1: 64, 2: 192, 3: 384, 4: 256, 5: 256}
# RESNET_CONV_LAYERS = {1: 256, 2: 512, 3: 1024, 4: 2048}

# VALID_ALEXNET_LAYERS = [1, 2, 3, 4, 5, 'best']
ALEXNET_LAYER_NUMS = {1: 2, 2: 5, 3: 7, 4: 9, 5: 12}
LAYER_1_RANDOM_FILTERS = 3000
VIT_BLOCK_NUM = 11

def iden_generator(*args) -> str:
    """Generates a model identifier by concatenating arbitrary arguments with underscores."""
    return '_'.join(str(arg) for arg in args if arg!='')



def load_identifier(model_name: str, dataset: str, features: int = None, layers: Union[int, str] = None) -> str:
    """Generates a model identifier based on the model configuration."""
    if layers == 'best':
        return iden_generator(model_name, f'dataset={dataset}')
    else:
        return iden_generator(model_name, f'features={features}', f'layers={layers}', f'dataset={dataset}')



def load_full_identifier(model_name: str, dataset: str, layers: int, features: int = None,
                         principal_components: int = False, non_linearity: str = 'relu', init_type: str = 'kaiming_uniform') -> str:
    """Constructs a full identifier for a model, including optional specifications."""
    identifier = load_identifier(model_name, dataset, features, layers)
    extension = (f'principal_components={principal_components}' if principal_components != False else '') + \
                      (f'{non_linearity}' if non_linearity != 'relu' else '') + \
                      (f'{init_type}' if init_type != 'kaiming_uniform' else '')
    return iden_generator(identifier, extension)




def get_best_layer_path(best_layer, dataset, region, subject):
    if 'alexnet' in best_layer:
        if 'untrained' in best_layer:
            file_path = Path(PREDS_PATH) / f'alexnet_untrained_dataset={dataset}_{region}_{subject}.pkl'
        else:
            file_path = Path(PREDS_PATH) / f'alexnet_trained_dataset={dataset}_{region}_{subject}.pkl'
    elif 'vit' in best_layer:
        if 'untrained' in best_layer:
            file_path = Path(PREDS_PATH) / f'vit_untrained_dataset={dataset}_{region}_{subject}.pkl'
        else:
            file_path = Path(PREDS_PATH) / f'vit_trained_dataset={dataset}_{region}_{subject}.pkl'            
    elif 'resnet50' in best_layer:
        if 'untrained' in best_layer:
            file_path = Path(PREDS_PATH) / f'resnet50_untrained_dataset={dataset}_{region}_{subject}.pkl'
        else:
            file_path = Path(PREDS_PATH) / f'resnet50_trained_dataset={dataset}_{region}_{subject}.pkl'  
    else:
        print('unidentified model')
        return
    return file_path



def load_model(model_name: str, features: int = None, layers: int = None, device='cuda') -> object:
    """Instantiates and builds a model based on the given model name and specifications."""
    match model_name:
        case 'expansion':
            from code_.model_activations.models.expansion import Expansion5L
            return Expansion5L(filters_5=features,device=device).build()
        
        case 'expansion_linear':
            from code_.model_activations.models.expansion import Expansion5L
            return Expansion5L(filters_5=features, non_linearity='none',device=device).build()
        
        case 'fully_connected':
            from code_.model_activations.models.fully_connected import FullyConnected5L
            return FullyConnected5L(features_5=features,device=device).build()
        
        case 'resnet50_trained':
            from code_.model_activations.models.resnet50_barlowtwins import ResNet50
            return ResNet50(features_layer=layers,device=device).build()
        
        case 'alexnet_trained':
            from code_.model_activations.models.alexnet import AlexNet
            if layers not in ALEXNET_LAYER_NUMS:
                raise ValueError("Invalid layer number for Alexnet model.")
            return AlexNet(features_layer=ALEXNET_LAYER_NUMS[layers],device=device).build()
            
        case 'alexnet_untrained':
            from code_.model_activations.models.alexnet_untrained import AlexNetU
            if layers not in ALEXNET_LAYER_NUMS:
                raise ValueError("Invalid layer number for Alexnet model.")
            return AlexNetU(features_layer=ALEXNET_LAYER_NUMS[layers],device=device).build()
        
        case 'vit':
            from code_.model_activations.models.vit import CustomViT
            return CustomViT(out_features=features, block=VIT_BLOCK_NUM,device=device).build()
        
        case 'fully_random':
            from code_.model_activations.models.expansion import Expansion5L
            return Expansion5L(filters_1=LAYER_1_RANDOM_FILTERS, filters_5=features,device=device).build()
        
        case 'vit_trained':
            from code_.model_activations.models.vit_trained import ViT
            return ViT(block=layers, device=device).build()
        
        case 'vit_untrained':
            from code_.model_activations.models.vit_untrained import ViTU
            return ViTU(block=layers, device=device).build()
        
        case _:
            raise ValueError(f"Unknown model name: {model_name}")
