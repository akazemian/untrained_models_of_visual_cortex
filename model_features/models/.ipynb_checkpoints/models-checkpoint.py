import warnings
warnings.filterwarnings('ignore')
import os
import sys
ROOT = os.getenv('MODELS_ROOT_PATH')
from alexnet import Alexnet
from alexnet_untrained import AlexnetU
from expansion import Expansion5L
from fully_connected import FullyConnected5L
from expansion_fully_random import FullyRandom5L
from ViT import CustomViT


def load_iden(model_name, dataset, block = None, features=None, layers=None, random_filters=None):
    
    if model_name in ['expansion','fully_connected', 'expansion_linear']:
        
        if model_name == 'fully_connected':
                if layers == 3:
                    features = features*9*9
                else:
                    features = features*6*6
                    
        return f'{model_name}_features={features}_layers={layers}_dataset={dataset}'
    
    
        
    elif 'alexnet' in model_name:
            match layers:
                case 1:
                    return f'{model_name}_conv{layers}_layers={layers}_features=64_gpool=False_dataset={dataset}'
                case 2:
                    return f'{model_name}_conv{layers}_layers={layers}_features=192_gpool=False_dataset={dataset}'
                case 3:
                    return f'{model_name}_conv{layers}_layers={layers}_features=384_gpool=False_dataset={dataset}'
                case 4:
                    return f'{model_name}_conv{layers}_layers={layers}_features=256_gpool=False_dataset={dataset}'
                case 5:
                    return f'{model_name}_conv{layers}_layers={layers}_features=256_gpool=False_dataset={dataset}'
                case 'best':
                    return f'{model_name}_gpool=False_dataset={dataset}'
    
    
    elif 'ViT' in model_name:
            return f'{model_name}_features={features}_dataset={dataset}'
            #return f'{model_name}_block={block}_dataset={dataset}'
    
    
    elif 'fully_random' in model_name:
            return f'{model_name}_{random_filters}_features={features}_layers={layers}_dataset={dataset}'
        
    
    else:
            print('model name not known')
            
    
    
def load_full_iden(model_name, feature, layers, random_filter, dataset, 
                   component=None, non_linearity='relu', initializer='kaiming_uniform'):
    
    
    identifier = load_iden(model_name=model_name, features=feature, layers=layers, random_filters=random_filter,
                           dataset=dataset)
    if component is not None:
        identifier += f'_principal_components={component}'
    if non_linearity != 'relu':
        identifier += f'_{non_linearity}'
    if initializer != 'kaiming_uniform':
        identifier += f'_{initializer}'
    
    return identifier


    
def load_model(model_name, block = None, features=None, layers=None, random_filters=None):
    
    match model_name:
        
        case 'expansion':
            return Expansion5L(filters_5=features).Build()
    
    
        case 'expansion_linear':
                return Expansion5L(filters_5=features, non_linearity='none').Build()
    
    
        case 'fully_connected':
                return FullyConnected5L(features_5=features).Build()
        
        case '_alexnet':
            
            match layers:
                
                case 1:
                    return Alexnet(features_layer =2).Build()
                case 2:
                    return Alexnet(features_layer =5).Build()
                case 3:
                    return Alexnet(features_layer =7).Build()
                case 4:
                    return Alexnet(features_layer =9).Build()
                case 5:
                    return Alexnet().Build()
                    

        case 'alexnet_untrained':
            
            match layers:
                
                case 1:
                    return AlexnetU(features_layer =2).Build()
                case 2:
                    return AlexnetU(features_layer =5).Build()
                case 3:
                    return AlexnetU(features_layer =7).Build()
                case 4:
                    return AlexnetU(features_layer =9).Build()
                case 5:
                    return AlexnetU().Build()

    
        case 'ViT':
            return CustomViT(use_wavelets=False, out_features = features, block = 11).Build()
            
    
        case 'fully_random':
            return FullyRandom5L(filters_1=random_filters, filters_5=features).Build()
            
            

    return
