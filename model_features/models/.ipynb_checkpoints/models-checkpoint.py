import warnings
warnings.filterwarnings('ignore')
import os
import sys
sys.path.append(os.getenv('MODELS_ROOT_PATH'))
from alexnet import Alexnet
# from alexnet_untrained import AlexnetU
from expansion import Expansion5L
from fully_connected import FullyConnected5L
from vit import CustomViT


def load_iden(model_name, dataset, block = None, features=None, layers=None):
    
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
    
    
    elif 'vit' in model_name:
            return f'{model_name}_features={features}_dataset={dataset}'
    
    
    elif 'fully_random' in model_name:
            return f'{model_name}_3000_features={features}_layers={layers}_dataset={dataset}'
        
    
    else:
            print('model name not known')
            
    
    
def load_full_iden(model_name, features, layers, dataset, 
                   components=None, non_linearity='relu', init_type='kaiming_uniform'):
    
    
    identifier = load_iden(model_name=model_name, features=features, layers=layers, 
                           dataset=dataset)
    if components is not None:
        identifier += f'_principal_components={components}'
    if non_linearity != 'relu':
        identifier += f'_{non_linearity}'
    if init_type != 'kaiming_uniform':
        identifier += f'_{init_type}'
    
    return identifier


    
def load_model(model_name, block = None, features=None, layers=None, random_filters=None):
    
    match model_name:
        
        case 'expansion':
            return Expansion5L(filters_5=features).Build()
    
    
        case 'expansion_linear':
                return Expansion5L(filters_5=features, non_linearity='none').Build()
    
    
        case 'fully_connected':
                return FullyConnected5L(features_5=features).Build()
        
        case 'alexnet':
            
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

    
        case 'vit':
            return CustomViT(out_features = features, block = 11).Build()
            
    
        case 'fully_random':
            return Expansion5L(filters_1=3000,filters_5=features).Build()
            
            

    return
