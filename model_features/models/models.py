import warnings
warnings.filterwarnings('ignore')
import os
import sys
ROOT = os.getenv('BONNER_ROOT_PATH')
from model_features.models.alexnet import Alexnet
from model_features.models.alexnet_untrained import AlexnetU
from model_features.models.expansion import Expansion3L, Expansion5L
from model_features.models.fully_connected import FullyConnected3L, FullyConnected5L
from model_features.models.expansion_linear import Expansion3LLinear, Expansion5LLinear


def load_iden(model_name, dataset, features=None, layers=None):
    
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
    
    else:
            print('model name not known')
            
    
    
    
    
def load_model(model_name, features=None, layers=None):
    
    match model_name:
        
        case 'expansion':
            match layers:
                case 3:
                    return Expansion3L(filters_3=features).Build()
                case 5:
                    return Expansion5L(filters_5=features).Build()
    
    
        case 'expansion_linear':
            match layers:
                case 3:
                    return Expansion3LLinear(filters_3=features).Build()
                case 5:
                    return Expansion5LLinear(filters_5=features).Build()
    
    
        case 'fully_connected':
            match layers:
                case 3:
                    return FullyConnected3L(features_3=features).Build()
                case 5:
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
    
    
    
