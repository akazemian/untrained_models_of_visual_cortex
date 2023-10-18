
import warnings
warnings.filterwarnings('ignore')
import os
import sys
ROOT = os.getenv('BONNER_ROOT_PATH')
from model_features.models.alexnet import Alexnet
from model_features.models.alexnet_untrained import AlexnetU
from model_features.models.expansion_3_layers import Expansion
from model_features.models.expansion_fully_random import FullyRandom
from model_features.models.fully_connected import FullyConnected
from model_features.models.fully_connected_3_layers import FullyConnected3L
import torchvision


def load_model_dict(name, gpool=True):
    
    match name:

        case 'expansion_10':
            return {
                'iden':'expansion_model',
                'model':Expansion(filters_3=10, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':10,
                'hook':None,
                'gpool':gpool
            }


        case 'expansion_100': 
            return {
                'iden':'expansion_model',
                'model':Expansion(filters_3=100, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':100,
                'hook':None,
                'gpool':gpool
            }

        case 'expansion_1000': 
            return {
                'iden':'expansion_model',
                'model':Expansion(filters_3=1000, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':1000,
                'hook':None,
                'gpool':gpool
            }

        case 'expansion_10000': 
            return {
                'iden':'expansion_model',
                'model':Expansion(filters_3=10000, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':10000,
                'hook':None,
                'gpool':gpool
                }


        case 'expansion_first_256_pcs': 
            return {
                'iden':'expansion_model',
                'model':Expansion(filters_3=10000, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':10000,
                'hook':'pca',
                'gpool':gpool
                }
        
        case 'expansion_linear': 
            return {
                'iden':'expansion_model_linear',
                'model':Expansion(filters_3=10000, non_linearity='none', gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':10000,
                'hook':None,
                'gpool':gpool
                }

        case 'fully_random': 
            return {
                'iden':'expansion_model_fully_random',
                'model':FullyRandom().Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':10000,
                'hook':None,
                'gpool':gpool
                }



        case 'fully_connected_10': 
            return {
                'iden':'fully_connected',
                'model':FullyConnected(features=10).Build(),
                'layers': ['last'], 
                'num_layers':1,
                'num_features':10,
                'hook':None,
                'gpool':gpool
                }
        
        case 'fully_connected_100': 
            return {
                'iden':'fully_connected',
                'model':FullyConnected(features=100).Build(),
                'layers': ['last'], 
                'num_layers':1,
                'num_features':100,
                'hook':None,
                'gpool':gpool
                }
        
        case 'fully_connected_1000': 
            return {
                'iden':'fully_connected',
                'model':FullyConnected(features=1000).Build(),
                'layers': ['last'], 
                'num_layers':1,
                'num_features':1000,
                'hook':None,
                'gpool':gpool
                }
        
        case 'fully_connected_10000': 
            return {
                'iden':'fully_connected',
                'model':FullyConnected().Build(),
                'layers': ['last'], 
                'num_layers':1,
                'num_features':10000,
                'hook':None,
                'gpool':gpool
                }


        case 'fully_connected_3_layers_10': 
            return {
                'iden':'fully_connected_3_layers',
                'model':FullyConnected3L(features_3 = 10).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':10,
                'hook':None,
                'gpool':gpool
                }

        case 'fully_connected_3_layers_100': 
            return {
                'iden':'fully_connected_3_layers',
                'model':FullyConnected3L(features_3 = 100).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':100,
                'hook':None,
                'gpool':gpool
                }        
        
        case 'fully_connected_3_layers_1000': 
            return {
                'iden':'fully_connected_3_layers',
                'model':FullyConnected3L(features_3 = 1000).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':1000,
                'hook':None,
                'gpool':gpool
                } 
        
        case 'fully_connected_3_layers_10000': 
            return {
                'iden':'fully_connected_3_layers',
                'model':FullyConnected3L().Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':10000,
                'hook':None,
                'gpool':gpool
                }

        case 'alexnet_conv1':
            return {
                'iden':'alexnet_conv1',
                'model':Alexnet(features_layer =2, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':1,
                'num_features':64,
                'hook':None,
                'gpool':gpool
            }            


        case 'alexnet_conv2':
            return {
               'iden':'alexnet_conv2',
                'model':Alexnet(features_layer =5, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':2,
                'num_features':192,
                'hook':None,
                'gpool':gpool
            }   


        case 'alexnet_conv3':
            return {
                'iden':'alexnet_conv3',
                'model':Alexnet(features_layer =7, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':384,
                'hook':None,
                'gpool':gpool
            }  


        case 'alexnet_conv4':
            return {
                'iden':'alexnet_conv4',
                'model':Alexnet(features_layer =9, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':4,
                'num_features':256,
                'hook':None,
                'gpool':gpool
            }   


        case 'alexnet_conv5':
            return {
                'iden':'alexnet_conv5',
                'model':Alexnet(gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':5,
                'num_features':256,
                'hook':None,
                'gpool':gpool
            } 


        case 'alexnet_untrained_conv1':
            return {
                'iden':'alexnet_untrained_conv1',
                'model':AlexnetU(features_layer =2, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':1,
                'num_features':64,
                'hook':None,                
                'gpool':gpool
            }            


        case 'alexnet_untrained_conv2':
            return {
               'iden':'alexnet_untrained_conv2',
                'model':AlexnetU(features_layer =5, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':2,
                'num_features':192,
                'hook':None,
                'gpool':gpool
            }   


        case 'alexnet_untrained_conv3':
            return {
                'iden':'alexnet_untrained_conv3',
                'model':AlexnetU(features_layer =7, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':384,
                'hook':None,
                'gpool':gpool
            }  


        case 'alexnet_untrained_conv4':
            return {
                'iden':'alexnet_untrained_conv4',
                'model':AlexnetU(features_layer =9, gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':4,
                'num_features':256,
                'hook':None,
                'gpool':gpool
            }   


        case 'alexnet_untrained_conv5':
            return {
                'iden':'alexnet_untrained_conv5',
                'model':AlexnetU(gpool = gpool).Build(),
                'layers': ['last'], 
                'num_layers':5,
                'num_features':256,
                'hook':None,
                'gpool':gpool
            } 