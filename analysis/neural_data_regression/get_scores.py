# getting activations for a specific dataset from a specific model. Output is an xarray with dims: features x presentation (stimulus_id)
# from kymatio.torch import Scattering2D
import xarray as xr
import os 
import sys
import torchvision

PATH = '/home/atlask/Desktop/MB_Lab_Project/'
sys.path.append(PATH)


from tools.processing import *
from models.call_model import *
from tools.loading import *
from analysis.neural_data_regression.tools.extractor import Activations
from scipy.io import loadmat
import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
from analysis.neural_data_regression.tools.regression import *
from analysis.neural_data_regression.tools.scorer import *
from models.call_model import EngineeredModel
import torchvision
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression
import random
from sklearn.linear_model import Ridge
from models.all_models.model_2L import EngineeredModel2L
from models.all_models.model_3L import EngineeredModel3L
from models.all_models.model_3L_pca import EngineeredModel3LPCA
from models.all_models.model_4L import EngineeredModel4L
from models.all_models.alexnet_untrained_wide import AlexnetU
from models.all_models.alexnet_untrained_wide_pca import AlexnetUPCA

torch.manual_seed(0)
torch.cuda.manual_seed(0)
untrained_alexnet = torchvision.models.alexnet(pretrained=False)
 
ROOT = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(ROOT,'activations')   


# define constants
DATASET = 'naturalscenes_zscored_processed'
REGIONS = ['V4']

# DATASET = 'majajhong'
# REGIONS = ['V4','IT']

MODE = 'train'
MAX_POOL = False
PCA = False
RANDOM_PROJ = False

ALPHAS = [10**i for i in range(2,5)]

    
MODEL_DICT = {
                      
                      
              'alexnet_u_wide_100000_nsd_pca_256_components':{'model':AlexnetUPCA(n_components=256).Build(),
              'layers': ['last'], 'preprocess':PreprocessRGBLarge},  
    
            #   'model_3L_mp_1000':{'model':EngineeredModel3L(filters_3=1000).Build(),
            #   'layers': ['last'], 'preprocess':PreprocessGSSmall},  
    
              # 'model_3L_mp_20000':{'model':EngineeredModel3L(filters_3=10000).Build(),
              # 'layers': ['last'], 'preprocess':PreprocessGS},  
              
#                'model_3L_PCA_more_pcs':{'model':EngineeredModel3LPCA().Build(),
#              'layers': ['last'], 'preprocess':PreprocessGS},      
    
#               'alexnet_untrained_mp':{'model': untrained_alexnet,
#               'layers': ['features.12'], 'preprocess':PreprocessRGB},
              
            #  'alexnet_mp_GS':{'model': torchvision.models.alexnet(pretrained=True),
            #   'layers': ['features.12'], 'preprocess':PreprocessGSLarge},
                

}
 

for alpha in ALPHAS:
        

    print('alpha',alpha)
    regression_model = Ridge(alpha=alpha)


    for region in REGIONS:
        for model_name, model_info in MODEL_DICT.items():

        
 
            for layer in model_info['layers']:
                print('layer',layer)



                activations_identifier = model_name + '_' + DATASET
                activations = Activations(model=model_info['model'],
                                    layer_names=[layer],
                                    dataset=DATASET,
                                    preprocess=model_info['preprocess'],
                                    max_pool=MAX_POOL,
                                    pca = PCA,
                                    random_proj = RANDOM_PROJ,
                                    mode = MODE
                                    )                   
                activations.get_array(ACTIVATIONS_PATH,activations_identifier)     




                scores_identifier = model_name + '_' + DATASET + '_' + region + '_' + MODE + '_' + f'Ridge(alpha={alpha})' 
                scorer(model_name=model_name,
                       activations_identifier=activations_identifier,
                       scores_identifier=scores_identifier,
                       regression_model=regression_model,
                       dataset=DATASET,
                       mode=MODE,
                       regions=[region]
                      )
