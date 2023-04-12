# getting activations for a specific dataset from a specific model. Output is an xarray with dims: features x presentation (stimulus_id)
# from kymatio.torch import Scattering2D
import xarray as xr
import os 
import sys
import torchvision

PATH = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(PATH)
# from models.kymatio import *

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
from models.all_models.model_3L_get_pcs import EngineeredModel3LPCs

from models.all_models.alexnet_untrained_wide_1 import AlexnetU1
import pickle   
    
    
# define paths
PATH_TO_PCA = '/data/atlas/pca'
ACTIVATIONS_PATH = '/data/atlas/activations'
      
    
DATASET = 'naturalscenes_zscored_processed'    
MAX_POOL = True

N_COMPONENTS_L2 = 500
N_COMPONENTS_L3 = 1000

# models    
MODEL_DICT = {
    
              
              f'model_2L_mp_5000_nsd_pca_{N_COMPONENTS_L2}_components':{'model':EngineeredModel2L(filters_2=5000).Build(),
              'layers': ['last'], 'preprocess':PreprocessGS,'n_components':N_COMPONENTS_L2},
    
              # f'model_3L_mp_10000_nsd_pca_{N_COMPONENTS_L3}_components':{'model':EngineeredModel3L(filters_2=5000,filters_3=10000).Build(),
              # 'layers': ['last'], 'preprocess':PreprocessGS,'n_components':N_COMPONENTS_L3},  
    
              f'model_3L_mp_10000_nsd_pca_{N_COMPONENTS_L3}_components':{'model':EngineeredModel3LPCs(filters_2=5000,filters_3=10000).Build(),
              'layers': ['last'], 'preprocess':PreprocessGS,'n_components':N_COMPONENTS_L3}, 
    
    #               'model_3L_mp_50000_nsd_pca':{'model':EngineeredModel3L(filters_3=10000,batches_3 = 5).Build(),
              # 'layers': ['last'], 'preprocess':PreprocessGS},
    
    
#               'alexnet_mp_nsd_pca':{'model': torchvision.models.alexnet(pretrained=True),
#               'layers': ['features.12'], 'preprocess':PreprocessRGB},
              
              
#               'alexnet_untrained_wide_10000_nsd_pca':{'model':AlexnetU1(filters_5 = 10000).Build(),
#               'layers': ['mp5'], 'preprocess':PreprocessRGB},
              
}





        
        

for model_name, model_info in MODEL_DICT.items():
       
    
    print(model_name)
    activations_identifier = model_name + '_' + DATASET
        
    if os.path.exists(os.path.join(os.path.join(PATH_TO_PCA,model_name))):
        print(f'pcs are already saved in {PATH_TO_PCA} as {model_name}')
        
    else:

        print('obtaining pcs...')
              
        for layer in model_info['layers']:


            activations = Activations(model = model_info['model'],
                                layer_names = [layer],
                                dataset = DATASET,
                                preprocess = model_info['preprocess'],
                                mode = 'pca',
                                max_pool = MAX_POOL,
                                pca = False,
                                random_proj=False
                                )                   
            activations.get_array(ACTIVATIONS_PATH,activations_identifier)     


            X = xr.open_dataset(os.path.join(ACTIVATIONS_PATH,activations_identifier)).x.values
            pca = PCA(n_components = model_info['n_components'])
            pca.fit(X)


            file = open(os.path.join(PATH_TO_PCA,model_name), 'wb')
            pickle.dump(pca, file, protocol=4)
            file.close()
            print(f'pcs are now saved in {PATH_TO_PCA} as {model_name}')


