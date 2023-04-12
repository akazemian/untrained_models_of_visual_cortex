# getting activations for a specific dataset from a specific model. Output is an xarray with dims: features x presentation (stimulus_id)
# from kymatio.torch import Scattering2D
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tables
from sklearn.linear_model import Ridge
import os 

import sys
path = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(path)

from analysis.neural_data_regression.tools.regression import *
from analysis.neural_data_regression.tools.scorer import *
from analysis.neural_data_regression.tools.extractor import Activations, Activations3Layer
from tools.loading import *
from tools.processing import *
from tools.utils import get_best_alpha 
from models.call_model import *
from models.call_model import EngineeredModel
from models.all_models.model_3L import EngineeredModel3L



# define paths
ACTIVATIONS_PATH = '/data/atlas/activations'
PATH_TO_CORE_ACTIVATIONS = '/home/akazemi3/Desktop/MB_Lab_Project/models/all_models/core_activations'    
PATH_TO_BETAS = f'/data/atlas/regression_betas/'


# define constants
DATASET = 'naturalscenes_zscored_processed'
REGIONS= ['V4']

MODEL = EngineeredModel3L(filters_3 = 20000, batches_3=5).Build()
MODEL_NAME = 'model_final_mp_3l_100000_all'
ACTIVATIONS_IDEN = MODEL_NAME + '_' + DATASET

CORE_ACTIVATIONS_NAME = 'model_final_mp_all'
CORE_ACTIVATIONS_IDEN =  CORE_ACTIVATIONS_NAME + '_' + DATASET
CORE_ACTIVATIONS_ALPHAS = [0] + [10**i for i in range(11)] 

MODE = 'test'
#ALPHAS = [0] + [10**i for i in range(11)] 
#ALPHAS = [10**i for i in range(6,11)] 
ALPHAS = [10**9]

for alpha in ALPHAS:
    

    activations = Activations3Layer(model=MODEL,
                        layer_names=['c3'],
                        dataset=DATASET,
                        max_pool=True,
                        preprocess = None
                        )   
    
    activations.get_array(path = ACTIVATIONS_PATH, 
                          identifier = ACTIVATIONS_IDEN, 
                          regions = REGIONS, 
                          core_activations_iden = CORE_ACTIVATIONS_IDEN,
                          core_activations_alphas = CORE_ACTIVATIONS_ALPHAS
                         )


    regression_model = Ridge(alpha=alpha)
    scores_identifier = ACTIVATIONS_IDEN + '_' + 'V4' + '_' + f'Ridge(alpha={alpha})' 
    scorer(model_name=MODEL_NAME,
           activations_identifier=ACTIVATIONS_IDEN,
           scores_identifier=scores_identifier,
           regression_model=regression_model,
           dataset=DATASET,
           mode=MODE,
           regions=REGIONS
          )




