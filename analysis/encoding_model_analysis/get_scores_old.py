# getting activations for a specific dataset from a specific model. Output is an xarray with dims: features x presentation (stimulus_id)
# from kymatio.torch import Scattering2D
import os 
import sys
ROOT_DIR = os.getenv('MB_ROOT_PATH')
sys.path.append(ROOT_DIR)

import xarray as xr
import torch
import torchvision
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
import random
from models.all_models.model_3L_abs_blurpool_avgpool_gabor import ExpansionModelGabor
from models.all_models.model_2L_abs_blurpool_avgpool import ExpansionModel2L
from models.all_models.model_3L_abs_blurpool_avgpool import ExpansionModel
from models.all_models.model_4L_abs_blurpool_avgpool import ExpansionModel4L
from models.all_models.scat_transform_kymatio import ScatTransformKymatio
from models.all_models.alexnet import Alexnet
from models.all_models.alexnet_u import AlexnetU

from models.all_models.model_3L_abs_blurpool_avgpool_pca import ExpansionModelPCA
from models.all_models.fully_connected import FCModel
from models.all_models.fully_connected_3layers import FCModel3L
from models.all_models.fully_random import FRModel
from tools.processing import *
from tools.utils import get_activations_iden, get_scores_iden
from analysis.encoding_model_analysis.tools.extractor import Activations
from analysis.encoding_model_analysis.tools.regression import *
from analysis.encoding_model_analysis.tools.scorer import *


torch.manual_seed(0)
torch.cuda.manual_seed(0)
untrained_alexnet = torchvision.models.alexnet(pretrained=False)
alexnet_pytorch =  torchvision.models.alexnet(pretrained=True)

ROOT = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(ROOT,'activations')   


# define constants
# DATASET = 'naturalscenes'
# REGIONS = ['general']

DATASET = 'naturalscenes'
REGIONS = ['general']

MAX_POOL = False
MODE = 'train'


# 'model name':{
# 'iden':str,
# 'model':nn.Module,
# 'layers': int, 
# 'preprocess':'function', 
# 'num_layers':int,
# 'num_features':int,
# 'dim_reduction_type':str,
# 'n_dims':int,
# 'max_dims':int,
# 'pca_dataset':str,
# 'max_pool':bool,
# 'alphas': list}


MODEL_DICT = {       


    
       'scattering kymatio j2':{
                'iden':'st_kymatio_j=2',
                'model':ScatTransformKymatio(J=2, L=4, M=96, N=96, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=96).PreprocessRGB, 
                'num_layers':2,
                'num_features':'n',
                'n_dims': None,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(1,4)]},   

    
       'scattering kymatio j3':{
                'iden':'st_kymatio_j=3',
                'model':ScatTransformKymatio(J=3, L=8, M=96, N=96, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=96).PreprocessRGB, 
                'num_layers':2,
                'num_features':'n',
                'n_dims': None,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(1,4)]},   

    
       'scattering kymatio j4':{
                'iden':'st_kymatio_j=4',
                'model':ScatTransformKymatio(J=4, L=8, M=96, N=96, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=96).PreprocessRGB, 
                'num_layers':2,
                'num_features':'n',
                'n_dims': None,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(1,4)]},   
}
 


for region in REGIONS:
    
    for model_name, model_info in MODEL_DICT.items():

        print(model_name)

        activations_identifier = get_activations_iden(model_info, DATASET, MODE)
        print(activations_identifier)

        activations = Activations(model=model_info['model'],
                                layer_names=model_info['layers'],
                                dataset=DATASET,
                                preprocess=model_info['preprocess'],
                                mode = MODE,
                                batch_size = 100
                            )           
        
        activations.get_array(ACTIVATIONS_PATH,activations_identifier)   
        
        for alpha in model_info['alphas']:
            
            print('alpha',alpha)
            regression_model = Ridge(alpha=alpha)            
            scores_identifier = get_scores_iden(model_info, activations_identifier, region, DATASET, MODE, alpha)
            
            scorer(model_name=model_info['iden'],
                   activations_identifier=activations_identifier,
                   scores_identifier=scores_identifier,
                   regression_model=regression_model,
                   dataset=DATASET,
                   mode=MODE,
                   regions=[region])

