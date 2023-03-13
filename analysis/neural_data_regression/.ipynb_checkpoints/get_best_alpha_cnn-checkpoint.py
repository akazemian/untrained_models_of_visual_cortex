# getting activations for a specific dataset from a specific model. Output is an xarray with dims: features x presentation (stimulus_id)
# from kymatio.torch import Scattering2D
import xarray as xr
import os 
import sys
import torchvision

path = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(path)
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
from models.call_model import EngineeredModel
import torchvision
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression


    
#torch.manual_seed(0)

# dataset = 'majajhong'
# regions = ['V4','IT']

dataset = 'naturalscenes_zscored_processed'
regions = ['V1','V2','V3','V4']
layers = ['features.2','features.5','features.7', 'features.9', 'features.12']
preprocess= PreprocessRGB

model_list = {
    'alexnet_mp':torchvision.models.alexnet(pretrained=True),
    #'alexnet_untrained_mp':torchvision.models.alexnet(pretrained=False),
}


df_max_alphas = pd.read_csv(f'/home/akazemi3/Desktop/MB_Lab_Project/analysis/neural_data_regression/max_alphas_cnns_{dataset}_mp_layerwise.csv')
dict_max_alphas = df_max_alphas.set_index(['region','model']).to_dict()['alpha']



for layer in layers:
    
    print('layer',layer)
    
    for region in regions:
        print('region',region)


        for model_name, model in model_list.items():


            alpha = dict_max_alphas[(region, model_name+f'_{layer}')]
            regression_model = Ridge(alpha=alpha)
            print('regression_model',regression_model)



            # get activations  
            activations = Activations(model=model,
                                layer_names=[layer],
                                dataset=dataset,
                                preprocess=preprocess
                                )                  

            activations_identifier = model_name + '_' + dataset + '_' + layer
            activations.get_array(activations_path,activations_identifier)     


            scores_identifier = activations_identifier + f'_{region}' + f'_Ridge(alpha={alpha})'
            scorer(model_name=model_name,
                   activations_identifier=activations_identifier,
                   scores_identifier=scores_identifier,
                   regression_model=regression_model,
                   dataset=dataset,
                   mode='test',
                   regions=[region]
                  )
