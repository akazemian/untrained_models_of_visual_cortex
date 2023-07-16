import sys
# from kymatio.torch import Scattering2D
import os 
path = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(path)
results_path = os.path.join(path, 'Results/Encoding_Performance')
import torch 
from torch import nn
import numpy as np
from sklearn import linear_model
from scipy.stats import pearsonr
from model_tools.activations import PytorchWrapper
import math
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
import brainscore.benchmarks as bench
from brainscore.metrics.regression import linear_regression, ridge_regression
from model_tools.brain_transformation.neural import LayerScores
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from tools.processing import *
#from models.call_model import *
from tools.loading import *
# from scattering_transform.scattering.ST import *
import importlib
import random
import shutil
from models.call_model import EngineeredModel

import pickle
#os.environ['RESULTCACHING_DISABLE'] = "1"

#os.environ['BRAINIO_HOME']= "/data/shared/.cache/brainscore/brainio"




data = {'name':'dicarlo.MajajHong2015public.','regions':['V4','IT']}
#data = {'name':'movshon.FreemanZiemba2013public.','regions':['V1','V2']}

# model = Scattering2D(J=5, shape=(96, 96))
# model_name = 'scattering_J=5'
# layers = ['logits']  


# model = EngineeredModel2Gabor().Build()
# model_name = 'engineered_eng2_gabor_'
# layers = ['logits'] 


# model = EngineeredModel2().Build()
# model_name = 'engineered_sc_2'
# layers = ['logits']  

# model = hmax.HMAX(universal_patch_set=os.path.join(path,'pytorch_hmax/universal_patch_set.mat'))
# model_name = 'hmax'
# layers = ['logits'] 


# model_name = 'scattering_sc'
# H, W = 96, 96
# model = ScatteringTransform(M=H, N=W, J=4, L=8).Build()
# layers = ['s2'] 


# model = kymatio2d(J=2, input_shape=(96, 96),layer='all').Build()
# model_name = 'kymatio_J=2'
# layers = ['logits']  

model_scores_path = '/data/atlas/model_scores'

import torchvision
random.seed(0)
model_list = {
    # 'model_final_test':EngineeredModel().Build(),
    # 'alexnet_test':torchvision.models.alexnet(pretrained=True),
    'alexnet_untrained_test_3':torchvision.models.alexnet(pretrained=False),
     }

alphas = [1]
for alpha in alphas:
    for model_name, model in model_list.items():

        layers = ['features.12'] if 'alexnet' in model_name else ['last']

        model_iden = model_name + '_' + data['name'] 
        preprocess = PreprocessRGB if 'alexnet' in model_name else PreprocessGS



        if os.path.exists(os.path.join(model_scores_path,f'{model_iden}_Ridge(alpha={alpha})')):
            print(f'model scores are already saved in {model_scores_path} as {model_iden}_Ridge(alpha={alpha})')

        # if os.path.exists(os.path.join(model_scores_path,f'{model_iden}_pls')):
        #     print(f'model scores are already saved in {model_scores_path} as {model_iden}_pls')
        else:
            print(f'obtaining model scores...')
            ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                        coords={'region': (['r_values'], [])
                                                     })
            for region in data['regions']: 

                benchmark_iden = data['name'] + region + '-pls'

                activations_model = PytorchWrapper(model=model,
                                                   preprocessing=preprocess,
                                                   identifier=model_iden)

                benchmark = bench.load(benchmark_iden)
                benchmark._identifier = benchmark.identifier.replace('pls', f'ridge_alpha={alpha}')
                benchmark._similarity_metric.regression = ridge_regression(sklearn_kwargs = {'alpha':alpha})


                model_scores = LayerScores(model_identifier=activations_model.identifier,
                                       activations_model=activations_model,
                                       visual_degrees=8)

                score = model_scores(benchmark=benchmark,layers=layers,prerun=True)                
                r_values = score.raw.raw.values.reshape(-1)
                #r_values = score.raw.raw.values.squeeze().mean(axis=0)
                

                #file = open(f'/data/atlas/model_scores/{model_name}_brain_score_scroes_{region}','wb')
                #pickle.dump(score, file)

                ds_tmp = xr.Dataset(data_vars=dict(r_value=(["r_values"], r_values)),
                                                    coords={'region': (['r_values'], 
                                                                       [region for i in range(len(r_values))])
                                                                 })

                ds = xr.concat([ds,ds_tmp],dim='r_values')


            ds.to_netcdf(os.path.join(model_scores_path,f'{model_iden}_Ridge(alpha={alpha})'))
            print(f'model scores are now saved in {model_scores_path} as {model_iden}_Ridge(alpha={alpha})')
