# model wrapper
import sys
import os
ROOT_DIR = os.getenv('MB_ROOT_PATH')
sys.path.append(ROOT_DIR)
from models.all_models.model_3L_abs_blurpool_avgpool import ExpansionModel
from models.all_models.model_3L_abs_blurpool_avgpool_pca import ExpansionModelPCA
from models.all_models.alexnet import Alexnet

from tools.loading import *
from tools.processing import *
import torch
from sklearn import svm
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import top_k_accuracy_score as top_k
from analysis.neural_data_regression.tools.utils import get_activations_iden, get_scores_iden
from analysis.neural_data_regression.tools.extractor import Activations
import torchvision
from train import train


DATA_DIR = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(DATA_DIR,'activations') 
DATASET, MODE = 'imagenet21kval', None
MAX_POOL = True


model_info = {'iden':'alexnet_test',
              'model':Alexnet(global_mp=MAX_POOL).Build(),
              'layers': ['last'], 
              'preprocess':Preprocess(im_size=224).PreprocessRGB, 
              'num_layers':5,
              'num_features':256,
              'dim_reduction_type':None,
              'max_pool':MAX_POOL}
     
# model_info = {'iden':'expansion_model',
#               'model':ExpansionModel(filters_3 = 10000, gpool = MAX_POOL).Build(),
#               'layers': ['last'], 
#               'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#               'num_layers':3,
#               'dim_reduction_type':None,
#               'num_features':10000,
#               'max_pool':MAX_POOL}


# model_info = { 
#                 'iden':'expansion_model_kaiming_normal',
#                 'model':ExpansionModel(init_type = 'kaiming_normal', filters_3 = 1000, gpool = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':1000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'n_dims': None,
#                 'alphas': 'diff_initializations'} 
    
activations_identifier = get_activations_iden(model_info, DATASET, MODE)
print(activations_identifier)

# get model activations  
activations = Activations(model=model_info['model'],
                        layer_names=model_info['layers'],
                        dataset=DATASET,
                        preprocess=model_info['preprocess'],
                        mode = MODE)

activations.get_array(ACTIVATIONS_PATH,activations_identifier) 
data = xr.open_dataset(f'/data/atlas/activations/{activations_identifier}')

top_1, top_5 = train(features = np.array(data.x.values), labels = np.array(data.stimulus_id), 
                     estimator_type = 'svm', shuffle = True, num_folds=5)

print('top 1 accuracy:', top_1)
print('top 5 accuracy:', top_5) 


