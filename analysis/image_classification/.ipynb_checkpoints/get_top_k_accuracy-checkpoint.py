# model wrapper
import sys
import os
ROOT_DIR = os.getenv('MB_ROOT_PATH')
sys.path.append(ROOT_DIR)
from models.all_models.model_3L_abs_blurpool_avgpool import ExpansionModel
from models.all_models.alexnet import Alexnet

from tools.loading import *
from tools.processing import *
import torch
from sklearn import svm
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import top_k_accuracy_score as top_k
from analysis.encoding_model_analysis.tools.utils import get_activations_iden, get_scores_iden
from analysis.encoding_model_analysis.tools.extractor import Activations
import torchvision
from train import train

DATA_DIR = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(DATA_DIR,'activations') 
DATASET, MODE = 'places', None
HOOK = 'pca'
MAX_POOL = True


model_info =   {
                'iden':'expansion_model_final',
                'model':ExpansionModel(filters_3=10000).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':3,
                'num_features':10000,
                'max_pool':MAX_POOL,
# 

                # 'iden':'alexnet_conv5',
                # 'model':Alexnet().Build(),
                # 'layers': ['last'], 
                # 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                # 'num_layers':5,
                # 'num_features':256,
                # 'dim_reduction_type':None,
                # 'max_pool':MAX_POOL,


}
     
 
    
activations_identifier = get_activations_iden(model_info, DATASET, MODE)
print(activations_identifier)

# get model activations  
activations = Activations(model=model_info['model'],
                        layer_names=model_info['layers'],
                        dataset=DATASET,
                        preprocess=model_info['preprocess'],
                        mode = MODE,
                        _hook = HOOK,
                         batch_size = 50)

activations.get_array(ACTIVATIONS_PATH,activations_identifier) 

data = xr.open_dataset(os.path.join(ACTIVATIONS_PATH,activations_identifier))
# labels = np.array(get_places_cats())

num_cats = 150
cat_dict = load_places_cats()
cats = np.unique(get_places_cats())
cat_dict_subset = {k: v for k, v in cat_dict.items() if v in cats[:num_cats]}

val_images_subset = list(cat_dict_subset.keys())
data_subset = data.where(data.stimulus_id.isin(val_images_subset),drop=True)
labels_subset = np.array([cat_dict_subset[i] for i in val_images_subset])
    
top_1, top_5 = train(features = np.array(data_subset.x.values), labels = labels_subset, 
                     estimator_type = 'svm', shuffle = True, num_folds=5)


print('top 1 accuracy:', top_1)
print('top 5 accuracy:', top_5) 





    
    