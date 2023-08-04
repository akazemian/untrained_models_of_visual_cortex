    
# define paths
import os 
import sys
ROOT_DIR = os.getenv('MB_ROOT_PATH')
sys.path.append(ROOT_DIR)
# define paths
ROOT_DATA = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(ROOT_DATA,'activations')


import xarray as xr
import torchvision
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from tools.processing import *
from tools.loading import *
from analysis.encoding_model_analysis.tools.extractor import Activations
from analysis.encoding_model_analysis.tools.regression import *
from analysis.encoding_model_analysis.tools.scorer import *
from models.all_models.model_3L_abs_blurpool_avgpool import ExpansionModel
from tools.utils import get_activations_iden
from models.all_models.alexnet import Alexnet
from models.all_models.alexnet_u import AlexnetU    
    
#DATASET = 'imagenet21k'    
DATASET = 'places'
MAX_POOL = True
N_COMPONENTS =  1000
MODE = 'pca'
PATH_TO_PCA = os.path.join(ROOT_DATA,'pca')

# models    
MODEL_DICT = {


    'expansion model 3L 10000':{
                'iden':'expansion_model_final',
                'model':ExpansionModel(filters_3=10000).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':3,
                'num_features':10000,
                'max_pool':MAX_POOL}, 
}





        
        

for model_name, model_info in MODEL_DICT.items():
       
    
    print(model_name)
    activations_identifier = get_activations_iden(model_info, DATASET, MODE)
        
    if os.path.exists(os.path.join(os.path.join(PATH_TO_PCA,activations_identifier))):
        print(f'pcs are already saved in {PATH_TO_PCA} as {activations_identifier}')
        
    else:

        print('obtaining pcs...')
              
        for layer in model_info['layers']:


            activations = Activations(model=model_info['model'],
                                    layer_names=model_info['layers'],
                                    dataset=DATASET,
                                    preprocess=model_info['preprocess'],
                                    mode = MODE,
                                    batch_size = 100
                                )           

            activations.get_array(ACTIVATIONS_PATH,activations_identifier)   


            X = xr.open_dataset(os.path.join(ACTIVATIONS_PATH,activations_identifier)).x.values
            pca = PCA(N_COMPONENTS)
            pca.fit(X)


            file = open(os.path.join(PATH_TO_PCA,activations_identifier), 'wb')
            pickle.dump(pca, file, protocol=4)
            file.close()
            print(f'pcs are now saved in {PATH_TO_PCA} as {activations_identifier}')


