    
# define paths
import os 
import sys
ROOT_DIR = os.getenv('MB_ROOT_PATH')
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
# define paths
ROOT_DATA = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(ROOT_DATA,'activations')


import xarray as xr
from tools.processing import *
from models.call_model import *
from tools.loading import *
from analysis.neural_data_regression.tools.extractor import Activations
import xarray as xr
import torchvision
from analysis.neural_data_regression.tools.regression import *
from analysis.neural_data_regression.tools.scorer import *
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from models.all_models.model_2L import EngModel2L
from models.all_models.model_3L import EngModel3L
# from models.all_models.alexnet_untrained_wide import AlexnetU
# from models.all_models.model_3L_abs import EngModel3LAbs
# from models.all_models.model_3L_abs_blurpool import EngModel3LAbsBP
from models.all_models.model_3L_abs_blurpool_avgpool import EngModel3LAbsBPAP

import pickle
alexnet =  torchvision.models.alexnet(pretrained=True)
      
    
DATASET = 'naturalscenes'    
MAX_POOL = True
N_COMPONENTS =  5000
#PATH_TO_PCA = os.path.join(ROOT_DATA,'pca_mp') if MAX_POOL else os.path.join(ROOT_DATA,'pca')
PATH_TO_PCA = os.path.join(ROOT_DATA,'pca')

# models    
MODEL_DICT = {
    
              
              # f'model_2L_mp_5000_nsd_pca_{N_COMPONENTS_L2}_components':{'model':EngineeredModel2L(filters_2=5000).Build(),
              # 'layers': ['last'], 'preprocess':PreprocessGS,'n_components':N_COMPONENTS_L2},
    
            #   f'model_RGB_3L_10000_nsd_pca_{N_COMPONENTS}_components':{'model':EngineeredModel3L(filters_3=10000).Build(),
            #   'layers': ['last'], 'preprocess':PreprocessRGBSmall,'n_components':N_COMPONENTS},  

              # f'model_3L_mp_10000_nsd_pca_{N_COMPONENTS_L3}_components':{'model':EngineeredModel3LPCs(filters_2=5000,filters_3=10000).Build(),
              # 'layers': ['last'], 'preprocess':PreprocessGS,'n_components':N_COMPONENTS_L3}, 
    
#             'model_vone_224':{
#                 'model':EngineeredModel3LVOne(im_size=224).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB,
#             }
#                 'model abs 3x3 bp 224':{
#                 'iden':'model_abs_3x3_bp_224',
#                 'model':EngModel3LAbsBP(filters_3=10000).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':10000,
#                 'pca':True,
#                 'pca_dataset':'nsd',
#                 'num_pca_components':None,
#                 'max_pca_components':5000,
#                 'pca_type':None,
#                 'max_pool':True,
#                 'alphas': [10**i for i in range(2,5)]}, 
    
    
    'model abs 3x3 bp 224 ap 10000 filters':{
                'iden':'model_abs_3x3_bp_224_ap',
                'model':EngModel3LAbsBPAP(filters_3 = 10000, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessGS, 
                'num_layers':3,
                'num_features':10000,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(1,5)]},  
    
                # 'alexnet':{
                # 'iden':'alexnet',
                # 'model':alexnet,
                # 'layers': ['features.12'], 
                # 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                # 'num_layers':5,
                # 'num_features':256,
                # 'pca':True,
                # 'pca_dataset':'nsd',
                # 'num_pca_components':None,
                # 'max_pca_components':5000,
                # 'pca_type':None,
                # 'max_pool':True,
                # 'alphas': [10**i for i in range(2,5)]}, 
              
}





        
        

for model_name, model_info in MODEL_DICT.items():
       
    
    print(model_name)
    activations_identifier = model_info['iden']
    if MAX_POOL:
        activations_identifier = activations_identifier + '_' + 'mp'
    activations_identifier = activations_identifier + '_' + 'pca' + '_' + str(N_COMPONENTS) + '_' + DATASET
        
    if os.path.exists(os.path.join(os.path.join(PATH_TO_PCA,activations_identifier))):
        print(f'pcs are already saved in {PATH_TO_PCA} as {activations_identifier}')
        
    else:

        print('obtaining pcs...')
              
        for layer in model_info['layers']:


            activations = Activations(model= model_info['model'],
                                layer_names = [layer],
                                dataset = DATASET,
                                preprocess = model_info['preprocess'],
                                mode = 'pca',
                                max_pool = MAX_POOL,
                                )                   
            activations.get_array(ACTIVATIONS_PATH,activations_identifier)     


            X = xr.open_dataset(os.path.join(ACTIVATIONS_PATH,activations_identifier)).x.values
            pca = PCA(n_components = N_COMPONENTS)
            pca.fit(X)


            file = open(os.path.join(PATH_TO_PCA,activations_identifier), 'wb')
            pickle.dump(pca, file, protocol=4)
            file.close()
            print(f'pcs are now saved in {PATH_TO_PCA} as {activations_identifier}')


