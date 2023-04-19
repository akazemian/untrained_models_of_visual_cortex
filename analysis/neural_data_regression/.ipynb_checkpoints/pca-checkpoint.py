    
# define paths
import os 
import sys
ROOT_DIR = os.getenv('MB_ROOT_PATH')
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
# define paths
ROOT_DATA = os.getenv('MB_DATA_PATH')
PATH_TO_PCA = os.path.join(ROOT_DATA,'pca')
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
from models.all_models.model_2L import EngineeredModel2L
from models.all_models.model_3L import EngineeredModel3L
from models.all_models.model_3L_pca import EngineeredModel3LPCA
from models.all_models.model_3L_get_pcs import EngineeredModel3LPCs
from models.all_models.alexnet_untrained_wide import AlexnetU
from models.all_models.model_3L_vone import EngineeredModel3LVOne
import pickle   
    
    

      
    
DATASET = 'naturalscenes'    
MAX_POOL = True
N_COMPONENTS =  5000

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
              
    
            'alexnet u wide':{
                'iden':'alexnet_u_wide',
                'model':AlexnetU().Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                }, 
    
              # f'alexnet_u_wide_mp_10000_nsd_pca_5000_components':{'model':AlexnetU(filters_5 = 10000).Build(),
              # 'layers': ['mp5'], 'preprocess':PreprocessRGBLarge},
              
}





        
        

for model_name, model_info in MODEL_DICT.items():
       
    
    print(model_name)
    activations_identifier = model_info['iden'] + '_' + 'mp' + '_' + 'pca' + '_' + str(N_COMPONENTS) + '_' + DATASET
        
    if os.path.exists(os.path.join(os.path.join(PATH_TO_PCA,activations_identifier))):
        print(f'pcs are already saved in {PATH_TO_PCA} as {activations_identifier}')
        
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
            pca = PCA(n_components = N_COMPONENTS)
            pca.fit(X)


            file = open(os.path.join(PATH_TO_PCA,activations_identifier), 'wb')
            pickle.dump(pca, file, protocol=4)
            file.close()
            print(f'pcs are now saved in {PATH_TO_PCA} as {activations_identifier}')


