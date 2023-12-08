import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')

from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_evaluation.utils import get_activations_iden, get_best_layer_iden
from model_features.activation_extractor import Activations
from model_features.models.models import load_model_dict
import gc

# define local variables

DATASET = 'majajhong'
REGIONS = ['V4','IT']

# DATASET = 'naturalscenes'
# REGIONS = ['general']#'V1','V2','V3','V4']

DEVICE = 'cuda' 
GLOBAL_POOL = False
MODEL_NAME = 'alexnet'
MODEL_LAYERS = ['alexnet_conv1','alexnet_conv2','alexnet_conv3','alexnet_conv4','alexnet_conv5']
    
# model_name = 'alexnet_untrained'
# models = ['alexnet_untrained_conv1',
#           'alexnet_untrained_conv2','alexnet_untrained_conv3','alexnet_untrained_conv4',
#           'alexnet_untrained_conv5']     
        
class ModelEvaluator:
    
    def __init__(self, model_name, model_layers, dataset, regions, device='cuda', global_pool=True):
        
        self.model_name = model_name
        self.model_layers = model_layers
        self.dataset = dataset
        self.regions = regions
        self.device = device
        self.global_pool = global_pool

    
    def __call__(self):
        self.run_evaluation()
        

    def run_evaluation(self):
        
        activation_iden_list = []
        
        for model_layer in self.model_layers:
            
            print('model layer: ', model_layer)
            
            model_info = self._load_model(model_layer)
            
            activations_identifier = get_activations_iden(model_info, self.dataset)
            self._get_activations(model_info)
            idens_list.append(activations_identifier)
            
        
        for region in self.regions:
            scores_iden = get_best_layer_iden(self.model_name, self.dataset, region, self.global_pool)
            self._get_scores(model_info, region, activation_iden_list, scores_iden)
        
        
                
    def _load_model(self, model_name):
        model_info = load_model_dict(model_name, gpool=self.global_pool)
        model_info['hook'] = None
        return model_info

    
    
    def _get_activations(self, model_info, iden):
                
        Activations(model=model_info['model'],
                    layer_names=model_info['layers'],
                    hook=model_info['hook'],
                    dataset=self.dataset,
                    device=self.device,
                    batch_size=10,
                    compute_mode='fast').get_array(iden)
        

    def _get_scores(model_info, region, activation_iden_list, scores_iden):
                
        encoding_score = EncodingScore(model_name=model_info['iden'],
                        activations_identifier=activation_iden_list,
                        dataset=DATASET,
                        region=region,
                        best_layer=True).get_scores(scores_iden)
                        
        gc.collect()

        
        
if __name__ == "__main__":

    evaluator = ModelEvaluator(model_name=MODEL_NAME, 
                               model_layers=MODEL_LAYERS,
                               dataset=DATASET, 
                               regions=REGIONS, 
                               device=DEVICE, 
                               global_pool=GLOABL_POOL)
    evaluator()  