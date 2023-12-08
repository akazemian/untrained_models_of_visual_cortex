import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')

from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_evaluation.utils import get_activations_iden
from model_features.activation_extractor import Activations
from model_features.models.models import load_model_dict
import gc

# define local variables

# DATASET = 'majajhong'
# REGIONS = ['V4','IT']

DATASET = 'naturalscenes'
REGIONS = ['general']
DEVICE = 'cuda' 
GLOBAL_POOL = True 
MODELS = ['expansion_first_256_pcs','alexnet_conv1','alexnet_conv3','alexnet_conv5',
          'alexnet_untrained_conv1', 'alexnet_untrained_conv3','alexnet_untrained_conv5'
          ]         
        
        
        
class ModelEvaluator:
    
    def __init__(self, models, dataset, regions, device='cuda', global_pool=True):
        
        self.models = models
        self.dataset = dataset
        self.regions = regions
        self.device = device
        self.global_pool = global_pool
        self.models = models

    
    def __call__(self):
        self.run_evaluation()
        
    
    def run_evaluation(self):
        
        for model_name in self.mdoels:
            
            print('model: ', model_name)
            model_info = self._load_model(model_name)

            if model_name == 'expansion_first_256_pcs':
                model_info['hook'] = 'pca'
                os.system('python Desktop/random_models_of_visual_cortex/model_evaluation/eigen_analysis/compute_pcs.py')

            for region in self.regions:
                self._get_activations_and_scores(model_info, region)


                
                
    def _load_model(self, model_name):
        model_info = load_model_dict(model_name, gpool=self.global_pool)
        model_info['hook'] = None
        return model_info

    
    
    
    def _get_activations_and_scores(self, model_info, region):
        
        activations_identifier = get_activations_iden(model_info, self.dataset)
        scores_identifier = activations_identifier + '_' + region
        
        
        Activations(model=model_info['model'],
                    layer_names=model_info['layers'],
                    hook=model_info['hook'],
                    dataset=self.dataset,
                    device=self.device,
                    batch_size=10,
                    compute_mode='fast').get_array(activations_identifier)
        
        
        EncodingScore(model_name=model_info['iden'],
                      activations_identifier=activations_identifier,
                      dataset=self.dataset,
                      region=region,
                      device=self.device).get_scores(scores_identifier)
        gc.collect()


        
        
        

if __name__ == "__main__":

    evaluator = ModelEvaluator(models=MODELS, 
                               dataset=DATASET, 
                               regions=REGIONS, 
                               device=DEVICE, 
                               global_pool=GLOABL_POOL)
    evaluator()  
