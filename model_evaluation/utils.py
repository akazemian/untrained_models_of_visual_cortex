
def get_activations_iden(model_info, dataset, mode):
    
        model_name = model_info['iden'] 
        
        activations_identifier = model_name + '_' + f'{model_info["num_layers"]}_layers' + '_' + f'{model_info["num_features"]}_features' 

        if mode == 'pca':
            return activations_identifier + '_' + dataset + '_' + 'pca'
                  
        else:
            return activations_identifier + '_' + dataset 



def get_scores_iden(model_info, activations_identifier, region, dataset, mode, alpha=None):        
    
    return activations_identifier + '_' + region + '_' + mode









