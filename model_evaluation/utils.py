
def get_activations_iden(model_info, dataset):
    
        model_name = model_info['iden'] 
        
        activations_identifier = model_name + '_' + f'{model_info["num_layers"]}_layers' + '_' + f'{model_info["num_features"]}_features' 

        return activations_identifier + '_' + dataset 










