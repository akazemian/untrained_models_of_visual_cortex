
def get_activations_iden(model_info, dataset):
    
        model_name = model_info['iden'] 
        
        activations_identifier = model_name + '_' + f'{model_info["num_layers"]}_layers' + '_' + f'{model_info["num_features"]}_features' 

        if model_info['gpool'] == False:
            activations_identifier = activations_identifier + '_gpool=False'        
        
        if model_info['hook'] == 'pca':
            activations_identifier = activations_identifier + '_principal_components'
        
        return activations_identifier + '_' + dataset











