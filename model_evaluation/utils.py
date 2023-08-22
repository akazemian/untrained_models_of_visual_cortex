
def get_activations_iden(model_info, dataset, hook=None):
    
        model_name = model_info['iden'] 
        
        activations_identifier = model_name + '_' + f'{model_info["num_layers"]}_layers' + '_' + f'{model_info["num_features"]}_features' 

        try:
            if model_info['hook'] == 'pca':
                return activations_identifier + '_' + dataset + '_principal_components'
            else:
                print('invalid hook')
                return
            
        except KeyError:
                return activations_identifier + '_' + dataset 










