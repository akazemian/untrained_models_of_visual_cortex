import os
import pickle

import xarray as xr
from dotenv import load_dotenv


from code_.model_activations.models.utils import load_model, load_full_identifier
from code_.model_activations.activation_extractor import Activations
# from config import setup_logging

# setup_logging()
load_dotenv()

CACHE = os.getenv("CACHE")
DATA = os.getenv("DATA")

def compute_model_pcs(model_name:str, features:int, layers:int, batch_size:int,
                      dataset:str, components:int, device:str):
    
    activations_identifier = load_full_identifier(model_name=model_name, 
                                                  features=features, 
                                                  layers=layers, 
                                                  dataset=dataset)
    # load model
    model = load_model(model_name=model_name, 
                       features=features, 
                       layers=layers,
                       device=device)
    
    # extract activations
    Activations(model=model, 
                dataset=dataset, 
                batch_size=batch_size,
                device= device).get_array(activations_identifier)   
    
    # load the saved activations
    data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier),
                             engine='netcdf4')
    
    pca_iden = load_full_identifier(model_name=model_name, 
                                    features=features,
                                    layers=5,
                                    dataset=dataset, 
                                    principal_components = components)
        
    # compute PCs given the dataset
    if dataset == 'naturalscenes':
        from code_.eigen_analysis.utils import _PCA
        from code_.encoding_score.benchmarks.nsd import filter_activations
        IDS_PATH = pickle.load(open(os.path.join(DATA,'naturalscenes', 'nsd_ids_unshared_sample=30000'), 'rb')) 
        NSD_UNSHARED_SAMPLE = [image_id.strip('.png') for image_id in IDS_PATH]
        data = filter_activations(data, NSD_UNSHARED_SAMPLE)
        _PCA(n_components = components)._fit(pca_iden, data, batch_size=64)
        del _PCA
        
    
    elif dataset == 'majajhong':
        from code_.eigen_analysis.utils import _PCA
        from code_.encoding_score.benchmarks.majajhong import load_activations
        data = load_activations(activations_identifier, mode = 'train')
        _PCA(n_components = components)._fit(pca_iden, data)
        del _PCA
        
    
    else: 
        from code_.eigen_analysis.utils import _PCA
        data = data.values
        _PCA(n_components = components)._fit(pca_iden, data)
        del _PCA




