
import sys
import xarray as xr
import numpy as np
import torch
import os
from ..regression import *
sys.path.append("/home/akazemi3/MB_Lab_Project") 
from tools.loading import get_image_labels
import random 
from sklearn.decomposition import PCA
from tqdm import tqdm 
import pickle 


ROOT = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(ROOT,'activations')
NEURAL_DATA_PATH = os.path.join(ROOT,'neural_data')
DATASET = 'naturalscenes'
PATH_TO_NSD_SHARED_IDS = os.path.join(NEURAL_DATA_PATH,'nsd_shared_ids')
file = open(PATH_TO_NSD_SHARED_IDS, 'rb')
SHARED_IDS = pickle.load(file)


import warnings
warnings.filterwarnings('ignore')

import scipy
from sklearn.preprocessing import scale
from tools.scorers.function_types import Regression
import random    
random.seed(0)
    

    
def get_subset(X, dim_reduction_type, n_dims):
    
    if dim_reduction_type == None:
        return X
    
    elif dim_reduction_type in ['pca','spca']:
        N = X.shape[0]
        X = X[:,:n_dims,...]
        X_subset = X.reshape(N,-1)
        print(X_subset.shape)
        return X_subset

    elif dim_reduction_type == 'rp':
        idx = random.sample(range(X.shape[1]),n_dims)
        X = X[:,idx,...]
        X = X.reshape(X.shape[0],-1)
        print(X.shape)
        return X
    
    
    
    
def nsd_scorer_unshared_cv(model_name: str, 
                           activations_identifier: str, 
                           scores_identifier: str, 
                           regression_model: Regression, 
                           regions: list = ['V1','V2','V3','V4'], 
                           dim_reduction_type:str = None,
                           n_dims:int = None) -> xr.Dataset:
    
        """
        
        Predicts neural responses using model activations through cross validated regression, and obtains a similarity score (Pearson r). 
        * This function uses only the unshared data from each subject from the NSD fmri dataset.


        Parameters
        ----------
        model_name : 
            The name of the model used for regression
            
        activations_identifier : 
            The name of the activations file
        
        scores identifier:
            The name used to save the file containing the scores
            
        regression_model: 
            Model used for regression
        
        regions:
            NSD ROIs from which neural responses are predicted 
            
        save_betas:
            Whether regression betas are stored

        Returns
        -------
        An xarray dataset containing the voxel-wise pearson r correlation scores per region per subject 
        
        """
        
        activations_data = xr.open_dataarray(os.path.join(ACTIVATIONS_PATH,activations_identifier))  
        
        ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                    coords={'subject': (['r_values'], []),
                                            'region': (['r_values'], [])
                                             })

        for region in regions:   

            print('region: ',region)
            for subject in range(8):

                print('subject: ',subject)

                ids, y = load_nsd_data(mode ='unshared',
                                       subject = subject,
                                       region = region,
                                       return_ids = True)


                X = filter_activations(data = activations_data,
                                       ids = ids)
                
                
                X = get_subset(X,dim_reduction_type,n_dims)
                
                y_true, y_predicted = regression_cv(x=X,
                                                    y=y,
                                                    model=regression_model,
                                                    scores_identifier=scores_identifier)
                r = torch.stack(
                   [pearson_r(y_true_, y_predicted_) for y_true_, y_predicted_ in zip(y_true, y_predicted)]
                ).mean(dim=0)

                
                subject_list = [subject for i in range(len(r))]
                region_list = [region for i in range(len(r))]

                ds_tmp = xr.Dataset(data_vars=dict(r_value=(["r_values"], r)),
                                            coords={'subject': (['r_values'], subject_list),
                                                    'region': (['r_values'], region_list)
                                                     })
                ds = xr.concat([ds,ds_tmp],dim='r_values')

        ds['name'] = scores_identifier
        return ds  


    
    
    
    
def nsd_scorer_shared_cv(model_name: str, 
                         activations_identifier:str, 
                         scores_identifier:str, 
                         regression_model: Regression, 
                         regions : list = ['V1','V2','V3','V4'], 
                         dim_reduction_type:str = None,
                         n_dims:int = None) -> xr.Dataset:
    
        """
        
        Predicts neural responses using model activations through cross validated regression, and obtains a similarity score (Pearson r). 
        * This function uses only the data shared across all subjects from the NSD fmri dataset.
        
        """
        
        activations_data = xr.open_dataarray(os.path.join(ACTIVATIONS_PATH,activations_identifier))         
        X = filter_activations(ids = SHARED_IDS,  data = activations_data)
        
        X = get_subset(X,dim_reduction_type,n_dims)
        
        ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                    coords={'subject': (['r_values'], []),
                                            'region': (['r_values'], [])
                                             })

        for region in regions:   

            print('region: ',region)
            for subject in range(8):

                print('subject: ',subject)

                y = load_nsd_data(mode = 'shared', subject = subject,
                                 region = region, return_ids = False)
                
                
                y_true, y_predicted = regression_cv(x=X,
                                                    y=y,
                                                    model=regression_model,
                                                    scores_identifier=scores_identifier)
                r = torch.stack(
                   [pearson_r(y_true_, y_predicted_) for y_true_, y_predicted_ in zip(y_true, y_predicted)]
                ).mean(dim=0)

                
                subject_list = [subject for i in range(len(r))]
                region_list = [region for i in range(len(r))]

                ds_tmp = xr.Dataset(data_vars=dict(r_value=(["r_values"], r)),
                                            coords={'subject': (['r_values'], subject_list),
                                                    'region': (['r_values'], region_list)
                                                     })
                ds = xr.concat([ds,ds_tmp],dim='r_values')

        ds['name'] = scores_identifier
        return ds 
    
    
    
    
    
    
def nsd_scorer_all(model_name: str, 
                   activations_identifier: str, 
                   scores_identifier: str, 
                   regression_model: Regression,
                   regions: list,
                   dim_reduction_type:str = None,
                   n_dims:int = None) -> xr.Dataset:
    
        """

        Predicts neural responses using model activations, and obtains a similarity score (Pearson r). 
        * This function uses the unshared data from each subject as the train set and the data shared across subjects as the test set.

        """    

        activations_data = xr.open_dataarray(os.path.join(ACTIVATIONS_PATH,activations_identifier))    
        ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                        coords={'subject': (['r_values'], []),
                                                'region': (['r_values'], [])
                                                 })
        for region in regions:   

            print('region: ',region)
            for subject in range(8):

                print('subject: ',subject)

                ids, y_train = load_nsd_data(mode = 'unshared', subject = subject,
                                             region = region, return_ids = True)

                y_test = load_nsd_data(mode = 'shared', subject = subject,
                                       region = region, return_ids = False)

                X_train = filter_activations(ids = ids,  data = activations_data)
                X_test = filter_activations(ids = SHARED_IDS,  data = activations_data)
                
                X_train = get_subset(X_train, dim_reduction_type, n_dims)
                X_test = get_subset(X_test, dim_reduction_type, n_dims)


                y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                             x_test=X_test,
                                                             y_train=y_train,
                                                             y_test=y_test,
                                                             model= regression_model)
                r = pearson_r(y_true,y_predicted)

                subject_list = [subject for i in range(len(r))]
                region_list = [region for i in range(len(r))]

                ds_tmp = xr.Dataset(data_vars=dict(r_value=(["r_values"], r)),
                                            coords={'subject': (['r_values'], subject_list),
                                                    'region': (['r_values'], region_list)
                                                     })
                ds = xr.concat([ds,ds_tmp],dim='r_values')

        ds['name'] = scores_identifier
        return ds


     
    
        
def load_nsd_data(mode: str, subject: int, region: str, return_ids: bool = True) -> torch.Tensor:
        
        """
        
        Loads the neural data from disk for a particular subject and region.


        Parameters
        ----------
        mode:
            The type of neural data to load ('shared' or 'unshared')
            
        subject:
            The subject number 
        
        region:
            The region name
            
        return_ids: 
            Whether the image ids are returned 
        

        Returns
        -------
        A Tensor of Neural data, or Tensor of Neural data and stimulus ids
        
        """
        
        file_name = f'subject_{subject}_{region}'
        file_path = os.path.join(NEURAL_DATA_PATH,DATASET,file_name)

        if mode == 'unshared':
            file_ext = '_unshared_new.nc'

        elif mode == 'shared':
            file_ext = '_shared_new.nc'

        data = xr.open_dataset(file_path + file_ext)
        ids = list(data.stimulus_id.values)
        data = torch.Tensor(data['x'].values).transpose(0,1)
        
        if return_ids:
            return ids, data
        else:
            return data
           
        
        
            
def filter_activations(data: xr.DataArray, ids: list) -> torch.Tensor:
            
        """
    
        Filters model activations using image ids.


        Parameters
        ----------
        data:
            Model activation data
            
        ids:
            image ids
        

        Returns
        -------
        A Tensor of model activations filtered by image ids
        
        """
        
        data = data.set_index({'presentation':'stimulus_id'})
        activations = data.sel(presentation=ids)
        activations = activations.sortby('presentation', ascending=True)

        return torch.Tensor(activations.values)
            

#NEURAL_DATA_PATH = '/data/rgautha1/cache/bonner-caching/neural-dimensionality/data/dataset=allen2021.natural_scenes/resolution=1pt8mm.preprocessing=fithrf_GLMdenoise_RR/roi=general/preprocessed/z_score=session.average_across_reps=True'

#             p = f'/data/rgautha1/cache/bonner-caching/neural-dimensionality/data/dataset=allen2021.natural_scenes/resolution=1pt8mm.preprocessing=fithrf_GLMdenoise_RR/roi={region}/preprocessed/z_score=session.average_across_reps=True/subject={subject}.nc'
            
#             neural_data = xr.open_dataarray(p)
            
#             y_train = neural_data.where(~neural_data.stimulus_id.isin(SHARED_IDS),drop=True)
#             print(y_train.values.shape)
            
#             y_test = neural_data.where(neural_data.stimulus_id.isin(SHARED_IDS),drop=True) 
#             print(y_test.values.shape)
            
            # unshared_ids = list(y_train.stimulus_id.values)
            # y_train, y_test = torch.Tensor(y_train.values), torch.Tensor(y_test.values)
