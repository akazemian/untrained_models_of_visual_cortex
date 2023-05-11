
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
import warnings
warnings.filterwarnings('ignore')
import scipy
from sklearn.preprocessing import scale
from tools.scorers.function_types import Regression

ACTIVATIONS_PATH = '/data/atlas/activations'
NEURAL_DATA_PATH = '/data/atlas/neural_data'
MODEL_SCORES_PATH = '/data/atlas/model_scores'
TRAIN_IDS = "/home/akazemi3/Desktop/MB_Lab_Project/analysis/neural_data_regression/train_ids_majajhong"
TEST_IDS = "/home/akazemi3/Desktop/MB_Lab_Project/analysis/neural_data_regression/test_ids_majajhong"

DATASET = 'majajhong'
SUBJECTS = ['Chabo','Tito']

            
    
    
    
def majajhong_scorer_all_cv(model_name: str, 
                            activations_identifier: str, 
                            scores_identifier:str, 
                            regression_model: Regression, 
                            regions: list = ['V4','IT'], 
                            save_betas: bool = False) -> xr.Dataset:
        

                
            
            ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                    coords={'subject': (['r_values'], []),
                                            'region': (['r_values'], [])
                                             })
            
            X = load_activations(activations_identifier, mode = 'cv')

            
            for region in regions:   

                print('region: ',region)
                for subject in SUBJECTS:

                        print('subject: ',subject)


                        y = load_mjh_data(subject = subject, region = region)

                        y_true, y_predicted = regression_cv(x = X, y = y,
                                                            model = regression_model)
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
    
    

    
    
    
    
    
def majajhong_scorer_subset_cv(model_name: str, 
                            activations_identifier: str, 
                            scores_identifier:str, 
                            regression_model: Regression, 
                            regions: list = ['V4','IT'], 
                            save_betas: bool = False) -> xr.Dataset:
        
    
    
    X = load_activations(activations_identifier, mode = 'train')

    ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                    coords={'subject': (['r_values'], []),
                                            'region': (['r_values'], [])
                                             })
    for region in regions:   

        print('region: ',region)
        
        for subject in SUBJECTS:

            print('subject: ',subject)

            y = load_mjh_data(subject = subject, region = region, mode = 'train')
            
            y_true, y_predicted = regression_cv(x = X, y = y, model = regression_model)
            
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






def majajhong_scorer_all(model_name: str, 
                            activations_identifier: str, 
                            scores_identifier:str, 
                            regression_model: Regression, 
                            regions: list, 
                            save_betas: bool = False) -> xr.Dataset:
    
    
    X_train = load_activations(activations_identifier, mode = 'train')
    X_test = load_activations(activations_identifier, mode = 'test')    
    
    ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                    coords={'subject': (['r_values'], []),
                                            'region': (['r_values'], [])
                                             })
    for region in regions:   

        print('region: ',region)
        for subject in SUBJECTS:

            print('subject: ',subject)

            y_train = load_mjh_data(subject = subject, region = region, mode = 'train')
            y_test = load_mjh_data(subject = subject, region = region, mode = 'test')
            
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




     
def load_mjh_data(subject: str, region: str, mode: bool = None) -> torch.Tensor:
        
        """
        
        Loads the neural data from disk for a particular subject and region.


        Parameters
        ----------
            
        subject:
            The subject number 
        
        region:
            The region name
            
        mode:
            The type of neural data to load ('train' or 'test')
        

        Returns
        -------
        A Tensor of Neural data
        
        """
        
        file_name = f'SUBJECT_{subject}_REGION_{region}'
        file_path = os.path.join(NEURAL_DATA_PATH,DATASET,file_name)
        neural_data = xr.open_dataset(file_path)

        
        if mode == 'train':
            with open(TRAIN_IDS, "rb") as fp:   
                train_ids = pickle.load(fp) 
            neural_data = neural_data.where(neural_data.stimulus_id.isin(train_ids),drop=True)
        
        
        elif mode == 'test':
            with open(TEST_IDS, "rb") as fp:   
                test_ids = pickle.load(fp) 
            neural_data = neural_data.where(neural_data.stimulus_id.isin(test_ids),drop=True)
            
        
        neural_data = neural_data.sortby('stimulus_id', ascending=True)
        neural_data = torch.Tensor(neural_data['x'].values.squeeze())
        
        return neural_data

            
           
        
            
def load_activations(activations_identifier: str, mode: bool = None) -> torch.Tensor:
            
        """
    
        Loads model activations.


        Parameters
        ----------
            
        activations_identifier:
            model activations identifier
        
            
        mode:
            The type of neural data to load ('train' or 'test')
        

        Returns
        -------
        A Tensor of activation data
        
        """
        
        activations_data = xr.open_dataset(os.path.join(ACTIVATIONS_PATH,activations_identifier))
        activations_data = activations_data.set_index({'presentation':'stimulus_id'})


        if mode == 'train':

            with open(TRAIN_IDS, "rb") as fp:   
                train_ids = pickle.load(fp)
            activations_data = activations_data.sel(presentation=train_ids)
        
        
        elif mode == 'test':

            with open(TEST_IDS, "rb") as fp:   
                test_ids = pickle.load(fp)
            activations_data = activations_data.sel(presentation=test_ids)    
        
        
        
        activations_data = activations_data.sortby('presentation', ascending=True)
        
        return torch.Tensor(activations_data.x.values)
    
    
    
    
# def activations_pca(model_name,activations_path,activations_identifier):    
    
#         activations_data = xr.open_dataset(os.path.join(activations_path,activations_identifier)).sortby('stimulus_id', ascending=True)    
#         pca = _PCA(n_components=1000, svd_solver='auto',pca_file=f'pca_{model_name}_1000_components_10000_images.joblib')
#         pca_features = pca(torch.Tensor(activations_data['x'].values))
        
#         activations_data_pca = xr.Dataset(
#             data_vars=dict(x=(["presentation", "features"], pca_features)),
#             coords={'stimulus_id': (['presentation'], activations_data.stimulus_id.values)})
        
#         return activations_data_pca
    
    
    
    
    
