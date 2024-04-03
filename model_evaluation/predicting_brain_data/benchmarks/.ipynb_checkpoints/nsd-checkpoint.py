
from ..regression.regression import regression_shared_unshared, pearson_r
from ..regression.torch_cv import TorchRidgeGCV
from sklearn.linear_model import Ridge

import sys
import xarray as xr
import numpy as np
import torch
import os
import random 
from tqdm import tqdm 
import pickle 
import warnings
warnings.filterwarnings('ignore')
import random    
random.seed(0)
import scipy.stats as st
import gc
from scipy.sparse import csr_matrix
from pathlib import Path

ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)
from config import CACHE, NSD_NEURAL_DATA, NSD_SAMPLE_IMAGES    

SHARED_IDS_PATH = os.path.join(ROOT, 'image_tools','nsd_ids_shared')
SHARED_IDS = pickle.load(open(SHARED_IDS_PATH, 'rb'))
SHARED_IDS = [image_id.strip('.png') for image_id in SHARED_IDS]

SAMPLE_IDS = pickle.load(open(NSD_SAMPLE_IMAGES, 'rb'))
SAMPLE_IDS = [image_id.strip('.png') for image_id in SAMPLE_IDS]

PREDS_PATH = '/data/atlas/.cache/beta_predictions'

ALPHA_RANGE = [10**i for i in range(10)]
   
    
    
def normalize(X, X_min=None, X_max=None, use_min_max=False):
    
    if use_min_max:
        X_normalized = (X - X_min) / (X_max - X_min)
        return X_normalized
    
    else:
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_normalized = (X - X_min) / (X_max - X_min)
        return X_normalized, X_min, X_max





# def nsd_scorer_subjects(activations_identifier: str, 
#                region: str,
#                device: str,
#                subject:int):

                
#         ds = xr.Dataset(data_vars=dict(r_value=(["neuroid"], [])),
#                                 coords={'x':(['neuroid'], []), 
#                                         'y':(['neuroid'], []), 
#                                         'z':(['neuroid'], []),
#                                          })

#         #load X_train and y_train
#         X_train = xr.open_dataset(os.path.join(CACHE,'activations', activations_identifier + f'_subject={subject}'), 
#                                     engine='netcdf4').x.values
            
#         _ , neural_data_train, var_name_train = load_nsd_data(mode = 'unshared',
#                                                     subject = subject,
#                                                     region = region)
#         y_train = neural_data_train[var_name_train].values
#         # corss validated ridge regression on training data to find optimal penalty term
#         regression = TorchRidgeGCV(
#             alphas=ALPHA_RANGE,
#             fit_intercept=True,
#             scale_X=False,
#             scoring='pearsonr',
#             store_cv_values=False,
#             alpha_per_target=False,
#             device=device)

#         #regression.to('cpu')
#         regression.fit(X_train, y_train)
#         best_alpha = float(regression.alpha_)
#         print('best alpha:',best_alpha)
#         print('best score:',regression.score_)

#         #load X_test and y_test
#         X_test = xr.open_dataset(os.path.join(CACHE,'activations',f'{activations_identifier}_shared_images'), 
#                                                       engine='netcdf4').x.values

#         _ , neural_data_test, var_name_test = load_nsd_data(mode ='shared',
#                                                             subject = subject,
#                                                             region = region)           
#         y_test = neural_data_test[var_name_test].values

#         model= Ridge(alpha=best_alpha)
        
#         X_train = X_train.astype(np.float32)
#         X_test = X_test.astype(np.float32)
#         y_train = y_train.astype(np.float32)
#         y_test = y_test.astype(np.float32)
    
#         model.fit(X_train, y_train)
#         gc.collect()
#         y_predicted = model.predict(X_test)
        
#         with open(os.path.join(PREDS_PATH,f'{activations_identifier}_{region}_{subject}.pkl'), 'wb') as file:
#                 pickle.dump(y_predicted, file)
                
#         r = pearson_r(torch.Tensor(y_test),torch.Tensor(y_predicted))
        
#         ds_tmp = xr.Dataset(data_vars=dict(r_value=(["neuroid"], r)),
#                                     coords={'x':(['neuroid'],neural_data_test.x.values),
#                                             'y':(['neuroid'],neural_data_test.y.values),
#                                             'z':(['neuroid'], neural_data_test.z.values),
#                                              })

#         ds = xr.concat([ds,ds_tmp], dim='neuroid')   

#         ds['name'] = activations_identifier + '_' + region 
#         return ds            
            

     

# def nsd_scorer_principal_components(activations_identifier: str,region: str,device: str, n_components:int):
    
            
#             activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier), engine='h5netcdf')  
        
#             for subject in tqdm(range(8)):
    
                
#                 ids_train, neural_data_train, var_name_train = load_nsd_data(mode ='unshared',
#                                                             subject = subject,
#                                                             region = region)
                
#                 X_train = filter_activations(data = activations_data, ids = ids_train)  
                
#                 ###### NORMALIZE ####
#                 X_train, X_min, X_max = normalize(X_train)
#                 X_train = np.nan_to_num(X_train)
#                 #####################
                
                
#                 y_train = neural_data_train[var_name_train].values
                
                
#                 X_train = X_train[:,:n_components]
#                 print(X_train.shape)
                
#                 regression = TorchRidgeGCV(
#                     alphas=ALPHA_RANGE,
#                     fit_intercept=True,
#                     scale_X=False,
#                     scoring='pearsonr',
#                     store_cv_values=False,
#                     alpha_per_target=False,
#                     device=device)
                
#                 regression.fit(X_train, y_train)
#                 best_alpha = float(regression.alpha_)
#                 print('best alpha:',best_alpha)
                
                
#                 ids_test, neural_data_test, var_name_test = load_nsd_data(mode ='shared',
#                                                                           subject = subject,
#                                                                           region = region)           
#                 X_test = filter_activations(data = activations_data, ids = ids_test)                
                
#                 ### NORMALIZE #####
#                 X_test = normalize(X_test, X_min, X_max, use_min_max=True)
#                 X_test = np.nan_to_num(X_test)
#                 #################################
                
#                 y_test = neural_data_test[var_name_test].values                   
                
#                 X_test = X_test[:,:n_components]
#                 print(X_test.shape)
                
#                 y_true, y_predicted = regression_shared_unshared(x_train=X_train,
#                                                                  x_test=X_test,
#                                                                  y_train=y_train,
#                                                                  y_test=y_test,
#                                                                  model= Ridge(alpha=best_alpha))
                
                
#                 file_path = Path(PREDS_PATH) / f'{activations_identifier}_principal_components={n_components}_{region}_{subject}.pkl'
#                 if not file_path.exists():
#                     print('saving ',file_path)
#                     with open(file_path, 'wb') as file:
#                         pickle.dump(y_predicted, file)
#                 else:
#                     print(file_path, ' exists')

    
#             return             


def nsd_scorer(activations_identifier: str, 
               region: str,
              device: str):
    

        activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier), engine='h5netcdf')  


        for subject in tqdm(range(8)):

            
            file_path = Path(PREDS_PATH) / f'{activations_identifier}_{region}_{subject}.pkl'
            if not file_path.exists():
                print('saving ',file_path)
                
                ids_train, neural_data_train, var_name_train = load_nsd_data(mode ='unshared',
                                                        subject = subject,
                                                        region = region)
            
                X_train = filter_activations(data = activations_data, ids = ids_train)  
                
                X_train, X_min, X_max = normalize(X_train) # normalize
                X_train = np.nan_to_num(X_train) # omit nans
                
                y_train = neural_data_train[var_name_train].values
                print('neural data shape:',y_train.shape)
                
                
                regression = TorchRidgeGCV(
                    alphas=ALPHA_RANGE,
                    fit_intercept=True,
                    scale_X=False,
                    scoring='pearsonr',
                    store_cv_values=False,
                    alpha_per_target=False,
                    device=device)
                
                regression.fit(X_train, y_train)
                best_alpha = float(regression.alpha_)
                print('best alpha:',best_alpha)
                
                
                ids_test, neural_data_test, var_name_test = load_nsd_data(mode ='shared',
                                                                          subject = subject,
                                                                          region = region)           
                X_test = filter_activations(data = activations_data, ids = ids_test)                
                
                X_test = normalize(X_test, X_min, X_max, use_min_max=True) # normalize
                X_test = np.nan_to_num(X_test) # omit nans
                
                y_test = neural_data_test[var_name_test].values                   
                
                y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                                 x_test=X_test,
                                                                 y_train=y_train,
                                                                 y_test=y_test,
                                                                 model= Ridge(alpha=best_alpha))
                
    
                with open(file_path, 'wb') as file:
                        pickle.dump(y_predicted, file)
            else:
                print(file_path, ' exists')
                
        return            
            







def get_best_model_layer(activations_identifier, region, device):
    
        best_score = 0
        
        for iden in activations_identifier:
            
            print('getting scores for:',iden)
            activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',iden), engine='h5netcdf')  
        
            scores = []
            alphas = [] 
            
            for subject in tqdm(range(8)):

                regression = fit_model_for_subject_roi(subject, region, activations_data, device)                
                scores.append(regression.score_.mean())
                alphas.append(float(regression.alpha_))
            
            mean_layer_score = sum(scores)/len(scores)
            
            if  mean_layer_score > best_score:
                best_score = mean_layer_score
                best_layer = iden
                best_alphas = alphas
            print('best_layer:', best_layer, 'best_alphas:', best_alphas)
            
            gc.collect()            
        
        return best_layer, best_alphas
            
            
            
            
            

def nsd_get_best_layer_scores(activations_identifier: list, region: str, device:str):
    

        best_layer, best_alphas = get_best_model_layer(activations_identifier, region, device)            
        activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',best_layer), engine='h5netcdf')  
        
        for subject in tqdm(range(8)):

            file_path = Path(PREDS_PATH) / f'alexnet_gpool=False_dataset=naturalscenes_{region}_{subject}.pkl'
            if not file_path.exists():
                print('saving ',file_path)
                
                ids_train, neural_data_train, var_name_train = load_nsd_data(mode ='unshared',
                                                        subject = subject,
                                                        region = region)
                X_train = filter_activations(data = activations_data, ids = ids_train)       
                y_train = neural_data_train[var_name_train].values
    
    
                ids_test, neural_data_test, var_name_test = load_nsd_data(mode ='shared',
                                                                          subject = subject,
                                                                          region = region)           
    
                X_test = filter_activations(data = activations_data, ids = ids_test)               
                y_test = neural_data_test[var_name_test].values                    
                
                y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                                 x_test=X_test,
                                                                 y_train=y_train,
                                                                 y_test=y_test,
                                                                 model= Ridge(alpha=best_alphas[subject]))
                
    
                with open(file_path, 'wb') as file:
                        pickle.dump(y_predicted, file)
            else:
                print(file_path, ' exists')
            

        return        
        
        
            

                
    
    
    
def fit_model_for_subject_roi(subject:int, region:str, activations_data:xr.DataArray(), device:str):
            
            ids_train, neural_data_train, var_name_train = load_nsd_data(mode ='unshared',
                                                                         subject = subject,
                                                                         region = region)

            X_train = filter_activations(data = activations_data, ids = ids_train)       
            y_train = neural_data_train[var_name_train].values

            regression = TorchRidgeGCV(
                alphas=ALPHA_RANGE,
                fit_intercept=True,
                scale_X=False,
                scoring='pearsonr',
                store_cv_values=False,
                alpha_per_target=False,
                device=device)
            
            regression.fit(X_train, y_train)
            return regression
            
            
            
            
            
            
    
    
    
            
            
def load_nsd_data(mode: str, subject: int, region: str, return_data=True) -> torch.Tensor:
        
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
        if region not in ['general','V1','V2','V3','V4']:
            ds = filter_roi(subject=subject,roi=region)
            var_name = f'allen2021.natural_scenes.preprocessing=fithrf_GLMdenoise_RR.roi=general.z_score=session.average_across_reps=True.subject={subject}'
            
        else:
            path = os.path.join(NSD_NEURAL_DATA,f'roi={region}/preprocessed/z_score=session.average_across_reps=True/subject={subject}.nc')
            ds = xr.open_dataset(path, engine='h5netcdf')
            var_name = f'allen2021.natural_scenes.preprocessing=fithrf_GLMdenoise_RR.roi={region}.z_score=session.average_across_reps=True.subject={subject}'

        
        
        if mode == 'unshared':
                mask = ~ds.presentation.stimulus_id.isin(SHARED_IDS)
                ds = ds.sel(presentation=ds['presentation'][mask])
    

        elif mode == 'shared':
                mask = ds.presentation.stimulus_id.isin(SHARED_IDS)
                ds = ds.sel(presentation=ds['presentation'][mask])
            
            
        ids = list(ds.presentation.stimulus_id.values)
            
        if return_data:
            return ids, ds, var_name
        else:
            return ids
        
        
            
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

        return activations.values
            
        
        
        
def set_new_coord(ds):
    
    new_coord_data = np.core.defchararray.add(np.core.defchararray.add(ds.x.values.astype(str), ds.y.values.astype(str)), ds.z.values.astype(str))
    new_coord = xr.DataArray(new_coord_data, dims=['neuroid'])
    ds = ds.assign_coords(xyz=new_coord)
    
    return ds



def filter_roi(subject,roi):

    ds_source = xr.open_dataset(f'/data/rgautha1/cache/bonner-caching/neural-dimensionality/data/dataset=allen2021.natural_scenes/betas/resolution=1pt8mm/preprocessing=fithrf/z_score=True/roi={roi}/subject={subject}.nc',engine='h5netcdf')

    ds_target = xr.open_dataset(os.path.join(NSD_NEURAL_DATA,f'roi=general/preprocessed/z_score=session.average_across_reps=True/subject={subject}.nc'), engine='h5netcdf')

    ds_source = set_new_coord(ds_source)
    ds_target = set_new_coord(ds_target)

    source_ids = ds_source['xyz'].values
    mask = ds_target['xyz'].isin(source_ids)
    ds_target = ds_target.sel(neuroid=ds_target['neuroid'][mask])
    
    return ds_target


