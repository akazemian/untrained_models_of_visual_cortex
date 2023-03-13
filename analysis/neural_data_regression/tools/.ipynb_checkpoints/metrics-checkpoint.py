import xarray as xr
import numpy as np
import torch
import os
from .regression import *
from sklearn.linear_model import Ridge



activations_path = '/data/atlas/activations'
neural_data_path = '/data/atlas/neural_data'
model_scores_path = '/data/atlas/model_scores'

def get_rvalues(identifier,dataset,regions,ridge_alpha):
    '''getting pearson r values between a models predictions and actual neural responses for a specidic dataset.''' 

    regression_model = Ridge(alpha=ridge_alpha)
    dataset_list = ['object2vec','naturalscenes','dicarlo.MajajHong2015public.','movshon.FreemanZiemba2013public.']
    if os.path.exists(os.path.join(model_scores_path,f'{identifier}_{regression_model}')):
        print(f'model scores are already saved in {model_scores_path} as {identifier}_{regression_model}')

    else:

        print('obtaining model scores...')
        assert dataset in dataset_list, f"the dataset should be one of {dataset_list}"

        if dataset== 'object2vec':
            num_subjects = 4
        elif dataset == 'naturalscenes':
            num_subjects = 8
        else:
            num_subjects = 1

        activations_data = xr.open_dataset(os.path.join(activations_path,identifier))    
        features = torch.Tensor(activations_data['x'].values)

        ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                    coords={'subject': (['r_values'], []),
                                            'region': (['r_values'], [])
                                             })

        for region in regions:   

            print('region: ',region)
            for subject in range(num_subjects):

                    print('subject: ',subject)

                    if num_subjects == 1:
                        data = xr.open_dataset(os.path.join(neural_data_path,dataset,f'REGION_{region}'))
                    else:
                        data = xr.open_dataset(os.path.join(neural_data_path,dataset,f'SUBJECT_{subject}_REGION_{region}'))
                    responses = torch.Tensor(data['x'].values)

                    y_true, y_predicted = regression_cv_concatenated(x=features,y=responses,model=regression_model)
                    r = pearson_r(y_true,y_predicted)

                    subject_list = [subject for i in range(len(r))]
                    region_list = [region for i in range(len(r))]

                    ds_tmp = xr.Dataset(data_vars=dict(r_value=(["r_values"], r)),
                                                coords={'subject': (['r_values'], subject_list),
                                                        'region': (['r_values'], region_list)
                                                         })
                    ds = xr.concat([ds,ds_tmp],dim='r_values')

        ds.to_netcdf(os.path.join(model_scores_path,f'{identifier}_{regression_model}'))
        print(f'model scores are now saved in {model_scores_path} as {identifier}_{regression_model}')
        return