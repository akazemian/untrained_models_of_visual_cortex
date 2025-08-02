import os
import sys
import xarray as xr
import numpy as np
# Ensure the root directory is correctly set up in the environment variables
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)

# Import the config after adding ROOT to the path
from config import NSD_NEURAL_DATA
from model_evaluation.predicting_brain_data.preprocessing.tools_nsd_data import average_betas_across_reps, z_score_betas_within_sessions

regions = ['early visual stream', 'midventral visual stream', 'ventral visual stream']

for region in regions:
    
    region_path = os.path.join(NSD_NEURAL_DATA, region)
    if not os.path.exists(region_path):
        os.mkdir(region_path)
        
    if not os.path.exists(os.path.join(region_path,'preprocessed_new_1')):
        os.mkdir(os.path.join(region_path,'preprocessed_new_1'))

    if not os.path.exists(os.path.join(region_path,'raw')):
        os.mkdir(os.path.join(region_path,'raw'))
        
    for subject in range(8):
        
        #raw_data = xr.open_dataset(f'/data/shared/for_atlas_3/roi={region}/subject={subject}.nc')

        #preprocessed_data = (average_betas_across_reps(z_score_betas_within_sessions(raw_data.beta))).to_dataset()
        
        #raw_data.to_netcdf(os.path.join(NSD_NEURAL_DATA, region, 'raw',f'subject={subject}.nc'))
        preprocessed_data = xr.open_dataset(os.path.join(NSD_NEURAL_DATA, region, 'preprocessed',f'subject={subject}.nc'),engine='h5netcdf')
        ids = list(preprocessed_data.stimulus.values)
        formatted_ids = [f"image{num:05d}" for num in ids]
        preprocessed_data = preprocessed_data.assign_coords(stimulus=('presentation', formatted_ids))

        preprocessed_data.to_netcdf(os.path.join(NSD_NEURAL_DATA, region, 'preprocessed_new_1',f'subject={subject}.nc'),engine='h5netcdf')
