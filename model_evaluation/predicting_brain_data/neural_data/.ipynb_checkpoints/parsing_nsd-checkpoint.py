import xarray as xr
import numpy as np
import os
from NSD_parsing_functions import *

NSD_raw_path = '/data/shared/brainio/bonner-datasets/allen2021.natural_scenes.1pt8mm.fithrf_GLMdenoise_RR.nc'
NSD_local_path = '/data/atlas/neural_data/'
zscored_path = os.path.join(NSD_local_path,'naturalscenes_zscored')
processed_zscored_path = os.path.join(NSD_local_path,'naturalscenes_zscored_processed')

#rois = ['V1','V2','V3','V4']
rois = ["general"]

for i in range(8):
    for roi in rois:
        betas = open_betas_by_roi(
        filepath=NSD_raw_path,
        subject=i,
        roi=roi).load()
        betas.to_netcdf(os.path.join(zscored_path, f"subject_{i}_{roi}.nc"))
    print(i)


subjects = []
for i in range(8):
    subjects.append(open_subject_assembly(subject= i))
shared_ids = compute_shared_stimulus_ids(subjects)



for i in range(8):
    
    for roi in rois:
        
        da = xr.open_dataset(os.path.join(zscored_path, f"subject_{i}_{roi}.nc"))
        da = z_score_betas_within_sessions(da,da)
        da = average_betas_across_reps(da)
        
        unshared = filter_betas_by_stimulus_id(da, stimulus_ids=shared_ids, exclude = True) # z_score before splitting shared and unshared
        unshared = unshared.drop(['x','y','z']).rename({'betas':'x'})
        unshared.to_netcdf(os.path.join(processed_zscored_path, f"subject_{i}_{roi}_unshared_new.nc")) 

        shared = filter_betas_by_stimulus_id(da, stimulus_ids=shared_ids, exclude = False) # z_score before splitting shared and unshared
        shared = shared.drop(['x','y','z']).rename({'betas':'x'})
        shared.to_netcdf(os.path.join(processed_zscored_path, f"subject_{i}_{roi}_shared_new.nc")) 
        
        print(i, roi)
