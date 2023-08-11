import xarray as xr
import numpy as np
import os

raw_data_path = '/data/shared/brainio/brain-score/assy_dicarlo_MajajHong2015_public.nc'
save_path = '/data/atlas/neural_data/majajhong'



# get shared stimulus ids
regions = ['IT','V4']
subjects = ['Tito','Chabo']
# saving each region from each subject separately 

for subject in subjects:
    for region in regions:
        
        da = xr.open_dataset(raw_data_path)
        # get data for shared images
        da = da.where(da.animal.isin(subject),drop=True)

        # get region's voxels
        da_region = da.where((da.region == region), drop=True)

        l = list(da_region.coords)
        # remove all other regions
        l.remove('image_id') # keep stimulus id
        da_region = da_region.drop(l) # drop other coords
        # get the average voxel response per image for region's voxels
        da_region = da_region.groupby('image_id').mean()
        da_region = da_region.rename({'image_id':'stimulus_id',
                                      'dicarlo.MajajHong2015.public':'x'})
        da_region.to_netcdf(os.path.join(save_path,f'SUBJECT_{subject}_REGION_{region}'))