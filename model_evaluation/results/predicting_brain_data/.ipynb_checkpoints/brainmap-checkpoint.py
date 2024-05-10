"""
The following brain surface plot functions are based on nilearn's libraries and adapted from Adapted from https://github.com/cvnlab/nsdcode 
"""


import xarray as xr
import nilearn.plotting
import nibabel as nib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
import copy
import numpy as np
from loguru import logger
import boto3
from collections.abc import Collection
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)

IDENTIFIER = "allen2021.natural_scenes"
BUCKET_NAME = "natural-scenes-dataset"
CACHE_PATH = f'/data/atlas/brain_map/{IDENTIFIER}'
os.environ['AWS_SHARED_CREDENTIALS_FILE'] = '/data/shared/aws/credentials'



def to_dataarray(
    filepath: Path,
    *,
    dims = ("x", "y", "z"),
    flatten = {"neuroid": ("x", "y", "z")},
) -> xr.DataArray:
    """Format an NII file as a DataArray.

    Args:
        filepath: path to NII file [must be a 3D array (x, y, z) or 4D array e.g. (presentation, x, y, z)]
        flatten: whether to flatten all the spatial dimensions into a "neuroid" dimension

    Returns:
        brain volume
    """
    nii = nib.load(filepath).get_fdata()

    nii = xr.DataArray(
        data=nii,
        dims=dims,
    )
    nii = nii.assign_coords(
        {dim: (dim, np.arange(nii.sizes[dim], dtype=np.uint8)) for dim in dims}
    )
    if flatten:
        assert len(flatten) == 1
        dim, dims_to_flatten = copy.deepcopy(flatten).popitem()
        nii = nii.stack({dim: dims_to_flatten}, create_index=False)
    return nii





def load_brain_mask(*, subject: int, resolution: str):
    """Load and format a Boolean brain mask for the functional data.

    Args:
        subject: subject ID
        resolution: "1pt8mm" or "1mm"

    Returns:
        Boolean brain mask
    """
    filepath = (
        Path("nsddata")
        / "ppdata"
        / f"subj{subject + 1:02}"
        / f"func{resolution}"
        / "brainmask.nii.gz"
    )
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return to_dataarray(CACHE_PATH / filepath, flatten=None).astype(bool, order="C")





def download_from_s3(
    s3_path: Path,
    *,
    bucket: str,
    local_path: Path = None,
    use_cached: bool = True,
) -> None:
    """Download file(s) from S3.

    Args:
        s3_path: path of file in S3
        bucket: S3 bucket name
        local_path: local path of file
        use_cached: use existing file or re-download, defaults to True
    """
    if local_path is None:
        local_path = s3_path
    if (not use_cached) or (not local_path.exists()):
        s3 = boto3.client("s3")
        logger.debug(f"Downloading {s3_path} from S3 bucket {bucket} to {local_path}")
        local_path.parent.mkdir(exist_ok=True, parents=True)
        with open(local_path, "wb") as f:
            s3.download_fileobj(bucket, str(s3_path), f)
    else:
        logger.debug(
            "Using previously downloaded file at"
            f" {local_path} instead of downloading {s3_path} from S3 bucket"
            f" {bucket}"
        )
        
        
        


MNI_ORIGIN = np.asarray([183 - 91, 127, 73]) - 1
MNI_RESOLUTION = 1


def load_transformation(
    subject: int, *, source_space: str, target_space: str, suffix: str
) -> np.ndarray:
    filepath = (
        Path("nsddata")
        / "ppdata"
        / f"subj{subject + 1:02}"
        / "transforms"
        / f"{source_space}-to-{target_space}{suffix}"
    )

    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    transformation = nib.load(CACHE_PATH / filepath).get_fdata()
    return transformation


def load_native_surface(
    subject: int, *, hemisphere: str, surface_type: str = "w-g.pct.mgh"
) -> Path:
    filepath = (
        Path("nsddata")
        / "freesurfer"
        / f"subj{subject + 1:02}"
        / f"surf"
        / f"{hemisphere}.{surface_type}"
    )
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return CACHE_PATH / filepath


def _interpolate(
    volume: np.ndarray, *, coordinates: np.ndarray, interpolation_type: str = "cubic"):
    """
    Wrapper for ba_interp3. Normal calls to ba_interp3 assign values to interpolation points that lie outside the original data range. We ensure that coordinates outside the original field-of-view (i.e. if the value along a dimension is less than 1 or greater than the number of voxels in the original volume along that dimension) are returned as NaN and coordinates that have any NaNs are returned as NaN.

    Args:
        volume: 3D matrix (can be complex-valued)
        coordinates: (3, N) matrix coordinates to interpolate at
        interpolation_type: "nearest", "linear", or "cubic"
    """
    
    if interpolation_type== "cubic":
        order = 3
    elif interpolation_type== "linear":
        order = 1
    elif interpolation_type== "nearest":
        order = 0
    else:
        raise ValueError("interpolation method not implemented")

    # bad locations must get set to NaN
    bad = np.any(np.isinf(coordinates), axis=0)
    coordinates[:, bad] = -1

    # out of range must become NaN, too
    bad = np.any(
        np.c_[
            bad,
            coordinates[0, :] < 0,
            coordinates[0, :] > volume.shape[0] - 1,
            coordinates[1, :] < 0,
            coordinates[1, :] > volume.shape[1] - 1,
            coordinates[2, :] < 0,
            coordinates[2, :] > volume.shape[2] - 1,
        ],
        axis=1,
    ).astype(bool)

    transformed_data = map_coordinates(
        np.nan_to_num(volume).astype(np.float64),
        coordinates,
        order=order,
        mode="nearest",
    )
    transformed_data[bad] = np.nan

    return transformed_data


def _transform(
    data: np.ndarray,
    *,
    transformation: np.ndarray,
    interpolation_type: str,
    target_type: str):
    """_summary_

    Args:
        data: data to be transformed from one space to another
        transformation: transformation matrix
        interpolation_type: passed to _interpolate
        target_type: "volume" or "surface"

    Returns:
        Transformed data
    """
    target_shape = transformation.shape[:3]

    coordinates = np.c_[
        transformation[..., 0].ravel(order="F"),
        transformation[..., 1].ravel(order="F"),
        transformation[..., 2].ravel(order="F"),
    ].T

    coordinates -= 1  # Kendrick's 1-based indexing.

    data_ = _interpolate(
        data, coordinates=coordinates, interpolation_type=interpolation_type
    )
    data_ = np.nan_to_num(data_)
    if target_type == "volume":
        data_ = data_.reshape(target_shape, order="F")

    return data_


def convert_ndarray_to_nifti1image(
    data: np.ndarray,
    *,
    resolution: float = MNI_RESOLUTION,
    origin: np.ndarray = MNI_ORIGIN,
) -> nib.Nifti1Image:
    header = nib.Nifti1Header()
    header.set_data_dtype(data.dtype)

    affine = np.diag([resolution] * 3 + [1])
    if origin is None:
        origin = (([1, 1, 1] + np.asarray(data.shape)) / 2) - 1
    affine[0, -1] = -origin[0] * resolution
    affine[1, -1] = -origin[1] * resolution
    affine[2, -1] = -origin[2] * resolution

    return nib.Nifti1Image(data, affine, header)


def transform_volume_to_mni(
    data: np.ndarray, *, subject: int, source_space: str, interpolation_type: str):
    transformation = load_transformation(
        subject=subject, source_space=source_space, target_space="MNI", suffix=".nii.gz"
    )
    transformed_data = _transform(
        data=data,
        transformation=transformation,
        target_type="volume",
        interpolation_type=interpolation_type,
    )
    return transformed_data


def transform_volume_to_native_surface(
    data: np.ndarray,
    *,
    subject,
    source_space,
    interpolation_type = "cubic",
    layers = (
        "layerB1",
        "layerB2",
        "layerB3"),
    average_across_layers= True):
    
    native_surface = {}
    for hemisphere in ("lh", "rh"):
        native_surface[hemisphere] = {}
        for layer in layers:
            transformation = load_transformation(
                subject=subject,
                source_space=f"{hemisphere}.{source_space}",
                target_space=layer,
                suffix=".mgz",
            )

            native_surface[hemisphere][layer] = _transform(
                data,
                transformation=transformation,
                target_type="surface",
                interpolation_type=interpolation_type,
            )

        if average_across_layers:
            native_surface[hemisphere] = {
                "average": np.vstack(list(native_surface[hemisphere].values())).mean(
                    axis=0
                )
            }
    return native_surface




def reshape_dataarray_to_brain(
    data: xr.DataArray, *, subject: int, resolution: str):
    
    brain_shape = load_brain_mask(subject=subject, resolution=resolution).shape
    if data.ndim == 2:
        output_shape = (data.shape[0], *brain_shape)
    else:
        output_shape = brain_shape

    output = np.full(output_shape, fill_value=np.nan)
    output[..., data["x"].values, data["y"].values, data["z"].values] = data.values
    return output





def convert_dataarray_to_nifti1image(
    data: xr.DataArray,
    *,
    subject: int,
    resolution: str,
    interpolation_type: str = "cubic",
    cond:str = None,
    scores_path:str = None,
    scores_path_2:str = None
    ):
    
    
    if cond == 'average':
        return convert_ndarray_to_nifti1image(average_across_brains(
                    scores_path = scores_path,
                    resolution=resolution,
                    interpolation_type = interpolation_type))
    
    elif cond == 'difference_of_means':
        model_1 = average_across_brains(scores_path, resolution, interpolation_type)
        model_2 = average_across_brains(scores_path_2, resolution, interpolation_type)
        return convert_ndarray_to_nifti1image(model_1 - model_2)
        
    
    elif cond == 'difference':
     
        model_1= load_model_for_brain(scores_path = scores_path, resolution = resolution, interpolation_type = interpolation_type,)
        model_2= load_model_for_brain(scores_path = scores_path_2, resolution = resolution, interpolation_type = interpolation_type,)
        return convert_ndarray_to_nifti1image(model_1 - model_2)



    else:
        return convert_ndarray_to_nifti1image(
            transform_volume_to_mni(
                data=reshape_dataarray_to_brain(
                    data=data,
                    subject=subject,
                    resolution=resolution,
                ),
                subject=subject,
                source_space=f"func{resolution[:-2]}",
                interpolation_type=interpolation_type,
            ),
        )
    



# def average_across_brains(
#     scores_path:str, 
#     resolution: str,
#     interpolation_type: str,
#     ):
    
#     s= 0 
#     data = process_data_to_plot(scores_path = scores_path, subject = s)
#     mni = transform_volume_to_mni(
#             data=reshape_dataarray_to_brain(
#                 data=data,
#                 subject=s,
#                 resolution=resolution,
#             ),
#             subject=s,
#             source_space=f"func{resolution[:-2]}",
#             interpolation_type=interpolation_type,
#         )
    
#     for s in range(1,8):
#         data = process_data_to_plot(scores_path = scores_path, subject = s)
#         mni += transform_volume_to_mni(
#             data=reshape_dataarray_to_brain(
#                 data=data,
#                 subject=s,
#                 resolution=resolution,
#             ),
#             subject=s,
#             source_space=f"func{resolution[:-2]}",
#             interpolation_type=interpolation_type,
#         )
        
#     return mni/8


# def load_model_for_brain(
#     scores_path:str, 
#     resolution: str,
#     interpolation_type: str,
#     ):
    
#     data = process_data_to_plot(scores_path = scores_path, subject = subject)
#     return transform_volume_to_mni(
#             data=reshape_dataarray_to_brain(
#                 data=data,
#                 subject=subject,
#                 resolution=resolution,
#             ),
#             subject=subject,
#             source_space=f"func{resolution[:-2]}",
#             interpolation_type=interpolation_type,
#         )



def plot_brain_map(
    data: xr.DataArray,
    *,
    subject: int,
    resolution: str = "1pt8mm",
    name:str,
    cmap='cold_hot',
    vmax=0.3,
    cond:str = None,    
    scores_path: str= None,
    scores_path_2: str= None,
    
    **kwargs):
    volume = convert_dataarray_to_nifti1image(
        data, subject=subject, resolution=resolution, 
        cond = cond, scores_path = scores_path, scores_path_2 = scores_path_2,
    )
    fig, _ = nilearn.plotting.plot_img_on_surf(
        volume,
        views=["lateral", "medial", "ventral"],
        hemispheres=["left", "right"],
        colorbar=True,
        inflate=True,
        surf_mesh='fsaverage',
        threshold=np.finfo(np.float32).resolution,
        **kwargs,
        vmax= vmax,
        cmap=cmap
    )
    fig.savefig(name,dpi=300, transparent=True)
    plt.close()
    return 




