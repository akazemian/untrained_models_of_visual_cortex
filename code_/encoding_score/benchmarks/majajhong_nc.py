import xarray as xr
import numpy as np
from scipy.stats import pearsonr
import xarray as xr
import pandas as pd
import pickle
import os

from config import MAJAJ_FULL_DATA, RESULTS


def split_half_reliability_across_stimuli(
    ds,
    var_name: str = None,
    image_dim: str = "presentation",
    image_coord: str = "image_id",
    random_state: int = None
) -> np.ndarray:
    """
    Given an xarray.Dataset or DataArray with dims
    ('neuroid', 'presentation', [maybe 'time_bin']), compute
    split‐half reliability per neuroid by:
    
      1. For each stimulus (unique image_id):
         • randomly split its repeats into two halves
         • compute mean response of each half (averaging over repeats 
           and over any extra dims like time_bin)
      2. Build two arrays half1, half2 of shape (n_stimuli, n_neuroids)
      3. For each neuroid i, compute Pearsonr(half1[:,i], half2[:,i])
    
    Returns
    -------
    reliabilities : np.ndarray, shape (n_neuroids,)
        The split‐half Pearson r for each neuroid.
    """
    # 1) extract the DataArray
    if isinstance(ds, xr.Dataset):
        if var_name is None:
            if len(ds.data_vars) != 1:
                raise ValueError(
                    "Dataset has multiple data_vars; please pass the one you want via var_name"
                )
            da = next(iter(ds.data_vars.values()))
        else:
            da = ds[var_name]
    else:
        da = ds

    # 2) setup
    rng = np.random.default_rng(random_state)
    stim_ids = np.unique(da[image_coord].values)
    n_stim = len(stim_ids)
    n_neuroids = da.sizes["neuroid"]

    half1 = np.full((n_stim, n_neuroids), np.nan, dtype=float)
    half2 = np.full((n_stim, n_neuroids), np.nan, dtype=float)

    # 3) loop over stimuli
    for si, sid in enumerate(stim_ids):
        sub = da.where(da[image_coord] == sid, drop=True)
        n_rep = sub.sizes[image_dim]
        if n_rep < 2:
            continue

        perm = rng.permutation(n_rep)
        h = n_rep // 2
        idx1, idx2 = perm[:h], perm[h:]

        # mean across repeats
        m1 = sub.isel({image_dim: idx1}).mean(dim=image_dim)
        m2 = sub.isel({image_dim: idx2}).mean(dim=image_dim)

        # extract numpy arrays (shape (neuroid,))
        arr1 = m1.values.squeeze()   # → shape (n_neuroids,)
        arr2 = m2.values.squeeze()

        half1[si, :] = arr1
        half2[si, :] = arr2

    # 4) compute Pearson r across stimuli for each neuroid
    reliabilities = np.full(n_neuroids, np.nan, dtype=float)
    for neu in range(n_neuroids):
        x = half1[:, neu]
        y = half2[:, neu]
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() >= 2:
            reliabilities[neu] = pearsonr(x[mask], y[mask])[0]

    return reliabilities


def get_majajhong_nc(region):
    
    ds = xr.open_dataset(MAJAJ_FULL_DATA)
    ds_chabo = ds.where((ds["animal"] == "Chabo") & (ds["region"] == region), drop=True)
    ds_tito  = ds.where((ds["animal"] == "Tito")  & (ds["region"] == region), drop=True)

    # Compute split‐half reliability
    nc_chabo = split_half_reliability_across_stimuli(ds_chabo)
    nc_tito  = split_half_reliability_across_stimuli(ds_tito)

    # Apply Spearman–Brown correction
    cc_chabo = 2 * nc_chabo / (1 + nc_chabo)
    cc_tito  = 2 * nc_tito  / (1 + nc_tito)

    # Average across all neuroids and both subjects
    combined = np.concatenate([cc_chabo, cc_tito])
    mean_score= np.nanmean(np.sqrt(combined))

    rows = []

    # Append a row
    rows.append({
        "model":     "noise_ceiling",
        "features":  None,
        "pcs":       None,
        "init_type": None,
        "nl_type":   None,
        "score":     mean_score,
        "lower":     None,
        "upper":     None,
        "region":    region
    })

    # ─── Build final DataFrame ───────────────────────────────────────────────────
    df_results = pd.DataFrame(rows, columns=[
        "model", "features", "pcs", "init_type", "nl_type",
        "score", "lower", "upper", "region"
    ])

    # ─── Display or save ────────────────────────────────────────────────────────
    file_path = os.path.join(RESULTS, f'nc-results-majajhong-{region}-df.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(df_results, file)
    print(f'noise ceilings are saved in {RESULTS}')
    return


