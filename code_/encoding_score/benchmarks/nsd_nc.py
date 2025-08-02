import xarray as xr
import numpy as np
from scipy.stats import pearsonr
import xarray as xr
import pandas as pd
import pickle
import os

from config import NSD_NC_DATA, RESULTS

def parse_k(rep_str: str) -> int:
    # rep_str is like "n_repetitions=2.0-1" or "n_repetitions=3.1-2"
    # split on '=' then take the part before the dot
    k_part = rep_str.split('=')[1]       # "2.0-1" or "3.0-1"
    k = int(float(k_part.split('-')[0])) # float->int gives 2 or 3
    return k

def get_nsd_nc(region):
    scores = []
    for subject in range(8):
        nc_ds = xr.open_dataset(os.path.join(NSD_NC_DATA, f'roi={region}/all_stimuli.subject={subject}.nc'))
        k_vals = [parse_k(r) for r in nc_ds.repetition.values]
        # make an xarray so it will broadcast along neuroid
        k_da   = xr.DataArray(
            k_vals,
            dims = ['repetition'],
            coords = {'repetition': nc_ds.repetition}
        )
        nc_corrected = (k_da * nc_ds) / (1 + (k_da - 1) * nc_ds)
        mean_scores= np.nanmean(np.sqrt(nc_corrected.correlation.values))
        scores.append(mean_scores)    

        # nc_final = nc_corrected.mean(dim='repetition')
        # scores.append(nc_final.correlation.values.mean())
    
    mean_score = sum(scores)/len(scores)
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
    file_path = os.path.join(RESULTS, f'nc-results-naturalscenes-{region}-df.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(df_results, file)
    print(f'noise ceilings are saved in {RESULTS}')
    return
