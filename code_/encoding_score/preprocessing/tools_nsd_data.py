import xarray as xr
import numpy as np


def groupby_reset(
    x: xr.DataArray,
    *,
    groupby_coord: str,
    groupby_dim,
) -> xr.DataArray:
    return (
        x.reset_index(groupby_coord)
        .rename({groupby_coord: groupby_dim})
        .assign_coords({groupby_coord: (groupby_dim, x[groupby_coord].data)})
        .drop_vars(groupby_dim)
    )
    
    
def z_score_betas_within_sessions(betas: xr.DataArray) -> xr.DataArray:
    def z_score(betas: xr.DataArray) -> xr.DataArray:
        mean = betas.mean("presentation")
        std = betas.std("presentation")
        return (betas - mean) / std

    return (
        betas.load()
        .groupby("session")
        .map(func=z_score, shortcut=True)
        .assign_attrs(betas.attrs)
        .rename(betas.name)
    )


def z_score_betas_within_runs(betas: xr.DataArray) -> xr.DataArray:
    # even-numbered trials (i.e. Python indices 1, 3, 5, ...) had 62 trials
    # odd-numbered trials (i.e. Python indices 0, 2, 4, ...) had 63 trials
    n_runs_per_session = 12
    n_sessions = len(np.unique(betas["session"]))
    run_id = []
    for i_session in range(n_sessions):
        for i_run in range(n_runs_per_session):
            n_trials = 63 if i_run % 2 == 0 else 62
            run_id.extend([i_run + i_session * n_runs_per_session] * n_trials)
    betas["run_id"] = ("presentation", run_id)

    def z_score(betas: xr.DataArray) -> xr.DataArray:
        mean = betas.mean("presentation")
        std = betas.std("presentation")
        return (betas - mean) / std

    return (
        betas.load()
        .groupby("run_id")
        .map(func=z_score, shortcut=True)
        .assign_attrs(betas.attrs)
        .rename(betas.name)
    )


def average_betas_across_reps(betas: xr.DataArray) -> xr.DataArray:
    """Average the provided betas across repetitions of stimuli.

    Args:
        betas: betas

    Returns:
        averaged betas
    """
    return groupby_reset(
        betas.load()
        .groupby("stimulus")
        .mean()
        .assign_attrs(betas.attrs)
        .rename(betas.name),
        groupby_coord="stimulus",
        groupby_dim="presentation",
    ).transpose("presentation", "neuroid")