import xarray as xr
import fnmatch
import xesmf as xe
import pandas as pd
import numpy as np
import glob
import os


def remove_nan(dataset, var):
    print("averaging and removing nan for:", var)

    for i in range(0, len(dataset.time)):
        avg = np.nanmean(dataset.variables[var][i])
        dataset.variables[var][i] = dataset.variables[var][i].fillna(avg)


def select_var(ds: xr.Dataset, var) -> xr.Dataset:
    """
    Filter the dataset, by leaving only the desired variables in it. The original dataset
    information, including original coordinates, is preserved.

    :param ds: The dataset from which to perform selection.
    :param var: One or more variable names to select and preserve in the dataset. \
    All of these are valid 'var_name' 'var_name1,var_name2,var_name3' ['var_name1', 'var_name2']. \
    One can also use wildcards when doing the selection. E.g., choosing 'var_name*' for selection \
    will select all variables that start with 'var_name'. This can be used to select variables \
    along with their auxiliary variables, to select all uncertainty variables, and so on.
    :return: A filtered dataset
    """
    if not var:
        return ds

    var_names = var
    dropped_var_names = list(ds.data_vars.keys())

    for pattern in var_names:
        keep = fnmatch.filter(dropped_var_names, pattern)
        for name in keep:
            dropped_var_names.remove(name)

    return ds.drop(dropped_var_names)


for file in glob.glob("/Volumes/LaCie/historical/*hist.nc"):
    print('opening dataset %s' % file)
    base = os.path.basename(file)
    ds = xr.open_dataset(file)

    print("creating new datasets with only needed variable")

    hs_only = select_var(ds, ['hs'])
    uas_only = select_var(ds, ['uwnd'])
    vas_only = select_var(ds, ['vwnd'])
    ice_only = select_var(ds, ['ice'])
    t0m1_only = select_var(ds, ['t0m1'])
    dir_only = select_var(ds, ['dir'])
    fp_only = select_var(ds, ['fp'])
    MAPSTA_only = select_var(ds, ['MAPSTA'])

    # removes all nan values from files

    remove_nan(hs_only, 'hs')
    remove_nan(uas_only, 'uwnd')
    remove_nan(vas_only, 'vwnd')
    remove_nan(ice_only, 'ice')
    remove_nan(t0m1_only, 't0m1')
    remove_nan(dir_only, 'dir')
    remove_nan(fp_only, 'fp')

    # duplicates MAPSTA variable for every time step
    time_coord = MAPSTA_only.time
    MAPSTA_drop_time = MAPSTA_only.drop("time")
    MAPSTA_dup = MAPSTA_drop_time.expand_dims(time=time_coord)

    new_hs_filename = '/Volumes/LaCie/nonan/hs/hs_nonan_%s' % base
    new_uas_filename = '/Volumes/LaCie/nonan/uas/uas_nonan_%s' % base
    new_vas_filename = '/Volumes/LaCie/nonan/vas/vas_nonan_%s' % base
    new_ice_filename = '/Volumes/LaCie/nonan/ice/ice_nonan_%s' % base
    new_t0m1_filename = '/Volumes/LaCie/nonan/t0m1/t0m1_nonan_%s' % base
    new_dir_filename = '/Volumes/LaCie/nonan/dir/dir_nonan_%s' % base
    new_fp_filename = '/Volumes/LaCie/nonan/fp/fp_nonan_%s' % base
    new_MAPSTA_filename = '/Volumes/LaCie/nonan/MAPSTA/MAPSTA_nonan_%s' % base

    filenames = [new_hs_filename, new_vas_filename, new_uas_filename, new_ice_filename, new_t0m1_filename,
                 new_dir_filename, new_fp_filename, new_MAPSTA_filename]
    new_ds = [hs_only, vas_only, uas_only, ice_only, t0m1_only, dir_only, fp_only, MAPSTA_dup]

    for f, d in zip(filenames, new_ds):
        print('saving to ', f)
        d.load().to_netcdf(path=f)
        print('finished saving')

    print('deleting %s' % file)
    os.remove(file)
    print('finished deleting file')