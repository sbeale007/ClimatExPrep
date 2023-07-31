import logging
import os
from timeit import default_timer as timer
import hydra
from dask.distributed import Client
import multiprocessing
import xarray as xr
import numpy as np
from datetime import timedelta

import dask

from ClimatExPrep.preprocess_helpers import (
    load_grid,
    slice_time,
    crop_field,
    coarsen_lr,
    train_test_split,
    homogenize_names,
    match_longitudes,
    compute_standardization,
    write_to_zarr,
    unit_change,
    log_transform,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def start(cfg) -> None:
    cores = int(multiprocessing.cpu_count())
    print(f"Using {cores} cores")

    with Client(
        # n_workers=16,
        # threads_per_worker=2,
        processes=False,
        # memory_limit="16GB",
        dashboard_address=cfg.compute.dashboard_address,
    ):
        logging.info("Now processing the following variables:")
        logging.info(cfg.vars.keys())
        for var in cfg.vars:
            start = timer()
            # process lr data first

            logging.info(f"Loading {var} LR input dataset...")
            lr_small_path_list = cfg.vars[var].lr_small.path
            lr_big_path_list = cfg.vars[var].lr_big.path
            hr_grid_ref = cfg.hr_grid_ref_path

            lr_small = load_grid(lr_small_path_list, cfg.engine, chunks=200)
            lr_big = load_grid(lr_big_path_list, cfg.engine, chunks=200)
            hr_ref = load_grid(hr_grid_ref, cfg.engine)

            logging.info("Homogenizing dataset keys...")
            keys = {"data_vars": cfg.vars, "coords": cfg.coords, "dims": cfg.dims}
            for key_attr, config in keys.items():
                lr_small = homogenize_names(lr_small, config, key_attr)
                lr_big = homogenize_names(lr_big, config, key_attr)
                hr_ref = homogenize_names(hr_ref, config, key_attr)

            # logging.info("Matching longitudes...")
            # lr = match_longitudes(lr) if cfg.vars[var].lr.is_west_negative else lr
            # hr_ref = match_longitudes(hr_ref)

            logging.info("Slicing time dimension...")
            lr_small = slice_time(lr_small, cfg.time.full.start, cfg.time.full.end)
            lr_big = slice_time(lr_big, cfg.time.full.start, cfg.time.full.end)

            # Crop the field to the given size.
            # logging.info("Cropping field...")
            # lr = crop_field(lr, cfg.spatial.scale_factor, cfg.spatial.x, cfg.spatial.y)
            # lr = lr.drop(["lat", "lon"])

            # Coarsen the low resolution dataset.
            if var != "MAPSTA":
                logging.info("Coarsening low resolution dataset...")
                lr_small = coarsen_lr(lr_small, cfg.spatial.scale_factor_small)
                lr_big = coarsen_lr(lr_big, cfg.spatial.scale_factor_big)

            # Train test split
            logging.info("Splitting dataset...")
            train_lr_small, test_lr_small = train_test_split(lr_small, cfg.time.test_years)
            test_lr_small, val_lr_small = train_test_split(test_lr_small, cfg.time.validation_years)

            train_lr_big, test_lr_big = train_test_split(lr_big, cfg.time.test_years)
            test_lr_big, val_lr_big = train_test_split(test_lr_big, cfg.time.validation_years)

            # Standardize the dataset.
            logging.info("Standardizing dataset...")
            if cfg.vars[var].standardize:
                logging.info(f"Standardizing {var}...")
                train_lr_small = compute_standardization(train_lr_small, var)
                test_lr_small = compute_standardization(test_lr_small, var, train_lr_small)
                val_lr_small  = compute_standardization(val_lr_small, var, train_lr_small)

                train_lr_big = compute_standardization(train_lr_big, var)
                test_lr_big = compute_standardization(test_lr_big, var, train_lr_big)
                val_lr_big  = compute_standardization(val_lr_big, var, train_lr_big)

            # Write the output to disk.
            logging.info("Writing smaller test output...")
            write_to_zarr(test_lr_small, f"{cfg.vars[var].output_path}/{var}_test_lr_small")
            logging.info("Writing smaller train output...")
            write_to_zarr(train_lr_small, f"{cfg.vars[var].output_path}/{var}_train_lr_small")
            logging.info("Writing smaller validation output...")
            write_to_zarr(val_lr_small, f"{cfg.vars[var].output_path}/{var}_validation_lr_small")

            logging.info("Writing larger test output...")
            write_to_zarr(test_lr_big, f"{cfg.vars[var].output_path}/{var}_test_lr_big")
            logging.info("Writing larger train output...")
            write_to_zarr(train_lr_big, f"{cfg.vars[var].output_path}/{var}_train_lr_big")
            logging.info("Writing larger validation output...")
            write_to_zarr(val_lr_big, f"{cfg.vars[var].output_path}/{var}_validation_lr_big")

            end = timer()
            logging.info("Done LR!")
            logging.info(f"Time elapsed: {timedelta(seconds=end-start)}")