from dask.distributed import Client
import xarray as xr
import torch
from datetime import timedelta
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
import hydra
import logging
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    # Define for loop that iterates over sets, resolutions, and variables
    # and saves each time step as a torch tensor to write to a pytorch file
    # format.
    for res in ["hr"]:
        start = timer()
        for s in ["train", "test", "validation"]:
            logging.info(f"Loading {s} {res} dataset...")
            for var in cfg.vars:
                output_path = cfg.vars[var].output_path
                with xr.open_zarr(f"{output_path}/{var}_{s}_{res}.zarr/", chunks={"time": 1000}) as ds:
                # Create parent dir if it doesn't exist for each variable
                    if not os.path.exists(f"{output_path}/{s}/{var}/{res}"):
                        logging.info(
                            f"Creating directory: {output_path}/{s}/{var}/{res}"
                        )
                        os.makedirs(f"{output_path}/{s}/{var}/{res}")

                    logging.info(f"Saving {s} {res} {var} to torch tensors...")
                    logging.info(f"Writing to {output_path}/{s}/{var}/{res}")
                    for i in tqdm(np.arange(ds.time.size), desc=f"{s} {res} {var}"):
                        arr = ds[var].transpose("time", "lat", "lon")[i, ...].values
                        x = torch.tensor(np.array(arr))
                        assert not torch.isnan(x).any(), f"NaNs found in {s} {res} {var} {i}"
                        torch.save(x, f"{output_path}/{s}/{var}/{res}/{var}_{i}.pt")
            end = timer()
            logging.info(f"Finished {res} dataset in {timedelta(seconds=end-start)}")
    for res in ["lr"]:
        start = timer()
        for s in ["train", "test", "validation"]:
            logging.info(f"Loading {s} {res} dataset...")
            for var in ["uas", "vas"]:
                output_path = cfg.vars[var].output_path
                with xr.open_zarr(f"{output_path}/{var}_{s}_{res}.zarr/", chunks={"time": 1000}) as ds:
                # Create parent dir if it doesn't exist for each variable
                    if not os.path.exists(f"{output_path}/{s}/{var}/{res}"):
                        logging.info(
                            f"Creating directory: {output_path}/{s}/{var}/{res}"
                        )
                        os.makedirs(f"{output_path}/{s}/{var}/{res}")

                    logging.info(f"Saving {s} {res} {var} to torch tensors...")
                    logging.info(f"Writing to {output_path}/{s}/{var}/{res}")
                    for i in tqdm(np.arange(ds.time.size), desc=f"{s} {res} {var}"):
                        arr = ds[var].transpose("time", "lat", "lon")[i, ...].values
                        x = torch.tensor(np.array(arr))
                        assert not torch.isnan(x).any(), f"NaNs found in {s} {res} {var} {i}"
                        torch.save(x, f"{output_path}/{s}/{var}/{res}/{var}_{i}.pt")
        end = timer()
        logging.info(f"Finished {res} dataset in {timedelta(seconds=end-start)}")

if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    with Client(n_workers=16, threads_per_worker=2, processes=False, dashboard_address=8787, memory_limit='4GB'):
        main()
