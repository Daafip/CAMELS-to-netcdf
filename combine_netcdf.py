import numpy as np
from pathlib import Path, PosixPath
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr

def main():
    files_daymet = list((Path.cwd() / "Output" / "daymet").glob("*nc"))
    files_maurer = list((Path.cwd() / "Output" / "maurer").glob("*nc"))
    files_nldas = list((Path.cwd() / "Output" / "nldas").glob("*nc"))
    file_lst = []
    names = files_daymet + files_maurer + files_nldas
    for file in names:
        basin_id = file.name[19:27]
        ds = xr.open_dataset(file)
        ds = ds.expand_dims(dim={'basin_id':np.array([(str('camels_'+ basin_id)).encode()],dtype=np.dtype('|S64'))})
        file_lst.append(ds)
    
    ds = xr.merge(file_lst)
    ds.to_netcdf("combined.nc")

if __name__ == "__main__":
    main()