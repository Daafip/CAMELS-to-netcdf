import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse

def from_camels_txt(fn: Path, 
                    basin_id: str, 
                    source: str, 
                    start_time: str, 
                    end_time: str , 
                    directory: Path, 
                    alpha:float, 
                    catchment_char: pd.Series,
                    streamflow_file_path: Path,
                    ) -> xr.Dataset:
    """Load forcing data from a txt file into an xarray dataset.

    Requirements:
        Must be in the same format as the CAMELS dataset:

        3 lines containing: lat, elevation and area.

        4th line with headers: 'Year Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)'

        Takes from the 5th line onwards with \t delimiter.

        Will convert date to pandas.Timestamp()

        Then convert from pandas to a xarray.

    Returns:
        ds: xr.Dataset
            Dataset with forcing data.
    """
    data = {}
    with open(fn, 'r') as fin:
        line_n = 0
        for line in fin:
            if line_n == 0:
                data["lat"] = float(line.strip())
            elif line_n == 1:
                data["elevation(m)"] = float(line.strip())
            elif line_n == 2:
                data["area basin(m^2)"] = float(line.strip())
            elif line_n == 3:
                header = line.strip()
            else:
                break
            line_n += 1

    headers = header.split(' ')[3:]
    headers[0] = "YYYY MM DD HH"

    # read with pandas
    df = pd.read_csv(fn, skiprows=4, delimiter="\t", names=headers)
    df.index = df.apply(lambda x: pd.Timestamp(x["YYYY MM DD HH"][:-3]), axis=1)
    df = df.drop(columns="YYYY MM DD HH")
    df.index.name = "time"

    # rename
    new_names = [item.split('(')[0] for item in list(df.columns)]
    rename_dict = dict(zip(headers[1:], new_names))
    df.rename(columns=rename_dict, inplace=True)
    rename_dict2 = {'prcp': 'pr',
                    'tmax': 'tasmax',
                    'tmin': 'tasmin'}
    df.rename(columns=rename_dict2, inplace=True)

    # add attributes
    attrs = {"title": "Basin mean forcing data",
             "history": "Created by David Haasnoot for eWatercycle using CAMELS dataset",
             "data_source": "CAMELS was compiled by A. Newman et al. `A large-sample watershed-scale hydrometeorological dataset for the contiguous USA`",
             "url_source_data": "https://dx.doi.org/10.5065/D6MW2F4D",
             "units": "daylight(s), precipitation(mm/day), mean radiation(W/m2), snow water equivalen(mm), temperature max(C), temperature min(C), temperature mean(c), vapour pressure(Pa), streamflow(mm/day)",
             'alpha': alpha, 
             }

    # add the data lines with catchment characteristics to the description
    attrs.update(data)

    ds = xr.Dataset(data_vars=df,
                    attrs=attrs)
    ds = ds.expand_dims(dim={'data_source':np.array([(source).encode()],dtype=np.dtype('|S64'))})
    ds = ds.assign_coords({'data_source':np.array([source],dtype=np.dtype('|S64'))})
    # Potential Evaporation conversion using srad & tasmin/maxs
    ds['evspsblpot'] = calc_pet(ds['srad'],
                         ds["tasmin"].values,
                         ds["tasmax"].values,
                         ds["time.dayofyear"].values,
                         alpha,
                         ds.attrs['elevation(m)'],
                         ds.attrs['lat']
                         )
    ds['tas'] = (ds["tasmin"] + ds["tasmax"]) / 2
    
    # load streamflow
    cubic_ft_to_cubic_m = 0.0283168466
    basin_area = data["area basin(m^2)"]
    new_header = ['hru_id', 'Year', 'Month', 'Day', 'Streamflow(cubic feet per second)', 'QC_flag']
    new_header_dict = dict(list(zip(range(len(new_header)), new_header)))
    df_q = pd.read_fwf(streamflow_file_path, delimiter=' ', encoding='utf-8', header=None)
    df_q = df_q.rename(columns=new_header_dict)
    df_q['Streamflow(cubic feet per second)'] = df_q['Streamflow(cubic feet per second)'].apply(
        lambda x: np.nan if x == -999.00 else x)
    df_q['Q (m3/s)'] = df_q['Streamflow(cubic feet per second)'] * cubic_ft_to_cubic_m
    df_q['Q'] = df_q['Q (m3/s)'] / basin_area * 3600 * 24 * 1000  # m3/s -> m/s ->m/d -> mm/d
    df_q.index = df_q.apply(lambda x: pd.Timestamp(f'{int(x.Year)}-{int(x.Month)}-{int(x.Day)}'), axis=1)
    df_q.index.name = "time"
    df_q.drop(columns=['Year', 'Month', 'Day', 'Streamflow(cubic feet per second)'], inplace=True)
    df_q = df_q.dropna(axis=0)

    ds['streamflow'] = df_q['Q']

    # add attrs
    for col in catchment_char.index:
        ds[col] = catchment_char[col]

    ds, ds_name = crop_ds(basin_id=basin_id, 
                              source=source, 
                              start_time=start_time, 
                              end_time=end_time,
                              directory=directory, 
                              ds=ds
                              )
    
    ds = ds.expand_dims(dim={'basin_id':np.array([(str('camels_'+ basin_id)).encode()],dtype=np.dtype('|S64'))})
    ds = ds.assign_coords({'basin_id':np.array([basin_id],dtype=np.dtype('|S64'))})
    return ds

def crop_ds(basin_id: str, source: str, start_time: str, end_time: str , directory:Path, ds: xr.Dataset):
    start = pd.Timestamp(get_time(start_time)).tz_convert(None)
    end = pd.Timestamp(get_time(end_time)).tz_convert(None)
    ds = ds.isel(time=(ds['time'].values >= start) & (ds['time'].values <= end))

    ds_name = f"basin_mean_forcing_{basin_id}_{source}.nc"
    out_dir = directory / ds_name
    if not out_dir.exists():
        ds.to_netcdf(out_dir)

    return ds, ds_name

# from ewatercycle utils
def get_time(time_iso: str) -> datetime:
    """Return a datetime in UTC.

    Convert a date string in ISO format to a datetime
    and check if it is in UTC.
    """
    time = parse(time_iso)
    if not time.tzname() == "UTC":
        raise ValueError(
            "The time is not in UTC. The ISO format for a UTC time "
            "is 'YYYY-MM-DDTHH:MM:SSZ'"
        )
    return time


def calc_pet(s_rad, t_min, t_max, doy, alpha, elev, lat) -> np.ndarray:
    """Calculates Potential Evaporation using Priestly–Taylor PET estimate, callibrated with longterm P-T trends from the camels data set (alpha).

    Parameters:
        s_rad: np.ndarray
            net radiation estimate in W/m^2. Function converts this to J/m^2/d
        t_min: np.ndarray
            daily minimum temperature (degree C)
        t_max: np.ndarray
            daily maximum temperature (degree C)
        doy: np.ndarray
            day of year: use `xt.DataArray.dt.dayofyear` - used to approximate daylight amount
        alpha: float
            factor callibrated from longterm P-T trend compensating for lack of other data.
        elev: float
            elevation in m as provided by camels
        lat: float
            latitude in degree

    Assumptions:
        G = 0 in a day: no loss to ground.

    Returns:
        pet: np.ndarray
            Array containing PET estimates in mm/day

    Reference:
        based on code from:
                kratzert et al. 2022
                `NeuralHydrology --- A Python library for Deep Learning research in hydrology,
                Frederik Kratzert and Martin Gauch and Grey Nearing and Daniel Klotz
                <https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/datautils/pet.py>`_
                https://doi.org/10.21105/joss.04050
        Who base on `allen et al. (1998) 'FOA 56' <https://appgeodb.nancy.inra.fr/biljou/pdf/Allen_FAO1998.pdf>`_ &
        `Newman et al (2015) <https://hess.copernicus.org/articles/21/5293/2017/>`_

    """
    G = 0
    LAMBDA = 2.45  # MJ/kg

    s_rad = s_rad * 0.0864  # conversion Wm-2 -> MJm-2day-1
    albedo = 0.23  # planetary albedo
    in_sw_rad = (1 - albedo) * s_rad

    # solar declination
    sol_dec = 0.409 * np.sin((2 * np.pi) / 365 * doy - 1.39)  # Equation 24 FAO-56 Allen et al. (1998)

    # Sunset hour angle
    lat = lat * (np.pi / 180)  # degree to rad
    term = -np.tan(lat) * np.tan(sol_dec)
    term[term < -1] = -1
    term[term > 1] = 1
    sha = np.arccos(term)

    # Inverse relative distance between Earth and Sun:
    ird = 1 + 0.033 * np.cos((2 * np.pi) / 365 * doy)  # Equation 23 FAO-56 Allen et al. (1998)

    # Extraterrestrial Radiation -  Equation 21 FAO-56 Allen et al. (1998)
    et_rad = ((24 * 60) / np.pi * 0.082 * ird) * (
                sha * np.sin(lat) * np.sin(sol_dec) + np.cos(lat) * np.cos(sol_dec) * np.sin(sha))

    # Clear sky radiation Equation 37 FAO-56 Allen et al. (1998)
    cs_rad = (0.75 + 2 * 10e-5 * elev) * et_rad

    # Actual vapor pressure estimated using min temperature - Equation 48 FAO-56 Allen et al. (1998
    avp = 0.611 * np.exp((17.27 * t_min) / (t_min + 237.3))

    # Net outgoing long wave radiation - Equation 49 FAO-56 Allen et al. (1998)
    term1 = ((t_max + 273.16) ** 4 + (t_min + 273.16) ** 4) / 2  # conversion in K in equation
    term2 = 0.34 - 0.14 * np.sqrt(avp)
    term3 = 1.35 * s_rad / cs_rad - 0.35
    stefan_boltzman = 4.903e-09
    out_lw_rad = stefan_boltzman * term1 * term2 * term3

    # psychrometer constant (kPa/C) - varies with altitude
    temp = (293.0 - 0.0065 * elev) / 293.0
    atm_pressure = np.power(temp, 5.26) * 101.3  # Equation 7 FAO-56 Allen et al. (1998)
    gamma = 0.000665 * atm_pressure

    # Slope of saturation vapour pressure curve Equation 13 FAO-56 Allen et al. (1998)
    t_mean = 0.5 * (t_min + t_max)
    s = 4098 * (0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3))) / ((t_mean + 237.3) ** 2)

    rn = in_sw_rad - out_lw_rad
    pet = ((alpha / LAMBDA) * s * (rn - G)) / (s + gamma)
    return pet * 0.408  # energy to evap