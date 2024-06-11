**Caravan** is an open community dataset of meteorological forcing data, catchment attributes, and discharge data for catchments around the world. Additionally, Caravan provides code to derive meteorological forcing data and catchment attributes from the same data sources in the cloud, making it easy for anyone to extend Caravan to new catchments. The vision of Caravan is to provide the foundation for a truly global open source community resource that will grow over time. 

**Note: This is uses a pre-release of the Caravan dataset, which is currently under revision.**

To regenerate this compressed and structured version from the original data [this script](https://gist.github.com/BSchilperoort/256751fe2ea060c50b103f72026590a2) can be used, which reads in the individual NetCDF files and combines them into one. 

In the v5 update, the original forcing for CAMELS-USA was added using the code in [this repo](https://github.com/Daafip/CAMELS-to-netcdf). These 671 basins now also include 3 additional forcing sources: `daymet`, `nldas` and `maurer`. These can be selected using the `data_source` dimension in the netCDF file.

## Dataset structure

The dataset is structured as follows:

- `camels` for CAMELS (US) 
- `camelsaus` for CAMELS-AUS
- `camelsbr` for CAMELS-BR
- `camelscl` for CAMELS-CL
- `camelsgb` for CAMELS-GB
- `hysets` for HYSETS
- `lamah` for LamaH-

The shapefiles (zipped) follow the same naming scheme. 

## Technical details

### Units of time series data

Era5: 
| Variable name                 | Description                               | Aggregation          | Unit  |
|-------------------------------|-------------------------------------------|----------------------|-------|
| snow_depth_water_equivalent   | Snow-Water Equivalent                     | Daily min, max, mean | mm    |
| surface_net_solar_radiation   | Surface net solar radiation               | Daily min, max, mean | W/m2  |
| surface_net_thermal_radiation | Surface net thermal radiation             | Daily min, max, mean | W/m2  |
| surface_pressure              | Surface pressure                          | Daily min, max, mean | kPa   |
| temperature_2m                | 2m air temperature                        | Daily min, max, mean | °C    |
| u_component_of_wind_10m       | U-component of wind at 10m                | Daily min, max, mean | m/s   |
| v_component_of_wind_10m       | V-component of wind at 10m                | Daily min, max, mean | m/s   |
| volumetric_soil_water_layer_1 | volumetric soil water layer 1 (0-7cm)     | Daily min, max, mean | m3/m3 |
| volumetric_soil_water_layer_2 | volumetric soil water layer 2 (7-28cm)    | Daily min, max, mean | m3/m3 |
| volumetric_soil_water_layer_3 | volumetric soil water layer 3 (28-100cm)  | Daily min, max, mean | m3/m3 |
| volumetric_soil_water_layer_4 | volumetric soil water layer 4 (100-289cm) | Daily min, max, mean | m3/m3 |
| total_precipitation           | Total precipitation                       | Daily sum            | mm    |
| potential_evaporation         | Potential Evapotranspiration              | Daily sum            | mm    |
| streamflow                    | Observed streamflow                       | Daily min, max, mean | mm/d  |

Maurer, NLDAS, DAYMET:
| Variable   | Description           | unit   |
|------------|-----------------------|--------|
| dayl       | daylight              | s      |
| pr         | precipitation         | mm/d   |
| srad       | mean radiation        | W/m2   |
| swe        | snow water equivalent | mm     |
| tasmax     | temperature max       | degC   |
| tasmin     | temperature min       | degC   |
| vp         | vapourpressure        | Pa     |
| evspsblpot | potential evaporation | mm/d   |
| tas        | temerature mean       | degC   |
| streamflow | observed streamflow   | mm/d   |

### Catchment attributes - Caravan

Refer to the [Caravan paper](https://www.nature.com/articles/s41597-023-01975-w) for a detailed list of all static catchment attributes. Generally, there are two different sets of catchment attributes that are shared in two different files:

1. `attributes_hydroatlas`: Attributes derived from the HydroATLAS dataset. Refer to the "BasinATLAS Catalogue" of HydroATLAS (see [here](https://www.hydrosheds.org/hydroatlas)) for an explanation of the features.
2. `attributes_caravan`: Attributes (climate indices) derived from ERA5-Land timeseries for the Caravan dataset. See Table 4 in the [Caravan paper](https://www.nature.com/articles/s41597-023-01975-w) for details.
3. `attributes_other`: Metadata information, such as station name, gauge latitude and logitude, country and area. See Table 5 in the [Caravan paper](https://www.nature.com/articles/s41597-023-01975-w) for details.

Theses have been integrated into the NetCDF files. 

### Catchment attributes - CAMELS-USA

The attributes are desribed in the NetCDF file itself, more info can also be found at the [data souce](https://gdex.ucar.edu/dataset/camels.html). The meta data can be downloaded here seperately.

### Time zones
All data in Caravan are in the local time of the corresponding catchment.
We ignore any possible Daylight Saving Time, i.e., data for a given location is always in non-DST time, regardless of the date.

### Schapefiles

Each entity has a gauge_id which corresponds to that in the NetCDF files. 

## Extend Caravan

To extend Caravan to new catchments, see the guides linked in the [GitHub repository](https://github.com/kratzert/Caravan).

## License

**Caravan & CAMELS-USA (the data) is published under the CC-BY-4.0 license.**
**The code for Caravan (under `code/` and the [GitHub repository](https://github.com/kratzert/Caravan)) is published under BSD 3-Clause.**
**The code for CAMELS-USA (the [GitHub repository](https://github.com/naddor/camels)) is published open source.**

If you use Caravan in your research, it would be appreciated to not only cite Caravan itself, but also the source datasets, to pay respect to the amount of work that was put into the creation of these datasets and that made Caravan possible in the first place.

## How to cite

Please cite the Caravan using the following reference.

```bib
@article{kratzert2023caravan,
  title={Caravan-A global community dataset for large-sample hydrology},
  author={Kratzert, Frederik and Nearing, Grey and Addor, Nans and Erickson, Tyler and Gauch, Martin and Gilon, Oren and Gudmundsson, Lukas and Hassidim, Avinatan and Klotz, Daniel and Nevo, Sella and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={61},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

Please cite the Camels using the following reference.

```bib 
@misc{CAMELS_2022,
    title={CAMELS: Catchment Attributes and MEteorology for Large-sample Studies}
    author={Newman, Andrew and Sampson, Kevin and Clark, Martyn and Bock, A. and Viger, R. J. and Blodgett, D. and Addor, N. and Mizukami, M.}
    url={https://gdex.ucar.edu/dataset/camels.html}, 
    journal={Ucar.edu}, 
    publisher={UCAR/NCAR - GDEX}, 
    year={2022}, 
    month={Jun} 
}
```

‌
