# CAMELS to netCDF

This repo takes the work by Newman et al: 

_Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21(10), 5293–5313, doi:10.5194/hess-21-5293-2017 , 2017._ 

and converts the textfiles of the original data set found here https://gdex.ucar.edu/dataset/camels.html and converts this to netCDF.

The main purpose was to make them more accessible for eWaterCycle, using the [OpenDAP service](https://opendap.4tu.nl/thredds/catalog/data2/djht/ca13056c-c347-4a27-b320-930c2a4dd207/1/catalog.html) from the data.4tu.nl, see [doi: 10.4121/ca13056c-c347-4a27-b320-930c2a4dd207.v1](https://doi.org/10.4121/ca13056c-c347-4a27-b320-930c2a4dd207.v1) for more info. 

The netCDF files are large, so not hosted here: but are accessible via the openDAP link provided. They might be also be added in an updated version of the data set by Newman et al. 

Please cite the original author if used: 

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
