import h5py
import netCDF4 as nc
import numpy as np
import pygrib
import xarray as xr

__all__ = ['_open_by_engine', '_array_by_engine', '_attribute_by_engine', '_pick_engine', '_check_var_in_dataset',
           '_array_to_stat_list']

NETCDF_EXTENSIONS = ('.nc', '.nc4')
GRIB_EXTENSIONS = ('.grb', '.grib', '.grib2')
HDF_EXTENSIONS = ('.h5', '.hd5', '.hdf5')
GEOTIFF_EXTENSIONS = ('.gtiff', '.tiff', 'tif')


def _open_by_engine(path: str, engine: str = None, backend_kwargs: dict = None) -> np.array:
    if engine is None:
        engine = _pick_engine(path)
    if backend_kwargs is None:
        backend_kwargs = dict()
    if engine == 'xarray':
        return xr.open_dataset(path, backend_kwargs=backend_kwargs)
    elif engine == 'netcdf4':
        return nc.Dataset(path, 'r')
    elif engine == 'cfgrib':
        return xr.open_dataset(path, engine='cfgrib', backend_kwargs=backend_kwargs)
    elif engine == 'pygrib':
        a = pygrib.open(path)
        return a.read()
    elif engine == 'h5py':
        return h5py.File(path, 'r')
    elif engine == 'rasterio':
        return xr.open_rasterio(path)
    else:
        raise ValueError(f'Unable to open file, unsupported engine: {engine}')


def _array_by_engine(open_file, var: str or int, h5_group: str = None):
    if isinstance(open_file, xr.Dataset):  # xarray, cfgrib
        return open_file[var].data
    elif isinstance(open_file, xr.DataArray):  # rasterio
        if isinstance(var, int):
            return open_file.data
        return open_file[var].data
    elif isinstance(open_file, nc.Dataset):  # netcdf4
        return open_file[var][:]
    elif isinstance(open_file, list):  # pygrib
        return open_file[var].values
    elif isinstance(open_file, h5py.File) or isinstance(open_file, h5py.Dataset):  # h5py
        if h5_group is not None:
            open_file = open_file[h5_group]
        return open_file[var][:]  # might need to use [...] for string data
    else:
        raise ValueError(f'Unrecognized opened file dataset: {type(open_file)}')


def _attribute_by_engine(open_file, var: str, attribute: str, h5_group: str = None):
    if isinstance(open_file, xr.Dataset) or isinstance(open_file, xr.DataArray):  # xarray, cfgrib, rasterio
        return open_file[var].attrs[attribute]
    elif isinstance(open_file, nc.Dataset):  # netcdf4
        return open_file[var].getncattr(attribute)
    elif isinstance(open_file, list):  # pygrib
        return open_file[var][attribute]
    elif isinstance(open_file, h5py.File) or isinstance(open_file, h5py.Dataset):  # h5py
        if h5_group is not None:
            open_file = open_file[h5_group]
        return open_file[var].attrs[attribute].decode('UTF-8')
    else:
        raise ValueError(f'Unrecognized opened file dataset: {type(open_file)}')


def _pick_engine(path: str) -> str:
    if path.startswith('http'):  # reading from opendap
        return 'xarray'
    if any(path.endswith(i) for i in NETCDF_EXTENSIONS):
        return 'netcdf4'
    if any(path.endswith(i) for i in GRIB_EXTENSIONS):
        return 'cfgrib'
    elif any(path.endswith(i) for i in HDF_EXTENSIONS):
        return 'h5py'
    if any(path.endswith(i) for i in GEOTIFF_EXTENSIONS):
        return 'rasterio'
    else:
        raise ValueError(f'File name does not match known files extensions, engine could not be guessed: {path}')


def _check_var_in_dataset(open_file, var, h5_group):
    if isinstance(open_file, xr.Dataset) or isinstance(open_file, nc.Dataset):  # xarray, netcdf4
        return bool(var in open_file.variables)
    elif isinstance(open_file, list):  # pygrib comes as lists of messages
        return bool(var <= len(open_file))
    elif isinstance(open_file, h5py.File) or isinstance(open_file, h5py.Dataset):  # h5py
        if h5_group is not None:
            open_file = open_file[h5_group]
        return bool(var in open_file.keys())
    elif isinstance(open_file, xr.DataArray):
        return bool(var <= open_file.band.shape[0])
    else:
        raise ValueError(f'Unrecognized opened file dataset: {type(open_file)}')


def _array_to_stat_list(array: np.array, statistic: str) -> list:
    list_of_stats = []
    # add the results to the lists of values and times
    if array.ndim == 1 or array.ndim == 2:
        if statistic == 'mean':
            list_of_stats.append(np.nanmean(array))
        elif statistic == 'median':
            list_of_stats.append(np.nanmedian(array))
        elif statistic == 'max':
            list_of_stats.append(np.nanmax(array))
        elif statistic == 'min':
            list_of_stats.append(np.nanmin(array))
        elif statistic == 'sum':
            list_of_stats.append(np.nansum(array))
        elif statistic == 'std':
            list_of_stats.append(np.nanstd(array))
        elif '%' in statistic:
            list_of_stats.append(np.nanpercentile(array, int(statistic.replace('%', ''))))
        else:
            raise ValueError(f'Unrecognized statistic, {statistic}. Use stat_type= mean, min or max')
    elif array.ndim == 3:
        for v in array:
            list_of_stats += _array_to_stat_list(v, statistic)
    else:
        raise ValueError('Too many dimensions in the array. You probably did not mean to do stats like this')
    return list_of_stats
