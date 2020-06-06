import h5py
import netCDF4 as nc
import numpy as np
import xarray as xr
from PIL import TiffImagePlugin, Image

__all__ = ['_open_by_engine', '_array_by_engine', '_pick_engine', '_check_var_in_dataset', '_array_to_stat_list']


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
    elif engine == 'h5py':
        return h5py.File(path, 'r')
    elif engine in ('PIL', 'pillow'):
        return Image.open(path, 'r')
    elif engine == 'rasterio':
        return xr.open_rasterio(path)
    else:
        raise ValueError(f'Unable to open file, unsupported engine: {engine}')


def _array_by_engine(open_file, var: str, h5_group: str = None):
    if isinstance(open_file, xr.Dataset):  # xarray, cfgrib, rasterio
        return open_file[var].data
    elif isinstance(open_file, nc.Dataset):  # netcdf4
        return open_file[var][:]
    elif isinstance(open_file, h5py.File) or isinstance(open_file, h5py.Dataset):  # h5py
        if h5_group is not None:
            open_file = open_file[h5_group]
        return open_file[var][:]  # might need to use [...] for string data
    elif isinstance(open_file, TiffImagePlugin.TiffImageFile):  # geotiff
        return np.array(open_file)
    else:
        raise ValueError(f'Unrecognized opened file dataset: {type(open_file)}')


def _pick_engine(path: str) -> str:
    if path.endswith('.nc') or path.endswith('.nc4'):
        return 'netcdf4'
    elif path.endswith('.grb') or path.endswith('.grib'):
        return 'cfgrib'
    elif path.endswith('.gtiff') or path.endswith('.tiff') or path.endswith('tif'):
        return 'rasterio'
    elif path.endswith('.h5') or path.endswith('.hd5') or path.endswith('.hdf5'):
        return 'h5py'
    else:
        raise ValueError(f'File path does not match known files extensions, engine could not be guessed: {path}')


def _check_var_in_dataset(open_file, variable, h5_group):
    if isinstance(open_file, xr.Dataset) or isinstance(open_file, nc.Dataset):  # xarray, netcdf4
        return bool(variable in open_file.variables)
    elif isinstance(open_file, h5py.File) or isinstance(open_file, h5py.Dataset):  # h5py
        if h5_group is not None:
            open_file = open_file[h5_group]
        return bool(variable in open_file.keys())
    elif isinstance(open_file, TiffImagePlugin.TiffImageFile):  # geotiff
        return False
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
