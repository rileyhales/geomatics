import datetime
import os
import tempfile

import geopandas
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask

from ._utils import _open_by_engine, _array_by_engine, _pick_engine, _check_var_in_dataset, _array_to_stat_list
from .data import gen_affine

__all__ = ['point', 'bounding_box', 'polygons', 'full_array_stats']
ALL_STATS = ('mean', 'median', 'max', 'min', 'sum', 'std')
ALL_ENGINES = ('xarray', 'netcdf4', 'cfgrib', 'pygrib', 'h5py', 'rasterio')


def point(files: list, var: str or int, coords: tuple, dims: tuple, t_dim: str = 'time', fill_value: int = -9999,
          time_from: str = 'var', strp: str = False, meta: str = False, ustring: str = 'detect',
          engine: str = None, h5_group: str = None, xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values at the grid cell closest to a specified point.

    Args:
        files (list): A list of absolute paths to netcdf, grib, or hdf5 files (even if len==1)
        var (str or int): The name of a variable as it is stored in the file (e.g. often 'temp' or 'T' instead of
            Temperature) or the band number if you are using grib files and you specify the engine as pygrib
        coords (tuple): A tuple of the coordinates for the location of interest in the order (x, y, z)
        dims (tuple): A tuple of the names of the (x, y, z) variables in the data files which you specified coords for.
            Defaults to common x, y, z variables names: ('lon', 'lat', 'depth')
            X dimension names are usually 'lon', 'longitude', 'x', or similar
            Y dimension names are usually 'lat', 'latitude', 'y', or similar
            Z dimension names are usually 'depth', 'elevation', 'z', or similar
        t_dim (str): Name of the time variable if it is used in the files. Default: 'time'
        fill_value (int): The value used for filling no_data spaces in the source file's array. Default: -9999
        time_from (str): How time data is stored in the files. The value of this variable may require you specify
            others parameters to succesfully parse the times. The options are:
            - 'var': uses the value stored in the time variable
            - 'units': uses the value in the time variable and interpreted as a date according to the unit string which
                should follow the patter: unit_of_time since YYYY-MM-DD HH:MM:SS. you may specify the units string in
                the `ustring` parameter or else it will attempt to detect the string in each file.
            - 'filename': attempt to parse the correct date for the data from the file name. you must also provide the
                `strp` parameter which is used by datetime.strptime
        strp: A string compatible with datetime.strptime for extracting datetime pattern in file names
        engine: the python package used to power the file reading. Defaults to best for the type of input data
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame with an index, datetime column, and a column of values for each stat specified
    """
    if engine is None:
        engine = _pick_engine(files[0])

    # get information to slice the array with
    dim_order, slices = _slicing_info(files[0], var, coords, None, dims, t_dim, engine, xr_kwargs, h5_group)

    # make the return item
    results = dict(datetime=[], values=[])

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = _open_by_engine(file, engine, xr_kwargs)
        results['datetime'] += list(_handle_time_steps(opened_file, file, t_dim, strp, h5_group))

        # extract the appropriate values from the variable
        vs = _array_by_engine(opened_file, var, h5_group)[slices]
        if vs.ndim == 0:
            if vs == fill_value:
                vs = np.nan
            results['values'].append(vs)
        elif vs.ndim == 1:
            vs[vs == fill_value] = np.nan
            for v in vs:
                results['values'].append(v)
        else:
            raise ValueError('There are too many dimensions')
        if engine != 'pygrib':
            opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(results)


def bounding_box(files: list,
                 var: str or int,
                 min_coords: tuple,
                 max_coords: tuple,
                 dims: tuple,
                 t_dim: str = 'time',
                 strp: str = False,
                 stats: str or list = 'mean',
                 fill_value: int = -9999,
                 engine: str = None,
                 h5_group: str = None,
                 xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values based on values within a bounding box specified by your coordinates.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file (e.g. often 'temp' or 'T' instead of Temperature) or the
            band number if you are using grib files and you specify the engine as pygrib
        min_coords: A tuple of the minimum coordinates for the region of interest in the order (x, y, z)
        max_coords: A tuple of the maximum coordinates for the region of interest in the order (x, y, z)
        dims: A tuple of the names of the (x, y, z) variables in the data files, the same you specified coords for.
            Defaults to common x, y, z variables names: ('lon', 'lat', 'depth')
            X dimension names are usually 'lon', 'longitude', 'x', or similar
            Y dimension names are usually 'lat', 'latitude', 'y', or similar
            Z dimension names are usually 'depth', 'elevation', 'z', or similar
        t_dim: Name of the time variable if it is used in the files. Default: 'time'
        strp: A string compatible with datetime.strptime for extracting datetime pattern in file names
        stats: How to reduce the values within the bounding box into a single value for the timeseries.
            Options include: mean, median, max, min, sum, std, a percentile (e.g. 25%) or all.
            Provide a list of strings (e.g. ['mean', 'max']), or a comma separated string (e.g. 'mean,max,min')
        fill_value: The value used for filling no_data spaces in the source file's array. Default: -9999
        engine: the python package used to power the file reading
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame with an index, datetime column, and a column of values for each stat specified
    """
    if engine is None:
        engine = _pick_engine(files[0])

    # interpret the choice of statistics provided
    stats = _gen_stat_list(stats)

    # get information to slice the array with
    dim_order, slices = _slicing_info(files[0], var, min_coords, max_coords, dims, t_dim, engine, xr_kwargs, h5_group)

    # make the return item
    results = dict(datetime=[])
    # add a list for each stat requested
    for stat in stats:
        results[stat] = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = _open_by_engine(file, engine, xr_kwargs)
        # get the times
        results['datetime'] += list(_handle_time_steps(opened_file, file, t_dim, strp, h5_group))

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = _array_by_engine(opened_file, var, h5_group=h5_group)[slices]
        vs[vs == fill_value] = np.nan
        for stat in stats:
            results[stat] += _array_to_stat_list(vs, stat)
        if engine != 'pygrib':
            opened_file.close()
    # return the data stored in a dataframe
    return pd.DataFrame(results)


def polygons(files: list,
             var: str or int,
             poly: str or dict,
             dims: tuple,
             t_dim: str = 'time',
             strp: str = False,
             stats: str = 'mean',
             fill_value: int = -9999,
             crs: str = '+proj=latlong',
             engine: str = None,
             h5_group: str = None,
             xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values within the boundaries of your polygon shapefile of the same coordinate system.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file (e.g. often 'temp' or 'T' instead of Temperature) or the
            band number if you are using grib files and you specify the engine as pygrib
        poly: the path to a shapefile or geojson in the same coordinate system and projection as the raster data
        dims: A tuple of the names of the (x, y, z) variables in the data files, the same you specified coords for.
            Defaults to common x, y, z variables names: ('lon', 'lat', 'depth')
            X dimension names are usually 'lon', 'longitude', 'x', or similar
            Y dimension names are usually 'lat', 'latitude', 'y', or similar
            Z dimension names are usually 'depth', 'elevation', 'z', or similar
        t_dim: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        strp: A string compatible with datetime.strptime for extracting datetime pattern in file names
        stats: How to reduce the values inside the polygons into a single value for the timeseries.
            Options include: mean, median, max, min, sum, std, a percentile (e.g. 25%) or all.
            Provide a list of strings (e.g. ['mean', 'max']), or a comma separated string (e.g. 'mean,max,min')
        fill_value: The value used for filling no_data spaces in the source file's array. Default: -9999
        crs: Coordinate Reference System of the source rasters used by rasterio.open().
            An EPSG such as 'EPSG:4326' or '+proj=latlong'
        engine: the python package used to power the file reading
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame with an index, datetime column, and a column of values for each stat specified
    """
    if engine is None:
        engine = _pick_engine(files[0])

    # interpret the choice of statistics provided
    stats = _gen_stat_list(stats)

    # get information to slice the array with
    dim_order, _ = _slicing_info(files[0], var, None, None, dims, t_dim, engine, xr_kwargs, h5_group)

    # generate an affine transform used in zonal statistics
    affine = gen_affine(files[0], dims[0], dims[1], engine=engine, xr_kwargs=xr_kwargs, h5_group=h5_group)

    file = _open_by_engine(files[0], engine, xr_kwargs)
    arr = np.squeeze(np.ones(_array_by_engine(file, var, h5_group).shape))
    tmp_geotiff = os.path.join(tempfile.gettempdir(), '_.tiff')
    with rasterio.open(
            tmp_geotiff,
            'w',
            driver='GTiff',
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            nodata=np.nan,
            crs=crs,
            transform=affine,
    ) as dst:
        dst.write(arr, 1)
    shp_file = geopandas.read_file(poly)

    mask, _ = rasterio.mask.mask(rasterio.open(tmp_geotiff, 'r'), shp_file.geometry)
    os.remove(tmp_geotiff)
    # np.nan_to_num(mask, copy=False)

    # make the return item
    results = dict(datetime=[])
    # add a list for each stat requested
    for stat in stats:
        results[stat] = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = _open_by_engine(file, engine, xr_kwargs)
        results['datetime'] += list(_handle_time_steps(opened_file, file, t_dim, strp, h5_group))

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = _array_by_engine(opened_file, var, h5_group)
        vs[vs == fill_value] = np.nan
        # modify the array as necessary
        if vs.ndim == 2:
            # if the values are in a 2D array, cushion it with a 3rd dimension so you can iterate
            vs = np.reshape(vs, [1] + list(np.shape(vs)))
            dim_order = 't' + dim_order
        if t_dim in dim_order:
            # roll axis brings the time dimension to the front so we can iterate over it -- may not work as expected
            vs = np.rollaxis(vs, dim_order.index('t'))

        # do zonal statistics on everything
        for slice_2d in vs:
            slice_2d = np.where(np.isnan(mask), np.nan, slice_2d * mask).squeeze()
            for stat in stats:
                results[stat] += _array_to_stat_list(slice_2d, stat)
        if engine != 'pygrib':
            opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(results)


def full_array_stats(files: list,
                     var: str or int,
                     t_dim: str = 'time',
                     strp: str = False,
                     stats: str or list = 'mean',
                     fill_value: int = -9999,
                     engine: str = None,
                     h5_group: str = None,
                     xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values based on values within a bounding box specified by your coordinates.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file (e.g. often 'temp' or 'T' instead of Temperature) or the
            band number if you are using grib files and you specify the engine as pygrib
        t_dim: Name of the time variable if it is used in the files. Default: 'time'
        strp: A string compatible with datetime.strptime for extracting datetime pattern in file names
        stats: How to reduce the array into a single value for the timeseries.
            Options include: mean, median, max, min, sum, std, a percentile (e.g. 25%) or all.
            Provide a list of strings (e.g. ['mean', 'max']), or a comma separated string (e.g. 'mean,max,min')
        fill_value: The value used for filling no_data spaces in the source file's array. Default: -9999
        engine: the python package used to power the file reading
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame with an index, datetime column, and a column of values for each stat specified
    """
    if engine is None:
        engine = _pick_engine(files[0])

    # interpret the choice of statistics provided
    stats = _gen_stat_list(stats)

    # make the return item
    results = dict(datetime=[])
    # add a list for each stat requested
    for stat in stats:
        results[stat] = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = _open_by_engine(file, engine, xr_kwargs)
        results['datetime'] += list(_handle_time_steps(opened_file, file, t_dim, strp, h5_group))

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = _array_by_engine(opened_file, var, h5_group=h5_group)
        vs[vs == fill_value] = np.nan
        for stat in stats:
            results[stat] += _array_to_stat_list(vs, stat)
        if engine != 'pygrib':
            opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(results)


# Auxiliary utilities
def _slicing_info(path: str,
                  var: str,
                  min_coords: tuple or None,
                  max_coords: tuple or None,
                  dims: tuple,
                  t_dim: str,
                  engine: str = None,
                  xr_kwargs: dict = None,
                  h5_group: str = None, ) -> tuple:
    if engine is None:
        engine = _pick_engine(path)
    # open the file to be read
    tmp_file = _open_by_engine(path, engine, xr_kwargs)

    # validate choice in variables
    if not _check_var_in_dataset(tmp_file, var, h5_group):
        raise ValueError(f'the variable "{var}" was not found in the file {path}')

    # if its a netcdf or grib, the dimensions should be included by xarray
    if engine in ('xarray', 'cfgrib'):
        dim_order = list(tmp_file[var].dims)
    elif engine == 'netcdf4':
        dim_order = list(tmp_file[var].dimensions)
    elif engine == 'pygrib':
        dim_order = ['latitudes', 'longitudes']
    elif engine == 'h5py':
        if h5_group is not None:
            tmp_file = tmp_file[h5_group]
        dim_order = [i.label for i in tmp_file[var].dims]
    elif engine == 'rasterio':
        dims = ['x', 'y', 'band']
        dim_order = list(tmp_file.dims)
        if min_coords is not None:
            if len(min_coords) == 2:
                min_coords = tuple(list(min_coords) + [1])
                if max_coords is not None:
                    max_coords = tuple(list(max_coords) + [1])
    else:
        raise ValueError(f'Unable to determine dims for engine: {engine}')

    for i, d in enumerate(dim_order):
        tmp = str(d)
        tmp = tmp.replace(t_dim, 't_dim')
        for j, dimname in enumerate(dims):
            tmp = tmp.replace(dimname, f'dim{j}')
        dim_order[i] = tmp
    dim_order = str.join(',', dim_order)

    if min_coords is None and max_coords is None:
        if engine != 'pygrib':
            tmp_file.close()
        return dim_order, None

    if max_coords is None:
        max_coords = (False, False, False,)

    # gather all the indices
    slices_dict = dict(t_dim=slice(None))

    if engine == 'pygrib':
        slices_dict['dim0'] = _find_nearest_slice_index(
            np.array(sorted(list(tmp_file[1].distinctLongitudes))), min_coords[0], max_coords[0])
        slices_dict['dim1'] = _find_nearest_slice_index(
            np.array(sorted(list(tmp_file[1].distinctLatitudes))), min_coords[0], max_coords[0])
    else:
        # SLICING THE --X-- COORDINATE
        steps = _array_by_engine(tmp_file, dims[0])
        if steps.ndim == 2:
            steps = steps[0, :]
        slices_dict['dim0'] = _find_nearest_slice_index(steps, min_coords[0], max_coords[0])

        # SLICING THE --Y-- COORDINATE
        if len(min_coords) >= 2:
            steps = _array_by_engine(tmp_file, dims[1])
            if steps.ndim == 2:
                steps = steps[:, 0]
            slices_dict['dim1'] = _find_nearest_slice_index(steps, min_coords[1], max_coords[1])

        # SLICING THE --Z-- AND OTHER ANY OTHER COORDINATES
        if len(min_coords) >= 3:
            for i in range(2, len(min_coords)):
                steps = _array_by_engine(tmp_file, dims[i])
                slices_dict[f'dim{i}'] = _find_nearest_slice_index(steps, min_coords[i], max_coords[i])

        if engine != 'pygrib':
            tmp_file.close()
    return dim_order, tuple([slices_dict[d] for d in dim_order.split(',')])


def _gen_stat_list(stats: str or list):
    if isinstance(stats, str):
        if stats == 'all':
            return ALL_STATS
        else:
            return stats.lower().replace(' ', '').split(',')
    if any(stat not in ALL_STATS for stat in stats):
        raise ValueError(f'Unrecognized stat requested. Choose from: {ALL_STATS}')


def _handle_time_steps(opened_file, file_path, t_dim, strp_string, h5_group):
    if strp_string:
        return [datetime.datetime.strptime(os.path.basename(file_path), strp_string), ]
    else:  # use the time variable
        ts = _array_by_engine(opened_file, t_dim, h5_group=h5_group)
        if isinstance(ts, np.datetime64):
            return [ts]
        if ts.ndim == 0:
            return ts
        else:
            dates = []
            for t in ts:
                dates.append(t)
            return dates


def _find_nearest_slice_index(list_of_values: np.array, val1, val2=False):
    try:
        assert list_of_values.ndim == 1
    except AssertionError as e:
        raise e

    min_val = list_of_values.min()
    max_val = list_of_values.max()
    if not max_val >= val1 >= min_val:
        raise ValueError(f'specified coordinate value ({val1}) is outside the min/max range: [{min_val}, {max_val}]')
    index1 = (np.abs(list_of_values - val1)).argmin()

    if val2 is False:
        return index1

    if not max_val >= val2 >= min_val:
        raise ValueError(f'specified coordinate value ({val2}) is outside the min/max range: [{min_val}, {max_val}]')
    index2 = (np.abs(list_of_values - val2)).argmin()
    if index1 == index2:
        return index1
    elif index1 < index2:  # we'll put this check in there just in case they got the coordinates wrong
        return slice(index1, index2)
    else:
        return slice(index2, index1)
