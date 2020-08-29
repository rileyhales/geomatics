import datetime
import os
import tempfile

import geopandas
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from dateutil.relativedelta import relativedelta

from ._utils import _open_by_engine, _array_by_engine, _attribute_by_engine, _pick_engine, \
    _check_var_in_dataset, _array_to_stat_list
from .data import gen_affine

__all__ = ['time_series']
ALL_STATS = ('mean', 'median', 'max', 'min', 'sum', 'std')
ALL_ENGINES = ('xarray', 'netcdf4', 'cfgrib', 'pygrib', 'h5py', 'rasterio')
RECOGNIZED_TIME_INTERVALS = ('years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds')


def time_series(
        files: list or tuple, var: str or int, dim_order: tuple, t_var: str = 'time', crs: str = '+proj=latlong',
        point: tuple = None, bound: tuple = None, array: bool = False, polys: str = None, stats: list or str = 'mean',
        **kwargs
) -> pd.DataFrame:
    """
    Creates a time series of values from arrays contained in netCDF, grib, hdf, or geotiff formats. Values in the series
    are extracted from a point, bounding box, or shapefile/geojson subset of the array, or are summary statistics of the
    entire array.

    The datetime for each value extracted can be assigned 4 ways and is assigned in this order of preference:
        1. When interp_units is True, interpret the values in the time variable as datetimes using the units attribute.
            You can specify the units string with the units_str kwarg and the origin_format kwarg if the date doesn't
            follow the conventional YYYY-MM-DD HH:MM:SS format.
        2. When a pattern is specified with strp_filename, the datetime extracted from the filename is applied to all
           values coming from that dataset
        3. The numerical values from the time variable are used without further interpretation
        4. The string file name is used if there is no time variable and no other options were provided

    Args:
        files (list): A list (even if len==1) of either absolute file paths to netcdf, grib, hdf5, or geotiff files or
            urls to an OPENDAP service (but beware the data transfer speed bottleneck)
        var (str or int): The name of a variable as it is stored in the file (e.g. often 'temp' or 'T' instead of
            Temperature) or the band number if you are using grib files and you specify the engine as pygrib
        dim_order (tuple): A tuple of the names of the (x, y, z) variables in the data files which you specified coords for.
            X dimension names are usually 'lon', 'longitude', 'x', or similar
            Y dimension names are usually 'lat', 'latitude', 'y', or similar
            Z dimension names are usually 'depth', 'elevation', 'z', or similar
        t_var (str): Name of the time variable if it is used in the files. Default: 'time'
        crs (str): an EPSG string, e.g. "EPSG:4326", for the raster data used for time series with the polys argument

        point (tuple): a tuple of coordinate values, listed in the same order as dim_order, for the data of interest.
        bound (tuple): a tuple containing 2 tuples, each in the format described for the point argument, one tuple
            should list the lower bound values and the other should list the upper bound (order of upper/lower bound
            tuples does not matter although the coordinate order in each must match the order in dim_order
        array (bool): when true, the specified stats are calculated on the entire array for each time step available
        polys (str): path to a spatial geometry file, such as a shapefile or geojson, which can be read by geopandas
        stats (list or str): How to reduce arrays of values to a single scalar value for the timeseries.
            Options include: mean, median, max, min, sum, std, a percentile (e.g. 25%) or all.
            Provide a list of strings (e.g. ['mean', 'max']), or a comma separated string (e.g. 'mean,max,min')

    Keyword Args:
        engine (str): the python package used to power the file reading. Defaults to best for the type of input data
        fill_value (int): The value used for filling no_data spaces in the source file's array. Default: -9999
        h5_group (str): if all variables in the hdf5 file are in the same group, specify the name of the group here
        xr_kwargs (dict): A dictionary of kwargs that you might need when opening complex grib files with xarray
        interp_units (bool): If your data conforms to the CF NetCDF standard for time data, choose True to
            convert the values in the time variable to datetime strings in the pandas output. The units string for the
            time variable of each file is checked separately unless you specify it in the unit_str parameter.
        unit_str (str): a CF Standard conformant string indicating how the spacing and origin of the time values.
            Only specify this if ALL files that you query will contain the same units string. This is helpful if your
            files do not contain a units string. Usually this looks like "step_size since YYYY-MM-DD HH:MM:SS" such as
            "days since 2000-01-01 00:00:00".
        origin_format (str): A datetime.strptime string for extracting the origin time from the units string. Defaults
            to '%Y-%m-%d %X'.
        strp_filename (str): A datetime.strptime string for extracting datetimes from patterns in file
            names.

    Returns:
        pandas.DataFrame with an index, datetime column, and a column of values for each stat specified
    """
    if not (isinstance(files, list) or isinstance(files, tuple)):
        raise TypeError(f'Expected list of strings (paths/urls) for the "files" argument. Got: {type(files)}')

    engine = kwargs.get('engine', _pick_engine(files[0]))
    h5_group = kwargs.get('h5_group', None)
    xr_kwargs = kwargs.get('xr_kwargs', None)
    fill_value = kwargs.get('fill_value', -9999)
    interp_units = kwargs.get('interp_units', False)
    unit_str = kwargs.get('unit_str', None)
    origin_format = kwargs.get('origin_format', None)
    strp_filename = kwargs.get('strp_filename', None)

    if point is not None:
        slices = _coords_to_slices(_open_by_engine(files[0]), dim_order, point, h5_group=h5_group)
        return _point(files, engine, var, slices, fill_value,
                      t_var, interp_units, unit_str, origin_format, strp_filename,
                      h5_group, xr_kwargs)
    elif bound is not None:
        slices = _coords_to_slices(_open_by_engine(files[0]), dim_order, bound[0], bound[1], h5_group=h5_group)
        return _bound(files, engine, var, slices, fill_value, stats,
                      t_var, interp_units, unit_str, origin_format, strp_filename,
                      h5_group, xr_kwargs)
    elif polys is not None:
        return _polys(files, engine, var, polys, fill_value, stats, crs,
                      t_var, interp_units, unit_str, origin_format, strp_filename,
                      h5_group, xr_kwargs)
    elif array:
        slices = _coords_to_slices(_open_by_engine(files[0]), dim_order, bound[0], bound[1], h5_group=h5_group)
        return _array()
    else:
        raise ValueError('Must provide the point, bound, polygon, or array parameter.')


def _point(files: list or tuple, engine: str, var: str or int, slices: tuple, fill_value: int,
           t_var: str, interp_units: bool, unit_str: str, origin_format: str, strp_filename: str,
           h5_group: str = None, xr_kwargs: dict = None, ) -> pd.DataFrame:
    # make the return item
    results = dict(datetime=[], values=[])

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = _open_by_engine(file, engine, xr_kwargs)
        results['datetime'] += list(_handle_time_steps(
            opened_file, file, t_var, interp_units, unit_str, origin_format, strp_filename, h5_group))

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


def _bound(files: list or tuple, engine: str, var: str or int, slices: tuple, fill_value: int, stats: list,
           t_var: str, interp_units: bool, unit_str: str, origin_format: str, strp_filename: str,
           h5_group: str = None, xr_kwargs: dict = None, ) -> pd.DataFrame:
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
        results['datetime'] += list(_handle_time_steps(
            opened_file, file, t_var, interp_units, unit_str, origin_format, strp_filename, h5_group))

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = _array_by_engine(opened_file, var, h5_group=h5_group)[slices]
        vs[vs == fill_value] = np.nan
        for stat in stats:
            results[stat] += _array_to_stat_list(vs, stat)
        if engine != 'pygrib':
            opened_file.close()
    # return the data stored in a dataframe
    return pd.DataFrame(results)


def _polys(files: list or tuple, engine: str, var: str or int, poly: str or dict, fill_value: int, stats: list,
           crs: str, t_var: str, interp_units: bool, unit_str: str, origin_format: str, strp_filename: str,
           h5_group: str = None, xr_kwargs: dict = None, ) -> pd.DataFrame:
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
        results['datetime'] += list(_handle_time_steps(
            opened_file, file, t_var, interp_units, unit_str, origin_format, strp_filename, h5_group))

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = _array_by_engine(opened_file, var, h5_group)
        vs[vs == fill_value] = np.nan
        # modify the array as necessary
        if vs.ndim == 2:
            # if the values are in a 2D array, cushion it with a 3rd dimension so you can iterate
            vs = np.reshape(vs, [1] + list(np.shape(vs)))
            dim_order = 't' + dim_order
        if t_var in dim_order:
            # roll axis brings the time dimension to the front so we can iterate over it -- may not work as expected
            vs = np.rollaxis(vs, dim_order.index(t_var))

        # do zonal statistics on everything
        for slice_2d in vs:
            slice_2d = np.where(np.isnan(mask), np.nan, slice_2d * mask).squeeze()
            for stat in stats:
                results[stat] += _array_to_stat_list(slice_2d, stat)
        if engine != 'pygrib':
            opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(results)


def _array(files: list or tuple, var: str or int, t_var: str = 'time', fill_value: int = -9999,
           stats: list or str = 'mean',
           interp_units: bool = False, unit_str: str = None, origin_format: str = None,
           strp_filename: str = None, engine: str = None, h5_group: str = None,
           xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values based on values within a bounding box specified by your coordinates.

    The datetime for each value extracted can be assigned 4 ways and is assigned in this order of preference:
        1. When interp_units is True, interpret the values in the time variable as datetimes using their units
        2. When a pattern is specified with strp_filename, the datetime extracted from the filename is applied to all
           values coming from that dataset
        3. The numerical values from the time variable are used without further interpretation
        4. The string file name is used if there is no time variable and no other options were provided

    Args:
        files (list): A list (even if len==1) of either absolute file paths to netcdf, grib, hdf5, or geotiff files or
            urls to an OPENDAP service (but beware the data transfer speed bottleneck)
        var (str or int): The name of a variable as it is stored in the file (e.g. often 'temp' or 'T' instead of
            Temperature) or the band number if you are using grib files and you specify the engine as pygrib
        t_var (str): Name of the time variable if it is used in the files. Default: 'time'
        stats (list or str): How to reduce the values within the bounding box into a single value for the timeseries.
            Options include: mean, median, max, min, sum, std, a percentile (e.g. 25%) or all.
            Provide a list of strings (e.g. ['mean', 'max']), or a comma separated string (e.g. 'mean,max,min')
        fill_value (int): The value used for filling no_data spaces in the source file's array. Default: -9999
        interp_units (bool): If your data conforms to the CF NetCDF standard for time data, choose True to
            convert the values in the time variable to datetime strings in the pandas output. The units string for the
            time variable of each file is checked separately unless you specify it in the unit_str parameter.
        unit_str (str): a CF Standard conformant string indicating how the spacing and origin of the time values.
            Only specify this if ALL files that you query will contain the same units string. This is helpful if your
            files do not contain a units string. Usually this looks like "step_size since YYYY-MM-DD HH:MM:SS" such as
            "days since 2000-01-01 00:00:00".
        origin_format (str): A datetime.strptime string for extracting the origin time from the units string. Defaults
            to '%Y-%m-%d %X'.
        strp_filename (str): A string compatible with datetime.strptime for extracting datetimes from patterns in file
            names.
        engine (str): the python package used to power the file reading. Defaults to best for the type of input data
        h5_group (str): if all variables in the hdf5 file are in the same group, specify the name of the group here
        xr_kwargs (dict): A dictionary of kwargs that you might need when opening complex grib files with xarray

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
        results['datetime'] += list(_handle_time_steps(
            opened_file, file, t_var, interp_units, unit_str, origin_format, strp_filename, h5_group))

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = _array_by_engine(opened_file, var, h5_group=h5_group)
        vs[vs == fill_value] = np.nan
        for stat in stats:
            results[stat] += _array_to_stat_list(vs, stat)
        if engine != 'pygrib':
            opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(results)


def _gen_stat_list(stats: str or list):
    if isinstance(stats, str):
        if stats == 'all':
            return ALL_STATS
        else:
            return stats.lower().replace(' ', '').split(',')
    if any(stat not in ALL_STATS for stat in stats):
        raise ValueError(f'Unrecognized stat requested. Choose from: {ALL_STATS}')


def _handle_time_steps(opened_file, file_path, t_var, interp_units, unit_str, origin_format, strp_filename, h5_group):
    if interp_units:  # convert the time variable array's numbers to datetime representations
        tvals = _array_by_engine(opened_file, t_var, h5_group=h5_group)
        if origin_format is None:
            origin_format = '%Y-%m-%d %X'
        if unit_str is None:
            unit_str = _attribute_by_engine(opened_file, t_var, 'units', h5_group=h5_group)
        return _delta_to_datetime(tvals, unit_str, origin_format)
    if strp_filename:  # strip the datetime from the file name
        return [datetime.datetime.strptime(os.path.basename(file_path), strp_filename), ]
    elif _check_var_in_dataset(opened_file, t_var, h5_group):  # use the time variable if it exists
        tvals = _array_by_engine(opened_file, t_var, h5_group=h5_group)
        if isinstance(tvals, np.datetime64):
            return [tvals]
        if tvals.ndim == 0:
            return tvals
        else:
            dates = []
            for t in tvals:
                dates.append(t)
            return dates
    else:
        return [os.path.basename(file_path), ]


def _delta_to_datetime(tvals: np.array, ustr: str, origin_format: str = '%Y-%m-%d %X') -> np.array:
    """
    Converts an array of numbered time delta values to datetimes. This is similar to the CF time utilities but more
    comprehensive on timedelta options recognized.

    Args:
        tvals (np.array): the numerical time delta values
        ustr (str): A time units string following the CF convention: <time units> since <YYYY-mm-dd HH:MM:SS>
        origin_format (str): a datetime.datetime.strptime string defining the origin time's format if not standard

    Returns:
        np.array of datetime.datetime objects
    """
    interval = ustr.split(' ')[0].lower()
    origin = datetime.datetime.strptime(ustr.split(' since ')[-1], origin_format)
    if interval == 'years':
        delta = relativedelta(years=1)
    elif interval == 'months':
        delta = relativedelta(months=1)
    elif interval == 'weeks':
        delta = relativedelta(weeks=1)
    elif interval == 'days':
        delta = relativedelta(days=1)
    elif interval == 'hours':
        delta = relativedelta(hours=1)
    elif interval == 'minutes':
        delta = relativedelta(minutes=1)
    elif interval == 'seconds':
        delta = relativedelta(seconds=1)
    elif interval == 'milliseconds':
        delta = datetime.timedelta(milliseconds=1)
    elif interval == 'microseconds':
        delta = datetime.timedelta(microseconds=1)
    else:
        raise ValueError(f'Unrecognized time interval: {interval}')
    datetime.timedelta()

    # the values in the time variable, scaled to a number of time deltas, plus the origin time
    a = tvals * delta + origin
    return np.array([i.strftime("%Y-%m-%d %X") for i in a])


def _coords_to_slices(sample_file, dim_order: tuple, coords_min: tuple, coords_max: tuple = False,
                      h5_group: str = None):
    slices = []

    for order, coord_var in enumerate(dim_order):
        vals = _array_by_engine(sample_file, coord_var, h5_group)
        if vals.ndim != 1:
            # todo try to reduce the array to 1 dimensional
            raise RuntimeError(f"The coordinate variable {coord_var} is 2 dimensional and couldn't be reduced to 1")

        min_val = vals.min()
        max_val = vals.max()

        val1 = coords_min[order]
        if not (isinstance(val1, int) or isinstance(val1, float)):
            slices.append(slice(None))
        if not max_val >= val1 >= min_val:
            raise ValueError(f'Coordinate value ({val1}) is outside the min/max range ({min_val}, '
                             f'{max_val}) for the dimension {coord_var}')
        index1 = (np.abs(vals - val1)).argmin()

        if not coords_max:
            slices.append(index1)
            continue

        val2 = coords_max[order]
        if not max_val >= val2 >= min_val:
            raise ValueError(f'Coordinate value ({val2}) is outside the min/max range ({min_val}, '
                             f'{max_val}) for the dimension {coord_var}')
        index2 = (np.abs(vals - val2)).argmin()

        # check each option in case the index is the same or in case the coords were provided backwards
        if index1 == index2:
            slices.append(index1)
        elif index1 < index2:
            slices.append(slice(index1, index2))
        else:
            slices.append(slice(index2, index1))
    return tuple(slices)
