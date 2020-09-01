import datetime
import os
import tempfile

import affine
import geopandas
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from dateutil.relativedelta import relativedelta

from ._utils import _open_by_engine, _array_by_engine, _attribute_by_engine, _pick_engine, \
    _check_var_in_dataset, _array_to_stat_list

__all__ = ['time_series', ]
ALL_STATS = ('mean', 'median', 'max', 'min', 'sum', 'std',)
ALL_ENGINES = ('xarray', 'netcdf4', 'cfgrib', 'pygrib', 'h5py', 'rasterio',)
RECOGNIZED_TIME_INTERVALS = ('years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds',)
X_VARIABLES = ('x', 'longitude', 'lon', 'degrees_east', 'eastings',)
Y_VARIABLES = ('y', 'latitude', 'lat', 'degrees_north', 'northings',)
Z_VARIABLES = ('z', 'depth', 'elevation', 'altitude',)


def time_series(
        files: list or tuple, var: str or int, dim_order: tuple, t_var: str = 'time', stats: list or str = 'mean',
        point: tuple = None, bound: tuple = None, polys: str = None, masks: np.array = None, array: bool = False,
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
        dim_order (tuple): A tuple of the names of the dimensions for `var`, listed in order.
        t_var (str): Name of the time variable if it is used in the files. Default: 'time'
        point (tuple): a tuple of coordinate values, listed in the same order as dim_order, for the data of interest.
        bound (tuple): a tuple containing 2 tuples, each in the format described for the point argument, one tuple
            should list the lower bound values and the other should list the upper bound (order of upper/lower bound
            tuples does not matter although the coordinate order in each must match the order in dim_order)
        polys (str): path to any spatial geometry file, such as a shapefile or geojson, which can be read by geopandas
        masks (np.array): a numpy array containing np.nan or 1 values of the same shape as the source data files (not
            including the shape of the time dimension, if applicable). Useful when you want to mask irregular subsets.
        array (bool): when true, the specified stats are calculated on the entire array for each time step available
        stats (list or str): How to reduce arrays of values to a single scalar value for the timeseries.
            Options include: mean, median, max, min, sum, std, a percentile (e.g. 25%) or all.
            Provide a list of strings (e.g. ['mean', 'max']), or a comma separated string (e.g. 'mean,max,min')

    Keyword Args:
        engine (str): the python package used to power the file reading. Defaults to best for the type of input data
        h5_group (str): if all variables in the hdf5 file are in the same group, specify the name of the group here
        xr_kwargs (dict): A dictionary of kwargs that you might need when opening complex grib files with xarray
        crs (str): an EPSG string, e.g. "EPSG:4326", for the raster data used for time series with the polys argument
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
        crs = kwargs.get('crs', '+proj=latlong')
        # todo x and y variables customizable
        # verify that the provided dimensions have an x and y variable
        if not any(str(x).lower() in X_VARIABLES for x in dim_order):
            raise ValueError('No spatial "x" coordinate variable found in the dim_order for this variable/data')
        if not any(str(y).lower() in Y_VARIABLES for y in dim_order):
            raise ValueError('No spatial "y" coordinate variable found in the dim_order for this variable/data')
        mask = _create_spatial_mask_array(files[0], polys, var, dim_order, crs, engine, h5_group, xr_kwargs)
        return _masks(files, engine, var, mask, fill_value, stats, dim_order,
                      t_var, interp_units, unit_str, origin_format, strp_filename,
                      h5_group, xr_kwargs)
    elif masks is not None:
        return _masks(files, engine, var, masks, fill_value, stats, dim_order,
                      t_var, interp_units, unit_str, origin_format, strp_filename,
                      h5_group, xr_kwargs)
    elif array:
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


def _masks(files: list or tuple, engine: str, var: str or int, mask: np.array, fill_value: int, stats: list,
           dim_order: tuple,
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
        results['datetime'] += list(_handle_time_steps(
            opened_file, file, t_var, interp_units, unit_str, origin_format, strp_filename, h5_group))

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vals = _array_by_engine(opened_file, var, h5_group)
        vals[vals == fill_value] = np.nan

        # if the dimensions are the same
        if vals.ndim == mask.ndim:
            vals = np.where(np.isnan(mask), np.nan, vals * mask).squeeze()
            for stat in stats:
                results[stat] += _array_to_stat_list(vals, stat)
        elif vals.ndim == mask.ndim + 1:
            if t_var in dim_order:
                # roll axis brings the time dimension to the "front" so we iterate over it in a for loop
                for time_step_array in np.rollaxis(vals, dim_order.index(t_var)):
                    time_step_array = np.where(np.isnan(mask), np.nan, time_step_array * mask).squeeze()
                    for stat in stats:
                        results[stat] += _array_to_stat_list(time_step_array, stat)
            else:
                raise RuntimeError(f'Wrong dimensions. mask dims: {mask.ndim}, data\'s dims {vals.ndim}, file: {file}')
        else:
            raise RuntimeError(f'Wrong dimensions. mask dims: {mask.ndim}, data\'s dims {vals.ndim}, file: {file}')

        if engine != 'pygrib':
            opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(results)


# todo fix the array stats function
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


def _create_spatial_mask_array(sample_file: str, poly: str, var: str, dim_order: tuple, crs: str,
                               engine, h5_group, xr_kwargs):
    x = None
    y = None
    for a in dim_order:
        if a in X_VARIABLES:
            x = a
        if a in Y_VARIABLES:
            y = a

    sample_data = _open_by_engine(sample_file, engine, xr_kwargs)
    x = _array_by_engine(sample_data, x, h5_group)
    y = _array_by_engine(sample_data, y, h5_group)
    var = _array_by_engine(sample_data, var, h5_group)
    if engine != 'pygrib':
        sample_data.close()

    # catch the case when people used 2d instead of the proper 1d coordinate dimensions
    if y.ndim == 2:
        y = y[:, 0]
    if x.ndim == 2:
        x = x[0, :]

    # write a temporary geotiff with a single value of 1 which we use to create the mask with rasterio
    tmp_geotiff = os.path.join(tempfile.gettempdir(), '_python_geomatics_temp.tiff')
    with rasterio.open(
            tmp_geotiff,
            'w',
            driver='GTiff',
            height=y.shape[0],
            width=x.shape[1],
            count=1,
            dtype=var.dtype,
            nodata=np.nan,
            crs=crs,
            transform=affine.Affine(x[1] - x[0], 0, x.min(), 0, y[0] - y[1], y.max()),
    ) as dst:
        dst.write(np.squeeze(np.ones(var.shape)), 1)

    # read the source spatial file and reproject to the correct crs
    shp_file = geopandas.read_file(poly).to_crs(crs=crs)
    # creates a mask of the shapefile (also returns the affine.Affine transform which we ignore)
    mask, _ = rasterio.mask.mask(rasterio.open(tmp_geotiff, 'r'), shp_file.geometry)
    # delete the temporary geotiff
    os.remove(tmp_geotiff)
    return mask


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
