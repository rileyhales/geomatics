import numpy as np
import pandas as pd
import rasterio
import rasterstats
import xarray as xr

from .data import detect_type

__all__ = ['point_series', 'box_series', 'shp_series']


def point_series(files: list, variable: str, coordinates: tuple, **kwargs) -> pd.DataFrame:
    """
    Creates a timeseries of values at the grid cell closest to a specified point.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        variable: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        coordinates: A tuple of the format (x_value, y_value) where the xy values are in units of the x and y
            coordinate variable

    Keyword Args:
        xvar: Name of the x coordinate variable used to spatial reference the array. Default: 'lon' (longitude)
        yvar: Name of the y coordinate variable used to spatial reference the array. Default: 'lat' (latitude)
        tvar: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files
        fill_value: The value used for filling no_data spaces in the array. Default: -9999

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.timeseries.point_series('/path/to/data/', 'AirTemp', (10, 20))
    """
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    t_var = kwargs.get('tvar', 'time')
    xr_kwargs = kwargs.get('xr_kwargs', {})
    fill_value = kwargs.get('fill_value', -9999)

    # determine what kind of spatial data was provided
    if isinstance(files, str):
        files = [files]
    datatype = detect_type(files[0])

    # get a list of the x&y coordinates using the first file as a reference
    xr_obj = _open(files[0], datatype, xr_kwargs)
    x_steps = xr_obj[x_var][:].data
    y_steps = xr_obj[y_var][:].data
    # determine the index in the netcdf's coordinates for the xy coordinate provided
    x_index = (np.abs(x_steps - round(float(coordinates[0]), 2))).argmin()
    y_index = (np.abs(y_steps - round(float(coordinates[1]), 2))).argmin()
    dim_order = _get_dimension_order(xr_obj[variable].dims, x_var, y_var, t_var)
    xr_obj.close()

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        xr_obj = _open(file, datatype, xr_kwargs)
        ts = xr_obj[t_var].data

        # extract the correct values from the array and
        vs = _slice_point(xr_obj[variable], dim_order, x_index, y_index)
        vs[vs == fill_value] = np.nan

        # add the results to the lists of values and times
        if vs.ndim == 0:
            values.append(vs)
        else:
            for v in vs:
                values.append(v)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)
        xr_obj.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def box_series(files: list, variable: str, coordinates: tuple, **kwargs) -> pd.DataFrame:
    """
    Creates a timeseries of values based on values within a bounding box specified by your coordinates.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        variable: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        coordinates: A tuple of the format (x_value, y_value) where the xy values are in units of the x and y
            coordinate variable

    Keyword Args:
        xvar: Name of the x coordinate variable used to spatial reference the array. Default: 'lon' (longitude)
        yvar: Name of the y coordinate variable used to spatial reference the array. Default: 'lat' (latitude)
        tvar: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The stats method to turn the values within the bounding box into a single value. Default: 'mean'

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.timeseries.box_series('/path/to/netcdf/', 'AirTemp', (10, 20, 15, 25))
    """
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    t_var = kwargs.get('tvar', 'time')
    xr_kwargs = kwargs.get('xr_kwargs', {})
    fill_value = kwargs.get('fill_value', -9999)
    stat = kwargs.get('stat_type', 'mean')

    # determine what kind of spatial data was provided
    if isinstance(files, str):
        files = [files]
    datatype = detect_type(files[0])

    # get a list of the x&y coordinates using the first file as a reference
    xr_obj = _open(files[0], datatype, xr_kwargs)
    x_steps = xr_obj[x_var][:].data
    y_steps = xr_obj[y_var][:].data
    # get the indices of the bounding box corners
    xmin = (np.abs(x_steps - float(coordinates[0]))).argmin()
    xmax = (np.abs(x_steps - float(coordinates[2]))).argmin()
    xmin_index = min(xmin, xmax)
    xmax_index = max(xmin, xmax)
    ymin = (np.abs(y_steps - float(coordinates[1]))).argmin()
    ymax = (np.abs(y_steps - float(coordinates[3]))).argmin()
    ymin_index = min(ymin, ymax)
    ymax_index = max(ymin, ymax)
    # which order are the dimensions for this variable
    dim_order = _get_dimension_order(xr_obj[variable].dims, x_var, y_var, t_var)
    xr_obj.close()

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        xr_obj = _open(file, datatype, xr_kwargs)
        ts = xr_obj[t_var].data

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        values_array = _slice_box(xr_obj[variable], dim_order, xmin_index, ymin_index, xmax_index, ymax_index)
        values_array[values_array == fill_value] = np.nan
        if stat == 'mean':
            vs = np.mean(values_array)
        elif stat == 'max':
            vs = np.max(values_array)
        elif stat == 'min':
            vs = np.min(values_array)
        else:
            raise ValueError('Unrecognized statistic, {0}. Choose stat_type= mean, min or max'.format(stat))

        # add the results to the lists of values and times
        if vs.ndim == 0:
            values.append(vs)
        else:
            for v in vs:
                values.append(v)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)
        xr_obj.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def shp_series(files: list, variable: str, shp_path: str, **kwargs) -> pd.DataFrame:
    """
    Creates a timeseries of values within the boundaries of your polygon shapefile of the same coordinate system.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        variable: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        shp_path: An absolute path to the .shp file in a shapefile. Must be in Geographic Coordinate System WGS 1984

    Keyword Args:
        xvar: Name of the x coordinate variable used to spatial reference the array. Default: 'lon' (longitude)
        yvar: Name of the y coordinate variable used to spatial reference the array. Default: 'lat' (latitude)
        tvar: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The stats method to turn the values within the bounding box into a single value. Default: 'mean'

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.timeseries.shp_series('/path/to/netcdf/', 'AirTemp', '/path/to/shapefile.shp')
    """
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    t_var = kwargs.get('tvar', 'time')
    xr_kwargs = kwargs.get('xr_kwargs', {})
    fill_value = kwargs.get('fill_value', -9999)
    stat = kwargs.get('stat_type', 'mean')

    # determine what kind of spatial data was provided
    if isinstance(files, str):
        files = [files]
    datatype = detect_type(files[0])

    # get a list of the x&y coordinates using the first file as a reference
    xr_obj = _open(files[0], datatype, xr_kwargs)
    nc_xs = xr_obj.variables[x_var][:]
    nc_ys = xr_obj.variables[y_var][:]
    affine = rasterio.transform.from_origin(nc_xs.min(), nc_ys.max(), nc_ys[1] - nc_ys[0], nc_xs[1] - nc_xs[0])
    dim_order = _get_dimension_order(xr_obj[variable].dims, x_var, y_var, t_var)
    xr_obj.close()

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        xr_obj = _open(file, datatype, xr_kwargs)
        ts = xr_obj[t_var][:].data

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        values_array = xr_obj[variable][:].data
        # roll axis brings the time dimension to the front so we can iterate over it
        for values_2d in np.rollaxis(values_array, dim_order.index('t')):
            # drop fill and no data entries
            values_2d[values_2d == fill_value] = np.nan
            # vertically flip array so the orientation is right (you just have to, try it)
            values_2d = values_2d[::-1]
            # actually do the gis to get the value within the shapefile
            stats = rasterstats.zonal_stats(shp_path, values_2d, affine=affine, nodata=np.nan, stats=stat)
            # if your shapefile has many polygons, you get many values back. weighted average of those values.
            tmp = [i[stat] for i in stats if i[stat] is not None]
            values.append(sum(tmp) / len(tmp))

        # add the timesteps to the list of times
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)

        xr_obj.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def _open(path, filetype, backend_kwargs=None):
    if backend_kwargs is None:
        backend_kwargs = dict()
    if filetype in 'netcdf':
        return xr.open_dataset(path)
    elif filetype == 'grib':
        return xr.open_dataset(path, engine='cfgrib', backend_kwargs=backend_kwargs)
    else:
        raise ValueError('unsupported file type')


def _slice_point(xarray_variable, dim_order, x_index, y_index):
    if dim_order == 'txy':
        return xarray_variable[:, x_index, y_index].data
    elif dim_order == 'tyx':
        return xarray_variable[:, y_index, x_index].data

    elif dim_order == 'xyt':
        return xarray_variable[x_index, y_index, :].data
    elif dim_order == 'yxt':
        return xarray_variable[y_index, x_index, :].data

    elif dim_order == 'xty':
        return xarray_variable[x_index, :, y_index].data
    elif dim_order == 'ytx':
        return xarray_variable[y_index, :, x_index].data

    elif dim_order == 'xy':
        return xarray_variable[x_index, y_index].data
    elif dim_order == 'yx':
        return xarray_variable[y_index, x_index].data
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice array.')


def _slice_box(xarray_variable, dim_order, xmin_index, ymin_index, xmax_index, ymax_index):
    if dim_order == 'txy':
        return xarray_variable[:, xmin_index:xmax_index, ymin_index:ymax_index].data
    elif dim_order == 'tyx':
        return xarray_variable[:, ymin_index:ymax_index, xmin_index:xmax_index].data

    elif dim_order == 'xyt':
        return xarray_variable[xmin_index:xmax_index, ymin_index:ymax_index, :].data
    elif dim_order == 'yxt':
        return xarray_variable[ymin_index:ymax_index, xmin_index:xmax_index, :].data

    elif dim_order == 'xty':
        return xarray_variable[xmin_index:xmax_index, :, ymin_index:ymax_index].data
    elif dim_order == 'ytx':
        return xarray_variable[ymin_index:ymax_index, :, xmin_index:xmax_index].data

    elif dim_order == 'xy':
        return xarray_variable[xmin_index:xmax_index, ymin_index:ymax_index].data
    elif dim_order == 'yx':
        return xarray_variable[ymin_index:ymax_index, xmin_index:xmax_index].data
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice array.')


def _get_dimension_order(dimensions, x_var, y_var, t_var):
    # check if the variable has 2 dimensions -- should be xy coordinates (spatial references)
    if len(dimensions) == 2:
        if dimensions == (x_var, y_var):
            return 'xy'
        elif dimensions == (y_var, x_var):
            return 'yx'
        else:
            raise ValueError('Unexpected dimension name. Specify with the xvar, yvar, tvar keyword arguments')
    # check if the variables has 3 dimensions ie xy and time coordinates (spatio-temporal references)
    elif len(dimensions) == 3:
        # cases where time is first
        if dimensions == (t_var, x_var, y_var):
            return 'txy'
        elif dimensions == (t_var, y_var, x_var):
            return 'tyx'

        # cases where time is last
        elif dimensions == (x_var, y_var, t_var):
            return 'xyt'
        elif dimensions == (y_var, x_var, t_var):
            return 'yxt'

        # unlikely, but, if time is in the middle
        elif dimensions == (x_var, t_var, y_var):
            return 'xty'
        elif dimensions == (y_var, t_var, x_var):
            return 'ytx'

        else:
            raise ValueError('Unexpected dimension name(s). Specify with the xvar, yvar, tvar keyword arguments')
    else:
        raise ValueError('Your data should have either 2 (x,y) or 3 (x,y,time) dimensions')
