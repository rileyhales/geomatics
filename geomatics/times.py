import os

import netCDF4 as nc
import numpy as np
import pandas as pd
import rasterstats

from ._utils import open_by_engine, array_by_engine, get_slicing_info, slice_array_cell, slice_array_range, pick_engine
from .data import gen_affine

__all__ = ['point_series', 'box_series', 'shp_series', 'gen_ncml']


# TIMESERIES FUNCTIONS
def point_series(files: list,
                 var: str,
                 coords: tuple,
                 x_var: str = 'lon',
                 y_var: str = 'lat',
                 t_var: str = 'time',
                 fill_value: int = -9999,
                 engine: str = None,
                 h5_group: str = None,
                 xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values at the grid cell closest to a specified point.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        coords: A tuple (x_value, y_value) where the xy values are the location you want to extract values for
            in units of the x and y coordinate variable
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'lon'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'lat'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        engine: the python package used to power the file reading
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame
    """
    if engine is None:
        engine = pick_engine(files[0])
    if engine == 'rasterio':
        x_var = 'x'
        y_var = 'y'

    # get information to slice the array with
    slicing_info = get_slicing_info(files[0], var, x_var, y_var, t_var, (coords,), engine, xr_kwargs, h5_group)
    dim_order = slicing_info['dim_order']
    x_idx = slicing_info['indices'][0][0]
    y_idx = slicing_info['indices'][0][1]

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = open_by_engine(file, engine, xr_kwargs)
        ts = array_by_engine(opened_file, t_var, h5_group=h5_group)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)

        # extract the appropriate values from the variable
        vs = slice_array_cell(array_by_engine(opened_file, var, h5_group), dim_order, x_idx, y_idx)
        if vs.ndim == 0:
            if vs == fill_value:
                vs = np.nan
            values.append(vs)
        else:
            vs[vs == fill_value] = np.nan
            for v in vs:
                values.append(v)

        opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def box_series(files: list,
               var: str,
               coords: tuple,
               x_var: str = 'lon',
               y_var: str = 'lat',
               t_var: str = 'time',
               fill_value: int = -9999,
               stat_type: str = 'mean',
               engine: str = None,
               h5_group: str = None,
               xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values based on values within a bounding box specified by your coordinates.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        coords: A tuple of the format ((min_x_value, min_y_value), (max_x_value, max_y_value)) where the xy values
            are in units of the x and y coordinate variable
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'lon'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'lat'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The method to turn the values within a bounding box into a single value. Eg mean, min, max
        engine: the python package used to power the file reading
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame
    """
    if engine == 'rasterio':
        x_var = 'x'
        y_var = 'y'

    # get information to slice the array with
    slicing_info = get_slicing_info(files[0], var, x_var, y_var, t_var, coords, engine, xr_kwargs, h5_group)
    dim_order = slicing_info['dim_order']
    xmin_idx = min(slicing_info['indices'][0][0], slicing_info['indices'][1][0])
    xmax_idx = max(slicing_info['indices'][0][0], slicing_info['indices'][1][0])
    ymin_idx = min(slicing_info['indices'][0][1], slicing_info['indices'][1][1])
    ymax_idx = max(slicing_info['indices'][0][1], slicing_info['indices'][1][1])

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = open_by_engine(file, engine, xr_kwargs)

        # get the times
        ts = array_by_engine(opened_file, t_var, h5_group=h5_group)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = slice_array_range(
            array_by_engine(opened_file, var, h5_group=h5_group), dim_order, xmin_idx, ymin_idx, xmax_idx, ymax_idx)
        vs[vs == fill_value] = np.nan
        # add the results to the lists of values and times
        if vs.ndim == 1 or vs.ndim == 2:
            if stat_type == 'mean':
                values.append(np.mean(vs))
            elif stat_type == 'max':
                values.append(np.max(vs))
            elif stat_type == 'min':
                values.append(np.min(vs))
            else:
                raise ValueError(f'Unrecognized statistic, {stat_type}. Use stat_type= mean, min or max')
        else:
            for v in vs:
                if stat_type == 'mean':
                    values.append(np.mean(v))
                elif stat_type == 'max':
                    values.append(np.max(v))
                elif stat_type == 'min':
                    values.append(np.min(v))
                else:
                    raise ValueError(f'Unrecognized statistic, {stat_type}. Use stat_type= mean, min or max')
        opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def shp_series(files: list,
               var: str,
               shp_path: str,
               x_var: str = 'lon',
               y_var: str = 'lat',
               t_var: str = 'time',
               fill_value: int = -9999,
               stat_type: str = 'mean',
               engine: str = None,
               h5_group: str = None,
               xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values within the boundaries of your polygon shapefile of the same coordinate system.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        shp_path: An absolute path to the .shp file in a shapefile. Must be same coord system as the raster data.
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'lon'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'lat'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The stats method to turn the values within the bounding box into a single value. Default: 'mean'
        engine: the python package used to power the file reading
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.timedata.shp_series([list, of, file, paths], 'AirTemp', '/path/to/shapefile.shp')
    """
    if engine == 'rasterio':
        x_var = 'x'
        y_var = 'y'

    # get information to slice the array with
    slicing_info = get_slicing_info(files[0], var, x_var, y_var, t_var, None, engine, xr_kwargs, h5_group)
    dim_order = slicing_info['dim_order']

    # generate an affine transform used in zonal statistics
    affine = gen_affine(files[0], x_var, y_var, engine=engine, xr_kwargs=xr_kwargs, h5_group=h5_group)

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = open_by_engine(file, engine, xr_kwargs)
        ts = array_by_engine(opened_file, t_var, h5_group=h5_group)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = array_by_engine(opened_file, var, h5_group)
        vs[vs == fill_value] = np.nan
        # modify the array as necessary
        if vs.ndim == 2:
            # if the values are in a 2D array, cushion it with a 3rd dimension so you can iterate
            vs = np.reshape(vs, [1] + list(np.shape(vs)))
            dim_order = 't' + dim_order
        if 't' in dim_order:
            # roll axis brings the time dimension to the front so we can iterate over it -- may not work as expected
            vs = np.rollaxis(vs, dim_order.index('t'))

        # do zonal statistics on everything
        for values_2d in vs:
            # actually do the gis to get the value within the shapefile
            stats = rasterstats.zonal_stats(shp_path, values_2d, affine=affine, nodata=np.nan, stats=stat_type)
            # if your shapefile has many polygons, you get many values back. weighted average of those values.
            tmp = [i[stat_type] for i in stats if i[stat_type] is not None]
            values.append(sum(tmp) / len(tmp))

        opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def gen_ncml(files: list, save_dir: str, time_interval: int) -> None:
    """
    Generates a ncml file which aggregates a list of netcdf files across the "time" dimension and the "time" variable.
    In order for the times displayed in the aggregated NCML dataset to be accurate, they must have a regular time step
    between measurments.

    Args:
        files: A list of absolute paths to netcdf files (even if len==1)
        save_dir: the directory where you would like to save the ncml
        time_interval: the time spacing between datasets in the units of the netcdf file's time variable
          (must be constont for ncml aggregation to work properly)

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.timedata.generate_timejoining_ncml('/path/to/netcdf/', '/path/to/save', 4)
    """
    ds = nc.Dataset(files[0])
    units_str = str(ds['time'].__dict__['units'])
    ds.close()

    # create a new ncml file by filling in the template with the right dates and writing to a file
    with open(os.path.join(save_dir, 'time_joined_series.ncml'), 'w') as ncml:
        ncml.write(
            '<netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2">\n' +
            '  <variable name="time" type="int" shape="time">\n' +
            '    <attribute name="units" value="' + units_str + '"/>\n' +
            '    <attribute name="_CoordinateAxisType" value="Time" />\n' +
            '    <values start="0" increment="' + str(time_interval) + '" />\n' +
            '  </variable>\n' +
            '  <aggregation dimName="time" type="joinExisting" recheckEvery="5 minutes">\n'
        )
        for file in files:
            ncml.write('    <netcdf location="' + file + '"/>\n')
        ncml.write(
            '  </aggregation>\n' +
            '</netcdf>'
        )
    return
