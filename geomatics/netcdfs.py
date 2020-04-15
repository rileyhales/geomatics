import datetime
import os

import dateutil
import netCDF4 as nc
import numpy as np
import pandas as pd
import rasterio
import rasterstats

from .__utilities import path_to_file_list

__all__ = ['inspect', 'guess_time_from_metadata', 'point_series', 'box_series', 'shp_series']


def inspect(path):
    """
    Prints lots of messages showing information about variables, dimensions, and metadata

    Args:
        path: The path to a single netcdf file.
    """
    nc_obj = nc.Dataset(path, 'r', clobber=False, diskless=True, persist=False)

    print("This is your netCDF python object")
    print(nc_obj)
    print()

    print("There are " + str(len(nc_obj.variables)) + " variables")       # The number of variables
    print("There are " + str(len(nc_obj.dimensions)) + " dimensions")     # The number of dimensions
    print()

    print('These are the global attributes of the netcdf file')
    print(nc_obj.__dict__)                                    # access the global attributes of the netcdf file
    print()

    print("Detailed view of each variable")
    print()
    for variable in nc_obj.variables.keys():                  # .keys() gets the name of each variable
        print('Variable Name:  ' + variable)              # The string name of the variable
        print('The view of this variable in the netCDF python object')
        print(nc_obj[variable])                               # How to view the variable information (netcdf obj)
        print('The data array stored in this variable')
        print(nc_obj[variable][:])                            # Access the numpy array inside the variable (array)
        print('The dimensions associated with this variable')
        print(nc_obj[variable].dimensions)                    # Get the dimensions associated with a variable (tuple)
        print('The metadata associated with this variable')
        print(nc_obj[variable].__dict__)                      # How to get the attributes of a variable (dictionary)
        print()

    for dimension in nc_obj.dimensions.keys():
        print(nc_obj.dimensions[dimension].size)              # print the size of a dimension

    nc_obj.close()                                            # close the file connection to the file
    return


def guess_time_from_metadata(nc_obj, t_var, date_string_format):
    """
    This method attempts to use the netCDF file's time variable to create a list of dates. A properly formatted time
    variable has a 'units' metadata entry which contains a string of the format "time_step since %Y-%m-%d %H:%M:%S".

    :return: pd.Series containing the timesteps
    """
    # try to build the correct times using the properly formatted time variable of a netCDF file
    units_str = str(nc_obj[t_var].__dict__['units'])

    # attempt to identify the timedelta
    timedelta = units_str.split(' ')[0]
    if timedelta.startswith('years'):
        timedelta = dateutil.relativedelta.relativedelta(years=1)
    elif timedelta.startswith('months'):
        timedelta = dateutil.relativedelta.relativedelta(months=1)
    elif timedelta.startswith('weeks'):
        timedelta = datetime.timedelta(weeks=1)
    elif timedelta.startswith('days'):
        timedelta = datetime.timedelta(days=1)
    elif timedelta.startswith('hours'):
        timedelta = datetime.timedelta(hours=1)
    elif timedelta.startswith('minutes'):
        timedelta = datetime.timedelta(minutes=1)
    elif timedelta.startswith('seconds'):
        timedelta = datetime.timedelta(seconds=1)
    else:
        raise ValueError("Timedelta was not specified and could not be guessed from the time variable's metadata")

    # attempt to identify the start date based on the rest of the string
    start_time = datetime.datetime.strptime(units_str, date_string_format)

    # create the series of dates using the start_time and the timedelta
    return [start_time + i * timedelta for i in nc_obj[t_var][:].data]


def point_series(path, variable, coordinates, **kwargs):
    """
    Creates a timeseries of values at the grid cell closest to a specified point.

    Args:
        path: Either 1) the absolute path to a directory containing netcdfs named by date or 2) the absolute path to
            a single netcdf containing many time values for a specified variable
        variable: The name of a variable as it is stored in the netcdf e.g. often 'temp' or 'T' instead of Temperature
        coordinates: A tuple of the format (x_value, y_value) where the xy values are in terms of the x and y
            coordinate variables used by the netcdf.

    Keyword Args:
        xvar: Name of the x coordinate variable used to spatial reference the netcdf array. Default: 'lon' (longitude)
        yvar: Name of the y coordinate variable used to spatial reference the netcdf array. Default: 'lat' (latitude)
        tvar: Name of the time coordinate variable used for time referencing the netcdf. Default: 'time'
        time_from_filename: The string containing the file naming pattern explained by the datetime documentation.
            Triggers attempting to identify the proper timestep for each file using the file's name. Will only work if
            each file contains exactly one time step.
        time_from_metadata: Triggers attempting to identify the proper timestep for each file using the values and
            metadata of the file's time variable.
        fill_value: The value used for filling no_data spaces in the array. Default: -9999

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.netcdfs.point_series('/path/to/netcdf/', 'AirTemp', (10, 20))
    """
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    t_var = kwargs.get('tvar', 'time')
    time_from_filename = kwargs.get('time_from_filename', False)
    time_from_metadata = kwargs.get('time_from_metadata', False)
    fill_value = kwargs.get('fill_value', -9999)

    # confirm that a valid path to data was provided
    files = path_to_file_list(path, 'nc')
    files.sort()

    # get a list of the x&y coordinates in the netcdfs using the first file as a reference
    nc_obj = nc.Dataset(files[0], 'r')
    nc_xs = nc_obj[x_var][:]
    nc_ys = nc_obj[y_var][:]
    # determine the index in the netcdf's coordinates for the xy coordinate provided
    x_index = (np.abs(nc_xs - round(float(coordinates[0]), 2))).argmin()
    y_index = (np.abs(nc_ys - round(float(coordinates[1]), 2))).argmin()
    dim_order = __get_dimension_order(nc_obj[variable].dimensions, x_var, y_var, t_var)
    nc_obj.close()

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        nc_obj = nc.Dataset(file, 'r')

        # attempt to determine the correct datetime to use in the timeseries
        if time_from_filename:
            new_times = [datetime.datetime.strptime(os.path.basename(file), time_from_filename)]
        elif time_from_metadata:
            new_times = list(guess_time_from_metadata(nc_obj[t_var], time_from_metadata))
        else:
            new_times = list(nc_obj[t_var][:].data)

        # extract the correct values from the array and
        new_values = __slice_point(nc_obj[variable], dim_order, x_index, y_index)
        new_values[new_values == fill_value] = np.nan
        new_values = list(new_values)

        # ensure the extraction went as planned
        if len(new_times) != len(new_values):
            raise ValueError('Extracted unequal times and values from file {0}'.format(file))

        times += new_times
        values += new_values

        # add to the list of timeseries
        nc_obj.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def box_series(path, variable, coordinates, **kwargs):
    """
    Creates a timeseries of values based on values within a bounding box specified by your coordinates.

    Args:
        path: Either 1) the absolute path to a directory containing netcdfs named by date or 2) the absolute path to
            a single netcdf containing many time values for a specified variable
        variable: The name of a variable as it is stored in the netcdf e.g. 'temp' instead of Temperature
        coordinates: A tuple of the format (min_x_value, min_y_value, max_x_value, max_y_value) where the xy values
            are in terms of the x and y coordinate variables used by the netcdf.

    Keyword Args:
        xvar: Name of the x coordinate variable used to spatial reference the netcdf array. Default: 'lon' (longitude)
        yvar: Name of the y coordinate variable used to spatial reference the netcdf array. Default: 'lat' (latitude)
        tvar: Name of the time coordinate variable used for time referencing the netcdf. Default: 'time'
        time_from_filename: The string containing the file naming pattern explained by the datetime documentation.
            Triggers attempting to identify the proper timestep for each file using the file's name. Will only work if
            each file contains exactly one time step.
        time_from_metadata: Triggers attempting to identify the proper timestep for each file using the values and
            metadata of the file's time variable.
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The stats method to turn the values within the bounding box into a single value. Default: 'mean'

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.netcdfs.box_series('/path/to/netcdf/', 'AirTemp', (10, 20, 15, 25))
    """
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    t_var = kwargs.get('tvar', 'time')
    time_from_filename = kwargs.get('time_from_filename', False)
    time_from_metadata = kwargs.get('time_from_metadata', False)
    fill_value = kwargs.get('fill_value', -9999)
    stat = kwargs.get('stat_type', 'mean')

    # confirm that a valid path to data was provided
    files = path_to_file_list(path, 'nc')
    files.sort()

    # get a list of the x&y coordinates using the first file as a reference
    nc_obj = nc.Dataset(files[0], 'r')
    nc_xs = nc_obj[x_var][:]
    nc_ys = nc_obj[y_var][:]
    # get the indices of the bounding box corners
    xmin_index = (np.abs(nc_xs - round(float(coordinates[0]), 2))).argmin()
    ymin_index = (np.abs(nc_ys - round(float(coordinates[1]), 2))).argmin()
    xmax_index = (np.abs(nc_xs - round(float(coordinates[2]), 2))).argmin()
    ymax_index = (np.abs(nc_ys - round(float(coordinates[3]), 2))).argmin()
    dim_order = __get_dimension_order(nc_obj[variable].dimensions, x_var, y_var, t_var)
    nc_obj.close()

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        nc_obj = nc.Dataset(file, 'r')

        # attempt to determine the correct datetime to use in the timeseries
        if time_from_filename:
            new_times = [datetime.datetime.strptime(os.path.basename(file), time_from_filename)]
        elif time_from_metadata:
            new_times = list(guess_time_from_metadata(nc_obj[t_var], time_from_metadata))
        else:
            new_times = list(nc_obj[t_var][:].data)

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        values_array = __slice_box(nc_obj[variable], dim_order, xmin_index, ymin_index, xmax_index, ymax_index)

        # roll axis brings the time dimension to the front so we can iterate over it
        new_values = []
        for values_2d in np.rollaxis(values_array, dim_order.index('t')):
            values_2d[values_2d == fill_value] = np.nan
            # get the specific value and append to the timeseries
            if stat == 'mean':
                new_values.append(float(values_2d.mean()))
            elif stat == 'max':
                new_values.append(float(max(values_2d)))
            elif stat == 'min':
                new_values.append(float(min(values_2d)))
            else:
                raise ValueError('Unrecognized statistic, {0}. Choose stat_type= mean, min or max'.format(stat))

        # ensure the extraction went as planned
        if len(new_times) != len(new_values):
            raise ValueError('Extracted unequal times and values from file {0}'.format(file))

        times += new_times
        values += new_values

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def shp_series(path, variable, shp_path, **kwargs):
    """
    Creates a timeseries of values within the boundaries of your polygon shapefile of the same coordinate system.

    Args:
        path: Either 1) the absolute path to a directory containing netcdfs named by date or 2) the absolute path to
            a single netcdf containing many time values for a specified variable
        variable: The name of a variable as it is stored in the netcdf e.g. 'temp' instead of Temperature
        shp_path: An absolute path to the .shp file in a shapefile. Must be in Geographic Coordinate System WGS 1984

    Keyword Args:
        xvar: Name of the x coordinate variable used to spatial reference the netcdf array. Default: 'lon' (longitude)
        yvar: Name of the y coordinate variable used to spatial reference the netcdf array. Default: 'lat' (latitude)
        tvar: Name of the time coordinate variable used for time referencing the netcdf. Default: 'time'
        time_from_filename: The string containing the file naming pattern explained by the datetime documentation.
            Triggers attempting to identify the proper timestep for each file using the file's name. Will only work if
            each file contains exactly one time step.
        time_from_metadata: Triggers attempting to identify the proper timestep for each file using the values and
            metadata of the file's time variable.
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The stats method to turn the values within the bounding box into a single value. Default: 'mean'

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.netcdfs.shp_series('/path/to/netcdf/', 'AirTemp', '/path/to/shapefile.shp')
    """
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    t_var = kwargs.get('tvar', 'time')
    time_from_filename = kwargs.get('time_from_filename', False)
    time_from_metadata = kwargs.get('time_from_metadata', False)
    fill_value = kwargs.get('fill_value', -9999)
    stat = kwargs.get('stat_type', 'mean')

    # confirm that a valid path to data was provided
    files = path_to_file_list(path, 'nc')
    files.sort()

    # open the netcdf determine the affine transformation of the netcdf grids
    nc_obj = nc.Dataset(files[0], 'r')
    nc_xs = nc_obj.variables[x_var][:]
    nc_ys = nc_obj.variables[y_var][:]
    affine = rasterio.transform.from_origin(nc_xs.min(), nc_ys.max(), nc_ys[1] - nc_ys[0], nc_xs[1] - nc_xs[0])
    dim_order = __get_dimension_order(nc_obj[variable].dimensions, x_var, y_var, t_var)
    nc_obj.close()

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        nc_obj = nc.Dataset(file, 'r')

        # attempt to determine the correct datetime to use in the timeseries
        if time_from_filename:
            new_times = [datetime.datetime.strptime(os.path.basename(file), time_from_filename)]
        elif time_from_metadata:
            new_times = list(guess_time_from_metadata(nc_obj[t_var], time_from_metadata))
        else:
            new_times = list(nc_obj[t_var][:].data)

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        new_values = []
        values_array = nc_obj[variable][:]
        # roll axis brings the time dimension to the front so we can iterate over it
        for values_2d in np.rollaxis(values_array, dim_order.index('t')):
            # drop fill and no data entries
            values_2d[values_2d == fill_value] = np.nan
            # vertically flip array so the orientation is right (you just have to, try it)
            values_2d = values_2d[::-1]
            # actually do the gis to get the value within the shapefile
            stats = rasterstats.zonal_stats(shp_path, values_2d, affine=affine, nodata=np.nan, stats=stat)
            # if your shapefile has many polygons, you get many values back. weighted average of those values.
            print(stats)
            tmp = [i[stat] for i in stats if i[stat] is not None]
            new_values.append(sum(tmp) / len(tmp))

        # ensure the extraction went as planned
        if len(new_times) != len(new_values):
            raise ValueError('Extracted unequal times and values from file {0}'.format(file))

        times += new_times
        values += new_values

        nc_obj.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def __get_dimension_order(dimensions, x_var, y_var, t_var):
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


def __slice_point(nc_var, dim_order, x_index, y_index):
    if dim_order == 'txy':
        return nc_var[:, x_index, y_index].data
    elif dim_order == 'tyx':
        return nc_var[:, y_index, x_index].data

    elif dim_order == 'xyt':
        return nc_var[x_index, y_index, :].data
    elif dim_order == 'yxt':
        return nc_var[y_index, x_index, :].data

    elif dim_order == 'xty':
        return nc_var[x_index, :, y_index].data
    elif dim_order == 'ytx':
        return nc_var[y_index, :, x_index].data

    elif dim_order == 'xy':
        return nc_var[x_index, y_index].data
    elif dim_order == 'yx':
        return nc_var[y_index, x_index].data
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice netCDF array.')


def __slice_box(nc_var, dim_order, xmin_index, ymin_index, xmax_index, ymax_index):
    if dim_order == 'txy':
        return nc_var[:, xmin_index:xmax_index, ymin_index:ymax_index].data
    elif dim_order == 'tyx':
        return nc_var[:, ymin_index:ymax_index, xmin_index:xmax_index].data

    elif dim_order == 'xyt':
        return nc_var[xmin_index:xmax_index, ymin_index:ymax_index, :].data
    elif dim_order == 'yxt':
        return nc_var[ymin_index:ymax_index, xmin_index:xmax_index, :].data

    elif dim_order == 'xty':
        return nc_var[xmin_index:xmax_index, :, ymin_index:ymax_index].data
    elif dim_order == 'ytx':
        return nc_var[ymin_index:ymax_index, :, xmin_index:xmax_index].data

    elif dim_order == 'xy':
        return nc_var[xmin_index:xmax_index, ymin_index:ymax_index].data
    elif dim_order == 'yx':
        return nc_var[ymin_index:ymax_index, xmin_index:xmax_index].data
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice netCDF array.')
