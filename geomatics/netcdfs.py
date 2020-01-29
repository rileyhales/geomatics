import datetime
import os

import dateutil
import netCDF4
import numpy as np
import pandas as pd
import rasterio
import rasterstats


def point_series(paths, variable, coordinates, filename_pattern, **kwargs):
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    t_var = kwargs.get('tvar', 'time')

    # confirm that a valid path to data was provided
    paths = __path_to_file_list(paths)
    paths.sort()

    # get a list of the x&y coordinates in the netcdfs using the first file as a reference
    nc_obj = netCDF4.Dataset(paths[0], 'r')
    nc_xs = nc_obj[x_var][:]
    nc_ys = nc_obj[y_var][:]
    # determine the index in the netcdf's coordinates for the xy coordinate provided
    x_index = (np.abs(nc_xs - round(float(coordinates[0]), 2))).argmin()
    y_index = (np.abs(nc_ys - round(float(coordinates[1]), 2))).argmin()
    dim_order = __get_dimension_order(nc_obj[variable].dimensions, x_var, y_var, t_var)
    nc_obj.close()

    # make the return item
    timeseries = []

    # iterate over each file, extracting the
    for path in paths:
        # open the file
        nc_obj = netCDF4.Dataset(path, 'r')
        # attempt to determine the correct datetime to use in the timeseries
        time = os.path.basename(path)
        time = datetime.datetime.strptime(time, filename_pattern)
        # slice the array at the area you want, depends on the order of the dimensions
        val = float(__slice_point(nc_obj[variable], dim_order, x_index, y_index))
        timeseries.append((time, val))
        nc_obj.close()

    # sort the list by the 0 entry (the date), turn it into a pd dataframe, return it
    timeseries.sort(key=lambda tup: tup[0])
    return pd.DataFrame(timeseries, columns=['datetime', 'values'])


def box_series(paths, variable, coordinates, filename_pattern, **kwargs):
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    t_var = kwargs.get('tvar', 'time')
    fill_value = kwargs.get('fill_value', -9999)
    stat = kwargs.get('stat_type', 'mean')

    # confirm that a valid path to data was provided
    paths = __path_to_file_list(paths)

    # get a list of the x&y coordinates using the first file as a reference
    nc_obj = netCDF4.Dataset(paths[0], 'r')
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
    timeseries = []

    # extract values at each timestep
    for path in paths:
        # open the file
        nc_obj = netCDF4.Dataset(path, 'r')
        # determine the correct datetime to use in the timeseries
        time = os.path.basename(path)
        time = datetime.datetime.strptime(time, filename_pattern)
        # slice the array, drop nan values
        array = __slice_box(nc_obj[variable], dim_order, xmin_index, ymin_index, xmax_index, ymax_index)
        array[array == fill_value] = np.nan  # If you have fill values, change the comparator to git rid of it
        array = array.flatten()
        array = array[~np.isnan(array)]

        # get the specific value and append to the timeseries
        if stat == 'mean':
            timeseries.append((time, float(array.mean())))
        elif stat == 'max':
            timeseries.append((time, float(max(array))))
        elif stat == 'min':
            timeseries.append((time, float(min(array))))
        else:
            raise ValueError('Unrecognized statistic, {}. Choose stat_type= mean, min or max'.format(stat))

        nc_obj.close()

    # sort the list by the 0 entry (the date), turn it into a pd dataframe, return it
    timeseries.sort(key=lambda tup: tup[0])
    return pd.DataFrame(timeseries, columns=['datetime', 'values'])


def shp_series(paths, variable, shp_path, filename_pattern, **kwargs):
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    t_var = kwargs.get('tvar', 'time')
    fill_value = kwargs.get('fill_value', -9999)
    stat = kwargs.get('stat_type', 'mean')

    # confirm that a valid path to data was provided
    paths = __path_to_file_list(paths)

    # open the netcdf determine the affine transformation of the netcdf grids
    nc_obj = netCDF4.Dataset(paths[0], 'r')
    nc_xs = nc_obj.variables[x_var][:]
    nc_ys = nc_obj.variables[y_var][:]
    affine = rasterio.transform.from_origin(nc_xs.min(), nc_ys.max(), nc_ys[1] - nc_ys[0], nc_xs[1] - nc_xs[0])
    dim_order = __get_dimension_order(nc_obj[variable].dimensions, x_var, y_var, t_var)
    nc_obj.close()

    # make the return item
    timeseries = []

    # extract values at each timestep
    for path in paths:
        # open the file
        nc_obj = netCDF4.Dataset(path, 'r')
        # determine the correct datetime to use in the timeseries
        time = os.path.basename(path)
        time = datetime.datetime.strptime(time, filename_pattern)

        # this is the array of values for the nc_obj
        array = np.asarray(nc_obj.variables[variable][:])
        # if time was one of the dimensions, we need to remove it
        if 't' in dim_order:
            array = __slice_shape(array, dim_order)  # converting the array from 3D to 2D (removing the
        # drop fill and no data entries
        array[array == fill_value] = np.nan
        # vertically flip array so tiff orientation is right (you just have to, try it)
        array = array[::-1]
        # actually do the gis to get the value within the shapefile
        stats = rasterstats.zonal_stats(shp_path, array, affine=affine, nodata=np.nan, stats=stat)
        tmp = [i['mean'] for i in stats if i['mean'] is not None]
        timeseries.append((time, sum(tmp) / len(tmp)))

        nc_obj.close()

    # sort the list by the 0 entry (the date), turn it into a pd dataframe, return it
    timeseries.sort(key=lambda tup: tup[0])
    return pd.DataFrame(timeseries, columns=['datetime', 'values'])


def convert_to_geotiff(paths, variable, **kwargs):
    """
    :param variable: the short-code name of the variable within the netcdf
    :param kwargs: save_dir - the name of the directory to save the files to
    :return:
        output_files: list of the full paths of the tifs that were created
        geotransform: a dictionary of the dimensions of the output tifs
    """
    paths = __path_to_file_list(paths)

    # parse the optional argument from the kwargs
    save_dir = kwargs.get('save_dir', os.path.dirname(paths[0]))
    delete_sources = kwargs.get('delete_sources', False)
    fill_value = kwargs.get('fill_value', -9999)

    # open the first netcdf and collect georeferencing information
    nc_obj = netCDF4.Dataset(paths[0], 'r')
    lat = nc_obj.variables['lat'][:]
    lon = nc_obj.variables['lon'][:]
    lon_min = lon.min()
    lon_max = lon.max()
    lat_min = lat.min()
    lat_max = lat.max()
    data = nc_obj[variable][:]
    data = data[0]
    height = data.shape[0]
    width = data.shape[1]
    nc_obj.close()

    # Geotransform for each of the netcdf files
    gt = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    # A list of all the files that get written which can be returned
    output_files = []

    # Create a geotiff for each netcdf in the list of files
    for path in paths:
        # set the paths to open/save
        save_path = os.path.join(save_dir, os.path.basename(path) + '.tif')
        output_files.append(save_path)

        # open the netcdf and get the data array
        nc_obj = netCDF4.Dataset(path, 'r')
        array = np.asarray(nc_obj[variable][:])
        array = array[0]
        array[array == fill_value] = np.nan  # If you have fill values, change the comparator to git rid of it
        array = np.flip(array, axis=0)
        nc_obj.close()

        # if you want to delete the source netcdfs as you go
        if delete_sources:
            os.remove(path)

        # write it to a geotiff
        with rasterio.open(
                save_path,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                nodata=np.nan,
                crs='+proj=latlong',
                transform=gt,
        ) as dst:
            dst.write(array, 1)

    return output_files, dict(
        lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, height=height, width=width)


def __path_to_file_list(path):
    # check that a valid path was provided
    if isinstance(path, str):
        if os.path.isfile(path):
            return [path]
        elif os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.nc') or f.endswith('.nc4')]
            if len(files) == 0:
                raise FileNotFoundError('No netcdfs located within this directory')
            return files
        else:
            raise FileNotFoundError('No netcdf file or directory found at this path')
    elif isinstance(path, list):
        return path
    else:
        raise ValueError('Provide an absolute file path to a netcdf or directory of netcdf files, or a list of paths')


def __guess_timedelta(nc_obj, t_var, step=1):
    units = str(nc_obj[t_var].__dict__['units'])
    units = units.replace(' ', '').lower()
    if units.startswith('years'):
        return dateutil.relativedelta.relativedelta(years=step)
    elif units.startswith('months'):
        return dateutil.relativedelta.relativedelta(months=step)
    elif units.startswith('weeks'):
        return datetime.timedelta(weeks=step)
    elif units.startswith('days'):
        return datetime.timedelta(days=step)
    elif units.startswith('hours'):
        return datetime.timedelta(hours=step)
    elif units.startswith('minutes'):
        return datetime.timedelta(minutes=step)
    elif units.startswith('seconds'):
        return datetime.timedelta(seconds=step)
    else:
        raise ValueError("Timedelta was not specified and could not be guessed from the time variable's metadata")


def __get_dimension_order(dimensions, x_var, y_var, t_var):
    # check if the variable has 2 dimensions ie only xy coordinates (georeferenced)
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
        return nc_var[0, x_index, y_index].data
    elif dim_order == 'tyx':
        return nc_var[0, y_index, x_index].data

    elif dim_order == 'xyt':
        return nc_var[x_index, y_index, 0].data
    elif dim_order == 'yxt':
        return nc_var[y_index, x_index, 0].data

    elif dim_order == 'xty':
        return nc_var[x_index, 0, y_index].data
    elif dim_order == 'ytx':
        return nc_var[y_index, 0, x_index].data

    elif dim_order == 'xy':
        return nc_var[x_index, y_index].data
    elif dim_order == 'yx':
        return nc_var[y_index, x_index].data
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice netCDF array.')


def __slice_box(nc_var, dim_order, xmin_index, ymin_index, xmax_index, ymax_index):
    if dim_order == 'txy':
        return nc_var[0, xmin_index:xmax_index, ymin_index:ymax_index].data
    elif dim_order == 'tyx':
        return nc_var[0, ymin_index:ymax_index, xmin_index:xmax_index].data

    elif dim_order == 'xyt':
        return nc_var[xmin_index:xmax_index, ymin_index:ymax_index, 0].data
    elif dim_order == 'yxt':
        return nc_var[ymin_index:ymax_index, xmin_index:xmax_index, 0].data

    elif dim_order == 'xty':
        return nc_var[xmin_index:xmax_index, 0, ymin_index:ymax_index].data
    elif dim_order == 'ytx':
        return nc_var[ymin_index:ymax_index, 0, xmin_index:xmax_index].data

    elif dim_order == 'xy':
        return nc_var[xmin_index:xmax_index, ymin_index:ymax_index].data
    elif dim_order == 'yx':
        return nc_var[ymin_index:ymax_index, xmin_index:xmax_index].data
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice netCDF array.')


def __slice_shape(array, dim_order):
    if dim_order == 'txy':
        return array[0, :, :]
    elif dim_order == 'tyx':
        return array[0, :, :]

    elif dim_order == 'xyt':
        return array[:, :, 0]
    elif dim_order == 'yxt':
        return array[:, :, 0]

    elif dim_order == 'xty':
        return array[:, 0, :]
    elif dim_order == 'ytx':
        return array[:, 0, :]

    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice netCDF array.')
