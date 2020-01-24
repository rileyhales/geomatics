import datetime
import dateutil
import os
import netCDF4
import numpy
import rasterio
import rasterstats
import pandas

#todo
# allow the user to specify min, mean, max, mode for the box series
# handle dimensions in different orders when slicing the array


def point_series(paths, variable, coordinates, filename_pattern, **kwargs):
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')

    # confirm that a valid path to data was provided
    paths = __get_list_of_files(paths)
    paths.sort()

    # get a list of the x&y coordinates using the first file as a reference
    nc_obj = netCDF4.Dataset(paths[0], 'r')
    nc_xs = nc_obj[x_var][:]
    nc_ys = nc_obj[y_var][:]
    # get the index number of the x&y coordinate provided for the point
    x_index = (numpy.abs(nc_xs - round(float(coordinates[0]), 2))).argmin()
    y_index = (numpy.abs(nc_ys - round(float(coordinates[1]), 2))).argmin()
    nc_obj.close()

    # make the return item
    timeseries = []

    # if it is a single file, extract the values at each entry of the time variable's list
    # if len(paths) == 0:
        # nc_obj = netCDF4.Dataset(paths[0], 'r')
        # if not time_increment:
        #     time_increment = __guess_timedelta(nc_obj)
        # for t_step in nc_obj['time'][:]:
        #     # the time of this data is the start date plus the timedelta time the number of steps away it is.
        #     time = startdatetime + (time_increment * float(nc_obj['time'][t_step].data))
        #     # slice the array at the area you want, depends on the order of the dimensions
        #     timeseries.append((time, nc_obj[variable][t_step, y_index, x_index].data))
        # nc_obj.close()

    # if there were many files, iterate over each file
    for path in paths:
        # open the file
        nc_obj = netCDF4.Dataset(path, 'r')
        # attempt to determine the correct datetime to use in the timeseries
        time = os.path.basename(path)
        time = datetime.datetime.strptime(time, filename_pattern)
        # slice the array at the area you want, depends on the order of the dimensions
        val = float(nc_obj[variable][0, y_index, x_index].data)
        timeseries.append((time, val))
        nc_obj.close()

    # sort the list by the 0 entry (the date), turn it into a pandas dataframe, return it
    timeseries.sort(key=lambda tup: tup[0])
    return pandas.DataFrame(timeseries, columns=['datetime', 'values'])


def box_series(paths, variable, coordinates, startdatetime, **kwargs):
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    # now figure out what variables we were given to try to determine the timesteps
    filename_timestring = kwargs.get('filename_timestring', None)
    time_metadata_var = kwargs.get('time_metadata_var', 'time')
    time_attr = kwargs.get('time_attr', None)
    timestring_format = kwargs.get('timestring_format', None)
    time_increment = kwargs.get('time_increment', None)

    # confirm that a valid path to data was provided
    paths = __get_list_of_files(paths)

    # get a list of the x&y coordinates using the first file as a reference
    nc_obj = netCDF4.Dataset(paths[0], 'r')
    nc_xs = nc_obj[x_var][:]
    nc_ys = nc_obj[y_var][:]
    # get the indices of the bounding box corners
    xmin_index = (numpy.abs(nc_xs - round(float(coordinates[0]), 2))).argmin()
    ymin_index = (numpy.abs(nc_ys - round(float(coordinates[1]), 2))).argmin()
    xmax_index = (numpy.abs(nc_xs - round(float(coordinates[2]), 2))).argmin()
    ymax_index = (numpy.abs(nc_ys - round(float(coordinates[3]), 2))).argmin()
    nc_obj.close()

    # make the return item
    timeseries = []

    # extract values at each timestep
    for t_step, path in enumerate(paths):
        # open the file
        nc_obj = netCDF4.Dataset(path, 'r')

        # attempt to determine the correct datetime to use in the timeseries
        try:
            if filename_timestring:
                time = os.path.basename(path)
                time = datetime.datetime.strptime(time, filename_timestring)
            # elif time_attr and timestring_format:
            #     time = datetime.datetime.strptime(nc_obj['time'].__dict__[time_attr], timestring_format)
            # elif time_increment:
            #     time = startdatetime + (time_increment * float(nc_obj['time'][t_step].data))
        except Exception:
            raise Exception('Unable to parse the time from the ')

        # slice the array, drop nan values, get the mean, append to list of values
        array = nc_obj[variable][0, ymin_index:ymax_index, xmin_index:xmax_index].data
        array[array < -5000] = numpy.nan  # If you have fill values, change the comparator to git rid of it
        array = array.flatten()
        array = array[~numpy.isnan(array)]
        print(array)
        timeseries.append((time, float(array.mean())))
        nc_obj.close()

    # sort the list by the 0 entry (the date), turn it into a pandas dataframe, return it
    timeseries.sort(key=lambda tup: tup[0])
    return pandas.DataFrame(timeseries, columns=['datetime', 'values'])


def shp_series(paths, variable, shp_path, startdatetime, **kwargs):
    # for customizing the workflow for standards non-compliant netcdf files
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    # now figure out what variables we were given to try to determine the timesteps
    filename_timestring = kwargs.get('filename_timestring', None)
    time_metadata_var = kwargs.get('time_metadata_var', 'time')
    time_attr = kwargs.get('time_attr', None)
    timestring_format = kwargs.get('timestring_format', None)
    time_increment = kwargs.get('time_increment', None)

    # confirm that a valid path to data was provided
    paths = __get_list_of_files(paths)

    # open the netcdf and get metadata
    nc_obj = netCDF4.Dataset(paths[0], 'r')
    nc_xs = nc_obj.variables['lon'][:]
    nc_ys = nc_obj.variables['lat'][:]
    affine = rasterio.transform.from_origin(nc_xs.min(), nc_ys.max(), nc_ys[1] - nc_ys[0], nc_xs[1] - nc_xs[0])
    nc_obj.close()

    # make the return item
    timeseries = []

    # extract values at each timestep
    for t_step, path in enumerate(paths):
        # open the file
        nc_obj = netCDF4.Dataset(path, 'r')

        # attempt to determine the correct datetime to use in the timeseries
        try:
            if filename_timestring:
                time = os.path.basename(path)
                time = datetime.datetime.strptime(time, filename_timestring)
            # elif time_attr and timestring_format:
            #     time = datetime.datetime.strptime(nc_obj['time'].__dict__[time_attr], timestring_format)
            # elif time_increment:
            #     time = startdatetime + (time_increment * float(nc_obj['time'][t_step].data))
        except Exception:
            raise Exception('Unable to parse the time from the ')

        array = nc_obj.variables[variable][:]  # this is the array of values for the nc_obj
        array = numpy.asarray(array)[0, :, :]  # converting the array from 3D to 2D
        array[array < -9000] = numpy.nan  # use the comparator to drop nodata fills
        array = array[::-1]  # vertically flip array so tiff orientation is right (you just have to, try it)
        stats = rasterstats.zonal_stats(shp_path, array, affine=affine, nodata=numpy.nan, stats="mean")
        tmp = [i['mean'] for i in stats if i['mean'] is not None]
        timeseries.append((time, sum(tmp) / len(tmp)))

        nc_obj.close()

    # sort the list by the 0 entry (the date), turn it into a pandas dataframe, return it
    timeseries.sort(key=lambda tup: tup[0])
    return pandas.DataFrame(timeseries, columns=['datetime', 'values'])


def convert_to_geotiff(nc_path, var, **kwargs):
    """
    Turns netcdfs into geotiffs

    :param nc_path: path to a netcdf for a directory containing netcdf files
    :param var: the short-code name of the variable within the netcdf
    :param kwargs: save_dir - the name of the directory to save the files to
    :return:
        output_files: list of the full paths of the tifs that were created
        geotransform: a dictionary of the dimensions of the output tifs
    """

    # list all the files in the user specified directory
    if os.path.isdir(nc_path):
        files = os.listdir(nc_path)
        files = [i for i in files if i.endswith('.nc') or i.endswith('.nc4')]
        files.sort()
    elif os.path.isfile(nc_path):
        split = os.path.split(nc_path)
        nc_path = split[0]
        files = [split[1]]

    # parse the optional argument from the kwargs
    save_dir = kwargs.get('save_dir', nc_path)
    delete_sources = kwargs.get('delete_sources', False)

    # open the first netcdf and collect georeferencing information
    nc_obj = netCDF4.Dataset(os.path.join(nc_path, files[0]), 'r')
    lat = nc_obj.variables['lat'][:]
    lon = nc_obj.variables['lon'][:]
    lon_min = lon.min()
    lon_max = lon.max()
    lat_min = lat.min()
    lat_max = lat.max()
    data = nc_obj[var][:]
    data = data[0]
    height = data.shape[0]
    width = data.shape[1]
    nc_obj.close()

    # Geotransform for each of the netcdf files
    gt = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    # A list of all the files that get written which can be returned
    output_files = []

    # Create a geotiff for each netcdf in the list of files
    for nc in files:
        # set the paths to open/save
        path = os.path.join(nc_path, nc)
        save_path = os.path.join(save_dir, nc + '.tif')
        output_files.append(save_path)

        # open the netcdf and get the data array
        nc_obj = netCDF4.Dataset(path, 'r')
        data = numpy.asarray(nc_obj[var][:])
        data = data[0]
        data = numpy.flip(data, axis=0)
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
                nodata=numpy.nan,
                crs='+proj=latlong',
                transform=gt,
        ) as dst:
            dst.write(data, 1)

    return output_files, dict(
        lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, height=height, width=width)


def __get_list_of_files(path):
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


def __guess_timedelta(nc_dataset):
    units = str(nc_dataset['time'].__dict__['units'])
    units = units.replace(' ', '').lower()
    if units.startswith('years'):
        return dateutil.relativedelta.relativedelta(years=1)
    elif units.startswith('months'):
        return dateutil.relativedelta.relativedelta(months=1)
    elif units.startswith('weeks'):
        return datetime.timedelta(weeks=1)
    elif units.startswith('days'):
        return datetime.timedelta(days=1)
    elif units.startswith('hours'):
        return datetime.timedelta(hours=1)
    elif units.startswith('minutes'):
        return datetime.timedelta(minutes=1)
    elif units.startswith('seconds'):
        return datetime.timedelta(seconds=1)
    else:
        raise ValueError("Timedelta was not specified and could not be guessed from the time variable's metadata")


ts = box_series('/Users/rileyhales/thredds/gldas/raw', 'Tair_f_inst', [10, 10, 20, 20],
                datetime.date(year=2010, month=1, day=1), filename_timestring='GLDAS_NOAH025_M.A%Y%m.021.nc4')
print(ts)
