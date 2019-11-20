import datetime as dt
import os
import netCDF4
import numpy
import rasterio
import rasterstats


def point_series(var, coords, path, files):
    # return items
    timeseries = []

    # get a list of the lat/lon and units using a reference file
    nc_obj = netCDF4.Dataset(os.path.join(path, files[0]), 'r')
    nc_lons = nc_obj['lon'][:]
    nc_lats = nc_obj['lat'][:]
    # get the index number of the lat/lon for the point
    lon_indx = (numpy.abs(nc_lons - round(float(coords[0]), 2))).argmin()
    lat_indx = (numpy.abs(nc_lats - round(float(coords[1]), 2))).argmin()
    nc_obj.close()

    # extract values at each timestep
    for nc in files:
        # get the time value for each file
        nc_obj = netCDF4.Dataset(os.path.join(path, nc), 'r')
        time = dt.datetime.strptime(nc_obj['time'].__dict__['begin_date'], "%Y%m%d")
        # slice the array at the area you want
        val = float(nc_obj[var][0, lat_indx, lon_indx].data)
        timeseries.append((time, val))
        nc_obj.close()

    return timeseries


def box_series(var, coords, path, files):
    # return items
    values = []

    # get a list of the latitudes and longitudes and the units
    nc_obj = netCDF4.Dataset(os.path.join(path, str(files[0])), 'r')
    nc_lons = nc_obj['lon'][:]
    nc_lats = nc_obj['lat'][:]
    units = nc_obj[var].__dict__['units']
    # get a bounding box of the rectangle in terms of the index number of their lat/lons
    minlon = (numpy.abs(nc_lons - round(float(coords[0][1][0]), 2))).argmin()
    maxlon = (numpy.abs(nc_lons - round(float(coords[0][3][0]), 2))).argmin()
    maxlat = (numpy.abs(nc_lats - round(float(coords[0][1][1]), 2))).argmin()
    minlat = (numpy.abs(nc_lats - round(float(coords[0][3][1]), 2))).argmin()
    nc_obj.close()

    # extract values at each timestep
    for nc in files:
        # set the time value for each file
        nc_obj = netCDF4.Dataset(os.path.join(path, nc), 'r')
        time = dt.datetime.strptime(nc_obj['time'].__dict__['begin_date'], "%Y%m%d")
        # slice the array, drop nan values, get the mean, append to list of values
        array = nc_obj[var][0, minlat:maxlat, minlon:maxlon].data
        array[array < -5000] = numpy.nan  # If you have fill values, change the comparator to git rid of it
        array = array.flatten()
        array = array[~numpy.isnan(array)]
        values.append((time, float(array.mean())))

        nc_obj.close()

    return values


def shp_series(var, shp_path, path, files):
    # return items
    values = []

    # open the netcdf and get metadata
    nc_obj = netCDF4.Dataset(os.path.join(path, files[0]), 'r')
    lat = nc_obj.variables['lat'][:]
    lon = nc_obj.variables['lon'][:]
    affine = rasterio.transform.from_origin(lon.min(), lat.max(), lat[1] - lat[0], lon[1] - lon[0])
    nc_obj.close()

    # extract the timeseries by iterating over each netcdf
    for nc in files:
        # open the netcdf and get the data array
        nc_obj = netCDF4.Dataset(os.path.join(path, nc), 'r')
        time = dt.datetime.strptime(nc_obj['time'].__dict__['begin_date'], "%Y%m%d")

        var_data = nc_obj.variables[var][:]  # this is the array of values for the nc_obj
        array = numpy.asarray(var_data)[0, :, :]  # converting the data type
        array[array < -9000] = numpy.nan  # use the comparator to drop nodata fills
        array = array[::-1]  # vertically flip array so tiff orientation is right (you just have to, try it)

        stats = rasterstats.zonal_stats(shp_path, array, affine=affine, nodata=numpy.nan, stats="mean")
        tmp = [i['mean'] for i in stats if i['mean'] is not None]
        values.append((time, sum(tmp) / len(tmp)))

        nc_obj.close()

    return values


def to_geotiff(nc_path, var, **kwargs):
    """
    Turns a netcdf or directory of netcdfs into a geotiff

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

    return output_files, dict(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min,
                              lat_max=lat_max, height=height, width=width)
