import os

import netCDF4
import rasterio
import numpy as np

from .__utilities import path_to_file_list

__all__ = ['convert_netcdf']


def convert_netcdf(path, variable, **kwargs):
    """
    Args:
        path: Either 1) the absolute path to a directory containing netcdfs named by date or 2) the absolute path to
            a single netcdf containing many time values for a specified variable
        variable: The name of a variable as it is stored in the netcdf e.g. 'temp' instead of Temperature

    Keyword Args:
        xvar: Name of the x coordinate variable used to spatial reference the netcdf array. Default: 'lon' (longitude)
        yvar: Name of the y coordinate variable used to spatial reference the netcdf array. Default: 'lat' (latitude)
        save_dir: The directory to store the geotiffs to. Default: directory containing the netcdfs.
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        delete_source: Allows you to delete the source netcdfs as they are converted. Default: False

    Returns:
        1. A list of paths to the geotiff files created
        2. A dictionary that contains the affine geotransform information
    """
    files = path_to_file_list(path, 'nc')

    # parse the optional argument from the kwargs
    x_var = kwargs.get('xvar', 'lon')
    y_var = kwargs.get('yvar', 'lat')
    save_dir = kwargs.get('save_dir', os.path.dirname(files[0]))
    delete_sources = kwargs.get('delete_sources', False)
    fill_value = kwargs.get('fill_value', -9999)

    # open the first netcdf and collect georeferencing information
    nc_obj = netCDF4.Dataset(files[0], 'r')
    lat = nc_obj.variables[x_var][:]
    lon = nc_obj.variables[y_var][:]
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
    for file in files:
        # set the files to open/save
        save_path = os.path.join(save_dir, os.path.basename(file) + '.tif')
        output_files.append(save_path)

        # open the netcdf and get the data array
        nc_obj = netCDF4.Dataset(file, 'r')
        array = np.asarray(nc_obj[variable][:])
        array = array[0]
        array[array == fill_value] = np.nan  # If you have fill values, change the comparator to git rid of it
        array = np.flip(array, axis=0)
        nc_obj.close()

        # if you want to delete the source netcdfs as you go
        if delete_sources:
            os.remove(file)

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
