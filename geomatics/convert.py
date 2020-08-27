import datetime
import os

import affine
import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.enums import Resampling

from ._utils import _open_by_engine, _array_by_engine
from .data import gen_affine

__all__ = ['to_gtiffs', 'to_mb_gtiff', 'upsample_gtiff']


def to_gtiffs(files: list,
              var: str,
              engine: str = None,
              aff: affine.Affine = None,
              crs: str = 'EPSG:4326',
              x_var: str = 'lon',
              y_var: str = 'lat',
              xr_kwargs: dict = None,
              h5_group: str = None,
              fill_value: int = -9999,
              save_dir: str = False,
              delete_sources: bool = False) -> list:
    """
    Converts the array of data for a certain variable in a grib file to a geotiff.

    Args:
        files: A list of absolute paths to the appropriate type of files (even if len==1)
        var: The name of a variable as it is stored in the netcdf e.g. 'temp' instead of Temperature
        engine: the python package used to power the file reading
        aff: an affine.Affine transformation for the data if you already know what it is
        crs: Coordinate Reference System used by rasterio.open(). An EPSG such as 'EPSG:4326' or '+proj=latlong'
        x_var: Name of the x coordinate variable used to spatial reference the netcdf array. Default: 'lon'
        y_var: Name of the y coordinate variable used to spatial reference the netcdf array. Default: 'lat'
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray
        save_dir: The directory to store the geotiffs to. Default: directory containing the netcdfs.
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        delete_sources: Allows you to delete the source netcdfs as they are converted. Default: False

    Returns:
        A list of paths to the geotiff files created
    """
    if isinstance(files, str):
        files = [files, ]
    if aff is None:
        aff = gen_affine(files[0], x_var, y_var, engine=engine, xr_kwargs=xr_kwargs)

    # A list of all the files that get written which can be returned
    output_files = []

    # Create a geotiff for each netcdf in the list of files
    for file in files:
        # set the files to open/save
        if not save_dir:
            save_path = os.path.join(os.path.dirname(file), os.path.splitext(os.path.basename(file))[0] + '.tif')
        else:
            save_path = os.path.join(save_dir, os.path.basename(file) + '.tif')
        output_files.append(save_path)

        # open the netcdf and get the data array
        file_obj = _open_by_engine(file, engine, xr_kwargs)
        array = np.asarray(_array_by_engine(file_obj, var=var, h5_group=h5_group))
        array = np.squeeze(array)
        array[array == fill_value] = np.nan  # If you have fill values, change the comparator to git rid of it
        array = np.flip(array, axis=0)
        file_obj.close()

        # if you want to delete the source files as you go
        if delete_sources:
            os.remove(file)

        # write it to a geotiff
        with rasterio.open(
                save_path,
                'w',
                driver='GTiff',
                height=array.shape[0],
                width=array.shape[1],
                count=1,
                dtype=array.dtype,
                nodata=np.nan,
                crs=crs,
                transform=aff,
        ) as dst:
            dst.write(array, 1)

    return output_files


def to_mb_gtiff(files: list,
                var: str,
                engine: str = None,
                aff: affine.Affine = None,
                crs: str = 'EPSG:4326',
                x_var: str = 'lon',
                y_var: str = 'lat',
                xr_kwargs: dict = None,
                h5_group: str = None,
                fill_value: int = -9999,
                save_dir: str = False,
                save_name: str = False,
                delete_sources: bool = False) -> str:
    """
    Converts a 3D array of data for a certain variable in one or many files to a multiband geotiff.

    Args:
        files: A list of absolute paths to the appropriate type of files (even if len==1)
        var: The name of a variable as it is stored in the netcdf e.g. 'temp' instead of Temperature
        engine: the python package used to power the file reading
        aff: an affine.Affine transformation for the data if you already know what it is
        crs: Coordinate Reference System used by rasterio.open(). An EPSG such as 'EPSG:4326' or '+proj=latlong'
        x_var: Name of the x coordinate variable used to spatial reference the netcdf array. Default: 'lon'
        y_var: Name of the y coordinate variable used to spatial reference the netcdf array. Default: 'lat'
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray
        save_dir: The directory to store the geotiffs to. Default: directory containing the netcdfs.
        save_name: The name of the output geotiff file including the '.tif' extension.
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        delete_sources: Allows you to delete the source netcdfs as they are converted. Default: False

    Returns:
        A list of paths to the geotiff files created
    """
    if isinstance(files, str):
        files = [files, ]
    if aff is None:
        aff = gen_affine(files[0], x_var, y_var, engine=engine, xr_kwargs=xr_kwargs)

    if not save_name:
        save_name = 'multiband_collection.tif'
    if not save_dir:
        save_dir = os.path.join(os.path.dirname(files[0]), save_name)
    if not os.path.exists(save_dir):
        raise NotADirectoryError(f'Directory to save output file not found at path: {save_dir}')
    save_path = os.path.join(save_dir, save_name)

    # open the netcdf and get the data array
    file_obj = _open_by_engine(files[0], engine, xr_kwargs)
    array = np.squeeze(np.asarray(_array_by_engine(file_obj, var=var, h5_group=h5_group)))
    array[array == fill_value] = np.nan  # If you have fill values, change the comparator to git rid of it
    array = np.flip(array, axis=0)
    file_obj.close()

    # if you want to delete the source netcdfs as you go
    if delete_sources:
        os.remove(files[0])

    # write it to a geotiff
    with rasterio.open(
            save_path,
            'w',
            driver='GTiff',
            height=array.shape[0],
            width=array.shape[1],
            count=len(files),
            dtype=array.dtype,
            nodata=np.nan,
            crs=crs,
            transform=aff,
    ) as dst:
        # write the first array that we used to get the referencing information
        dst.write(array, 1)
        files.pop(0)

        # now add the rest of the arrays for the remaining files
        for i, file in enumerate(files):
            file_obj = _open_by_engine(file, engine, xr_kwargs)
            array = np.squeeze(np.asarray(_array_by_engine(file_obj, var=var, h5_group=h5_group)))
            array[array == fill_value] = np.nan  # If you have fill values, change the comparator to git rid of it
            array = np.flip(array, axis=0)
            file_obj.close()

            # if you want to delete the source files as you go
            if delete_sources:
                os.remove(file)

            dst.write(array, i + 2)

    return save_path


def upsample_gtiff(files: list, scale: float) -> list:
    """
    Performs array math to artificially increase the resolution of a geotiff. No interpolation of values. A scale
    factor of X means that the length of a horizontal and vertical grid cell decreases by X. Be careful, increasing the
    resolution by X increases the file size by ~X^2

    Args:
        files: A list of absolute paths to the appropriate type of files (even if len==1)
        scale: A positive integer used as the multiplying factor to increase the resolution.

    Returns:
        list of paths to the geotiff files created
    """
    # Read raster dimensions
    raster_dim = rasterio.open(files[0])
    width = raster_dim.width
    height = raster_dim.height
    lon_min = raster_dim.bounds.left
    lon_max = raster_dim.bounds.right
    lat_min = raster_dim.bounds.bottom
    lat_max = raster_dim.bounds.top
    # Geotransform for each resampled raster (east, south, west, north, width, height)
    affine_resampled = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width * scale, height * scale)
    # keep track of the new files
    new_files = []

    # Resample each GeoTIFF
    for file in files:
        rio_obj = rasterio.open(file)
        data = rio_obj.read(
            out_shape=(int(rio_obj.height * scale), int(rio_obj.width * scale)),
            resampling=Resampling.nearest
        )
        # Convert new resampled array from 3D to 2D
        data = np.squeeze(data, axis=0)
        # Specify the filepath of the resampled raster
        new_filepath = os.path.splitext(file)[0] + '_upsampled.tiff'
        new_files.append(new_filepath)
        # Save the GeoTIFF
        with rasterio.open(
                new_filepath,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                nodata=np.nan,
                crs=rio_obj.crs,
                transform=affine_resampled,
        ) as dst:
            dst.write(data, 1)

    return new_files


def tif_to_nc(tif: str, var: str, time: datetime.datetime, dtype: str = 'i2', fill: int or str = 0,
              compress: bool = False, level: int = 9) -> None:
    """
    Converts a single tif to a netcdf compliant with the Common Data Model (CDM) and therefore able to be used by the
    THREDDS data server. Your tif MUST be projected to a global coordinate system using latitude and longtiude for best
    results with THREDDS so you MUST reproject your tif before using this function.

    Args:
        tif (str): path to the tif to convert
        var (str): name to assign the netcdf variable where the geotiff information is stored
        time (datetime.datetime): the start time of the data in the tiff
        dtype (str): the netcdf datatype of the variable to store in the new netcdf: default to i2. consult
            https://unidata.github.io/netcdf4-python/netCDF4/index.html
        fill (int or str): the fill value to apply when using a masked array in the new variable's data array
        compress (bool): True = compress the netcdf, False = do not compress the new file
        level (int): An integer between 1 and 9. 1 = least compression and 9 = most compression

    Returns:
        None
    """
    # read the tif with the xarray wrapper to rasterio for convenience in coding
    a = _open_by_engine(tif, engine='rasterio')
    shape = a.values.shape

    # create the new netcdf
    new_nc = nc.Dataset(f'{os.path.splitext(tif)[0]}.nc4', 'w')

    # create latitude dimension, variable, add values, metadata
    new_nc.createDimension('lat', shape[1])
    new_nc.createVariable('lat', 'f', ('lat',))
    new_nc['lat'].axis = "lat"
    new_nc['lat'][:] = a.y.values
    new_nc['lat'].units = "degrees_north"

    # create longitude dimension, variable, add values, metadata
    new_nc.createDimension('lon', shape[2])
    new_nc.createVariable('lon', 'f', ('lon',))
    new_nc['lon'][:] = a.x.values
    new_nc['lon'].axis = "lon"
    new_nc['lon'].units = "degrees_east"

    # create time dimension, variable, add values AND specify the units string (essential for thredds)
    new_nc.createDimension('time', 1)
    new_nc.createVariable('time', 'i2', ('time',))
    new_nc['time'].long_name = 'time'
    new_nc['time'].units = f'days since {time.strftime("%Y-%m-%d %X")}'
    new_nc['time'].calendar = 'standard'
    new_nc['time'].axis = 'T'

    # now create the variable which holds the tif's array (use a.values[0] because first dim is the band #)
    if compress:
        new_nc.createVariable(var, dtype, ('time', 'lat', 'lon'), fill_value=fill)
    else:
        new_nc.createVariable(var, dtype, ('time', 'lat', 'lon'), fill_value=fill, complevel=level, zlib=True)
    new_nc[var].axis = "lat lon"
    new_nc[var][:] = a.values

    # save and close the new_nc
    new_nc.sync()
    new_nc.close()
    return
