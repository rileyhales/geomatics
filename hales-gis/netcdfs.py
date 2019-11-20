import os
import netCDF4
import numpy
import rasterio
from rasterio.enums import Resampling


def to_geotiff(nc_path, var, **kwargs):
    """
    turns every netcdf in a directory into a a geotiff

    netcdfs must be
    - correctly formatted with the lat and lon variables and dimensions
    - the variable specified needs to be
    """
    # parse the optional arguments from the kwargs
    save_dir = kwargs.get('save_dir', nc_path)
    rs_fctr = kwargs.get('resample_factor', None)

    # list all the files in the user specified directory
    # todo make this work for a single file or an entire directory
    files = os.listdir(nc_path)
    files = [i for i in files if i.endswith('.nc') or i.endswith('.nc4')]
    files.sort()

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

    # Create a geotiff for each netcdf in the list of files
    for nc in files:
        # set the paths to open/save
        path = os.path.join(nc_path, nc)
        save_path = os.path.join(save_dir, nc + '.tif')

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

        # if we're resampling, rewrite the
        if rs_fctr:
            gt = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width * rs_fctr, height * rs_fctr)

            with rasterio.open(save_path) as dataset:
                data = dataset.read(
                    out_shape=(height * 10, width * 10),
                    resampling=Resampling.nearest
                )

            # Convert new resampled array from 3D to 2D
            data = numpy.squeeze(data, axis=0)

            # Save the GeoTIFF
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

    return


to_geotiff('/Users/rileyhales/NLDAS/2015event/', 'APCP')
