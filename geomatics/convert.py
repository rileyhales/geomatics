import os

import affine
import numpy as np
import rasterio
import shapefile
from rasterio.enums import Resampling

from ._utils import open_by_engine, array_by_engine
from .data import gen_affine

__all__ = ['geojson_to_shapefile', 'to_geotiff', 'upsample_geotiff']


def geojson_to_shapefile(geojson: dict, savepath: str) -> None:
    """
    Turns a valid dict, json, or geojson containing polygon data in a geographic coordinate system into a shapefile

    Args:
        geojson: a valid geojson as a dictionary or json python object. try json.loads for strings
        savepath: the full file path to save the shapefile to, including the file_name.shp
    """
    # create the shapefile
    fileobject = shapefile.Writer(target=savepath, shpType=shapefile.POLYGON, autoBalance=True)

    # label all the columns in the .dbf
    geomtype = geojson['features'][0]['geometry']['type']
    if geojson['features'][0]['properties']:
        for attribute in geojson['features'][0]['properties']:
            fileobject.field(str(attribute), 'C', '30')
    else:
        fileobject.field('Name', 'C', '50')

    # add the geometry and attribute data
    for feature in geojson['features']:
        # record the geometry
        if geomtype == 'Polygon':
            fileobject.poly(polys=feature['geometry']['coordinates'])
        elif geomtype == 'MultiPolygon':
            for i in feature['geometry']['coordinates']:
                fileobject.poly(polys=i)

        # record the attributes in the .dbf
        if feature['properties']:
            fileobject.record(**feature['properties'])
        else:
            fileobject.record('unknown')

    # close writing to the shapefile
    fileobject.close()

    # create a prj file
    if savepath.endswith('.shp'):
        savepath.replace('.shp', '')
    with open(savepath + '.prj', 'w') as prj:
        prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
                  'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')

    return


def to_geotiff(files: list,
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
        file_obj = open_by_engine(file, engine, xr_kwargs)
        array = np.asarray(array_by_engine(file_obj, var=var, h5_group=h5_group))
        array = np.squeeze(array)
        array[array == fill_value] = np.nan  # If you have fill values, change the comparator to git rid of it
        array = np.flip(array, axis=0)
        file_obj.close()

        # if you want to delete the source netcdfs as you go
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


def upsample_geotiff(files: list, scale: float) -> list:
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
