import h5py
import netCDF4 as nc
import xarray as xr

from ._utils import _open_by_engine, _array_by_engine

__all__ = ['netcdf', 'grib', 'hdf5', 'geotiff', 'georeferencing']


def netcdf(path: str) -> None:
    """
    Prints some useful summary information about a netcdf

    Args:
        path: The path to a netcdf file.
    """
    nc_obj = nc.Dataset(path, 'r', clobber=False, diskless=True, persist=False)

    print("This is your netCDF python object")
    print(nc_obj)
    print()

    print(f"There are {len(nc_obj.variables)} variables")  # The number of variables
    print(f"There are {len(nc_obj.dimensions)} dimensions")  # The number of dimensions
    print()

    print('These are the global attributes of the netcdf file')
    print(nc_obj.__dict__)  # access the global attributes of the netcdf file
    print()

    print("Detailed view of each variable")
    print()
    for variable in nc_obj.variables.keys():  # .keys() gets the name of each variable
        print('Variable Name:  ' + variable)  # The string name of the variable
        print('The view of this variable in the netCDF python object')
        print(nc_obj[variable])  # How to view the variable information (netcdf obj)
        print('The data array stored in this variable')
        print(nc_obj[variable][:])  # Access the numpy array inside the variable (array)
        print('The dimensions associated with this variable')
        print(nc_obj[variable].dimensions)  # Get the dimensions associated with a variable (tuple)
        print('The metadata associated with this variable')
        print(nc_obj[variable].__dict__)  # How to get the attributes of a variable (dictionary)
        print()

    for dimension in nc_obj.dimensions.keys():
        print(nc_obj.dimensions[dimension].size)  # print the size of a dimension

    nc_obj.close()  # close the file connection to the file
    return


def grib(path: str, xr_kwargs: dict = None) -> None:
    """
    Prints a summary of the information available when you open a grib with pygrib

    Args:
        path: The path to any grib file
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray
    """
    gb = xr.open_dataset(path, engine='cfgrib', backend_kwargs=xr_kwargs)
    print('These are the variables')
    print(gb.variables)
    print('These are the dimensions')
    print(gb.dims)
    return


def hdf5(path: str) -> None:
    """
    Prints lots of messages showing information about variables, dimensions, and metadata

    Args:
        path: The path to any HDF5 file
    """
    ds = h5py.File(path)
    print('The following groups/variables are contained in this HDF5 file')
    ds.visit(print)
    return


def geotiff(path: str) -> None:
    """
    Prints the information available when you open a geotiff with xarray

    Args:
        path: The path to any geotiff file
    """
    print(xr.open_rasterio(path))
    return


def georeferencing(file: str,
                   engine: str = None,
                   x_var: str = 'lon',
                   y_var: str = 'lat',
                   xr_kwargs: dict = None,
                   h5_group: str = None) -> dict:
    """
    Determines the information needed to create an affine transformation for a geo-referenced data array.

    Args:
        file: the absolute path to a netcdf or grib file
        engine: the python package used to power the file reading
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'lon' (longitude)
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'lat' (latitude)
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here

    Returns:
        A dictionary containing the information needed to create the affine transformation of a dataset.
    """
    # open the file to be read
    ds = _open_by_engine(file, engine, xr_kwargs)
    x_data = _array_by_engine(ds, x_var, h5_group)
    y_data = _array_by_engine(ds, y_var, h5_group)

    return {
        'x_first_val': x_data[0],
        'x_last_val': x_data[-1],
        'x_min': x_data.min(),
        'x_max': x_data.max(),
        'x_num_values': x_data.size,
        'x_resolution': x_data[1] - x_data[0],

        'y_first_val': y_data[0],
        'y_last_val': y_data[-1],
        'y_min': y_data.min(),
        'y_max': y_data.max(),
        'y_num_values': y_data.size,
        'y_resolution': y_data[1] - y_data[0]
    }
