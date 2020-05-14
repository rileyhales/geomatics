import h5py
import numpy as np
import xarray as xr
from PIL import TiffImagePlugin, Image

__all__ = ['open_by_engine', 'array_by_engine', 'pick_engine', 'get_slicing_info', 'slice_array_cell',
           'slice_array_range', 'check_var_in_dataset']


def open_by_engine(path: str, engine: str = None, backend_kwargs: dict = None) -> np.array:
    if engine is None:
        engine = pick_engine(path)
    if backend_kwargs is None:
        backend_kwargs = dict()
    if engine == 'xarray':
        return xr.open_dataset(path, backend_kwargs=backend_kwargs)
    elif engine == 'cfgrib':
        return xr.open_dataset(path, engine='cfgrib', backend_kwargs=backend_kwargs)
    elif engine == 'h5py':
        return h5py.File(path, 'r')
    elif engine in ('PIL', 'pillow'):
        return Image.open(path, 'r')
    elif engine == 'rasterio':
        return xr.open_rasterio(path)
    else:
        raise ValueError(f'Unable to open file, unsupported engine: {engine}')


def array_by_engine(open_file, var: str, h5_group: str = None):
    if isinstance(open_file, xr.Dataset):  # xarray
        return open_file[var].data
    elif isinstance(open_file, h5py.Dataset):  # h5py
        # not using open_file[:] because [:] can't slice string data but ... catches it all
        if h5_group is not None:
            open_file = open_file[h5_group]
        return open_file[...]
    elif isinstance(open_file, TiffImagePlugin.TiffImageFile):  # geotiff
        return np.array(open_file)
    else:
        raise ValueError(f'Unrecognized opened file dataset: {type(open_file)}')


def pick_engine(path: str) -> str:
    if path.endswith('.nc') or path.endswith('.nc4'):
        return 'xarray'
    elif path.endswith('.grb') or path.endswith('.grib'):
        return 'cfgrib'
    elif path.endswith('.gtiff') or path.endswith('.tiff') or path.endswith('tif'):
        return 'rasterio'
    elif path.endswith('.h5') or path.endswith('.hd5') or path.endswith('.hdf5'):
        return 'h5py'
    else:
        raise ValueError(f'File does not match known files extension patterns: {path}')


def get_slicing_info(path: str,
                     var: str,
                     x_var: str,
                     y_var: str,
                     t_var: str,
                     coords: tuple,
                     engine: str = None,
                     xr_kwargs: dict = None,
                     h5_group: str = None, ) -> dict:
    if engine is None:
        engine = pick_engine(path)
    # open the file to be read
    tmp_file = open_by_engine(path, engine, xr_kwargs)

    # validate choice in variables
    if not check_var_in_dataset(tmp_file, var, h5_group):
        raise ValueError(f'the variable "{var}" was not found in the file {path}')

    # get a list of the x&y coordinates
    x_steps = array_by_engine(tmp_file, x_var)
    y_steps = array_by_engine(tmp_file, y_var)

    # if the coordinate data was stored in 2d arrays instead of 1d lists of steps
    if x_steps.ndim == 2:
        # select the first row
        x_steps = x_steps[0, :]
    if y_steps.ndim == 2:
        # select the first column
        y_steps = y_steps[:, 0]

    assert x_steps.ndim == 1
    assert y_steps.ndim == 1

    # if its a netcdf or grib, the dimensions should be included by xarray
    if engine in ('xarray', 'cfgrib', 'netcdf4', 'rasterio'):
        dims = list(tmp_file[var].dims)
        for i, dim in enumerate(dims):
            dims[i] = str(dim).replace(x_var, 'x').replace(y_var, 'y').replace(t_var, 't')
        dims = str.join('', dims)

    # guess the dimensions based on the shape of the variable array and length of the x/y steps
    elif engine == 'hdf5':
        if h5_group is not None:
            tmp_file = tmp_file[h5_group]
        shape = list(array_by_engine(tmp_file, engine, var).shape)
        for i, length in enumerate(shape):
            if length == len(x_steps):
                shape[i] = 'x'
            elif length == len(y_steps):
                shape[i] = 'y'
            else:
                shape[i] = 't'
        dims = str.join('', shape)
    else:
        dims = False

    tmp_file.close()

    if coords is None:
        return dict(dim_order=dims)

    # gather all the indices
    indices = []
    x_min = x_steps.min()
    x_max = x_steps.max()
    y_min = y_steps.min()
    y_max = y_steps.max()
    for coord in coords:
        # first verify that the location is in the bounds of the coordinate variables
        x, y = coord
        x = float(x)
        y = float(y)
        if x < x_min or x > x_max:
            raise ValueError(f'specified x value ({x}) is outside the bounds of the data: [{x_min}, {x_max}]')
        if y < y_min or y > y_max:
            raise ValueError(f'specified x value ({y}) is outside the bounds of the data: [{y_min}, {y_max}]')
        # then calculate the indicies and append to the list of indices
        indices.append(((np.abs(x_steps - x)).argmin(), (np.abs(y_steps - y)).argmin()), )

    return dict(indices=indices, dim_order=dims)


def slice_array_cell(array, dim_order, x_idx, y_idx):
    if dim_order == 'txy':
        return array[:, x_idx, y_idx]
    elif dim_order == 'tyx':
        return array[:, y_idx, x_idx]

    elif dim_order == 'xyt':
        return array[x_idx, y_idx, :]
    elif dim_order == 'yxt':
        return array[y_idx, x_idx, :]

    elif dim_order == 'xty':
        return array[x_idx, :, y_idx]
    elif dim_order == 'ytx':
        return array[y_idx, :, x_idx]

    elif dim_order == 'xy':
        return array[x_idx, y_idx]
    elif dim_order == 'yx':
        return array[y_idx, x_idx]
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice array.')


def slice_array_range(array, dim_order, xmin_index, ymin_index, xmax_index, ymax_index):
    if dim_order == 'txy':
        return array[:, xmin_index:xmax_index, ymin_index:ymax_index]
    elif dim_order == 'tyx':
        return array[:, ymin_index:ymax_index, xmin_index:xmax_index]

    elif dim_order == 'xyt':
        return array[xmin_index:xmax_index, ymin_index:ymax_index, :]
    elif dim_order == 'yxt':
        return array[ymin_index:ymax_index, xmin_index:xmax_index, :]

    elif dim_order == 'xty':
        return array[xmin_index:xmax_index, :, ymin_index:ymax_index]
    elif dim_order == 'ytx':
        return array[ymin_index:ymax_index, :, xmin_index:xmax_index]

    elif dim_order == 'xy':
        return array[xmin_index:xmax_index, ymin_index:ymax_index]
    elif dim_order == 'yx':
        return array[ymin_index:ymax_index, xmin_index:xmax_index]
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice array.')


def check_var_in_dataset(open_file, variable, h5_group):
    if isinstance(open_file, xr.Dataset):  # xarray
        return bool(variable in open_file.variables)
    elif isinstance(open_file, h5py.Dataset):  # h5py
        # not using open_file[:] because [:] can't slice string data but ... catches it all
        if h5_group is not None:
            open_file = open_file[h5_group]
        return bool(variable in open_file.keys())
    elif isinstance(open_file, TiffImagePlugin.TiffImageFile):  # geotiff
        return False
    else:
        raise ValueError(f'Unrecognized opened file dataset: {type(open_file)}')
