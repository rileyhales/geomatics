import os
import glob


def path_to_file_list(path, filetype):
    if filetype in ('netcdf', 'nc', 'nc4'):
        filters = ['*.nc', '*.nc4']
    elif filetype in ('grib', 'grb'):
        filters = ['*.grib', '*.grb']
    elif filetype in ('geotiff', 'tiff'):
        filters = ['*.geotiff', '*.gtiff', '*.tiff']
    else:
        raise ValueError('Unconfigured filter type')

    # check that a valid path was provided
    if isinstance(path, str):
        if not os.path.exists(path):
            raise FileNotFoundError('No files or directory found at this path')
        elif os.path.isfile(path):
            return [path]
        elif os.path.isdir(path):
            files = []
            for filter in filters:
                files += glob.glob(os.path.join(path, filter))
            if len(files) == 0:
                raise FileNotFoundError('No located within this directory')
            return files
    elif isinstance(path, list):
        return path
    else:
        raise ValueError('Provide an absolute file path to a file or directory of files, or a list of paths')
