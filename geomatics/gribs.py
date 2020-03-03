import pygrib

__all__ = ['inspect']


def inspect(path, band_number=None):
    """
    Prints lots of messages showing information about variables, dimensions, and metadata

    Args:
        path: The path to a netcdf file.
        band_number: (optional) A list of the elements stored in this
    """
    grib = pygrib.open(path)
    grib.seek(0)
    print('This is a summary of all the information in your grib file')
    print(grib.read())

    if band_number:
        print()
        print('The keys for this variable are:')
        print(grib[band_number].keys())
        print()
        print('The data stored in this variable are:')
        print(grib[band_number].values)

    return
