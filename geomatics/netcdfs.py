import os

import netCDF4 as nc

__all__ = ['inspect', 'generate_timejoining_ncml']


def inspect(path):
    """
    Prints lots of messages showing information about variables, dimensions, and metadata

    Args:
        path: The path to a single netcdf file.
    """
    nc_obj = nc.Dataset(path, 'r', clobber=False, diskless=True, persist=False)

    print("This is your netCDF python object")
    print(nc_obj)
    print()

    print("There are " + str(len(nc_obj.variables)) + " variables")       # The number of variables
    print("There are " + str(len(nc_obj.dimensions)) + " dimensions")     # The number of dimensions
    print()

    print('These are the global attributes of the netcdf file')
    print(nc_obj.__dict__)                                    # access the global attributes of the netcdf file
    print()

    print("Detailed view of each variable")
    print()
    for variable in nc_obj.variables.keys():                  # .keys() gets the name of each variable
        print('Variable Name:  ' + variable)              # The string name of the variable
        print('The view of this variable in the netCDF python object')
        print(nc_obj[variable])                               # How to view the variable information (netcdf obj)
        print('The data array stored in this variable')
        print(nc_obj[variable][:])                            # Access the numpy array inside the variable (array)
        print('The dimensions associated with this variable')
        print(nc_obj[variable].dimensions)                    # Get the dimensions associated with a variable (tuple)
        print('The metadata associated with this variable')
        print(nc_obj[variable].__dict__)                      # How to get the attributes of a variable (dictionary)
        print()

    for dimension in nc_obj.dimensions.keys():
        print(nc_obj.dimensions[dimension].size)              # print the size of a dimension

    nc_obj.close()                                            # close the file connection to the file
    return


# todo
def generate_timejoining_ncml(path, save_dir, time_interval=False):
    """
    Generates a ncml file which aggregates a list of netcdf files across the "time" dimension and the "time" variable.
    In order for the times displayed in the aggregated NCML dataset to be accurate, they must have a regular time step
    between measurments.

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.netcdfs.generate_timejoin_ncml('/path/to/netcdf/', '/path/to/save', time_interval=False)
    """
    ds = nc.Dataset(files[0])
    units_str = str(ds['time'].__dict__['units'])
    ds.close()

    # todo
    # attempt to guess the time interval
    if not time_interval:
        times = []
        for file in files:
            times += guess_time_from_metadata(nc.Dataset(file))
        times.sort()
        time_interval = rd.relativedelta(times[1], times[0])
        inconsistent = 0
        for i in range(len(times) - 1):
            if rd.relativedelta(times[i + 1], times[i]) != time_interval:
                inconsistent += 1
        if inconsistent:
            raise RuntimeWarning(
                'Inconsistent time step found on {0} steps, aggregated times will be wrong.'.format(inconsistent))

    # create a new ncml file by filling in the template with the right dates and writing to a file
    with open(os.path.join(save_dir, 'time_joined_series.ncml'), 'w') as ncml:
        ncml.write(
            '<netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2">\n' +
            '  <variable name="time" type="int" shape="time">\n' +
            '    <attribute name="units" value="' + units_str + '"/>\n' +
            '    <attribute name="_CoordinateAxisType" value="Time" />\n' +
            '    <values start="0" increment="' + time_interval + '" />\n' +
            '  </variable>\n' +
            '  <aggregation dimName="time" type="joinExisting" recheckEvery="5 minutes">\n'
        )
        for file in files:
            ncml.write('    <netcdf location="' + file + '"/>\n')
        ncml.write(
            '  </aggregation>\n' +
            '</netcdf>'
        )
    return
