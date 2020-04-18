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


#todo

# import pprint
# grib = pygrib.open('/Users/rileyhales/SpatialData/tmp/gfs_2020041712_2020041718.grb')
# grib.seek(0)
# print(grib.read())
# pprint.pprint(grib[1].latitudes)
# print(grib[573])
# new_values, lats, lons = grib[573].data(lat1=10, lon1=20, lat2=20, lon2=30)
# print(new_values)
# print(lats)
# print(lons)

# ts = grib[573].values
# print(ts)
# grib.read(2)
# ts = grib[573].values
# print(ts)
#
# exit()
# print(pygrib.julian_to_datetime(grib[573].julianDay))
#
# print(grib[573].next())
# exit()


# new_values, _, _ = grib[573].data(lat1=10, lon1=20, lat2=10, lon2=20)
# print(new_values)
# import datetime
# import julian
# print(julian.from_jd(grib[573].julianDay))
# print(grib[573].dataTime)
# print(grib[573].dataDate)
# print(grib[573].dataDate)
# print(grib[573].julianDay)
# print(grib[573].year)
# print(grib[573].month)
# print(grib[573].day)
# print(grib[573].hour)
# print(grib[573].minute)
# print(grib[573].second)
# print()
# print(grib[573].endStep)
# print(grib[573].typeOfTimeIncrement)
# print()
# print(grib[573].yearOfEndOfOverallTimeInterval)
# print(grib[573].monthOfEndOfOverallTimeInterval)
# print(grib[573].dayOfEndOfOverallTimeInterval)
# print(grib[573].hourOfEndOfOverallTimeInterval)
# print(grib[573].minuteOfEndOfOverallTimeInterval)
# print(grib[573].secondOfEndOfOverallTimeInterval)

# pprint.pprint(grib[573].keys())

# pprint.pprint(grib.read())
# test = grib.select(name='Lon')
# print(test)