import datetime
import glob

import geomatics  # tests run on v0.9


def speed_test_optimization1():
    var = 'Tair_f_inst'
    coords = (10, 10)
    all_files = sorted(glob.glob('/Users/rileyhales/SpatialData/THREDDS/gldas/raw/*.nc4'))
    t1 = datetime.datetime.now()
    geomatics.timeseries.point(all_files, var, coords, ('lon', 'lat'), engine='netcdf4')
    t2 = datetime.datetime.now()
    geomatics.timeseries.point(all_files, var, coords, ('lon', 'lat'), engine='xarray')
    t3 = datetime.datetime.now()
    print((t2 - t1).total_seconds())
    print((t3 - t2).total_seconds())
    return


def speed_test_optimization2():
    var = 'Tair_f_inst'
    coords = (10, 10)
    all_files = sorted(glob.glob('/Users/rileyhales/SpatialData/THREDDS/gldas/raw/*.nc4'))
    list_each_file = list(map(lambda element: [element], all_files))
    t1 = datetime.datetime.now()
    geomatics.timeseries.point(all_files, var, coords, ('lon', 'lat'), engine='netcdf4')
    t2 = datetime.datetime.now()
    for file in list_each_file:
        geomatics.timeseries.point(file, var, coords, ('lon', 'lat'))
    t3 = datetime.datetime.now()
    print((t2 - t1).total_seconds())
    print((t3 - t2).total_seconds())
    return


if __name__ == '__main__':
    geomatics.timeseries.point(
        ['/Users/rileyhales/Downloads/gfs_test_file_20001010.grb'], 1, (10, 10), ('longitudes', 'latitudes'),
        engine='pygrib', strp='gfs_test_file_%Y%m%d.grb')
    speed_test_optimization1()
    speed_test_optimization2()
