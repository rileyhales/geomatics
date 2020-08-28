import glob

import geomatics

a = glob.glob('/Users/rileyhales/SpatialData/thredds/timeseries-workshop/*.nc4')

b = geomatics.timeseries.time_series(a, 'Tair_f_inst', ('time', 'lat', 'lon'), point=('time', 10, 10), engine='h5py',
                                     interp_units=True)
print(b)
