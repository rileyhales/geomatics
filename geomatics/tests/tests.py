import geomatics

path = '/Users/riley/code/geomatics/publication_data/grib_data/gfs_2020061518_2020061600.grb'
# geomatics.inspect.grib(path, xr_kwargs=dict(filter_by_keys={'typeOfLevel': 'surface'}))
import pygrib
a = pygrib.open(path)
a = a.read()
print(a)