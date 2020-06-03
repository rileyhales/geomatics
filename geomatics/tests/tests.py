import datetime
import glob

import geomatics

gldas_files = glob.glob('/Users/riley/spatialdata/thredds/gldas/raw/*.nc4')
st = datetime.datetime.now()
ts = geomatics.timeseries.polygons(gldas_files, 'Tair_f_inst', '/Users/riley/spatialdata/shapefiles/united_states.shp',
                                   ('lon', 'lat'), stats='all')
ed = datetime.datetime.now()
print((ed - st).total_seconds())
# ts.sort_values(by='datetime', inplace=True)
# ts.reset_index(drop=True, inplace=True)
# nwm_files = glob.glob('/Users/riley/spatialdata/national_water_model/nwm*.nc')
# ts = geomatics.timeseries.point(nwm_files, 'streamflow', (1000, ), ('feature_id', ))
ts.to_csv('/Users/riley/spatialdata/timeseries.csv', index=False)
print(ts)
