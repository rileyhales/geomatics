import os
import glob
import geomatics

path_to_save_gtiffs = os.path.join(os.path.dirname(__file__), 'geotiff_data')
netcdf_files = glob.glob(os.path.join(os.path.dirname(__file__), 'netcdf_data', '*.nc4'))
var = 'Tair_f_inst'
geomatics.convert.to_gtiffs(netcdf_files, var, save_dir=path_to_save_gtiffs)
