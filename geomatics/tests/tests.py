import geomatics
import tempfile
import os


def test_getdata(tmpdir):
    # collect gfs data
    gfs_files = geomatics.data.download_noaa_gfs(tmpdir, 3)
    # collect a shapefile
    geojson = geomatics.data.get_livingatlas_geojson('Northern Africa')
    shp_path = os.path.join(tmpdir, 'tmpshp.shp')
    geomatics.convert.geojson_to_shapefile(geojson, shp_path)
    return gfs_files, shp_path


if __name__ == '__main__':
    # tmpdir = tempfile.gettempdir()
    # gfs_files, shp_path = test_getdata(tmpdir)
    # os.removedirs(tmpdir)

    gfs_files = geomatics.data.download_noaa_gfs('/Users/rileyhales/SpatialData/tmp/', 3)
