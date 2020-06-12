import datetime
import glob
import json

import pandas as pd

import geomatics


def speed_test_by_engine(files: list, engine: str, poly: str):
    var = 'Tair_f_inst'
    coords = (10, 10)
    min_coords = (0, 10)
    max_coords = (30, 30)
    t1 = datetime.datetime.now()
    geomatics.timeseries.point(files, var, coords, ('lon', 'lat'), engine=engine)
    t2 = datetime.datetime.now()
    geomatics.timeseries.bounding_box(files, var, min_coords, max_coords, ('lon', 'lat'), engine=engine)
    t3 = datetime.datetime.now()
    geomatics.timeseries.polygons(files, var, poly, ('lon', 'lat'), engine=engine)
    t4 = datetime.datetime.now()
    geomatics.timeseries.full_array_stats(files, var, engine=engine)
    t5 = datetime.datetime.now()
    return (t2 - t1).total_seconds(), (t3 - t2).total_seconds(), (t4 - t3).total_seconds(), (t5 - t4).total_seconds()


if __name__ == '__main__':
    usa_geojson = geomatics.data.get_livingatlas_geojson('United States')
    filepath = '/Users/riley/spatialdata/geojson/united_states.json'
    with open(filepath, 'w') as file:
        file.write(json.dumps(usa_geojson))
    # netcdf sample data
    all_files = sorted(glob.glob('/Users/riley/spatialdata/sampledata/*.nc4'))
    data = {}
    for eng in ('xarray', 'netcdf4'):
        pt_times = []
        box_times = []
        ply_times = []
        fll_times = []
        for i in range(50):
            pt, bx, ply, fll = speed_test_by_engine(all_files, eng, filepath)
            pt_times.append(pt)
            box_times.append(bx)
            ply_times.append(ply)
            fll_times.append(fll)
            print(i)
        data[f'{eng}_point'] = pt_times
        data[f'{eng}_bbox'] = box_times
        data[f'{eng}_poly'] = ply_times
        data[f'{eng}_full'] = fll_times
    df = pd.DataFrame(data).to_csv('speed_test_results.csv', index=False)
