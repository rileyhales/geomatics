import glob
import json
import os

import pandas as pd

import geomatics
import geomatics.tests as tests


if __name__ == '__main__':
    var = 'tp'
    band_num = 30
    coords = (10, 10)
    min_coords = (0, 10)
    max_coords = (30, 30)
    xr_kwargs = dict(filter_by_keys={'typeOfLevel': 'surface'})
    usa_geojson = geomatics.data.get_livingatlas_geojson('United States')
    filepath = '/Users/riley/spatialdata/geojson/united_states.json'
    with open(filepath, 'w') as file:
        file.write(json.dumps(usa_geojson))
    # netcdf sample data
    path_to_sample_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'grib_data')
    files = sorted(glob.glob(os.path.join(path_to_sample_data, '*.grb')))
    data = {}
    strp = 'gfs_2020061518_%Y%m%d%H.grb'
    for engine in ('pygrib', 'cfgrib'):
        pt_times = []
        box_times = []
        ply_times = []
        fll_times = []
        for i in range(50):
            if engine == 'cfgrib':
                pt, bx, ply, fll = tests.speed_tests.test_engine(
                    files, var, coords, min_coords, max_coords, ('longitude', 'latitude'), engine, filepath, xr_kwargs)
            else:
                pt, bx, ply, fll = tests.speed_tests.test_engine(
                    files, band_num, coords, min_coords, max_coords, ('longitudes', 'latitudes'), engine, filepath, xr_kwargs, strp)
            pt_times.append(pt)
            box_times.append(bx)
            ply_times.append(ply)
            fll_times.append(fll)
            print(i)

        data[f'{engine}_point'] = pt_times
        data[f'{engine}_bbox'] = box_times
        data[f'{engine}_poly'] = ply_times
        data[f'{engine}_full'] = fll_times
    df = pd.DataFrame(data).to_csv('grib_test_results.csv', index=False)

    tests.speed_tests.make_bar_chart(df)
    stats_df = tests.speed_tests.compute_stats(df)
    stats_df.to_csv('grib_stats.csv')
    tests.speed_tests.make_stats_bar_chart(stats_df)
