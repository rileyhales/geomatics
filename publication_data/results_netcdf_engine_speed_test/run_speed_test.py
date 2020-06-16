import glob
import json

import pandas as pd

import geomatics
import geomatics.tests as tests


if __name__ == '__main__':
    var = 'Tair_f_inst'
    coords = (10, 10)
    min_coords = (0, 10)
    max_coords = (30, 30)
    dims = ('lon', 'lat')
    usa_geojson = geomatics.data.get_livingatlas_geojson('United States')
    filepath = '/Users/riley/spatialdata/geojson/united_states.json'
    with open(filepath, 'w') as file:
        file.write(json.dumps(usa_geojson))
    # netcdf sample data
    files = sorted(glob.glob('/Users/riley/spatialdata/sampledata/*.nc4'))
    data = {}
    for engine in ('xarray', 'netcdf4'):
        pt_times = []
        box_times = []
        ply_times = []
        fll_times = []
        for i in range(50):
            pt, bx, ply, fll = tests.speed_tests.test_engine(
                files, var, coords, min_coords, max_coords, dims, engine, filepath)
            pt_times.append(pt)
            box_times.append(bx)
            ply_times.append(ply)
            fll_times.append(fll)
            print(i)
        data[f'{engine}_point'] = pt_times
        data[f'{engine}_bbox'] = box_times
        data[f'{engine}_poly'] = ply_times
        data[f'{engine}_full'] = fll_times
    df = pd.DataFrame(data).to_csv('netcdf_test_results.csv', index=False)

    tests.speed_tests.make_bar_chart(df)
    stats_df = tests.speed_tests.compute_stats(df)
    stats_df.to_csv('netcdf_stats.csv')
    tests.speed_tests.make_stats_bar_chart(stats_df)
