import glob
import json
import os

import pandas as pd

import geomatics
import geomatics.tests as tests


if __name__ == '__main__':
    band_num = 1
    coords = (10, 10)
    min_coords = (0, 10)
    max_coords = (30, 30)
    usa_geojson = geomatics.data.get_livingatlas_geojson('United States')
    filepath = '/Users/riley/spatialdata/geojson/united_states.json'
    with open(filepath, 'w') as file:
        file.write(json.dumps(usa_geojson))
    # netcdf sample data
    path_to_sample_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'geotiff_data')
    files = sorted(glob.glob(os.path.join(path_to_sample_data, '*.tif')))
    data = {}
    strp = 'GLDAS_NOAH025_3H.A%Y%m%d.%H00.021.nc4.tif'
    for engine in ('rasterio', ):
        pt_times = []
        box_times = []
        ply_times = []
        fll_times = []
        for i in range(50):
            pt, bx, ply, fll = tests.speed.test_engine(
                files, band_num, coords, min_coords, max_coords, ('longitude', 'latitude'), engine, filepath, strp_filename=strp)
            pt_times.append(pt)
            box_times.append(bx)
            ply_times.append(ply)
            fll_times.append(fll)
            print(i)

        data[f'{engine}_point'] = pt_times
        data[f'{engine}_bbox'] = box_times
        data[f'{engine}_poly'] = ply_times
        data[f'{engine}_full'] = fll_times
    df = pd.DataFrame(data).to_csv('geotiff_test_results.csv', index=False)

    tests.speed.make_bar_chart(df)
    stats_df = tests.speed.compute_stats(df)
    stats_df.to_csv('geotiff_stats.csv')
    tests.speed.make_stats_bar_chart(stats_df)
