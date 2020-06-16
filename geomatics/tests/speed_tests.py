# adapted directly from the plotly documentation https://plotly.com/python/bar-charts/
# https://plotly.com/python/static-image-export/#vector-formats-svg-and-pdf
# conda install plotly
# conda install -c plotly plotly-orca==1.2.1 psutil requests

import datetime
import glob

import pandas as pd
import plotly.graph_objects as go

import geomatics


def test_engine(files: list, var: str or int, coords: tuple, min_coords, max_coords, dims: tuple, engine: str,
                poly: str, xr_kwargs: dict = None, strp: str = None):
    t1 = datetime.datetime.now()
    geomatics.timeseries.point(files, var, coords, dims, engine=engine, xr_kwargs=xr_kwargs, strp=strp)
    t2 = datetime.datetime.now()
    geomatics.timeseries.bounding_box(files, var, min_coords, max_coords, dims, engine=engine, xr_kwargs=xr_kwargs, strp=strp)
    t3 = datetime.datetime.now()
    geomatics.timeseries.polygons(files, var, poly, dims, engine=engine, xr_kwargs=xr_kwargs, strp=strp)
    t4 = datetime.datetime.now()
    geomatics.timeseries.full_array_stats(files, var, engine=engine, xr_kwargs=xr_kwargs, strp=strp)
    t5 = datetime.datetime.now()
    return (t2 - t1).total_seconds(), (t3 - t2).total_seconds(), (t4 - t3).total_seconds(), (t5 - t4).total_seconds()


def make_bar_chart(df: pd.DataFrame or str):
    if isinstance(df, str):
        df = pd.read_csv(df)
    x = ['Point Series', 'Bounding Box Series', 'Polygon Series', 'Full Array Series']
    df = df.mean(axis=0)
    layout = go.Layout(
        title='Average Execution Times',
        yaxis={'title': 'Time (seconds)'},
        xaxis={'title': 'Series Type'},
        barmode='group',
    )
    fig = go.Figure(data=[
        go.Bar(name='netCDF4 Engine', x=x, y=df.values[:4]),
        go.Bar(name='xarray Engine', x=x, y=df.values[4:])
    ], layout=layout)
    fig.write_image('netcdf_plot.svg')
    return


def compute_stats(df: pd.DataFrame or str, round_decimals_to: int = 3):
    if isinstance(df, str):
        df = pd.read_csv(df)
    stats = pd.DataFrame([df.max(axis=0), df.mean(axis=0), df.median(axis=0), df.min(axis=0), df.std(axis=0)],
                         index=['Max', 'Mean', 'Median', 'Min', 'St. Dev.'])

    per_file = pd.DataFrame(stats.values / len(df.index), columns=stats.columns,
                            index=['Max/File', 'Mean/File', 'Median/File', 'Min/File', 'St. Dev./File'])
    per_file.drop('St. Dev./File', inplace=True)

    stats = stats.append(per_file)
    stats = stats.round(round_decimals_to)

    return stats


def test_file_grouping():
    var = 'Tair_f_inst'
    coords = (10, 10)
    all_files = sorted(glob.glob('/Users/riley/spatialdata/sampledata/*.nc4'))
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
