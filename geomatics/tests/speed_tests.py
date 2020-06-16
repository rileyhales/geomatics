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


def make_bar_chart(df: pd.DataFrame or str, name_of_plot: str = 'Average Execution Times',
                   name_of_file: str = 'results_bar_plot.svg'):
    if isinstance(df, str):
        avg = pd.read_csv(df)
    else:
        avg = df.copy()
    columns = avg.columns
    avg = avg.mean(axis=0)
    x = ['Point Series', 'Bounding Box Series', 'Polygon Series', 'Full Array Series']
    layout = go.Layout(
        title=name_of_plot,
        yaxis={'title': 'Time (seconds)'},
        xaxis={'title': 'Series Type'},
        barmode='group',
    )
    eng1 = columns[0].split('_')[0]
    eng2 = columns[-1].split('_')[0]
    fig = go.Figure(data=[
        go.Bar(name=f'{eng1} Engine', x=x, y=avg.values[:4]),
        go.Bar(name=f'{eng2} Engine', x=x, y=avg.values[4:])
    ], layout=layout)
    fig.write_image(name_of_file)
    return


def make_stats_bar_chart(df: pd.DataFrame or str, name_of_file: str = 'results_stats_bar_plot.svg'):
    if isinstance(df, str):
        stats = pd.read_csv(df)
    else:
        stats = df.copy()
    print(stats)
    print(stats.index)
    stats = stats[stats.index == 'Mean/File']
    return make_bar_chart(stats, 'Average Execution Time Per File', name_of_file)


def compute_stats(df: pd.DataFrame or str, round_decimals_to: int = 3):
    if isinstance(df, str):
        df1 = pd.read_csv(df)
    else:
        df1 = df.copy()
    stats = pd.DataFrame([df1.max(axis=0), df1.mean(axis=0), df1.median(axis=0), df1.min(axis=0), df1.std(axis=0)],
                         index=['Max', 'Mean', 'Median', 'Min', 'St. Dev.'])
    per_file = pd.DataFrame(stats.values / len(df1.index), columns=stats.columns,
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
