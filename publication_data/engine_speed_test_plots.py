# adapted directly from the plotly documentation https://plotly.com/python/bar-charts/
# https://plotly.com/python/static-image-export/#vector-formats-svg-and-pdf
# conda install plotly plotly-orca psutil

import pandas as pd
import plotly.graph_objects as go


def make_bar_chart():
    df = pd.read_csv('netcdf_engine_speed_test_results.csv')
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
    fig.write_image('engine_speed_test_plot.svg')
    return


def compute_stats_table():
    df = pd.read_csv('netcdf_engine_speed_test_results.csv')
    stats = pd.DataFrame([df.max(axis=0), df.mean(axis=0), df.median(axis=0), df.min(axis=0), df.std(axis=0)],
                         index=['Max', 'Mean', 'Median', 'Min', 'St. Dev.'])
    stats = stats.round(2)
    per_file = pd.DataFrame(stats.values / 480, columns=stats.columns,
                            index=['Max', 'Mean', 'Median', 'Min', 'St. Dev.'])
    per_file.drop('St. Dev.', inplace=True)
    per_file = per_file.round(4)

    stats.to_csv('netcdf_engine_speed_test_stats.csv')
    per_file.to_csv('netcdf_engine_speed_test_stats_per_file.csv')
    return


if __name__ == '__main__':
    make_bar_chart()
    compute_stats_table()
