# adapted directly from the plotly documentation https://plotly.com/python/bar-charts/
import plotly.graph_objects as go
import pandas as pd


def make_bar_chart():
    df = pd.read_csv('engine_speed_test_results.csv')
    x = ['Point Series', 'Bounding Box Series', 'Polygon Series', 'Full Array Series']
    del df['Trial']
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
    fig.show()
    return


def compute_stats_table():
    df = pd.read_csv('engine_speed_test_results.csv')
    stats = pd.DataFrame([df.max(axis=0), df.mean(axis=0), df.median(axis=0), df.min(axis=0), df.std(axis=0)],
                         index=['Max', 'Mean', 'Median', 'Min', 'St. Dev.'])
    print(stats.head())
    return


if __name__ == '__main__':
    # make_bar_chart()
    compute_stats_table()
