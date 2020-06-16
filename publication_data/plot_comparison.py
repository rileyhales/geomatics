import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('combined_results.csv', index_col=0)
columns = df.columns
df = df[df.index == 'Mean/File']
x = ['Point Series', 'Bounding Box Series', 'Polygon Series', 'Full Array Series']
layout = go.Layout(
    title='Comparison of Engine Speeds Per File',
    yaxis={'title': 'Time (seconds)'},
    xaxis={'title': 'Series Type'},
    barmode='group',
)
bars = []
engs = len(columns) // 4
for i in range(engs):
    eng = columns[i * 4].split('_')[0]
    bars.append(go.Bar(name=f'{eng} Engine', x=x, y=df.values[0][4*i:4*(i+1)]))

fig = go.Figure(data=bars, layout=layout)
fig.write_image('total_comparison.png')
