# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:08:26 2023

@author: ashwin.bhandiwad
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Figure 6a - Sankey diagram

df = pd.read_table('../data/Fig6_summary.csv',delimiter=',')
attrs = pd.read_table('../data/Fig6_attrs.csv',delimiter=',')

struct = attrs['structure'].to_numpy()
struct_id = attrs['struct_id'].to_numpy()
color = attrs['color'].to_numpy()
xpos = attrs['x'].to_list()
ypos = attrs['y'].to_list()

source_name = df['Source'].to_numpy()
target_name = df['Target'].to_numpy()

source = []
target = []
line_col = []

for idx,value in enumerate(source_name):
    
    source_lookup = np.where(struct==value)[0][0]
    target_lookup = np.where(struct==target_name[idx])[0][0]
    
    source.append(struct_id[source_lookup])
    target.append(struct_id[target_lookup])
    line_col.append(color[source_lookup])


strength = df['strength'].to_list()


fig = go.Figure(go.Sankey(
    orientation='v',
    arrangement='snap',
    node=dict(
        label=struct,
        x=xpos,
        y=ypos,
        color=color,
        pad=10
    ),
    link=dict(
        arrowlen=5,
        source=source,
        target=target,
        color=line_col,
        value=strength 
    )
))

fig.update_layout(width=1500,
                  height=1000,
                  font=dict(size = 8, color = 'black'),
                  plot_bgcolor='white',
                  paper_bgcolor='white')
fig.write_html("../figures/Fig6.html")
fig.show(renderer='png')
fig.write_image('../figures/Fig6a_diagram.svg')

## Fig 6c - Chord diagram

# df = pd.read_table('../data/Fig6_summary_chord.csv',delimiter=',')
# attrs = pd.read_table('../data/Fig6_attrs_chord.csv',delimiter=',')

# struct = attrs['structure'].to_numpy()
# struct_id = attrs['struct_id'].to_numpy()
# color = attrs['color'].to_numpy()

# import holoviews as hv
# from holoviews import opts, dim
# from bokeh.sampledata.airport_routes import routes, airports

# hv.extension('bokeh')

# # Count the routes between Airports
# route_counts = routes.groupby(['SourceID', 'DestinationID']).Stops.count().reset_index()
# nodes = hv.Dataset(airports, 'AirportID', 'City')
# chord = hv.Chord((route_counts, nodes), ['SourceID', 'DestinationID'], ['Stops'])

# # Select the 20 busiest airports
# busiest = list(routes.groupby('SourceID').count().sort_values('Stops').iloc[-20:].index.values)
# busiest_airports = chord.select(AirportID=busiest, selection_mode='nodes')

