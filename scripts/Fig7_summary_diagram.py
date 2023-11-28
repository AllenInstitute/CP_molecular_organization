# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:08:26 2023

@author: ashwin.bhandiwad
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Figure 7a - 

df = pd.read_table('../data/Fig7_summary_ctxmerged.csv',delimiter=',')
attrs = pd.read_table('../data/Fig7_attrs.csv',delimiter=',')

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
fig.write_html("../figures/Fig7.html")
fig.show(renderer='png')
fig.write_image('../figures/Fig7_diagram.svg')


# # Fig 7b

# divs = sitk.ReadImage('../data/ccf_volumes/CP_bounds_CCFregistered.nrrd')

# sections = [640, 728, 839]
# coronal = [1320 - x for x in sections]

# fig,ax = plt.subplots(3,1,figsize=(60,20))
# axes = ax.flatten()

# for idx,subax in enumerate(axes):
    
#     img = divs[coronal[idx],:,:]
#     img_npy = sitk.GetArrayViewFromImage(img)
#     img_half = img_npy[100:500,:].T
#     subax.imshow(img_half,cmap="Greys")

# plt.savefig('../figures/Fig7b_CPdivisions_coronal_series.svg')

