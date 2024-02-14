# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:06:21 2023

@author: ashwin.bhandiwad
"""

import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def hex_to_rgb(hex_value):
    
    hex_value = hex_value.lstrip("#")
    
    return tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))

layers = ['L2/3','L5','L5','L6']
celltypes = ['IT','ET','IT','IT']

df_full = pd.read_table('../data/swc_arbor_stats.csv',sep=',')
df_full.dropna(subset=['dispersion'],inplace=True)

df = df_full#[df_filt['celltype']==celltypes[idx]]
df = df[df['layer']=='L5']
df = df[df['celltype'].isin(['ET','IT'])]

size = np.log10(df['size'].values)
dispersion = df['dispersion'].values

colors = [tuple([1,0,0]),tuple([0,0.33,1])]

# Scatter plot
fig = plt.scatter(size,dispersion, c=df['celltype'].map(dict(zip(df['celltype'].unique(), colors))), s=7)

# Labels and title
plt.xlabel('Log(number of terminals)')
plt.ylabel('Dispersion')
plt.xlim(0,2.5)
plt.ylim(0,200)

# Show the plot
plt.savefig('../figures/Fig4g_swc_L5_ITvsET.svg',dpi=300,rasterize=True)


df = df_full

color_table = pd.read_table('../data/harris_order_smgrouped.csv',sep=',')
color_table.dropna(subset=['color'],inplace=True)

size = np.log10(df['size'].values)
dispersion = df['dispersion'].values

colors = []
for datapoint in df['ccf_region']:
    
    region_color = color_table.loc[color_table['name']==datapoint,'color'].values[0]
    colors.append(region_color)

color_table_values = ['#'+ x for x in colors]

# Scatter plot
fig = plt.scatter(size,dispersion, c=color_table_values, s=7)

# Labels and title
plt.xlabel('Log(number of terminals)')
plt.ylabel('Dispersion')
plt.xlim(0,2.5)
plt.ylim(0,200)

# Show the plot
plt.savefig('../figures/Fig4f_swc_all_size_dispersion.svg',dpi=300)