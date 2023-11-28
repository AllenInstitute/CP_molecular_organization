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

layers = ['L2/3','L5','L5','L6']
celltypes = ['IT','ET','IT','IT']

df_full = pd.read_table('../data/swc_arbor_stats_ACAd_MOs.csv',sep=',')
df_full.dropna(subset=['dispersion'],inplace=True)

df_filt = df_full[df_full['ccf_region'].isin(['ACAd','MOs'])]


fig,ax = plt.subplots(2,2,figsize=(8,8))
font = {'family': 'Arial', 'size': 12}
plt.rc('font', **font)
ax = ax.flatten()

for idx,axes in enumerate(ax):

    df = df_filt[df_filt['celltype']==celltypes[idx]]
    df = df[df['layer']==layers[idx]]
    
    proj = pd.read_table('../data/swc_cp_projection_densities_v3.csv',sep=',')
    color_table = pd.read_table('../data/cortical_module_colors.txt',header=None)[0].to_list()
    
    size = np.log10(df['size'].values)
    dispersion = df['dispersion'].values
    
    # proj = proj.loc[proj['experiment_id'].isin(df['experiment_id'].values)]
    # projn = proj.to_numpy()
    
    # colors = sns.color_palette(color_table)
    colors = [tuple([1,0,0]),tuple([0,0.33,1])]
    
    # Scatter plot
    axes.scatter(size,dispersion, c=df['ccf_region'].map(dict(zip(df['ccf_region'].unique(), colors))), s=7)
    # Labels and title
    axes.set_xlabel('Log(number of terminals)')
    axes.set_ylabel('Dispersion')
    axes.set_xlim(0,2.5)
    axes.set_ylim(0,200)
    # Remove the spines (borders) around each subplot
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    # axes.spines['bottom'].set_visible(False)
    # axes.spines['left'].set_visible(False)

    # df['ccf_region'] = pd.Categorical(df['ccf_region'])
    
    # # Customize the colorbar and its labels
    # cbar = plt.colorbar()
    # cbar.set_ticks(np.unique(df['ccf_region'].cat.codes))
    # cbar.set_ticklabels(df['ccf_region'].cat.categories)
    


    
    # Remove the box around the plot
    # axes.box(False)
# plt.show()
# Show the plot
plt.savefig('../figures/ExtendedFig10_LACAd_MOs_size_dispersion_by_celltype.svg',dpi=300)