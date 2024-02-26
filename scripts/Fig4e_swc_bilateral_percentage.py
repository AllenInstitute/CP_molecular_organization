# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:34:32 2023

@author: ashwin.bhandiwad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


swc_df = pd.read_csv('../data/swc_cp_projection_densities.csv')
order = pd.read_csv('../data/harris_order.csv')
order = order['name'].to_numpy()

ctx_layers = ['L2/3','L5','L6']

for layer in ctx_layers:
    projections = swc_df.loc[(swc_df['celltype']=='IT') & (swc_df['layer']==layer)]

    ipsi_only = projections.loc[(swc_df['sum_ipsi']>0) & (swc_df['sum_contra']==0)]
    contra_only = projections.loc[(swc_df['sum_contra']>0) & (swc_df['sum_ipsi']==0)]
    bilateral = projections.loc[(swc_df['sum_contra']>0) & (swc_df['sum_ipsi']>0)]
    
    ipsi_mean = ipsi_only.groupby('ccf_region').count()
    contra_mean = contra_only.groupby('ccf_region').count()
    bilateral_mean = bilateral.groupby('ccf_region').count()
    
    ccf_regions = np.unique(swc_df['ccf_region'])
    
    contra = np.zeros(np.shape(ccf_regions))
    bilat = np.zeros(np.shape(ccf_regions))
    ipsi = np.zeros(np.shape(ccf_regions))
    
    for count,region in enumerate(ipsi_mean.index.to_numpy()):
        
        idx = np.where(ccf_regions==region)[0][0]
        if np.isin(region,ipsi_mean.index):
            ipsi[idx] = ipsi_mean['sum_ipsi'][count]
        else:
            ipsi[idx] = []
        
    for count,region in enumerate(contra_mean.index.to_numpy()):
        idx = np.where(ccf_regions==region)[0][0]
        if np.isin(region,contra_mean.index):
            contra[idx] = contra_mean['sum_contra'][count]
        else:
            contra[idx] = []
        
    for count,region in enumerate(bilateral_mean.index.to_numpy()):
        idx = np.where(ccf_regions==region)[0][0]
        if np.isin(region,bilateral_mean.index):
            bilat[idx] = bilateral_mean['sum_contra'][count] + bilateral_mean['sum_ipsi'][count]
        else:
            bilat[idx] = []
    
    
    ipsi_pct = 100*(ipsi/(ipsi+contra+bilat))
    contra_pct = 100*(contra/(ipsi+contra+bilat))
    bilat_pct = 100*(bilat/(ipsi+contra+bilat))
    
    if layer == ctx_layers[0]:
        df = pd.DataFrame({'ccf_region': ccf_regions})
    df[f'{layer}_ipsi_only_neurons'] =  ipsi
    df[f'{layer}_contra_neurons'] =  contra
    df[f'{layer}_bilateral_neurons'] =  bilat
    df[f'{layer}_percent_ipsi'] =  ipsi_pct
    df[f'{layer}_percent_contra'] =  contra_pct
    df[f'{layer}_percent_bilateral'] =  bilat_pct

df['ccf_region'] = pd.Categorical(df['ccf_region'],categories=order,ordered=True)
df.sort_values(by='ccf_region',inplace=True)
df.reset_index(inplace=True)
df.dropna(axis=0,thresh=14,inplace=True)

df.to_csv('../data/cortical_swc_ipsi_bilateral_ratio.csv')
font_color = '#525252'
hfont = {'family': 'Arial', 'size': 12}
facecolor = '#eaeaf2'
color_red = '#fd625e'
color_blue = '#01b8aa'
plt.rc('font', **hfont)

ccf_region = df['ccf_region'].to_numpy()
x = np.arange(len(ccf_region))

fig, ax = plt.subplots()
bottom = np.zeros(len(ccf_region))
multiplier = 0
width = 0.25

bars = ['ipsi_only_neurons','contra_neurons','bilateral_neurons']
for layer in ctx_layers:
    offset = width*multiplier
    
    for bar_id in bars:
        
        p = ax.bar(x+offset, df[f'{layer}_{bar_id}'].to_numpy(), label=bar_id, width=width, bottom=bottom)
        bottom += df[f'{layer}_{bar_id}'].to_numpy()
    
    bottom = np.zeros(len(ccf_region))
    multiplier += 1

ax.set_title("Single neuron distribution")
ax.set(xticks=list(x))
ax.set(xticklabels=ccf_region)
# ax.legend(loc="upper right")

plt.savefig('../figures/Fig4e_ipsi_bilateral_summary.svg')
