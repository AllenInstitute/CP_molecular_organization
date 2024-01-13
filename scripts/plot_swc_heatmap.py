# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:58:28 2023

@author: ashwin.bhandiwad
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def reorder_dataframe(df,order_list):

    df['ccf_region'] = pd.Categorical(df['ccf_region'], categories=order_list['name'].to_numpy(), ordered=True)

    return df.reset_index(drop=True)

def swc_heatmap(heatmap_table,order_list,hemisphere,tick=[],label=[]):
    
    plt.style.use('default')
    hfont = {'family': 'Arial', 'size': 12}
    plt.rc('font',**hfont)
    
    ax = sns.heatmap(heatmap_table, annot=False,vmax=75, cmap='plasma',rasterized=True,square=True)
    plt.title(hemisphere)
    plt.yticks(range(len(heatmap_table.index)),heatmap_table.index.to_list())
    plt.ylabel('Cortical source')
    plt.xlabel('Cell type')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Neurite count')
    
    plt.gcf().set_size_inches(5,10)
    plt.savefig(f'../figures/{hemisphere}_cortical_swc_projection.svg', dpi=100)
    plt.clf()

def swc_by_subdivision(heatmap_table,order_list,title):
    
    ipsi = heatmap_table.iloc[:,5:10]
    contra = heatmap_table.iloc[:,10:15]
    xticks = ['CPdm','CPvm','CPiv','CPl','CPp']
    hfont = {'family': 'Arial', 'size': 12}
    plt.rc('font',**hfont)
    
    fig, axes = plt.subplots(figsize=(10,5), ncols=2, sharey=True)
    fig.tight_layout()

    sns.heatmap(ipsi, annot=False,vmax=40, cmap='plasma',ax=axes[0],rasterized=True,square=True)
    axes[0].set_title('Ipsilateral', pad=15)
    axes[0].set(xticks=range(len(xticks)),xticklabels = xticks)
    sns.heatmap(contra, annot=False,vmax=40,ax=axes[1], cmap='plasma',rasterized=True,square=True)
    axes[1].set_title('Contralateral', pad=15)
    axes[1].set(yticks=range(len(heatmap_table)), yticklabels=heatmap_table.index)
    axes[1].set(xticks=range(len(xticks)),xticklabels = xticks)
    
    plt.gcf().set_size_inches(5,10)
    plt.savefig(f'../figures/{title}_cortical_swc_projection.svg', dpi=100)
    plt.clf()

df = pd.read_csv('../data/swc_cp_projection_densities_v3.csv')
order_list = pd.read_csv('../data/harris_order_SSp_MOp_merged.csv')
included_regions = np.unique(df['ccf_region'])
order_list = order_list[order_list['name'].isin(included_regions)]

for layer in ['L2/3', 'L5']:


    filt_df = df[df['layer'].isin([layer])]
    ordered_df = reorder_dataframe(filt_df,order_list)
    
    means = ordered_df.groupby('ccf_region').mean()
    means.reset_index(inplace=True)
    means.set_index('ccf_region',inplace=True)
    
    swc_by_subdivision(means,order_list,layer[:2])
    

df["cell_class"] = df[['layer', 'celltype']].agg(' '.join, axis=1)

focal_types = ['L2/3 IT','L5 ET', 'L5 IT', 'L6 CT', 'L6 IT']
filtered_df = df[df['cell_class'].isin(focal_types)]

ordered_df = reorder_dataframe(filtered_df,order_list)

means = ordered_df.groupby(['ccf_region','cell_class']).mean()
means.reset_index(inplace=True)

ipsi = means.pivot_table(index='ccf_region',columns='cell_class',values='sum_ipsi')
swc_heatmap(ipsi,order_list,'Ipsilateral')

contra = means.pivot_table(index='ccf_region',columns='cell_class',values='sum_contra')
swc_heatmap(contra,order_list,'Contralateral')