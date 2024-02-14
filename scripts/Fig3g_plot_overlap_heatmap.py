# -*- coding: utf-8 -*-
"""
Plot pairwise overlap in anterograde dataset for Fig 3g.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def insert_names(overlap_data,l5):
    
    overlap_data = overlap_data.assign(vol1_name=pd.Series(dtype=str))
    overlap_data = overlap_data.assign(vol2_name=pd.Series(dtype=str))
    
    l5_ids = l5['image-series-id'].to_numpy()

    for image_id in l5_ids:
        
        num_repeats_x = overlap_data.loc[(overlap_data['Volume1']==image_id)]
        num_repeats_y = overlap_data.loc[(overlap_data['Volume2']==image_id)]
        overlap_data.loc[overlap_data['Volume1']==image_id, 'vol1_name'] = np.repeat(l5.loc[l5['image-series-id']==image_id]['primary injection'].values,len(num_repeats_x))
        overlap_data.loc[overlap_data['Volume2']==image_id, 'vol2_name'] = np.repeat(l5.loc[l5['image-series-id']==image_id]['primary injection'].values,len(num_repeats_y))
    
    return overlap_data,l5_ids


def filter_sparse_projections(overlap_data,metadata,density_metadata,threshold=0.01):
    
    remove_list = []
    density_values = []
    for exp_id in overlap_data['Volume1'].to_list():
        fractional_density = density_metadata.loc[(density_metadata['experiment_id']==exp_id),'projection_density_ipsi']
        density_values.append(np.mean(fractional_density))
        if np.mean(fractional_density) < threshold:
            remove_list.append(exp_id) 
    
    zeros_list = list(np.unique(overlap_data.loc[(overlap_data['Dice_coefficient']>1),'Volume1']))
    remove_list += zeros_list
    
    for vol in remove_list:
        overlap_data.drop(overlap_data.loc[(overlap_data['Volume1']==vol) | (overlap_data['Volume2']==vol)].index, inplace=True)
        metadata.drop(metadata.loc[metadata['image-series-id']==vol].index,inplace=True)
        
    return overlap_data,metadata,density_values

def reorder_dataframe(overlap_data,order_list):
    
    overlap_data['vol1_name'] = pd.Categorical(overlap_data['vol1_name'], categories=order_list, ordered=True)
    overlap_data['vol2_name'] = pd.Categorical(overlap_data['vol2_name'], categories=order_list, ordered=True)
    overlap_data = overlap_data.sort_values(by=['vol1_name','Volume1','vol2_name','Volume2'])
    overlap_data = overlap_data.reset_index(drop=True)
    
    return overlap_data

def get_ticks(overlap_data,order_list,l5_ids):
    
    name = overlap_data['vol1_name'].to_numpy()
    ticks=[]
    idx=0
    for region in order_list:
        replicates = int(len(np.where(name==region)[0])/len(l5_ids))
        idx=idx+replicates
        ticks.append(idx)
        
    return ticks

def overlap_heatmap(heatmap_table,save_filename,tick=[],label=[]):
    
    ax = sns.heatmap(heatmap_table, annot=False,vmax=1, cmap='plasma',rasterized=True)
    plt.title('Cortical pairwise overlap projection')
    if len(tick)>0:
        plt.xticks(tick,label)
        plt.yticks(tick,label)
        # for tick in ax.get_xticks():
        #     ax.axvline(tick, color='white', linewidth=0.3)
        # for tick in ax.get_yticks():
        #     ax.axhline(tick, color='white', linewidth=0.3)
    plt.ylabel('Volume1')
    plt.xlabel('Volume2')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Dice coefficient')
    
    plt.gcf().set_size_inches(12,10)
    plt.savefig(save_filename, dpi=100)
    plt.clf()
    

path = '../data/'
overlap_data = pd.read_csv(path+'anterograde_cp_overlap_L5_cortical.csv',header=0)
order_list = pd.read_csv(path+'harris_order.csv')['name'].to_list()[:-1]

metadata = pd.read_csv(path+'anterograde_cortical_20230922.csv')

# Remove sparse projections based on fractional density in CP - requires data from Fig 3b.
density_metadata = pd.read_csv(path+'Fig3b_cp_cortical_anterograde_projections.csv')

# Some cells have Dice coefficient of Inf because of divide by zero.Removing them from analysis
overlap_data,metadata,density_values = filter_sparse_projections(overlap_data,metadata,density_metadata,threshold=0.01)

l5 = metadata[metadata['injected layer'].str.contains("L5 IT")==True]

overlap_data,l5_ids = insert_names(overlap_data,l5)

vol1_id = overlap_data['Volume1'].to_numpy()
assert(len(np.unique(l5_ids))==len(np.unique(vol1_id)))

overlap_data = reorder_dataframe(overlap_data,order_list)
ticks = get_ticks(overlap_data,order_list,l5_ids)

assert((overlap_data['Volume1'].unique()==overlap_data['Volume2'].unique()).all())

# Separated
heatmap_table = overlap_data.pivot_table(index='Volume1',columns='Volume2',values='Dice_coefficient')
heatmap_table = heatmap_table.reindex(index=overlap_data['Volume1'].unique(),columns=overlap_data['Volume2'].unique())

overlap_heatmap(heatmap_table,'../figures/Fig3e_L5_cortical_overlap.svg',ticks,order_list)

