# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:08:37 2023

@author: ashwin.bhandiwad
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox

def subset_subcortical(overlap_data):
    
    subcort_subset = pd.read_excel(path+'anterograde_annotated_Quanxin_subcorticalsubset.xlsx',sheet_name='Subcortical inputs to CP')
    subset_ids = subcort_subset['image series ID'].to_list()
            
    return overlap_data[overlap_data['Volume2'].isin(subset_ids)],subset_ids

def insert_names_subcortical(overlap_data,metadata):
    
    overlap_data = overlap_data.assign(vol2_name=pd.Series(dtype=str))
    
    metadata_ids = metadata['image-series-id'].to_numpy()

    for image_id in metadata_ids:

        num_repeats_y = overlap_data.loc[overlap_data['Volume2']==image_id]
        overlap_data.loc[overlap_data['Volume2']==image_id, 'vol2_name'] = np.repeat(metadata.loc[metadata['image-series-id']==image_id]['primary injection'].values,len(num_repeats_y))
        
    return overlap_data

def insert_names_cortical(overlap_data,ctx_metadata):
    
    overlap_data = overlap_data.assign(vol1_name=pd.Series(dtype=str))
    
    ids = ctx_metadata[ctx_metadata['mouse-line'].str.contains("C57BL/6J|Emx1-IRES-Cre")==True]['image-series-id'].to_numpy()

    for image_id in ids:
        
        num_repeats_x = overlap_data.loc[(overlap_data['Volume1']==image_id)]
        overlap_data.loc[overlap_data['Volume1']==image_id, 'vol1_name'] = np.repeat(ctx_metadata.loc[ctx_metadata['image-series-id']==image_id]['primary injection'].values,len(num_repeats_x))
    
    return overlap_data,ids


def reorder_dataframe(overlap_data,order_list_cortical,order_list_subcortical):
    
    
    overlap_data['vol1_name'] = pd.Categorical(overlap_data['vol1_name'], categories=order_list_cortical, ordered=True)
    overlap_data['vol2_name'] = pd.Categorical(overlap_data['vol2_name'], categories=order_list_subcortical, ordered=True)
    overlap_data = overlap_data.sort_values(by=['vol1_name','Volume1','vol2_name','Volume2'])
    overlap_data = overlap_data.reset_index(drop=True)
    
    return overlap_data

def generate_order_list(order):
    
    ont = pd.read_csv(path+'MouseAtlas_ontologies_notree.csv')
    for count,j in enumerate(order[:,1]):
        
        order[count,0] = ont.loc[(ont['Acronym']==j),'InDel'].values[0]
    order = order[order[:, 0].argsort()]
    order_list = list(order[:,1])

    return order_list

def get_ticks(overlap_data,column_name,order_list,ids):
    
    name = overlap_data[column_name].to_numpy()
    order_list = [x for x in order_list if x in name]
    ticks=[]
    idx=0
    for region in order_list:
        replicates = int(len(np.where(name==region)[0])/len(np.unique(ids)))
        idx=idx+replicates
        ticks.append(idx)
        
    return ticks,order_list

def overlap_heatmap(heatmap_table,save_filename,xtick,ytick,order_list_cortical,order_list_subcortical):
    
    ax = sns.heatmap(heatmap_table, annot=False,vmin=0,vmax=1, cmap='plasma',square=True,rasterized=True)
    if len(xtick)>0:
        plt.xticks(xtick,order_list_subcortical)
        plt.yticks(ytick,order_list_cortical)
        # for tick in ax.get_xticks():
        #     ax.axvline(tick, color='white', linewidth=0.3)
        # for tick in ax.get_yticks():
        #     ax.axhline(tick, color='white', linewidth=0.3)
    plt.ylabel('Subcortical source')
    plt.xlabel('Cortical source')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Dice coefficient')
    
    plt.gcf().set_size_inches(20,10)
    plt.savefig(save_filename, dpi=100)
    plt.clf()
    

    
path = '../data/'
overlap_data = pd.read_csv(path+'anterograde_cp_overlap_cortsub.csv')
overlap_data,subset_ids = subset_subcortical(overlap_data)
overlap_data.drop_duplicates(subset=['Volume1', 'Volume2'],inplace=True)
no_signal =  [288321385,310193233,312240825,178489574,301466249,292212456,182805965]
overlap_data = overlap_data[~overlap_data['Volume2'].isin(no_signal)]

sub_metadata = pd.read_csv(path+'anterograde_subcortical.csv')
sub_metadata = sub_metadata[sub_metadata['image-series-id'].isin(subset_ids)]
sub_metadata = sub_metadata[~sub_metadata['image-series-id'].isin(no_signal)]
ctx_metadata = pd.read_csv(path+'anterograde_cortical.csv')
ctx_metadata = ctx_metadata[~ctx_metadata['image-series-id'].isin(no_signal)]

subcortical_order = pd.read_csv(path+'subcortical_order.csv').to_numpy()

overlap_data,ids = insert_names_cortical(overlap_data,ctx_metadata)
overlap_data = insert_names_subcortical(overlap_data,sub_metadata)

order_list_cortical = pd.read_csv(path+'harris_order.csv')['name'].to_list()
order_list_subcortical = generate_order_list(subcortical_order)


vol1_id = overlap_data['Volume1'].to_numpy()
vol2_id = overlap_data['Volume2'].to_numpy()
sub_metadata_ids = sub_metadata['image-series-id'].to_numpy()

assert(len(np.unique(ids))==len(np.unique(vol1_id)))
assert(len(np.unique(sub_metadata_ids))==len(np.unique(vol2_id)))

ytick,order_list_cortical = get_ticks(overlap_data,'vol1_name',order_list_cortical,sub_metadata_ids) 
xtick,order_list_subcortical = get_ticks(overlap_data,'vol2_name',order_list_subcortical,ids)

overlap_data = reorder_dataframe(overlap_data,order_list_cortical,order_list_subcortical)
# transformed_data,lambda_func=boxcox(overlap_data['Overlapping_voxels'].to_numpy()+10e-5)

# overlap_data['Overlapping_voxels'] = transformed_data

overlap_data['Volume1'] = pd.Categorical(overlap_data['Volume1'],ordered=True)
overlap_data['Volume2'] = pd.Categorical(overlap_data['Volume2'],ordered=True)

heatmap_table = overlap_data.pivot_table(index='Volume1',columns='Volume2',values='Dice_coefficient')
heatmap_table = heatmap_table.reindex(index=overlap_data['Volume1'].unique(),columns=overlap_data['Volume2'].unique())
#heatmap_table = heatmap_table.transpose()

overlap_heatmap(heatmap_table,'../figures/cortical_subcortical_overlap_voxels.svg',xtick,ytick,order_list_cortical,order_list_subcortical)
