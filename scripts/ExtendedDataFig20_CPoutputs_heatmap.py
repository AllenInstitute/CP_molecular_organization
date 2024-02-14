# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:08:37 2023

@author: ashwin.bhandiwad
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def insert_names(overlap_data,metadata):
    
    overlap_vol1 = overlap_data['Volume1'].to_list()
    overlap_data['Volume1'] = [int(x[:-5]) for x in overlap_vol1]
    overlap_vol2 = overlap_data['Volume2'].to_list()
    overlap_data['Volume2'] = [int(x[:-5]) for x in overlap_vol2]
    overlap_data = overlap_data.assign(vol1_name=pd.Series(dtype=str))
    overlap_data = overlap_data.assign(vol2_name=pd.Series(dtype=str))
    
    metadata_ids = metadata['image series ID'].to_numpy()

    for image_id in metadata_ids:
        
        num_repeats_x = overlap_data.loc[overlap_data['Volume1']==image_id]
        num_repeats_y = overlap_data.loc[overlap_data['Volume2']==image_id]
        overlap_data.loc[overlap_data['Volume1']==image_id, 'vol1_name'] = np.repeat(metadata.loc[metadata['image series ID']==image_id]['manually annotated injection site'].values,len(num_repeats_x))
        overlap_data.loc[overlap_data['Volume2']==image_id, 'vol2_name'] = np.repeat(metadata.loc[metadata['image series ID']==image_id]['manually annotated injection site'].values,len(num_repeats_y))
        
    return overlap_data

def subset_subcortical():
    
    df = pd.read_csv(path+'anterograde_cp_overlap_subcortical.csv')
    subcort_subset = pd.read_excel(path+'anterograde_annotated_Quanxin_subcorticalsubset.xlsx',sheet_name='Subcortical inputs to CP')
    subset_ids = subcort_subset['image series ID'].to_list()

    subset = (df['Volume1'].isin(subset_ids)) & (df['Volume2'].isin(subset_ids))
    
    df = df[subset]
            
    return df,subset_ids

def reorder_dataframe(overlap_data,order_list):
    
    
    overlap_data['vol1_name'] = pd.Categorical(overlap_data['vol1_name'], categories=order_list, ordered=True)
    overlap_data['vol2_name'] = pd.Categorical(overlap_data['vol2_name'], categories=order_list, ordered=True)
    overlap_data = overlap_data.sort_values(by=['vol1_name','Volume1','vol2_name','Volume2'])
    overlap_data = overlap_data.reset_index(drop=True)
    
    return overlap_data


def get_ticks(overlap_data,order_list,num_replicates):
    
    name = overlap_data['vol1_name'].to_numpy()
    order_list = [x for x in order_list if x in name]
    ticks=[]
    idx=0
    for region in order_list:
        replicates = int(len(np.where(name==region)[0])/num_replicates)
        idx=idx+replicates
        ticks.append(idx)
        
    return ticks,order_list

def overlap_heatmap(heatmap_table,save_filename,tick=[],label=[]):
    
    ax = sns.heatmap(heatmap_table, annot=False,vmin=0,vmax=1, cmap='plasma',rasterized=True)
    if len(tick)>0:
        plt.xticks(tick,order_list)
        plt.yticks(tick,order_list)
        # for tick in ax.get_xticks():
        #     ax.axvline(tick, color='white', linewidth=0.3)
        # for tick in ax.get_yticks():
        #     ax.axhline(tick, color='white', linewidth=0.3)
    plt.ylabel('Source')
    plt.xlabel('Source')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Dice coefficient')
    
    plt.gcf().set_size_inches(12,10)
    plt.savefig(save_filename, dpi=100)
    plt.clf()
    

    
path = '../data/'
for region in ['GPe','GPi','SNr']:
    overlap_data = pd.read_csv(path+f'ExtendedDataFig20_cp_overlap_{region}.csv')
    
    metadata = pd.read_csv(path+'CP_anterograde_output.csv')
    plot_order = ['CPdm','CPvm','CPiv','CPl','CPp']
    
    metadata['manually annotated injection site'] = pd.Categorical(metadata['manually annotated injection site'],categories=plot_order,ordered=True)
    metadata.sort_values(by=['manually annotated injection site','mouse-line'],inplace=True)
    
    order_list = metadata['image series ID'].to_list()
    overlap_data = insert_names(overlap_data,metadata)
    vol1_id = overlap_data['Volume1'].to_numpy()
    vol2_id = overlap_data['Volume2'].to_numpy()
    
    assert(len(np.unique(vol2_id))==len(np.unique(vol1_id)))
    assert(len(np.unique(order_list))==len(np.unique(vol1_id)))

    overlap_data = reorder_dataframe(overlap_data,plot_order)
    
    overlap_data['Volume1'] = pd.Categorical(overlap_data['Volume1'],ordered=True)
    overlap_data['Volume2'] = pd.Categorical(overlap_data['Volume2'],ordered=True)
    
    ticks,order_list = get_ticks(overlap_data,plot_order,len(np.unique(vol1_id)))
    
    heatmap_table = overlap_data.pivot_table(index='Volume1',columns='Volume2',values='Dice_coefficient')
    heatmap_table = heatmap_table.reindex(index=overlap_data['Volume1'].unique(),columns=overlap_data['Volume1'].unique())
    
    overlap_heatmap(heatmap_table,f'../figures/ExtendedDataFig20_cp_overlap_{region}.svg',ticks,order_list)
