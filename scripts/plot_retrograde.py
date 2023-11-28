# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:38:24 2023

@author: ashwin.bhandiwad
"""
import os, re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def add_suffix(col_name, count):
    if count > 1:
        return f"{col_name}_{count}"
    else:
        return col_name
    
def reorder_columns(dataset,order_list):
    
    ordcols = dataset.columns
    # Count occurrences of each column name
    col_counts = Counter(ordcols)
    # Rename duplicated columns with suffixes
    ordcols_with_suffix = [add_suffix(col, col_counts[col]+idx) for idx,col in enumerate(ordcols)]
    
    # Set columns of dataset_cortical with suffixes for duplicated columns
    dataset.columns = ordcols_with_suffix
    
    ordered_cols = np.sort(np.array(ordcols_with_suffix))

    order_list = [item for item in order_list if any(item in df_row for df_row in dataset.index)]
    
    ordered_dataset = dataset.reindex(index=order_list)
    ordered_dataset = ordered_dataset.reindex(ordered_cols,axis=1)
    ordcols = np.sort(ordcols)
    return ordered_dataset,ordcols

def load_annotate():
    metadata = pd.read_excel(data_filename,sheet_name='metadata')
    division = assign_subdivision(metadata)
    metadata['division'] = pd.Series(division)
    metadata.to_csv('../data/retrograde_metadata.csv')
    
def set_order(ontology_filename,region_labels):
    
    ccf_order_table = pd.read_csv(ontology_filename)

    order = []
    ordered_labels = []
    for region in region_labels:
        order.append(int(ccf_order_table['order'].loc[ccf_order_table['name']==region]))
        ordered_labels.append(ccf_order_table['name'].loc[ccf_order_table['name']==region].values[0])
        
    zipped_pairs = zip(order, ordered_labels)
    ccf_order = [x for _, x in sorted(zipped_pairs)]
    return pd.api.types.CategoricalDtype(categories=ccf_order, ordered=True), ccf_order

def assign_subdivision(metadata):
    
    coords = metadata.iloc[:,2:5]
    division=[]
    for idx,row in coords.iterrows():
        
        row_coord = int(row['z']/10),int(row['x']/10),int(row['y']/10)
        blank_flag=0
        
        for j in range(0,len(ipsi_divisions)):
            
            div_mask = np.load(cp_mask_path+ipsi_divisions[j])
            in_volume = np.where((div_mask[0,:] == row_coord[0]) & (div_mask[1,:] == row_coord[1]) & (div_mask[2,:] == row_coord[2]))[0]
            if len(in_volume)>0:
                division.append(cp_divisions[j])
                print(f'{idx} {j} {row_coord[2]} {row_coord[1]} {1140-row_coord[0]} {cp_divisions[j]}')
                blank_flag += 1
        if blank_flag<1:
            print(f'{idx} {j} {row_coord[2]} {row_coord[1]} {1140-row_coord[0]} NA')
            division.append('NA')
    return division

def retrograde_heatmap(projection_table,col_order=None,hemisphere='Ipsilateral',source='Cortical',zmax=0.2,figsize = [20,20]):
    
    ax = sns.heatmap(projection_table, annot=False,vmax=zmax,rasterized=True,square=True,cmap='plasma')
    ax.set_yticks(range(np.shape(projection_table)[0]))
    ax.set_xticks(range(np.shape(projection_table)[1]))
    ax.set_yticklabels(list(projection_table.index))
    if col_order is None:
        ax.set_xticklabels(list(projection_table.columns),rotation=90)
    else:
        ax.set_xticklabels(col_order,rotation=90)
    plt.ylabel('Source')
    plt.xlabel('Targets')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Fractional density')
    plt.gcf().set_size_inches(figsize[0],figsize[1])
    plt.savefig(f'../figures/retrograde_{hemisphere}_{source}_fractional_density.svg', dpi=300)
    plt.clf()
    
def retrograde_layer_heatmap_divisions(projection_table,zmax=1):
    
    projection_table = projection_table.groupby('division').mean()
    projection_table = projection_table.iloc[:,2:-1].transpose()
    #dataset_subcortical = dataset.iloc[:,45:-1]

    ax = sns.heatmap(projection_table, annot=False,vmax=zmax,rasterized=True, square=True, cmap='plasma')
    ax.set_yticks(range(np.shape(projection_table)[0]))
    ax.set_yticklabels(list(projection_table.index))
    plt.ylabel('Source')
    plt.xlabel('Targets')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Fractional density')
    plt.gcf().set_size_inches(12,30)
    plt.savefig(f'../figures/retrograde_layer_fractional_density.svg',dpi=200)
    plt.clf()
    
def retrograde_layer_heatmap_layer(projection_table,zmax=1):
    
    df = projection_table.iloc[:,3:-1]
    
    df = df.mean(axis=0)
    layer_list = list(df.index)
    layer_num = []
    region= []
    region_order = ['region']
    for value in layer_list:
        layer_match = re.search(r'\d.*', value)
        if layer_match:
            layer_num.append(layer_match.group())
        region_match = re.search(r'^[A-Za-z]+', value)
        if region_match: 
            region.append(region_match.group())
            if region_order[-1] != region_match.group():
                region_order.append(region_match.group())
    region_order = region_order[1:]   
    dataset = pd.DataFrame({'region':region,'layer':layer_num,'projection_density':df.to_list()})
    
    ordered_names,ordered_labels = set_order('../data/harris_order.csv',region_order)
    
    dataset['region'] = dataset['region'].astype(ordered_names)
    
    heatmap_table = pd.pivot_table(dataset,values='projection_density',index='region',columns='layer')

    ax = sns.heatmap(heatmap_table, annot=False,vmax=1, rasterized=True,square=True, cmap='plasma')
    ax.set_yticks(range(0,len(region_order)))
    ax.set_yticklabels(ordered_labels)
    plt.ylabel('Laminar origin of presynaptic inputs to CP')
    plt.xlabel('Layer')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Fractional density')
    plt.gcf().set_size_inches(7,8)
    plt.savefig(f'../figures/retrograde_layer_combined_fractional_density.svg', dpi=200)
    plt.clf()
    
def plot_group(dataset,hemisphere):
    
    cortical_order = pd.read_csv('../data/harris_order_noSSp.csv')
    order_list = cortical_order['name'].to_list()
    
    dataset = dataset.drop(columns='CP')
    #dataset = dataset.groupby('division').mean()
    dataset_cortical = dataset.iloc[:,3:45].transpose()
    dataset_subcortical= dataset.iloc[:,46:-1].transpose()
    
    dataset_cortical.columns = dataset['celltype_div'].to_numpy()
    dataset_cortical,cortical_ordcols = reorder_columns(dataset_cortical,order_list)
    
    
    dataset_subcortical.columns = dataset['celltype_div'].to_list()
    dataset_subcortical.columns = list(np.sort(dataset_subcortical.columns))
    dataset_subcortical=dataset_subcortical.sort_index(axis=1)
    

    retrograde_heatmap(dataset_cortical,cortical_ordcols,hemisphere,'Cortical',0.2,[20,10])
    retrograde_heatmap(dataset_subcortical,None,hemisphere,'Subcortical',0.02,[20,15])

data_filename = '../data/retrograde_labeling.xlsx'
cp_mask_path = '../data/ccf_volumes/subdivisions/lookup/'

ipsi = pd.read_excel(data_filename,sheet_name='ipsi')

contra = pd.read_excel(data_filename,sheet_name='contra')
ipsi_layers = pd.read_excel(data_filename,sheet_name='ipsi_layers')

cp_code = np.linspace(5,10,6,dtype=int)
cp_divisions = ['CPvm','CPiv','CPbor','CPdm','CPl','CPp']

ipsi_divisions = [filename for filename in os.listdir(cp_mask_path) if 'ipsi' in filename]

#load_annotate()

metadata = pd.read_csv('../data/retrograde_metadata.csv')
for j in range(0,len(metadata)):
    ipsi.loc[(ipsi['image_series_id'] == metadata.iloc[j]['image_series_id']),'celltype_div'] = metadata.iloc[j]['celltype_div']
    contra.loc[(contra['image_series_id'] == metadata.iloc[j]['image_series_id']),'celltype_div'] = metadata.iloc[j]['celltype_div']
    ipsi_layers.loc[(ipsi_layers['image_series_id'] == metadata.iloc[j]['image_series_id']),'celltype_div'] = metadata.iloc[j]['celltype_div']

plot_group(ipsi,'Ipsilateral')
plot_group(contra,'Contralateral')
retrograde_layer_heatmap_layer(ipsi_layers)




    
