# -*- coding: utf-8 -*-
"""
Set of functions used to plot heatmaps split by CP subdivision.
Used in Fig 3b and ExtendedData Fig 11. 
"""
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def split_name(labels,split_list=['SSp','MOp','MOs']):
    split_labels = []
    for item in labels:
        label_components = re.split('-',item)
        if label_components[0] in split_list:
            split_labels.append(label_components[0])
        else:
            split_labels.append(item)
            
    return np.array(split_labels)

def ccf_order(ontology_filename,region_labels):
    
    ccf_order_table = pd.read_csv(ontology_filename)

    region_order = []
    ordered_labels = []
    for region in region_labels:
        region_order.append(int(ccf_order_table['order'].loc[ccf_order_table['name']==region]))
        ordered_labels.append(ccf_order_table['name'].loc[ccf_order_table['name']==region].values[0])

    zipped_pairs = zip(region_order, ordered_labels)
    ccf_order = [x for _, x in sorted(zipped_pairs)]
    return pd.api.types.CategoricalDtype(categories=ccf_order, ordered=True), ccf_order_table['name'].values

def normalize_by_volume(summary_ipsi,refvol_path='../data/ccf_volumes/lookup/'):
    
    subdivs = summary_ipsi.columns.to_list()
    
    for div in subdivs:
        ref_vol = np.load(f'{refvol_path}{div}_left_hemisphere.npy')
        vol_size = np.shape(ref_vol)[1]
        
        summary_ipsi[div] /= vol_size

    return summary_ipsi

def subset_subcortical():
    
    df = pd.read_csv(path+'cp_subcortical_anterograde_projections.csv')
    subcort_subset = pd.read_excel(path+'anterograde_annotated_Quanxin_subcorticalsubset.xlsx',sheet_name='Subcortical inputs to CP')
    subset_ids = subcort_subset['image series ID'].to_list()

    for idx,value in enumerate(df['experiment_id']):
        if value not in subset_ids:
            df=df.drop(idx)
            
    df.to_csv(path+'cp_subcortical_anterograde_projections.csv')

def anterograde_heatmap(projection_table,region_labels,hemisphere='Ipsilateral'):

    ax = sns.heatmap(projection_table, annot=False,vmax=7e-8, square=True, rasterized=True,cmap='plasma')
    ax.set_yticks(range(len(projection_table)))
    ax.set_yticklabels(region_labels)
    ax.set_xticklabels(list(projection_table.columns))
    plt.ylabel(f'Source')
    plt.xlabel('Target')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Fractional density')
    plt.gcf().set_size_inches(8,16)
    plt.savefig(f'../figures/Fig3b_anterograde_{hemisphere}_fractional_density.svg', dpi=200)
    plt.clf()
    
    
def celltype_heatmap(projection_table,hemisphere='Ipsilateral',source='Cortical',colormap='plasma',ymin=0,ymax=0.9):
    
    
    ax = sns.heatmap(projection_table,annot=False,vmin=ymin,vmax=ymax,rasterized=True,cmap=colormap,square=True)
    ax.set_yticks(range(len(projection_table)))
    ax.set_yticklabels(list(projection_table.index.categories))
    ax.set_xticklabels(['L2/3 IT','L4/5 IT','L5 ET','L5 IT','L6 CT','L6','PV','Sst'])
    plt.ylabel(f'Source')
    plt.xlabel('Cell-type')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Fractional density')
    plt.gcf().set_size_inches(8,16)
    plt.savefig(f'../figures/ExtendedData_{hemisphere}_celltype_fractional_density.svg', dpi=200)
    plt.clf()
    
def swc_heatmap(projection_table,savefile,colormap='plasma',ymin=0,ymax=50):
    
    ipsi = projection_table.iloc[:,5:10]
    contra= projection_table.iloc[:,10:15]
    
    grid_shape = np.shape(ipsi)
    
    region_labels = projection_table.index.values
    
    fig, axes = plt.subplots(1, 2,figsize=(10,10),sharey=True)

    sns.heatmap(ipsi,annot=False,vmin=ymin,vmax=ymax,rasterized=True,cmap=colormap,cbar=False,ax=axes[0],square=True)
    axes[0].set(yticks=range(len(projection_table)),yticklabels=region_labels,
                              xticklabels = ['CPiv','CPdm','CPl','CPp','CPvm'])
    
    sns.heatmap(contra,annot=False,vmin=ymin,vmax=ymax,rasterized=True,cmap=colormap,cbar=False,ax=axes[1],square=True)
    axes[1].set(yticks=range(len(projection_table)),yticklabels=region_labels,
                              xticklabels = ['CPiv','CPdm','CPl','CPp','CPvm'])
    # cbar = axes[1].collections[0].colorbar
    # cbar.set_label('Mean axon termini')

    plt.savefig(f'../figures/{savefile}.svg', dpi=200)
    plt.clf()
    
def csv_to_plot_matrix(projection_filepath,ccf_order_filepath,name_col='cortex_region',group='division'):
    
    df = pd.read_csv(projection_filepath)
    df['lateral'] = (df['projection_density_ipsi']-df['projection_density_contra'])#/(df['projection_density_ipsi']+df['projection_density_contra']+10e-7)
    df[name_col] = split_name(df[name_col].to_numpy())
    region_labels=np.unique(df[name_col].to_numpy())
    
    print(np.shape(df))

    summary_ccf = df.groupby([group,name_col]).mean()
    summary_ccf = summary_ccf.reset_index()
    ordered_names,ordered_labels = ccf_order(ccf_order_filepath,region_labels)

    summary_ccf[name_col] = summary_ccf[name_col].astype(ordered_names)

    summary_ipsi = summary_ccf.pivot(name_col,group,"projection_density_ipsi")
    summary_contra = summary_ccf.pivot(name_col,group,"projection_density_contra")
    lat = summary_ccf.pivot(name_col,group,"lateral")
    
    return summary_ipsi,summary_contra,lat,ordered_labels


path = '../data/'

# ## FIGURE 3 - Anterograde projections to CP subdivisions
# Corticostriatal projections by subdivision
summary_ipsi,summary_contra,_,region_labels = csv_to_plot_matrix(path+'Fig3b_cp_cortical_anterograde_projections.csv',
                                                      path+'harris_order_smgrouped.csv',name_col='cortex_region')

summary_ipsi = normalize_by_volume(summary_ipsi)
summary_contra = normalize_by_volume(summary_contra)

region_labels = list(summary_contra.index)

anterograde_heatmap(summary_ipsi,region_labels,'Ipsilateral')
anterograde_heatmap(summary_contra,region_labels,'Contralateral')

# # FIGURE 3 - Anterograde projections to CP by celltype

summary_ipsi,summary_contra,lat,region_labels = csv_to_plot_matrix(path+'cp_layer_projections_L5merged.csv',
                                                      path+'harris_order_smgrouped.csv',
                                                      name_col='cortex_region',
                                                      group='cell_type')

celltype_heatmap(summary_ipsi,'Ipsilateral')
celltype_heatmap(summary_contra,'Contralateral')
celltype_heatmap(lat,'Difference',colormap='vlag',ymin=-0.5,ymax=0.5)






