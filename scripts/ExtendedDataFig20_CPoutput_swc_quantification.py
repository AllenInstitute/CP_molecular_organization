"""
Set of functions used to categorize single neuron reconstructions by projections to GPe, GPi, and SNr
and to plot figures used in Extended Data Fig 20.
"""

import sys,nrrd,pickle
import numpy as np
import pandas as pd
sys.path.append('../src/')
from swc_tools import *
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def generate_region_mask(ontology_metadata,region_name):
    
    annotation = sitk.ReadImage('../data/ccf_volumes/annotation_10.nrrd')
    mask_region_id = ontology_metadata.loc[ontology_metadata['Acronym']==region_name]['ID'].to_list()[0]
    mask = annotation==mask_region_id
    
    mask_npy = sitk.GetArrayFromImage(mask)
    
    nonzero_vals = np.nonzero(mask_npy)
    
    return np.array([ nonzero_vals[2], nonzero_vals[1], nonzero_vals[0] ]).T

def swc_heatmap(stats,output_region):
    
    heatmap_table = stats.pivot_table(values=f'{output_region}_projecting',index='celltype',columns='CP_division')
    
    plt.style.use('default')
    hfont = {'family': 'Arial', 'size': 12}
    plt.rc('font',**hfont)
    
    ax = sns.heatmap(heatmap_table, annot=False,vmax=75, cmap='plasma',rasterized=True,square=True)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Terminals')
    
    plt.gcf().set_size_inches(5,10)
    plt.savefig(f'../figures/CPoutput_{output_region}.svg', dpi=100)
    plt.clf()
    
def draw_filled_sphere(point, radius, arr, shape, color = [255,255,255]):
    z, y, x = point
    xx, yy, zz = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    mask = xx**2 + yy**2 + zz**2 <= radius**2
    x_min, x_max = max(0, x - radius), min(shape[1], x + radius + 1)
    y_min, y_max = max(0, y - radius), min(shape[2], y + radius + 1)
    z_min, z_max = max(0, z - radius), min(shape[3], z + radius + 1)
    for x_coord in range(x_min, x_max):
        for y_coord in range(y_min, y_max):
            for z_coord in range(z_min, z_max):
                if mask[x_coord - x_min, y_coord - y_min, z_coord - z_min]:
                    arr[0,x_coord, y_coord, z_coord] = color[0]
                    arr[1,x_coord, y_coord, z_coord] = color[1]
                    arr[2,x_coord, y_coord, z_coord] = color[2]

    return arr


dir_path = Path('../data/Peng_2021_single_neurons/')
cp_mask_filename = '../data/ccf_volumes/CP_bounds_CCFregistered.nrrd' #Change to bounds nrrd to find CP division
ontology_metadata = pd.read_csv('../data/MouseAtlas_ontologies_notree.csv')

suffix = '**/*_reg.swc'

file_list = dir_path.glob(suffix)

## Classify CP output neurons by projection pattern
cp_lookup = {'6':'CPp', '7': 'CPdm', '8': 'CPl', '9': 'CPvm', '10': 'CPiv', '12': 'CPiv'}
celltype = []
for idx in range(len(save_df)):

    # save_df.iloc[idx]['CP_division'] = cp_lookup[save_df.iloc[idx]['CP_division']]
    save_df.iloc[idx]['GPe_projecting'] = int(save_df.iloc[idx]['GPe_projecting'])
    save_df.iloc[idx]['GPi_projecting'] = int(save_df.iloc[idx]['GPi_projecting'])
    save_df.iloc[idx]['SNr_projecting'] = int(save_df.iloc[idx]['SNr_projecting'])
    
    if (save_df.iloc[idx]['GPe_projecting']>0 and save_df.iloc[idx]['GPi_projecting'] == 0 and save_df.iloc[idx]['SNr_projecting'] == 0):
        celltype.append('Drd2')
    elif (save_df.iloc[idx]['GPi_projecting'] > 0 and save_df.iloc[idx]['SNr_projecting'] > 0 and save_df.iloc[idx]['GPe_projecting'] > 0):
        celltype.append('Drd1_GPeGPiSNr')
    elif (save_df.iloc[idx]['SNr_projecting'] > 0 and save_df.iloc[idx]['GPe_projecting'] > 0):
        celltype.append('Drd1_GPeSNr')
    elif (save_df.iloc[idx]['SNr_projecting'] > 0 and save_df.iloc[idx]['GPi_projecting'] > 0):
        celltype.append('Drd1_GPiSNr')
    elif (save_df.iloc[idx]['GPi_projecting'] > 0 and save_df.iloc[idx]['GPe_projecting'] > 0):
        celltype.append('Drd1_GPeGPi')
    elif (save_df.iloc[idx]['SNr_projecting'] > 0 and save_df.iloc[idx]['GPe_projecting'] == 0 and save_df.iloc[idx]['GPi_projecting'] == 0):
        celltype.append('Drd1_SNr')
    elif (save_df.iloc[idx]['SNr_projecting'] == 0 and save_df.iloc[idx]['GPe_projecting'] == 0 and save_df.iloc[idx]['GPi_projecting'] > 0):
        celltype.append('Drd1_GPi')
    else:
        celltype.append('')
save_df['celltype'] = celltype

save_df.to_csv('../data/ExtendedDataFig20_CP_single_neuron_output.csv')

save_df = pd.read_csv('../data/ExtendedDataFig20_CP_single_neuron_output.csv')

stats = save_df.groupby(by=['CP_division','celltype']).mean()

for output_region in ['GPe','GPi','SNr']:
    
    swc_heatmap(stats,output_region)

for region in ['GPe','GPi','SNr']:
    
    mask_coords = generate_region_mask(ontology_metadata,region)
    np.save(f'../data/ccf_volumes/{region}.npy',xyz_single_value(mask_coords))

GPe = np.load('../data/ccf_volumes/GPe.npy')
GPi = np.load('../data/ccf_volumes/GPi.npy')
SNr = np.load('../data/ccf_volumes/SNr.npy')

cp_mask = sitk.ReadImage(cp_mask_filename)
cp_npy = sitk.GetArrayViewFromImage(cp_mask)

cp_metadata = pd.read_csv('../data/ExtendedDataFig20_CP_single_neuron_output.csv')
file_list = save_df['file_name'].to_numpy()

df = np.array(['file_name','CP_division','GPe_projecting','GPi_projecting','SNr_projecting'])

drd1_vol = np.zeros((3,1320,800,570),dtype=np.int8)
drd2_vol = np.zeros((3,1320,800,570),dtype=np.int8)

colorid = {'6':[75,255,255],'7':[255,0,0], '8': [255,255,0], '9': [0,255,0], '10': [255, 75, 255], '10': [255, 75, 255], '12': [255, 75, 255]}
soma_coords = np.empty((1,3))

for file_name in file_list:

    swc_db = np.genfromtxt(file_name)
    soma = swc_db[0,2:5]
    soma = [round(x/10) for x in soma]
    
    cp_div = cp_npy[soma[2],soma[1],soma[0]]

    if cp_div>0:
        soma_coords = np.vstack((soma_coords,soma))
        
        nodes,_ = db_to_tree(swc_db)
        terminals = find_leaves(nodes)
        terminal_counts = np.empty(3)

        
        
        GPe_terminals = points_in_division(terminals,GPe)
        GPi_terminals = points_in_division(terminals,GPi)
        SNr_terminals = points_in_division(terminals,SNr)
    
        dataset = np.array([file_name.name,cp_div,len(GPe_terminals),len(GPi_terminals),len(SNr_terminals)])
        df = np.vstack((df,dataset))
        
        all_terminals = np.concatenate((GPe_terminals,GPi_terminals,SNr_terminals))
        for row in all_terminals:
            rowstring = str(row)
            x,y,z = [int(rowstring[:3]),int(rowstring[3:6]),int(rowstring[6:])]
            if cp_metadata.loc[cp_metadata['file_name']==file_name.name,'celltype'].values[0] == 'Drd1':
                drd1_vol[:,x,y,z] = colorid[str(cp_div)]
            elif cp_metadata.loc[cp_metadata['file_name']==file_name.name,'celltype'].values[0] == 'Drd2':
                drd2_vol[:,x,y,z] = colorid[str(cp_div)]
                
save_df = pd.DataFrame(df[1:,:],columns=df[0,:]) 
save_df['x'] = soma_coords[1:,2]
save_df['y'] = soma_coords[1:,1]
save_df['z'] = soma_coords[1:,0]


## Plot soma locations
shape = (3,1320,800,570)
radius = 4

arr = np.zeros(shape,dtype=np.int16)

points = save_df.iloc[:,-4:]

colors = {'Drd1': [255, 100, 0], 'Drd2': [0, 100, 255]}

for celltype in ['Drd1','Drd2']:
    
    celltype_soma = points.loc[points['celltype']==celltype]

    point_set = celltype_soma.iloc[:,1:].to_numpy()

    for point in point_set:
        point = [int(x) for x in point]
        arr = draw_filled_sphere(point,radius,arr,shape,color = colors[celltype])

# Save the resulting array to a nrrd
nrrd.write('../data/ExtendedDataFig20_CPoutputs_swc_soma.nrrd',arr)
