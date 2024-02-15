"""
Set of functions used to generate injection sites/swc soma locations for Figs 3,4,5. 

"""
import nrrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, Array

def hex_to_rgb(hex_value):
    
    hex_value = hex_value.lstrip("#")
    
    return tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))

def draw_color_sphere(point, rad, arr, shape, color):
    x, y, z = point
    rad = 2*int(rad)
    xx, yy, zz = np.ogrid[-rad:rad+1, -rad:rad+1, -rad:rad+1]
    mask = xx**2 + yy**2 + zz**2 <= rad**2
    x_min, x_max = max(0, x - rad), min(shape[1], x + rad + 1)
    y_min, y_max = max(0, y - rad), min(shape[2], y + rad + 1)
    z_min, z_max = max(0, z - rad), min(shape[3], z + rad + 1)
    for x_coord in range(x_min, x_max):
        for y_coord in range(y_min, y_max):
            for z_coord in range(z_min, z_max):
                if mask[x_coord - x_min, y_coord - y_min, z_coord - z_min]:
                    arr[0, x_coord, y_coord, z_coord] = color[0]
                    arr[1, x_coord, y_coord, z_coord] = color[1]
                    arr[2, x_coord, y_coord, z_coord] = color[2]
    
    return arr

def draw_filled_sphere(point, radius, arr, shape):
    x, y, z = point
    radius = int(radius)
    xx, yy, zz = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    mask = xx**2 + yy**2 + zz**2 <= radius**2
    x_min, x_max = max(0, x - radius), min(shape[0], x + radius + 1)
    y_min, y_max = max(0, y - radius), min(shape[1], y + radius + 1)
    z_min, z_max = max(0, z - radius), min(shape[2], z + radius + 1)
    for x_coord in range(x_min, x_max):
        for y_coord in range(y_min, y_max):
            for z_coord in range(z_min, z_max):
                if mask[x_coord - x_min, y_coord - y_min, z_coord - z_min]:
                    arr[x_coord, y_coord, z_coord] = 255
    
    return arr


path = Path('../data/')

## Retrograde injection coordinates

retrograde_metadata = pd.read_table(path / 'retrograde_metadata.csv',sep=',')

cp_bounds,_ = nrrd.read(path / 'ccf_volumes/cp_mask.nrrd')
shape = cp_bounds.shape
cp_bounds = np.tile(cp_bounds[..., np.newaxis], (1, 1, 1, 3))
radius = 5 #voxels

x = [(int(retrograde_metadata['x'][j]/10)) for j in range(retrograde_metadata.shape[0])]
y = [(int(retrograde_metadata['y'][j]/10)) for j in range(retrograde_metadata.shape[0])]
z = [(int(retrograde_metadata['z'][j]/10)) for j in range(retrograde_metadata.shape[0])]

zhemi=[]
for value in z:
    if value<570:
        zhemi.append(1140-value)
    else:
        zhemi.append(value)

injection_pts = np.vstack([x,y,zhemi]).transpose()

colors = retrograde_metadata.iloc[:,-3:].to_numpy()

idx = np.indices(shape)
x_idx, y_idx, z_idx = idx[1], idx[0], idx[2]

for j,point in enumerate(injection_pts):
    print(f'{j}: {point}')
    distances = np.sqrt((x_idx - point[0])**2 + (y_idx - point[1])**2 + (z_idx - point[2])**2)
    sphere = distances <= radius
    cp_bounds[sphere,:] = np.tile(colors[j], [len(cp_bounds[sphere,:]),1])

nrrd.write(str(path.resolve())+'/retrograde_r.nrrd',cp_bounds[:,:,:,0])
nrrd.write(str(path.resolve())+'/retrograde_g.nrrd',cp_bounds[:,:,:,1])
nrrd.write(str(path.resolve())+'/retrograde_b.nrrd',cp_bounds[:,:,:,2])


# ### Fig 3a - Cortical anterograde injection coordinates - for flatmapping

anterograde_metadata = pd.read_table(path / 'anterograde_annotated_Quanxin.csv',sep=',')
all_experiments = pd.read_table(path / 'all_experiments.csv',sep=',')

experiment_ids = anterograde_metadata['image-series-id'].to_numpy()

cortical_subset = all_experiments[all_experiments['id'].isin(experiment_ids)]

shape = (1320,800,1140)
arr = np.zeros(shape,dtype=np.int16)

injection_pts = cortical_subset.iloc[:,4:7].to_numpy(dtype=int)

radius = cortical_subset['injection_volume'].to_numpy()
radius = abs(np.log(radius/radius.max()))

for idx,point in enumerate(injection_pts):
    point = [int(x/10) for x in point]
    arr = draw_filled_sphere(point,radius[idx],arr,shape)
    

nrrd.write(str(path.resolve())+'/cortical_injections.nrrd',arr)

### Fig 3d - anterograde injections figure

cortical_metadata = pd.read_table(path / 'anterograde_cortical.csv',sep=',')
subcortical_metadata = pd.read_table(path / 'anterograde_subcortical.csv',sep=',')
all_experiments = pd.read_table(path / 'all_experiments.csv',sep=',')
ontology_metadata = pd.read_table(path / 'MouseAtlas_ontologies_notree.csv',sep=',')

experiment_ids = np.concatenate((cortical_metadata['image-series-id'].to_numpy(),subcortical_metadata['image-series-id'].to_numpy()))

subset = all_experiments[all_experiments['id'].isin(experiment_ids)]

shape = (3,1320,800,1140)

volume = np.zeros(shape,dtype=np.int16)

injection_pts = subset.iloc[:,4:7].to_numpy()
injection_pts = injection_pts/10
injection_pts = injection_pts.astype(np.int16)

radius = subset['injection_volume'].to_numpy()
radius = abs(np.log(radius/radius.max()))

color = np.empty((len(subset),3))
for idx,ccf_region in enumerate(subset['structure_abbrev'].values):
    
    ccf_color = ontology_metadata[ontology_metadata['Acronym'] == ccf_region]
    color[idx,:] = [ccf_color['red'].values, ccf_color['green'].values, ccf_color['blue'].values]


for idx,point in enumerate(injection_pts):
    
    volume = draw_color_sphere(point, radius[idx], volume, shape, color[idx])
    
nrrd.write('../data/Fig3a_injection_locations.nrrd',volume)

# # ### Fig 4a - Single neuron reconstruction soma positions

metadata = pd.read_table(path / 'Fig4_swc_cp_projection_densities.csv',sep=',')
output_filepath = '../data/swc_soma.nrrd'
points = metadata.iloc[:,5:8].to_numpy()
shape = (1320,800,1140)
radius = 2

arr = np.zeros(shape,dtype=int)

for point in points:
    
    arr = draw_filled_sphere(point,radius,arr,shape)
    print(point)

# Save the resulting array to a nrrd
nrrd.write('../data/swc_soma.nrrd',arr)


## Subcortical anterograde

metadata = pd.read_table(path / 'anterograde_subcortical.csv',sep=',')
metadata.set_index('image-series-id',inplace=True)

all_experiments = pd.read_table(path / 'all_experiments.csv',sep=',')
all_experiments.rename(columns = {'id': 'image-series-id'},inplace=True)
all_experiments.set_index('image-series-id',inplace=True)

site = all_experiments.iloc[:,2:6]

df=metadata.join(site)

cp_bounds,_ = nrrd.read(path / 'ccf_volumes/cp_mask.nrrd')
cp_bounds = cp_bounds.astype(np.int16)
shape = cp_bounds.shape
cp_bounds = np.tile(cp_bounds[..., np.newaxis], (1, 1, 1, 3))

x = [(int(df['injection_x'].values[j]/10)) for j in range(df.shape[0])]
y = [(int(df['injection_y'].values[j]/10)) for j in range(df.shape[0])]
z = [(int(df['injection_z'].values[j]/10)) for j in range(df.shape[0])]

zhemi=[]
for value in z:
    if value<570:
        zhemi.append(1140-value)
    else:
        zhemi.append(value)

injection_pts = np.vstack([x,y,zhemi]).transpose()
col_array = np.repeat([[255],[255],[255]],len(injection_pts),axis=1).transpose()

injection_vol = df['injection_volume'].to_numpy()
radius = 1-2*np.log(injection_vol)

idx = np.indices(shape)
x_idx, y_idx, z_idx = idx[0], idx[1], idx[2]

for j,point in enumerate(injection_pts):
    print(f'{j}: {point}')
    distances = np.sqrt((x_idx - point[0])**2 + (y_idx - point[1])**2 + (z_idx - point[2])**2)
    sphere = distances <= radius[j]
    cp_bounds[sphere,:] = np.tile(col_array[j,:], [len(cp_bounds[sphere,:]),1])

cp_bounds = np.moveaxis(cp_bounds,-1,0)
cp_bounds = cp_bounds[:,:,:,570:]
    
nrrd.write(str(path.resolve())+'/subcortical_injections.nrrd',cp_bounds)

# ## CP anterograde injections

metadata = pd.read_table(path / 'CP_injection_sites.csv',sep=',')
metadata.set_index('image-series-ID',inplace=True)

all_experiments = pd.read_table(path / 'all_experiments.csv',sep=',')
all_experiments.rename(columns = {'id': 'image series ID'},inplace=True)
all_experiments.set_index('image series ID',inplace=True)

site = all_experiments.iloc[:,2:6]

df=metadata.join(site)

cp_bounds,_ = nrrd.read(path / 'ccf_volumes/cp_mask.nrrd')
cp_bounds = cp_bounds.astype(np.int16)
shape = cp_bounds.shape
cp_bounds = np.tile(cp_bounds[..., np.newaxis], (1, 1, 1, 3))

x = [(int(df['injection_x'].values[j]/10)) for j in range(df.shape[0])]
y = [(int(df['injection_y'].values[j]/10)) for j in range(df.shape[0])]
z = [(int(df['injection_z'].values[j]/10)) for j in range(df.shape[0])]

zhemi=[]
for value in z:
    if value<570:
        zhemi.append(1140-value)
    else:
        zhemi.append(value)

injection_pts = np.vstack([x,y,zhemi]).transpose()
mouse_line = df['mouse-line'].to_numpy()
celltype = np.unique(mouse_line)
colors = pd.read_table(path / 'CPinjection_14colors.csv',header=None).to_numpy()
col_array = np.empty((len(mouse_line),3))
for idx,x in enumerate(mouse_line):
    hex_color = colors[np.where(celltype==x)][0][0]
    col_array[idx,:] = hex_to_rgb(hex_color)

injection_vol = df['injection_volume'].to_numpy()
radius = 1-2*np.log(injection_vol)

idx = np.indices(shape)
x_idx, y_idx, z_idx = idx[0], idx[1], idx[2]

for j,point in enumerate(injection_pts):
    print(f'{j}: {point}')
    distances = np.sqrt((x_idx - point[0])**2 + (y_idx - point[1])**2 + (z_idx - point[2])**2)
    sphere = distances <= radius[j]
    cp_bounds[sphere,:] = np.tile(col_array[j,:], [len(cp_bounds[sphere,:]),1])

cp_bounds = np.moveaxis(cp_bounds,-1,0)
cp_bounds = cp_bounds[:,:,:,570:]
    
nrrd.write(str(path.resolve())+'/cp_injection_sites.nrrd',cp_bounds)


# Extended data Fig 7a: cell-type projections data

cortical_metadata = pd.read_table(path / 'ExtendedDataFig7a_cp_layer_projections_L5merged.csv',sep=',')

all_experiments = pd.read_table(path / 'all_experiments.csv',sep=',')

experiment_ids = cortical_metadata['experiment_id'].to_numpy()

color_lookup = {}
unique_celltypes = np.unique(cortical_metadata['cell_type'])
color_list = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']

for idx,celltype in enumerate(unique_celltypes):
    color_lookup[celltype] = color_list[idx]
    
subset = all_experiments[all_experiments['id'].isin(experiment_ids)]
cortical_metadata_filtered = cortical_metadata[cortical_metadata['experiment_id'].isin(subset['id'].values)]

shape = (3,1320,800,1140)

volume = np.zeros(shape,dtype=np.int16)

injection_pts = subset.iloc[:,4:7].to_numpy()
injection_pts = injection_pts/10
injection_pts = injection_pts.astype(np.int16)

radius = subset['injection_volume'].to_numpy()
radius = abs(np.log(radius/radius.max()))

color = np.empty((len(subset),3))
for idx,celltype in enumerate(cortical_metadata_filtered['cell_type'].values):
    
    ccf_color = hex_to_rgb(color_lookup[celltype])
    color[idx,:] = [ccf_color[0], ccf_color[1], ccf_color[2]]


for idx,point in enumerate(injection_pts):
    # if point[2]<570:
    #     point[2] += int(570)
    volume = draw_color_sphere(point, radius[idx], volume, shape, color[idx])
    
nrrd.write('../data/ExtendedData7a_injection_locations_r.nrrd',volume[0,:,:,:])
nrrd.write('../data/ExtendedData7a_injection_locations_g.nrrd',volume[1,:,:,:])
nrrd.write('../data/ExtendedData7a_injection_locations_b.nrrd',volume[2,:,:,:])

