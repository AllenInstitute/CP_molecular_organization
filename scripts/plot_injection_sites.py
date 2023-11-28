# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:02:26 2023

@author: ashwin.bhandiwad
"""
import nrrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import ccf_streamlines.projection as ccf_proj
from pathlib import Path

def hex_to_rgb(hex_value):
    
    hex_value = hex_value.lstrip("#")
    
    return tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))

path = Path('../data/')

# ## Retrograde injection coordinates

# retrograde_metadata = pd.read_table(path / 'retrograde_metadata.csv',sep=',')

# cp_bounds,_ = nrrd.read(path / 'ccf_volumes/cp_mask.nrrd')
# shape = cp_bounds.shape
# cp_bounds = np.tile(cp_bounds[..., np.newaxis], (1, 1, 1, 3))
# radius = 5 #voxels

# x = [(int(retrograde_metadata['x'][j]/10)) for j in range(retrograde_metadata.shape[0])]
# y = [(int(retrograde_metadata['y'][j]/10)) for j in range(retrograde_metadata.shape[0])]
# z = [(int(retrograde_metadata['z'][j]/10)) for j in range(retrograde_metadata.shape[0])]

# zhemi=[]
# for value in z:
#     if value<570:
#         zhemi.append(1140-value)
#     else:
#         zhemi.append(value)

# injection_pts = np.vstack([x,y,zhemi]).transpose()

# colors = retrograde_metadata.iloc[:,-3:].to_numpy()

# idx = np.indices(shape)
# x_idx, y_idx, z_idx = idx[1], idx[0], idx[2]

# for j,point in enumerate(injection_pts):
#     print(f'{j}: {point}')
#     distances = np.sqrt((x_idx - point[0])**2 + (y_idx - point[1])**2 + (z_idx - point[2])**2)
#     sphere = distances <= radius
#     cp_bounds[sphere,:] = np.tile(colors[j], [len(cp_bounds[sphere,:]),1])

# nrrd.write(str(path.resolve())+'/retrograde_r.nrrd',cp_bounds[:,:,:,0])
# nrrd.write(str(path.resolve())+'/retrograde_g.nrrd',cp_bounds[:,:,:,1])
# nrrd.write(str(path.resolve())+'/retrograde_b.nrrd',cp_bounds[:,:,:,2])


# ### Anterograde injection coordinates

# anterograde_metadata = pd.read_table(path / 'anterograde_cortical.csv',sep=',')
# all_experiments = pd.read_table(path / 'all_experiments.csv',sep=',')

# experiment_ids = anterograde_metadata['image-series-id'].to_numpy()

# cortical_subset = all_experiments[all_experiments['id'].isin(experiment_ids)]

# volume = np.zeros([1320,800,1140],dtype=np.int16)

# injection_pts = cortical_subset.iloc[:,4:7].to_numpy()
# injection_pts = injection_pts/10

# radius = cortical_subset['injection_volume'].to_numpy()
# radius = abs(np.log(radius/radius.max()))

# idx = np.indices(np.shape(volume))
# x_idx, y_idx, z_idx = idx[1], idx[0], idx[2]

# for j,point in enumerate(injection_pts):
#     print(f'{j}')
#     distances = np.sqrt((x_idx - point[0])**2 + (y_idx - point[1])**2 + (z_idx - point[2])**2)
#     sphere = distances <= radius[j]
#     volume[sphere] = np.repeat(255, np.shape(volume[sphere]))

# nrrd.write(str(path.resolve())+'/cortical_injections.nrrd',volume)

# ### Single neuron reconstruction soma positions

# metadata = pd.read_table(path / 'cp_projection_densities.csv',sep=',')
# injection_pts = metadata.iloc[:,5:8].to_numpy()
# colors = metadata.iloc[:,-3:].to_numpy()

# volume = np.zeros([1320,800,1140,3],dtype=np.int16)

# idx = np.indices([1320,800,1140])
# x_idx, y_idx, z_idx = idx[1], idx[0], idx[2]

# for j,point in enumerate(injection_pts):
#     print(f'{j}: {point}')
#     distances = np.sqrt((x_idx - point[0])**2 + (y_idx - point[1])**2 + (z_idx - point[2])**2)
#     sphere = distances <= 5
#     volume[sphere,:] = np.tile(colors[j], [len(volume[sphere,:]),1])

# nrrd.write(str(path.resolve())+'/swc_soma_r.nrrd',volume[:,:,:,0])
# nrrd.write(str(path.resolve())+'/swc_soma_g.nrrd',volume[:,:,:,1])
# nrrd.write(str(path.resolve())+'/swc_soma_b.nrrd',volume[:,:,:,2])

## CP anterograde injections
metadata = pd.read_table(path / 'CP_injection_sites.csv',sep=',')
metadata.set_index('image series ID',inplace=True)

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

idx = np.indices([1320,800,1140],dtype=np.int16)
x_idx, y_idx, z_idx = idx[0], idx[1], idx[2]

for j,point in enumerate(injection_pts):
    print(f'{j}: {point}')
    distances = np.sqrt((x_idx - point[0])**2 + (y_idx - point[1])**2 + (z_idx - point[2])**2)
    sphere = distances <= radius[j]
    cp_bounds[sphere,:] = np.tile(col_array[j,:], [len(cp_bounds[sphere,:]),1])

cp_bounds = np.moveaxis(cp_bounds,-1,0)
cp_bounds = cp_bounds[:,:,:,570:]
    
nrrd.write(str(path.resolve())+'/cp_injection_sites_k.nrrd',cp_bounds)
