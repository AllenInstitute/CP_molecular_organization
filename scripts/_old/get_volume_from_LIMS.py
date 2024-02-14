# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:48:13 2023

@author: ashwin.bhandiwad
"""

import os, nrrd
import SimpleITK as sitk
from pathlib import Path

import pandas as pd
import numpy as np
import allensdk.internal.core.lims_utilities as lims_utilities
from scipy.ndimage import gaussian_filter


def get_LIMS_path(image_ids,volume_type='projection_density'):
    """
    Parameters
    ----------
    image_id : list of image IDs (int)

    Returns
    -------
    path_to_image: str; filepath to image ID in LIMS
    """


    query = 'SELECT storage_directory FROM image_series im '\
            'WHERE im.id IN {0}'.format(tuple(image_id))
        
    image_paths = lims_utilities.query(query)
    path_to_image= []
    for image in image_paths:
        
        dir_path = image['storage_directory']
        dir_path += 'grid/'+volume_type+'_10.nrrd'
        path_to_image.append(dir_path)
    
    return path_to_image

def radial_mask(x,y,z,shape=[1320,800,1140],radius=30):
    
    mask = np.zeros(shape,dtype=np.int8)
    mask[x-radius:x+radius,y-radius:y+radius,z-radius:z+radius] = 1
    
    return mask

def starter_cell_dist(starter_cells,impath,count):
    
    impath = Path(impath)
    impath_lnx = impath.as_posix()
    projection_density = sitk.ReadImage('/'+impath_lnx)
    
    x = int(coords[count][1]/10)
    y = int(coords[count][0]/10)
    z = int(coords[count][2]/10)
    
    cp_mask = radial_mask(x,y,z)
    cp_mask = sitk.PermuteAxes(sitk.GetImageFromArray(cp_mask),[2,1,0])
    cp_mask = sitk.Cast(cp_mask,projection_density.GetPixelID())
    cp_mask.CopyInformation(projection_density)
    cp_density = sitk.Multiply(projection_density,cp_mask)
    cp_density = sitk.PermuteAxes(cp_density,[2,1,0])
    
    density_npy = sitk.GetArrayViewFromImage(cp_density)
    
    density_gauss = gaussian_filter(density_npy,4)
    if len(np.nonzero(density_gauss)[0])>0:
        thresh = np.quantile(density_gauss[np.nonzero(density_gauss)],0.999)
        density_idx = np.where(density_gauss>thresh)
        starter_cells[:,density_idx[0],density_idx[1],density_idx[2]] = [np.repeat(color[count,0],len(density_idx[0])),
                                                                     np.repeat(color[count,1],len(density_idx[0])),
                                                                     np.repeat(color[count,2],len(density_idx[0]))]
    return starter_cells
    

df = pd.read_table('../data/retrograde_metadata.csv',delimiter=',')

image_id = df['image_series_id'].to_list()
path_to_image = get_LIMS_path(image_id)
color = df.iloc[:,-3:].to_numpy()
coords = df.iloc[:,3:6].to_numpy()

# cp_mask = sitk.ReadImage('../data/ccf_volumes/cp_mask.nrrd')
starter_cells = np.zeros([3,1320,800,1140],np.int8)

for count,impath in enumerate(path_to_image):
    
    starter_cells = starter_cell_dist(starter_cells,impath,count)
    print(f'{impath} complete')

nrrd.write('../data/starter_cells.nrrd',starter_cells)

