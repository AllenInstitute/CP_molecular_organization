# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:16:24 2023

@author: ashwin.bhandiwad
"""

import os, re
import numpy as np
import pandas as pd
from anytree import Node,RenderTree
import SimpleITK as sitk

def db_to_tree(swc_db):

    names=swc_db[:,0].astype(int)
    assert swc_db[0,-1]==-1
    soma_coords = [swc_db[0,2],swc_db[0,3],swc_db[0,4]]
    
    neuron = Node(f'p{names[0]}',
                  parent=None,
                  x=soma_coords[0],
                  y=soma_coords[1],
                  z=soma_coords[2])
    
    nodes={}
    nodes[neuron.name]=neuron
    for count,point in enumerate(names[1:,]):
        name = f'p{point}'
        if swc_db[count+1,-1]>0:
            parent_node=nodes[f'p{swc_db[count+1,-1].astype(int)}']
        else:
            parent_node=neuron
        nodes[name]=Node(
            name,
            parent=parent_node,
            x=swc_db[count+1,2],
            y=swc_db[count+1,3],
            z=swc_db[count+1,4])
        
    return nodes,soma_coords

def find_leaves(swc_graph,resolution=10):
    neurites = np.empty((0,3),dtype=int)
    for point in swc_graph:
        if swc_graph[point].is_leaf==True:
            x = int(swc_graph[point].x/resolution)
            y = int(swc_graph[point].y/resolution)
            z = int(swc_graph[point].z/resolution)
            neurites = np.vstack([neurites,np.array([x,y,z])])
    return neurites     

def sitk_load(image_filename,extension='nrrd'):
    """Load image from filename.
    
    Input:  image_filename (str) - image filename
            extension (str) - file type. Can be 'nrrd' or 'nii'. Default is nrrd
    
    Output: volume: SimpleITK image object
    
    """
    reader = sitk.ImageFileReader()
    if extension=='nrrd':
        reader.SetImageIO("NrrdImageIO")
    elif extension=='nii':
        reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(image_filename)
    
    volume = reader.Execute()
    
    if volume.GetSize()[0] == 1140:
        volume = sitk.PermuteAxes(volume,[2,1,0])
    
    return volume

def nonzero_coords(cp_volume):

    cp_volume_npy = sitk.GetArrayFromImage(sitk.PermuteAxes(cp_volume,[2,1,0]))
    cp_nonzero = np.nonzero(cp_volume_npy)

    return np.array(cp_nonzero).transpose()

def region_mask_lookup(file_list,path):
    
    for filename in file_list:
        cp_volume = sitk_load(path+filename)
        if 'cp_r' in filename:
            cp_coords_ipsi = xyz_single_value(nonzero_coords(cp_volume))
            cp_coords_contra = xyz_single_value(nonzero_coords(sitk.Flip(cp_volume,[False,False,True])))
            np.save(path+filename[:-5]+'_ipsi.npy',cp_coords_ipsi)
            np.save(path+filename[:-5]+'_contra.npy',cp_coords_contra)
        else:
            cp_coords = xyz_single_value(nonzero_coords(cp_volume))
            np.save(path+filename[:-4]+'npy',cp_coords)
        

def split_region_layer(acronym):

    layer = ''.join(re.findall(r'(\d+)',acronym))
    layer = 'L'+layer
    if len(layer)>2:
        layer = ''.join([layer[0:2],'/',layer[2]])

    region = re.split(r'(\d+)',acronym)[0]
        
    return region,layer

def flip_swc(swc_db,resolution=10,volume_shape=[1320,800,1140]):
    
    if int(swc_db[0,4]/resolution)>int(volume_shape[2]/2):
        swc_db[:,4] = (resolution*volume_shape[2])-swc_db[:,4]
        
    return swc_db

def xyz_single_value(coords):
    
    return np.array([int("".join(map(str, row))) for row in coords])

def points_in_division(points,ref_coords):
    
    return np.intersect1d(xyz_single_value(points), ref_coords)

if __name__ =='__main__':
    
    path = '../data/ccf_volumes/subdivisions/'
    filelist = os.listdir('../data/ccf_volumes/subdivisions/')
    region_mask_lookup(filelist,path)
    print('Filters swcs by projection to brain region mask and saves as csv')