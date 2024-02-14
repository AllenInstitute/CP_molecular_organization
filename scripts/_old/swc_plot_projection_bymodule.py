# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:56:43 2023

@author: ashwin.bhandiwad
"""

import sys
import numpy as np
import pandas as pd
sys.path.append('../src/')
from swc_tools import *
import SimpleITK as sitk
from multiprocessing import Pool

def neuron_3d(swc_db,volume):
    
    for point in swc_db:
        ccf_coords = tuple([int(n/10) for n in point[2:5]])
        if (ccf_coords[0]<1320 and ccf_coords[1]<800 and ccf_coords[2]<1140):
            volume[0,ccf_coords[0],ccf_coords[1],ccf_coords[2]] = 255
        
    return volume

def neurite_3d(neurite_list,volume):
    
    for point in neurite_list:
        if (point[0]<1320 and point[1]<1140 and point[2]<800):
            volume[point[0],point[1],point[2]] = 255

    return volume

def point_cloud(region):
    
    volume = np.zeros([1320,1140,800],dtype=np.int16)
    swc_path_mouselight = '//allen/programs/celltypes/workgroups/mousecelltypes/_UPENN_fMOST/morphology_data/202205111930_upload_registered_reconstructions/'
    swc_path_gao = '../data/ctx_swcs/'

    swc_df = pd.read_csv('../data/cp_projection_densities.csv')
    
    filtered_df = swc_df.loc[swc_df['ccf_region']==region]
    swc_filelist = filtered_df['experiment_id'].values

    sum_ipsi = np.sum(filtered_df.iloc[:,8:14].to_numpy(),axis=1)

    for count,swc_file in enumerate(swc_filelist):
    
        if swc_file[0] == 'A':
            swc_db = np.genfromtxt(swc_path_mouselight+swc_file+'.swc')
        else:
            swc_db = np.genfromtxt(swc_path_gao+swc_file+'_reg.swc')
            
        swc_db = flip_swc(swc_db)
        swc_graph,soma_coords = db_to_tree(swc_db)
        neurites = find_leaves(swc_graph,resolution=10)
        
        if sum_ipsi[count]>1:
            volume = neurite_3d(neurites,volume)
        
    volume = sitk.GetImageFromArray(volume)   
    volume = sitk.PermuteAxes(volume,[2,1,0])
    sitk.WriteImage(volume,f'../data/{region}_terminals.nrrd')
        
if __name__ =='__main__':
    
    regions = ['ORBm','ORBvl','ORBl','ACAv','ACAd','PL','ILA','AId','AIv']
    with Pool(processes=len(regions)) as pool:
        pool.map(point_cloud,regions)

