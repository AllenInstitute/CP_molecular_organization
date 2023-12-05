# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:41:25 2023

@author: ashwin.bhandiwad
"""

import csv,sys
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
sys.path.append('../src/')
from swc_tools import *
# volume = np.zeros((3,1320,800,1140),dtype=np.uint8)
# def neuron_3d(swc_db,volume):
    
#     for point in swc_db:
#         ccf_coords = tuple([int(n/10) for n in point[2:5]])
#         #volume[ccf_coords[0],ccf_coords[1],:] = 30
#         volume[0,ccf_coords] = 255

def set_globals():
    
    global swc_path_mouselight
    global swc_path_gao
    global cp_coords
    
    swc_path_mouselight = '//allen/programs/celltypes/workgroups/mousecelltypes/_UPENN_fMOST/morphology_data/202205111930_upload_registered_reconstructions/'
    swc_path_gao = '../data/ctx_swcs/'
    cp_coords = np.load('../data/ccf_volumes/lookup/cp_ipsi.npy')

def arbor_calculation(args):
    
    set_globals()
    
    [swc_file,ccf_id,ccf_region,layer,celltype] = args

    if swc_file[0] == 'A':
        swc_db = np.genfromtxt(swc_path_mouselight+swc_file+'.swc')
    else:
        swc_db = np.genfromtxt(swc_path_gao+swc_file+'_reg.swc')
        
    swc_db = flip_swc(swc_db)
    swc_graph,soma_coords = db_to_tree(swc_db)
    neurites = find_leaves(swc_graph,resolution=10)
    
    neurites_merge = np.array([int(f'{x[0]}{x[1]}{x[2]}') for x in neurites])
    
    _,_,terminal_idx = np.intersect1d(cp_coords, neurites_merge, assume_unique=False, return_indices=True)
    npoints = len(terminal_idx)
    
    if npoints>1:
        cp_terminals = neurites[terminal_idx,:]
        
        center = np.mean(cp_terminals,axis=0)
        cp_terminal_diff = cp_terminals - center
        
        terminal_point_distance = np.linalg.norm(cp_terminal_diff,axis=1)
        dispersion = np.median(terminal_point_distance)

    else:
        dispersion = ''
    
    row_data = [swc_file,ccf_id,ccf_region,layer,celltype,dispersion,npoints ]
    
    with open('../data/swc_arbor_stats_l23.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)
        



if __name__ == '__main__':

    swc_df = pd.read_csv('../data/cp_projection_densities.csv')

    swc_filelist = swc_df.loc[swc_df['layer']=='L2/3']

    args_list = [(swc_filelist.iloc[x]['experiment_id'], 
                  swc_filelist.iloc[x]['ccf_id'],
                  swc_filelist.iloc[x]['ccf_region'],
                  swc_filelist.iloc[x]['layer'],
                  swc_filelist.iloc[x]['celltype']) for x in range(872,len(swc_filelist))]

    with open('../data/swc_arbor_stats.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment_id','ccf_id','ccf_region','layer','celltype',
                          'dispersion','size'])
    
    print(cpu_count())
    with Pool(processes=20) as pool:
        pool.map(arbor_calculation,args_list)
