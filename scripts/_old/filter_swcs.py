# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:01:09 2023

@author: ashwin.bhandiwad
"""
import sys
sys.path.append('../src/')
from swc_tools import *

swc_path = '../data/ctx_swcs/'
swc_filelist = os.listdir(swc_path)

cp_coords = nonzero_coords('../data/ccf_volumes/cp_mask.nrrd')

annotation_volume = sitk_load('../data/ccf_volumes/annotation_10.nrrd')
annotation_lookup = pd.read_csv('../data/ccf_volumes/AdultMouseCCF_notree.csv')
resolution=10

cp_projecting = pd.DataFrame(columns=['experiment_id','ccf_id','ccf_region','layer','x','y','z'])

for swc_file in swc_filelist[1682:]:
    
    swc_db = np.genfromtxt(swc_path+swc_file)
    swc_graph,soma_coords = db_to_tree(swc_db)
    neurites = find_leaves(swc_graph,resolution)
    
    common_rows = np.intersect1d(neurites, cp_coords, assume_unique=False, return_indices=False)
    
    if len(common_rows)>50:
        
        x=int(soma_coords[0]/resolution)
        y=int(soma_coords[1]/resolution)
        z=int(soma_coords[2]/resolution)

        ccf_idx = annotation_lookup.index[annotation_lookup['ID']==annotation_volume[x,y,z]][0]
        ccf_id = annotation_lookup.loc[ccf_idx]['ID']
        ccf_acronym = annotation_lookup.loc[ccf_idx]['Acronym']
        ccf_region,layer = split_region_layer(ccf_acronym)
        cp_projecting.loc[len(cp_projecting)] = [swc_file[:-8],
                                                 ccf_id,
                                                 ccf_region,
                                                 layer,
                                                 x,y,z]
        print(swc_file)

cp_projecting.to_csv(swc_path+'cp_projecting_2.csv',index=False) 