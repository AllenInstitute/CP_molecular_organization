# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:01:09 2023

@author: ashwin.bhandiwad
"""
import csv,sys
import multiprocessing
import numpy as np
import pandas as pd
sys.path.append('../src/')
from swc_tools import *

def get_swc_filename():
    
    mouselight_swc_path = '//allen/programs/celltypes/workgroups/mousecelltypes/_UPENN_fMOST/morphology_data/202205111930_upload_registered_reconstructions/'
    ctx_swc_path = '../data/ctx_swcs/'
    
    metadata = pd.read_csv('../data/swc_cp_projection_densities_v2.csv')
    swc_list = metadata['experiment_id'].to_list()
    
    swc_filelist = []
    for value in swc_list:
        if value[0] =='A':
            swc_filelist.append(mouselight_swc_path+value+'.swc')
        else:
            swc_filelist.append(ctx_swc_path+value+'_reg.swc')
    
    return swc_filelist,swc_list

def process_swc_file(swc_file):
    swc_db = np.genfromtxt(swc_file)
    swc_db = flip_swc(swc_db)
    swc_graph,soma_coords = db_to_tree(swc_db)
    neurites = find_leaves(swc_graph,resolution=10)

    common_rows = np.intersect1d(neurites, cp_coords, assume_unique=False, return_indices=False)

    x=int(soma_coords[0]/resolution)
    y=int(soma_coords[1]/resolution)
    z=int(soma_coords[2]/resolution)


    ccf_idx = annotation_lookup.index[annotation_lookup['ID']==annotation_volume[x,y,z]][0]
    ccf_id = annotation_lookup.loc[ccf_idx]['ID']
    ccf_acronym = annotation_lookup.loc[ccf_idx]['Acronym']
    ccf_region,layer = split_region_layer(ccf_acronym)

    n_ipsi_pts = []
    n_contra_pts = []
    for division in division_list:
        ipsi_coords = np.load(region_lookup_path+division+'_ipsi.npy')
        ipsi = points_in_division(neurites,ipsi_coords)
        
        contra_coords = np.load(region_lookup_path+division+'_contra.npy')
        contra = points_in_division(neurites,contra_coords)
        n_ipsi_pts.append(len(ipsi))
        n_contra_pts.append(len(contra))

    row_info = [swc_file,ccf_id, ccf_region,layer,x,y,z,len(common_rows)]+n_ipsi_pts+n_contra_pts
    return row_info


def write_to_csv(batch_data):
    with open(save_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        for row_info in batch_data:
            writer.writerow(row_info)
        f.close()


def main():
    
    swc_filelist, swc_list = get_swc_filename()
    
    region_lookup_path = '../data/ccf_volumes/lookup/'
    
    annotation_volume = sitk_load('../data/ccf_volumes/annotation_10.nrrd')
    annotation_lookup = pd.read_csv('../data/ccf_volumes/AdultMouseCCF_notree.csv')
    
    save_filename = '../data/cp_projection_densities_v3.csv'
    column_names=['experiment_id','ccf_id','ccf_region','layer','x','y','z','total_pts',
              'CPdm_ipsi','CPvm_ipsi','CPiv_ipsi','CPl_ipsi','CPp_ipsi',
              'CPdm_contra','CPvm_contra','CPiv_contra','CPl_contra','CPp_contra']
    
    with open(save_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        f.close()
        
    cp_coords = np.load(region_lookup_path+'cp_mask.npy')
    
    division_list=['CPdm','CPvm','CPiv','CPl','CPp']
    resolution = 10
    batch_size = 10
    batch_size = 10
    pool = multiprocessing.Pool()
    for i in range(0, len(swc_filelist), batch_size):
        swc_batch = swc_filelist[i:i+batch_size]
        row_info_batch = pool.map(process_swc_file, swc_batch)
        write_to_csv(row_info_batch)
    pool.close()
    pool.join()


if __name__ == '__main__':
    
    main()
