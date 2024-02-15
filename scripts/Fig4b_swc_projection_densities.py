# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:01:09 2023

@author: ashwin.bhandiwad
"""
import csv,sys,re
import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
from neuron_morphology.swc_io import morphology_from_swc
from neuron_morphology.feature_extractor.data import Data
sys.path.append('../src/')
from swc_tools import *



region_lookup_path = '../data/ccf_volumes/lookup/'

annotation_volume = sitk_load('../data/ccf_volumes/annotation_10.nrrd')
annotation_lookup = pd.read_csv('../data/ccf_volumes/AdultMouseCCF_notree.csv')

division_list=['CPdm','CPvm','CPiv','CPl','CPp']


save_filename = '../data/cp_projection_densities_v5.csv'
column_names=['experiment_id','ccf_id','ccf_region','layer','x','y','z','total_pts',
          'CPdm_ipsi','CPvm_ipsi','CPiv_ipsi','CPl_ipsi','CPp_ipsi',
          'CPdm_contra','CPvm_contra','CPiv_contra','CPl_contra','CPp_contra']

with open(save_filename, mode='a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(column_names)
    f.close()
    
cp_coords = np.load(region_lookup_path+'cp_mask.npy')

def get_swc_filename(path=None):
    
    if path is None:
        mouselight_swc_path = '//allen/programs/celltypes/workgroups/mousecelltypes/_UPENN_fMOST/morphology_data/202205111930_upload_registered_reconstructions/'
        ctx_swc_path = '../data/CTX_VIS/'
        
        metadata = pd.read_csv('../data/swc_cp_projection_densities_v3.csv')
        swc_list = metadata['experiment_id'].to_list()
        swc_list = os.listdir(ctx_swc_path)
        
        swc_filelist = []
        for value in swc_list:
            if value[0] =='A':
                swc_filelist.append(mouselight_swc_path+value+'.swc')
            else:
                swc_filelist.append(ctx_swc_path+value)
    
    else:
        dir_path = Path(path)
        swc_files = dir_path.glob('**/*_reg.swc')
        
        swc_filelist = [str(path) for path in swc_files]
    
    return swc_filelist

def process_swc_file(swc_file,is_VIS=True):
    swc_db = np.genfromtxt(swc_file)
    experiment_id = re.split('/',swc_file)[-1]
    exp_id = re.split('\.',experiment_id)[0]
    swc_db = flip_swc(swc_db)
    
    assert swc_db[0,-1]==-1
    soma_coords = [swc_db[0,2],swc_db[0,3],swc_db[0,4]]
    resolution = 10

    if is_VIS==False:
        swc_graph = db_to_tree(swc_db)
        neurites = find_leaves(swc_graph,resolution)
        
    elif is_VIS==True:
        neurites = load_and_process_VIS(swc_file,resolution)
    
    neurite_coords = xyz_single_value(neurites)
    common_rows = np.intersect1d(neurite_coords, cp_coords, assume_unique=False, return_indices=False)
            
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
        ipsi_coords = np.load(region_lookup_path+division+'_contra.npy').T
        ipsi = points_in_division(neurites,ipsi_coords)
        
        contra_coords = np.load(region_lookup_path+division+'_ipsi.npy').T
        contra = points_in_division(neurites,contra_coords)
        n_ipsi_pts.append(len(ipsi))
        n_contra_pts.append(len(contra))
    
        row_info = [exp_id,ccf_id, ccf_region,layer,x,y,z,len(common_rows)]+n_ipsi_pts+n_contra_pts

    return row_info

def format_leaf_nodes(leaf_nodes,resolution=10):
    xyz_position = np.empty((0,3),dtype=int)
    for point in leaf_nodes:
        x = int(point['x']/resolution)
        y = int(point['y']/resolution)
        z = int(point['z']/resolution)
        xyz_position = np.vstack([xyz_position,np.array([x,y,z])])
        
    return xyz_position

def load_and_process_VIS(swc_file,resolution):
    
    swc_data = morphology_from_swc(swc_file)
    leaf_nodes = swc_data.get_leaf_nodes()
    neurites = format_leaf_nodes(leaf_nodes,resolution)
    
    return neurites

def write_to_csv(batch_data):
    with open(save_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        for row_info in batch_data:
            writer.writerow(row_info)
        f.close()


def main():
    swc_filelist = get_swc_filename('../data/CTX_VIS_not_using/')

    batch_size = 20
    pool = multiprocessing.Pool()
    for i in range(0, len(swc_filelist), batch_size):
        swc_batch = swc_filelist[i:i+batch_size]
        row_info_batch = pool.map(process_swc_file, swc_batch)
        write_to_csv(row_info_batch)
    pool.close()
    pool.join()


if __name__ == '__main__':
    
    main()
