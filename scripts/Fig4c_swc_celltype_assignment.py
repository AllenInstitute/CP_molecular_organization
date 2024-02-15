# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:18:15 2023

@author: ashwin.bhandiwad
"""

import csv,multiprocessing,nrrd,json,sys,os
import numpy as np
from pathlib import Path
from anytree import Node
from neuron_morphology.swc_io import morphology_from_swc
from neuron_morphology.feature_extractor.data import Data
sys.path.append('../src/')
from swc_tools import *



def json_to_tree(json_filename):
    
    with open(json_filename, 'r') as file:
        dataset = json.load(file)
        file.close()

    root_data = dataset['msg'][0]

    root = Node(name='997', 
                full_name=root_data.get('name'),
                acronym=root_data.get('acronym'),
                parent=None, 
                color=root_data.get('color_hex_triplet'))
    nodes = {'997' : root}
    tree_builder(root_data.get('children'),nodes)
    
    return root,nodes

def tree_builder(d,nodes):

    if isinstance(d,list):
        for item in d:
            if isinstance(item, dict):
                node_uid = str(item.get('id'))
                node = nodes[node_uid] = Node(name=node_uid,
                                       full_name=item.get('name'),
                                       acronym=item.get('acronym'),
                                       parent=nodes[str(item.get('parent_structure_id'))], 
                                       color=item.get('color_hex_triplet'))
                tree_builder(item.get('children'),nodes)
            
    return nodes

def divs_list(annotation_volume,ccf_tree,brain_divs):
    
    div_superset = {}
    volume = np.zeros(np.shape(annotation_volume),dtype=np.int16)
    
    for div in brain_divs:
        div_descendants = ccf_tree[str(div)].descendants
        descendants_list = []
        for node in div_descendants:
            descendants_list.append(float(node.name))
        
        div_superset[str(div)] = descendants_list
        volume[np.isin(annotation_volume,descendants_list)] = div
        
    return div_superset,volume

def assign_celltype(point_set,threshold=5):
    
    ctx = point_set[0]
    cnu = point_set[1]
    th = point_set[2]
    rest = sum(point_set[3:])
    

    if ((ctx>=threshold or cnu>=threshold or th>=threshold) and (rest>=threshold)):
        celltype = 'ET'
    elif ((ctx>=threshold or cnu>=threshold) and th>=threshold):
        celltype = 'CT'
    elif (ctx>=threshold or cnu>=threshold):
        celltype = 'IT'
    else:
        celltype = ''
    
    return celltype

def process_swc_file(swc_file,is_VIS=True):
    
    volume = np.load(filepath / 'celltype_divisions.npy')
    brain_bounds = np.shape(volume)
    
    def load_and_process_VIS(swc_file,resolution):
        
        swc_data = morphology_from_swc(swc_file)
        leaf_nodes = swc_data.get_leaf_nodes()
        neurites = format_leaf_nodes(leaf_nodes,resolution)
        
        return neurites
    
    def format_leaf_nodes(leaf_nodes,resolution=10):
        xyz_position = np.empty((0,3),dtype=int)
        for point in leaf_nodes:
            x = int(point['x']/resolution)
            y = int(point['y']/resolution)
            z = int(point['z']/resolution)
            xyz_position = np.vstack([xyz_position,np.array([x,y,z])])
            
        return xyz_position

    resolution = 10

    if is_VIS==False:
        swc_file = swc_file.replace('\\','/')
        swc_db = np.genfromtxt(swc_file)
        swc_db = flip_swc(swc_db)
        assert swc_db[0,-1]==-1
        swc_graph = db_to_tree(swc_db)
        neurites = find_leaves(swc_graph,resolution)
        
    elif is_VIS==True:
        neurites = load_and_process_VIS(swc_file,resolution)
    
    # if swc_file[-4:] == '.swc':
    #     swc_file = swc_file.replace('\\','/')
    #     swc_db = np.genfromtxt(swc_file)
    #     swc_db = flip_swc(swc_db)
    #     swc_graph,soma_coords = db_to_tree(swc_db)
    #     neurites = find_leaves(swc_graph,resolution=10)
    # else:
    #     row_info = [0]*8
    #     return row_info
    
    brain_divs = [688,623,549,1097,313,1065,512]
    locations=[]
    for point in neurites:
        if (point[0]<brain_bounds[0] and point[1]<brain_bounds[1] and point[2]<brain_bounds[2]):
            locations.append(volume[point[0],point[1],point[2]])
    
    neurite_points=[]
    for div in brain_divs:
        neurite_points.append(locations.count(div))
        
    celltype = assign_celltype(neurite_points,threshold=2)
    swc_filename = re.split('/',swc_file)[-1]
    row_info = [swc_filename[:-4]]+neurite_points+[celltype]
    #row_info = [swc_file[:-4]]+neurite_points+[celltype]

    return row_info

def write_to_csv(batch_data):
    with open(save_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        for row_info in batch_data:
            writer.writerow(row_info)
        f.close()
        
def main(swc_filelist):
    
    _,ccf_tree = json_to_tree(filepath / 'adult_mouse_ontology.json')

    divs_file = filepath / 'celltype_divisions.npy'
    if divs_file.exists():
        volume = np.load(divs_file)
    else:
        div_superset,volume = divs_list(annotation_volume,ccf_tree,brain_divs)
        np.save(divs_file,volume)

    pool = multiprocessing.Pool()
    for i in range(0, len(swc_filelist), batch_size):
        swc_batch = swc_filelist[i:i+batch_size]
        row_info_batch = pool.map(process_swc_file, swc_batch)
        write_to_csv(row_info_batch)
    pool.close()
    pool.join()

if __name__ == '__main__':
    
    batch_size = 10
    swc_path = '../data/CTX_VIS_not_using/'

    swc_filelist = os.listdir(swc_path)
    swc_filelist = [swc_path+x for x in swc_filelist]
    
    filepath = Path('../data/')
    annotation_volume,_ = nrrd.read(filepath / 'ccf_volumes/annotation_10.nrrd')

    brain_divs = [688,623,549,1097,313,1065,512] # Divisions: CTX, CNU, TH, HY,MB, HB, CB

    save_filename = filepath / 'swc_CTX_VIS.csv'
    column_names=['experiment_id','CTX','CNU','TH','HY','MB','HB','CB','predicted_celltype']

    with open(save_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        f.close()

    main(swc_filelist)
