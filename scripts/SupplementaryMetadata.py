# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:59:49 2023

@author: ashwin.bhandiwad
"""

import numpy as np
import pandas as pd

path = '../data/'
full = pd.read_csv(path+'all_experiments.csv')

# Anterograde metadata

# Add filtered cortical data
ant = pd.read_csv(path+'Fig3b_cp_cortical_anterograde_projections.csv')
ids = np.unique(ant['experiment_id'])

# Add subcortical data
subcort = pd.read_csv(path+'anterograde_subcortical.csv')
ids = np.hstack((ids,np.unique(subcort['image-series-id'])))

metadata = np.array(['Experiment ID','Structure abbreviation','Structure name',
                     'Mouse line','Sex','Injection hemisphere', 'Experiment URL'])
for item in ids:
    
    experiment_url = 'http://connectivity.brain-map.org/projection/experiment/'+str(item)
    
    info = full.loc[full['id']==item]
    injection_z = info['injection_z'].values[0]
    
    if injection_z < 5700:
        injection_hemisphere = 'left'
    else:
        injection_hemisphere = 'right'
        
    transgenic_line = info['transgenic_line']
    
    if pd.isna(transgenic_line).values[0]:
        transgenic_line = 'C57BL/6J'
    else:
        transgenic_line = transgenic_line.values[0]
    
    row = [item, 
           info['structure_abbrev'].values[0],
           info['structure_name'].values[0],
           transgenic_line,
           info['gender'].values[0],
           injection_hemisphere,
           experiment_url]
   
    metadata = np.vstack((metadata,row))

output = pd.DataFrame(data=metadata[1:,],columns=metadata[0,:])

output.to_csv('../data/SupplementalTable1_anterograde')

# Retrograde metadata
# Used Shenqin's metadata file


# Single neuron metadata

# df = pd.read_csv(path+'SupplementalTable3_single_neurons.csv')
