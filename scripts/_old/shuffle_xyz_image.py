# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:07:52 2023

@author: ashwin.bhandiwad
"""

import numpy as np
import pandas as pd
import os
import SimpleITK as sitk

base_path = '../data/input/subcortical/CP_output/'
df = pd.read_table('../data/CP_anterograde_output.csv',delimiter=',')

cp_regions = np.unique(df['manually annotated injection site'])

for region in ['GPe','GPi','SNr']:
    
    img_path = base_path+region+'/'
    file_list = os.listdir(img_path)
    experiment_list = [int(x[:-5]) for x in file_list]
    
    for cp_division in cp_regions:
        
        division_idx = np.where(df['manually annotated injection site']==cp_division)[0]
        
        vol = sitk.ReadImage(img_path+file_list[division_idx[0]])
        
        for idx in range(len(division_idx)-1):
            
            vol_add = sitk.ReadImage(img_path+file_list[division_idx[idx+1]])
            vol = sitk.Add(vol,vol_add)
            del vol_add
        
        vol = vol/len(division_idx)
    
        sitk.WriteImage(vol,f'{img_path}{cp_division}_agg.nrrd',True)
