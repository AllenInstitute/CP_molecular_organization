# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:08:17 2023

@author: ashwin.bhandiwad
"""

import pandas as pd
import numpy as np

metadata = pd.read_table('../data/anterograde_subcortical.csv',delimiter=',')
expt_lookup = pd.read_table('../data/all_experiments.csv',delimiter=',',index_col='id')

hemisphere = []
for j in range(0,len(metadata)):
    expt_info=expt_lookup.loc[metadata['image series ID'][j]]
    if expt_info['injection_z']>=5700:
        hemisphere.append('right')
    else:
        hemisphere.append('left')
        
metadata['injection hemisphere']=np.array(hemisphere)

metadata.to_csv('../data/anterograde_subcortical.csv')
