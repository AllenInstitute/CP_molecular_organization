# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:45:52 2023

@author: ashwin.bhandiwad
"""

import pandas as pd
import numpy as np

path = '../data/'

df = pd.read_csv(path+'Harris_ctx_anterograde_volumes.csv')
quant = pd.read_csv(path+'cp_anterograde_projections.csv')

not_quantified = df[~df['image id'].isin(quant['experiment_id'])]
df = df.drop(not_quantified.index,axis=0)

subdivisions = [5,6,7,8,9,10]
for division in subdivisions:
    quant_division=quant[quant['division']==division]
    
    injection_vol = quant_division['injection_volume'].to_numpy()
    
    df['injection_volume']=injection_vol
    # df[f'r{division}_intensity_ipsi']=quant_division['projection_strength_ipsi'].to_numpy()
    # df[f'r{division}_intensity_contra']=quant_division['projection_strength_contra'].to_numpy()
    df[f'r{division}_density_ipsi']=quant_division['projection_density_ipsi'].to_numpy()/injection_vol
    df[f'r{division}_density_contra']=quant_division['projection_density_contra'].to_numpy()/injection_vol

df.to_csv(path+'CP_anterograde_projection_quantification_injectionvolnormalized.csv',index=False)

