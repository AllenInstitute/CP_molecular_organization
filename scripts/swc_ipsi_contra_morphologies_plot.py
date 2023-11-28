# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:07:03 2023

@author: ashwin.bhandiwad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from swc_tools import *

def swc_coords(df,resolution=10):

    x = (df[:,2]/resolution).astype(int)
    y = (df[:,3]/resolution).astype(int)
    z = (df[:,4]/resolution).astype(int)
    
    return x,y,z

swc_df = pd.read_csv('../data/swc_cp_projection_densities.csv')
swc_df = swc_df[swc_df['ccf_region']=='MOs']
path = '../data/ctx_swcs/'


cells = {'L23' : ['195049_023','195788_004','194778_046','195617_009'], #ACAd
         'L5IT' : ['200323_002','195616_015','195616_001','195828_056'],
         'L5ET' : ['195047_033','192982_046','200194_097','200198_023'], # ipsi-contra ratio ~0.33
         'colors': [tuple([66,125,255]),tuple([0,200,28]),tuple([255,97,110]),tuple([255,166,0])]
         }

region = 'L5ET'
ipsi = cells[region]
img_xy = np.zeros((800,1140,3),dtype=int)
img_yz = np.zeros((1320,1140,3),dtype=int)

for n,cell_id in enumerate(ipsi):
    
    filename = f'{path}{cell_id}_reg.swc'
    df = pd.read_table(filename,sep=' ',header=None).to_numpy()
    df = flip_swc(df)
    x,y,z = swc_coords(df)
    
    radius = 5
    
    for point in range(len(x)):
        if point==0:
            img_xy[y[point]-radius:y[point]+radius,z[point]-radius:z[point]+radius,:] = cells['colors'][n]
            img_yz[x[point]-radius:x[point]+radius,z[point]-radius:z[point]+radius,:] = cells['colors'][n]
        else:
            img_xy[y[point],z[point],:] = cells['colors'][n]
            img_yz[x[point],z[point],:] = cells['colors'][n]


fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(60,40))

ax1.imshow(img_xy)
ax1.axis('off')
ax2.imshow(img_yz)
ax2.axis('off')
plt.savefig(f'../figures/{region}.svg')