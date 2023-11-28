# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:14:23 2023

@author: ashwin.bhandiwad
"""

import os
import SimpleITK as sitk
import numpy as np

path = '../data/ccf_volumes/'

cp_vol = sitk.ReadImage(path + 'CP_bounds_CCFregistered.nrrd')
cp_mask = sitk.ReadImage(path + 'cp_mask.nrrd')
cp_mask = sitk.Cast(cp_mask,cp_vol.GetPixelID())
cp_mask.CopyInformation(cp_vol)

cp_vol = sitk.Multiply(cp_vol,cp_mask)
cp_vol = sitk.PermuteAxes(cp_vol,[2,1,0])

cp_npy = sitk.GetArrayViewFromImage(cp_vol)

subdivisions = [6,7,8,9,10]
div_names = ['CPp','CPdm','CPl','CPvm','CPiv']

for idx,divs in enumerate(subdivisions):
    
    pts = np.where(cp_npy==divs)
    pts_arr = np.vstack((pts[0],pts[1],pts[2]))
    
    np.save(f'{path}lookup\\{div_names[idx]}_contra.npy',pts_arr)
    
    ipsi = np.array([1140 - x for x in pts[2]])
    pts_arr = np.vstack((pts[0],pts[1],ipsi))
    
    np.save(f'{path}lookup\\{div_names[idx]}_ipsi.npy',pts_arr)

# subdivision_vol_list=os.listdir(path)
# vols = subdivision_vol_list[0]
# for vols in subdivision_vol_list:
    
#     volume = sitk.ReadImage(path+vols)
#     pts_npy = sitk.GetArrayFromImage(volume)
#     pts_idx = np.nonzero(pts_npy)
    
#     np.save(f'{path}lookup\\{vols[:-5]}_contra.npy',pts_idx)
    
#     volume = sitk.Flip(volume,[False,False,True])
#     pts_npy = sitk.GetArrayFromImage(volume)
#     pts_idx = np.nonzero(pts_npy)
    
#     np.save(f'{path}lookup\\{vols[:-5]}_ipsi.npy',pts_idx)