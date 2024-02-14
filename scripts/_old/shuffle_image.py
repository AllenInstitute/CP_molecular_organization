# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:23:49 2023

@author: ashwin.bhandiwad
"""

import SimpleITK as sitk
import numpy as np

filename = '../data/cortical_injections.nrrd'

img = sitk.ReadImage(filename)
img = sitk.PermuteAxes(img,[2,1,0])

img_npy = sitk.GetArrayViewFromImage(img)

nonzero_vals = np.nonzero(img_npy)

img_shuffled = np.zeros_like(img_npy)

for idx in range(len(nonzero_vals[0])):
    
    img_shuffled[nonzero_vals[1][idx],nonzero_vals[0][idx],nonzero_vals[2][idx]] = 255

del img,img_npy
img = sitk.GetImageFromArray(img_shuffled)
img2 = sitk.PermuteAxes(img,[2,1,0])

sitk.WriteImage(img2,'../data/cortical_injections_shuffled.nrrd',True)
