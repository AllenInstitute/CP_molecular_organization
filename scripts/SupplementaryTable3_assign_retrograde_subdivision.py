# -*- coding: utf-8 -*-
"""
Generation of Supplementary Table 3 - assigning retrograde subdivision based on x,y,z coordinates

"""
import os, nrrd
import SimpleITK as sitk

import pandas as pd
import numpy as np

path = '../data/Supplementary_table_3_retrograde_injection_sites.csv'
annotation_path = '../data/ccf_volumes/CP_bounds_CCFregistered.nrrd'

metadata = pd.read_csv(path)

annotation_sitk = sitk.ReadImage(annotation_path)
annotation_sitk_asl = sitk.PermuteAxes(annotation_sitk,[2,1,0])

annotation = sitk.GetArrayViewFromImage(annotation_sitk_asl)
annotation_shape = np.shape(annotation)

x = [int(0.1*point) for point in metadata['x']]
y = [int(0.1*point) for point in metadata['y']]
z = [int(annotation_shape[-1]-0.1*point) for point in metadata['z']]

CP_div = []
for idx in range(len(x)):
    CP_div.append(annotation[y[idx],x[idx],z[idx]])
    
div_ids = {6: 'CPp', 7: 'CPdm', 8: 'CPl', 9: 'CPvm', 10: 'CPiv', 0: 'unknown'}

CP_div_labels = [div_ids[idx] for idx in CP_div]

metadata['CP_division'] = CP_div_labels

metadata.to_csv(path)