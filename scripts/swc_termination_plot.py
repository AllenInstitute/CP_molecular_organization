# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:37:19 2023

@author: ashwin.bhandiwad
"""

import os
import numpy as np
from anytree import Node,RenderTree
import SimpleITK as sitk

def sitk_load(annotation_image,extension='nrrd'):
    
    reader = sitk.ImageFileReader()
    if extension=='nrrd':
        reader.SetImageIO("NrrdImageIO")
    elif extension=='nii':
        reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(annotation_image)
    
    volume = reader.Execute()
    
    if volume.GetSize()[0] == 1140:
        volume = sitk.PermuteAxes(volume,[2,1,0])
    
    return volume

if __name__ =='__main__':

    swc_path = '../data/ctx_swcs/' #Directory where 
    swc_filelist = os.listdir(swc_path)
    
    brain_vol = np.zeros((1320,800,1140),dtype=int)
    resolution = 10
    
    for swc_file in swc_filelist:
        swc_db = np.genfromtxt(swc_path+swc_file)
        names=swc_db[:,0].astype(int)
        assert swc_db[0,-1]==-1
        
        neuron = Node(f'p{names[0]}',
                      parent=None,
                      x=swc_db[0,2],
                      y=swc_db[0,3],
                      z=swc_db[0,4])
        
        nodes={}
        nodes[neuron.name]=neuron
        for count,point in enumerate(names[1:,]):
            name = f'p{point}'
            nodes[name]=Node(
                name,
                parent=nodes[f'p{swc_db[count+1,-1].astype(int)}'],
                x=swc_db[count+1,2],
                y=swc_db[count+1,3],
                z=swc_db[count+1,4])
    
    brain_vol = sitk.GetImageFromArray(brain_vol)
    brain_vol = sitk.PermuteAxes(brain_vol,[2,1,0])
    # sitk.Show(brain_vol)
    
    mask = sitk_load('../data/ccf_volumes/cp_mask.nrrd')
    mask = sitk.Cast(mask,sitk.sitkInt32)
    brain_vol.SetDirection(mask.GetDirection())
    cp_termini = sitk.Multiply(brain_vol,mask)
