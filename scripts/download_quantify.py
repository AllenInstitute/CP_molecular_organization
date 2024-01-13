# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:27:14 2023

@author: ashwin.bhandiwad
"""
import os, re
import pandas as pd
import numpy as np
import SimpleITK as sitk
import urllib.request as request
        
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

def sitk_to_npy(image):
    
    permuted_image = sitk.PermuteAxes(image,[2,1,0])

    return sitk.GetArrayFromImage(permuted_image)

def npy_to_sitk(image_npy):
    
    image = sitk.GetImageFromArray(image_npy)
    
    return sitk.PermuteAxes(image,[2,1,0])

def multiply_images(image1,image2):

    image2 = sitk.Cast(image2,image1.GetPixelID())
    image2.SetSpacing(image1.GetSpacing())
    
    return sitk.Multiply(image1,image2)
    

def batch_compress(
        pathname='../data/input',
        mask_filename='../data/cp_mask.nrrd'
        ):
    
    
    filelist = os.listdir(pathname)
    mask = sitk_load(mask_filename)
    
    for files in filelist:
        if ~os.path.isdir(files):
            img = sitk_load(pathname+files)
            sitk.WriteImage(multiply_images(img,mask),f'{pathname}masked/{files}', True)
        
def subdivision_generator(filename='../data/ccf_volumes/CP_subdivisions_regd.nrrd',
                          mask_filename='../data/ccf_volumes/cp_mask.nrrd'):
    
    img = sitk_load(filename)
    mask = sitk_load(mask_filename)
    
    masked_img = multiply_images(img,mask)
    img_npy = sitk_to_npy(masked_img)
    del mask, img
    
    img_unique_vals = np.unique(img_npy)
    
    for subregion in img_unique_vals:
        if subregion>0:
            subset = np.zeros(np.shape(img_npy))
            subset[np.where(img_npy==subregion)] = 1
            
            sitk.WriteImage(npy_to_sitk(subset),f'{filename[:-5]}_r{subregion}.nrrd',True)
            print(f'Region {subregion} complete')

        
def download_and_mask(metadata_filename,path):
    
    metadata = pd.read_table(metadata_filename,delimiter=',')
    
    experiment_id = list(metadata['image-series-id'])
    injection_location = list(metadata['primary injection'])
    
    for this_file in experiment_id:
        download_link = f'http://api.brain-map.org/grid_data/download_file/{this_file}?image=projection_density&resolution=10'
        try:
            request.urlretrieve(download_link,filename=f'{path}full/{this_file}.nrrd')
        except:
            continue
    # ccf_annotation = sitk_to_npy(sitk_load('../data/ccf_volumes/annotation_10.nrrd'))
    # cp_id = 672
    
    # cp_mask = np.zeros(np.shape(ccf_annotation))
    # cp_mask[np.where(ccf_annotation==cp_id)]=1
    # cp_mask = npy_to_sitk(cp_mask)
    # del ccf_annotation
    # sitk.WriteImage(cp_mask,'../data/ccf_volumes/cp_mask.nrrd',True)
    
    # for j in os.listdir(f'{path}full/'):
    #     if re.split('\.',j)[1]=='nrrd':
    #         print(j)
    #         volume = sitk_load(f'{path}full/{j}')
    #         sitk.WriteImage(volume,f'{path}full/{j}',True)
       

if __name__ == '__main__':
    
    path = '../data/input/cortical/'
    # metadata_filename = '../data/anterograde_subcortical.csv'
    # download_and_mask(metadata_filename,path)
    batch_compress(path,'../data/ccf_volumes/cp_mask.nrrd')
    # subdivision_generator()
    
