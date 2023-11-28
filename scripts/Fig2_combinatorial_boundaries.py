# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:37:48 2023

@author: ashwin.bhandiwad
"""
import os
import numpy as np
import SimpleITK as sitk
# from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph



def gaussian_blur(input_image,sigma):
    
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(sigma)
    return gaussian.Execute(input_image)

def downsample_img(img, resize_factor):

    dimension = img.GetDimension()
    reference_physical_size = np.zeros(dimension)
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]
    
    reference_origin = img.GetOrigin()
    reference_direction = img.GetDirection()

    reference_size = [round(sz/resize_factor) for sz in img.GetSize()] 
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(reference_direction)

    transform.SetTranslation(np.array(reference_origin) - reference_origin)
  
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.CompositeTransform(transform)
    centered_transform.AddTransform(centering_transform)

    # sitk.Show(sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0))
    
    return sitk.Resample(img, reference_image, centered_transform, sitk.sitkLinear, 0.0)
    

path = '../data/Fig2/regd/cpp/'
mask_path = '../data/ccf_volumes/'
rescale_factor = 2

mask = sitk.ReadImage(mask_path+'cp_mask.nrrd')
mask = sitk.Cast(mask,sitk.sitkFloat32)
mask_downsample = downsample_img(mask,rescale_factor)
dims = mask_downsample.GetSize()
mask_downsample = mask_downsample[:,:,:int(dims[2]/2)]
nonzero = np.nonzero(sitk.GetArrayViewFromImage(mask_downsample))

file_list = os.listdir(path)
df = np.empty((len(nonzero[0])))
for file_name in file_list:

    img = sitk.ReadImage(path+file_name)
    img_downsample = downsample_img(img,rescale_factor)
    img_blur = gaussian_blur(img_downsample,2.0)
    
    img_blur = img_blur[:,:,:int(dims[2]/2)]
    
    mask_downsample.CopyInformation(img_blur)
    
    img_cp = sitk.Multiply(img_blur,mask_downsample)
    sitk.WriteImage(img_cp,path+file_name[:-3]+'masked.nrrd',True)
    img_cp_npy = sitk.GetArrayViewFromImage(img_cp)
    img_cp_npy = img_cp_npy[nonzero]
    df = np.vstack((df,img_cp_npy))
    print(f'{file_name} complete')

# df = df.transpose()
# df = df[:,1:]

# np.random.seed(0)
# connectivity = grid_to_graph(*dims,mask=np.array(nonzero)[0,:])
# ward = AgglomerativeClustering(n_clusters=6, linkage="ward").fit(df)
# label = ward.labels_

# # df_2d = manifold.SpectralEmbedding(n_components=2).fit_transform(df)
# # sitk.Show(img_cp)

# vol = np.zeros(dims)

# for k in range(len(nonzero[0])):
#     vol[nonzero[2][k],nonzero[1][k],nonzero[0][k]] = label[k]+1
    
# vol = sitk.PermuteAxes(sitk.GetImageFromArray(vol),[2,1,0])
# sitk.Show(vol)

# sitk.WriteImage(vol,path+'Fig2_clusters.nrrd',True)

# with open(path+'dataset.pkl', 'rb') as f:
#     x = pickle.load(f)