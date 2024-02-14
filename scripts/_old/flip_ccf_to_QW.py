import os
import SimpleITK as sitk
import numpy as np
import sys

path_dir = '../data/Fig2/'#merfish/'

# file_list = os.listdir(path_dir)
file_list = ['otof_ish.nrrd']
    
reference_file = '../data/ccf_volumes/avg_template_QW.nrrd' 
mask_file = '../data/ccf_volumes/cp_mask.nrrd' 

# reference is (A-P,S-I,L-R)
reference = sitk.ReadImage( reference_file )
rarr = sitk.GetArrayViewFromImage( reference )

mask = sitk.ReadImage( mask_file )
mask = sitk.Cast(mask,sitk.sitkFloat32)

for file_name in file_list:
    input_file = f'{path_dir}{file_name}'
    output_file = f'{path_dir}{file_name[:-5]}regd.nrrd'

    # input is (L-R,P-A,I-S)
    input_img = sitk.ReadImage( input_file )
    input_img.CopyInformation( mask )
    input_img = sitk.Multiply(input_img,mask)
    output = sitk.Flip( input_img, [True, True, True] )
    del input_img
    output = sitk.PermuteAxes( output, [2,0,1] )

    oarr = sitk.GetArrayViewFromImage( output )
    oarr = oarr[:799,:1319,:619]
    nonzero = np.nonzero(oarr)
    
    # shift by one in the L-R axis
    zarr = np.empty( rarr.shape, dtype=oarr.dtype )
    zarr[nonzero[0],nonzero[1],nonzero[2]] = oarr[nonzero]
    
    # col_zarr = np.zeros((3,799,1319,619),dtype=np.float32)
    # col_zarr[1,:,:,:] = zarr
    
    # write out to file
    output = sitk.GetImageFromArray( zarr )
    output.CopyInformation( reference )
    output = sitk.Cast(output,sitk.sitkInt16)
    sitk.WriteImage( output, output_file, True )
