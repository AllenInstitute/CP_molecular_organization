import nrrd
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom, gaussian_filter

def read_mask_resize(input_dir,vol,mask,scale=0.5):
    
    data_vol = sitk.ReadImage(input_dir+vol)
    mask = sitk.Cast(mask,data_vol.GetPixelID())
    data_vol.CopyInformation(mask)
    data_vol = sitk.Multiply(data_vol,mask)
    
    vol_npy = sitk.GetArrayViewFromImage(data_vol)
    vol_npy = gaussian_filter(vol_npy,2.0)
    resized_vol = zoom(vol_npy, [scale,scale,scale], order=0)
    
    return resized_vol

input_dir = '../data/Fig2/regd/'
mask_filename = '../data/ccf_volumes/cp_mask.nrrd'

mask = sitk.ReadImage(mask_filename)
orig_size=mask.GetSize()
size_20um = [int(x/2) for x in orig_size]

inputs = {'CPdm': ['Fos.nii.gz','526783054.nrrd','calb2.nii.gz'],
          'CPvm': ['col6a.nii.gz','ORBvl_aggregated.nrrd','sim1.nii.gz'],
          'CPl': ['astn2.nrrd','SSp-m_MOp-m_aggregated.nrrd','dlg3.nii.gz'],
          'CPiv': ['wfs1.nii.gz','299783689.nrrd','nr5a1.nii.gz'],
          'CPp': ['otof_ish.nrrd','AUDpo_aggregated.nrrd','tacr1.nii.gz']}

for subdiv in inputs:

    col_img = np.zeros((3,int(size_20um[2]/2),size_20um[1],size_20um[0]))
    
    volumes = inputs[subdiv]
    for idx,vol in enumerate(volumes):
        
        resized_image = read_mask_resize(input_dir,vol,mask)
        if idx==1:
            resized_image = np.flip(resized_image,axis=0)
        img_npy = resized_image[:285,:,:]
    
        col_img[idx,:,:,:] = img_npy
        del resized_image,img_npy
    
    nrrd.write(input_dir+subdiv+'_Fig2_coronal.nrrd',col_img)
    nrrd.write(input_dir+subdiv+'_Fig2_dorsal.nrrd',col_img.transpose(0,3,1,2))
    del col_img   
