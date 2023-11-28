# -*- coding: utf-8 -*-
"""
Anterograde projection summary stats.

 - Loads anterograde volume and masks by subdivision
 - Calculates projection strength and projection density
 - Saves as a csv

"""
import os,re,csv
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from multiprocessing import Pool, cpu_count
import numpy as np


def sitk_load(image_filename,extension='nrrd'):
    """Load image from filename.
    
    Input:  image_filename (str) - image filename
            extension (str) - file type. Can be 'nrrd' or 'nii'. Default is nrrd
    
    Output: volume: SimpleITK image object
            - Also outputs an error log
    
    """
    reader = sitk.ImageFileReader()
    if extension=='nrrd':
        reader.SetImageIO("NrrdImageIO")
    elif extension=='nii':
        reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(image_filename)
    
    try:    
        volume = reader.Execute()
    except RuntimeError as error_message:
        if "EOF encountered before end of element" not in str(error_message):
            with open('error_log.txt', 'a') as f:
                f.write(f"{image_filename}: error encountered{error_message}\n")
            f.close()
            volume = []
            pass
        else:
            volume = []
            pass  # Do nothing if there is an EOF error
    
    if len(volume)>0:
        if volume.GetSize()[0] == 1140:
            volume = sitk.PermuteAxes(volume,[2,1,0])
    
    return volume

def sitk_summary_stats(volume):
    """Use SimpleITK's LabelStatisticsImageFilter to extract summary stats for volume.
    
    Input: volume (SimpleITK Image)
    Outputs:
        count (float) - Number of nonzero voxels in volume
        sum (float) - Sum of all nonzero voxels in volume
        
    """
    label_stats = sitk.LabelStatisticsImageFilter()
    label_stats.Execute(volume,sitk.BinaryThreshold(volume,0,0,0,1))

    return label_stats.GetCount(1), label_stats.GetSum(1)

def initialize_csv(subdivisions):
    with open('../data/cp_cortical_anterograde_projections.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment_id','cortex_region','division','projection_strength_ipsi','projection_density_ipsi',
                         'projection_strength_contra','projection_density_contra','injection_volume'])
    

# Define the function that will be run in parallel
def projection_info(args):
    division,image_volume,volume_path,metadata,mask_count,all_experiments=args
    
    experiment_id = int(re.split('\.',image_volume)[0])
    projection_vol = sitk_load(volume_path+image_volume)

    injection_hemisphere = metadata.loc[metadata['image-series-id']==experiment_id]['injection hemisphere']
    if injection_hemisphere.item()=='left':
        vol_direction = projection_vol.GetDirection()
        projection_vol = sitk.Flip(projection_vol,[False,False,True])
        projection_vol.SetDirection(vol_direction)
        
    vol_pts = sitk.GetArrayFromImage(projection_vol)
    
    # Load and calculate contralateral projections
    contra_mask = np.load(f'../data/ccf_volumes/subdivisions/lookup/cp_r{division}_contra.npy')
    contra = vol_pts[contra_mask[0,:],contra_mask[1,:],contra_mask[2,:]]
    
    projection_density_contra = np.shape(np.nonzero(contra))[1]
    projection_strength_contra = np.sum(contra)
    projection_density_contra /= mask_count

    # Load and calculate ipsilateral projections
    ipsi_mask = np.load(f'../data/ccf_volumes/subdivisions/lookup/cp_r{division}_ipsi.npy')
    ipsi = vol_pts[ipsi_mask[0,:],ipsi_mask[1,:],ipsi_mask[2,:]]
    
    projection_density_ipsi = np.shape(np.nonzero(ipsi))[1]
    projection_strength_ipsi = np.sum(ipsi)
    projection_density_ipsi /= mask_count

    injection_volume = all_experiments.loc[experiment_id]['injection_volume']
    cortex_region = all_experiments.loc[experiment_id]['structure_abbrev']
    row_info = [experiment_id,cortex_region,division,projection_strength_ipsi,projection_density_ipsi,projection_strength_contra,projection_density_contra,injection_volume]

    with open('../data/cp_cortical_anterograde_projections.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_info)

# Define the main function
def main(metadata_file,volume_path):
    
    # Define the subdivisions and volume paths
    # Set up metadata and filepath
    
    vol_list = os.listdir(volume_path)
    metadata = pd.read_table(metadata_file,delimiter=',')
    subdivisions = [5,6,7,8,9,10] #Hardcoded but based on Quanxin's annotations.
    
    initialize_csv(subdivisions)

    # Query Allen SDK to get experiment metadata
    ncc = MouseConnectivityCache(manifest_file=Path('../data/') / 'manifest.json')
    all_experiments = ncc.get_experiments(dataframe=True)
    lookup_metadata = all_experiments[['injection_volume','structure_abbrev']].copy()
    
    # Each division is saved as a single hemisphere binary mask file
    mask_vol = np.load('../data/ccf_volumes/lookup/cp_mask.npy')#sitk.ReadImage(f'../data/ccf_volumes/subdivisions/CP_subdivisions_regd_r{division}.nrrd')  
    mask_count = np.shape(mask_vol)[0]/2#,_ = sitk.GetArrayFromImage(mask_vol).sum(), sitk.GetArrayFromImage(mask_vol).shape
    
    args_list = [(division,image_volume,volume_path,metadata,mask_count,lookup_metadata) for division in subdivisions for image_volume in vol_list]

    # Create a Pool of worker processes
    print(cpu_count())
    with Pool(processes=20) as pool:
        pool.map(projection_info,args_list)
        
if __name__ == '__main__':
    volume_path = '../data/input/cortical/masked/'
    metadata_file = '../data/anterograde_cortical.csv'
    main(metadata_file,volume_path)
