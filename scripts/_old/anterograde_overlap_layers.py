# -*- coding: utf-8 -*-
"""
Anterograde projection summary stats.

 - Loads anterograde volume and masks by subdivision
 - Calculates projection strength and projection density for a specific layer using an HPC system with multiple cores
 - Saves as a csv

"""
import os,re,csv
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from multiprocessing import Pool, cpu_count
import numpy as np


def sitk_load(image_filename,extension='nrrd',fliplr=False):
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

def load_flip_half_volume(volume_filename,injection_hemisphere):
    
    volume = sitk_load(volume_filename)
    if injection_hemisphere in ('Left','left'):
        vol_direction = volume.GetDirection()
        volume = sitk.Flip(volume,[False,False,True])
        volume.SetDirection(vol_direction)
    
    volume = sitk.Extract(volume,[volume.GetSize()[0],
                                     volume.GetSize()[1],
                                     volume.GetSize()[2]//2])
    
    return volume
    
def threshold_image(volume_filename,volume_path,injection_hemisphere):
    
    volume = load_flip_half_volume(volume_path+volume_filename+'.nrrd',injection_hemisphere)
    
    return sitk.BinaryThreshold(volume,lowerThreshold=10e-4)
    

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

def get_overlap(vol1,vol2):
    
    overlap = sitk.And(vol1,vol2)
    overlap_pts = np.nonzero(sitk.GetArrayFromImage(overlap))
        
    return np.shape(overlap_pts)

def get_dice_coef(vol1,vol2):
    
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(vol1, vol2)
    
    return overlap_measures_filter.GetDiceCoefficient()

# Define the function that will be run in parallel
def projection_info(args):
    
    vol1_filename,vol2_filename,injection_hemisphere,volume_path=args
    
    vol1 = threshold_image(vol1_filename,volume_path,injection_hemisphere)
    vol2 = threshold_image(vol2_filename,volume_path,injection_hemisphere)
    
    n_overlap = get_overlap(vol1,vol2)[1]
    dice = get_dice_coef(vol1,vol2)
    
    row_info = [vol1_filename,vol2_filename,n_overlap,dice]
    
    with open(f'../data/anterograde_cp_overlap_L5.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_info)


# Define the main function
def main(volume_path,metadata_file):

    df = pd.read_table(metadata_file,delimiter=',')
    
    l5_mask = df['injected layer'].isin(['L5 IT'])
    metadata = df[l5_mask]

    vol_list = metadata['image-series-id'].to_list()
    injection_hemisphere = metadata['injection hemisphere'].to_list()
    
    args_list = [(vol1,vol2,injection_hemisphere[i],injection_hemisphere[j],volume_path) for i,vol1 in enumerate(vol_list) for j,vol2 in enumerate(vol_list)]

    
    with open(f'../data/anterograde_cp_overlap_L5.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Volume1','Volume2','Overlapping_voxels','Dice_coefficient'])
    
    # Create a Pool of worker processes
    print(cpu_count())
    with Pool(processes=20) as pool:
        pool.map(projection_info,args_list)
    
if __name__ == '__main__':
    volume_path = '../data/input/cortical/masked/'
    metadata_file = '../data/anterograde_cortical.csv'
    main(volume_path,metadata_file)
