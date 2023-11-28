# -*- coding: utf-8 -*-
"""
Anterograde projection summary stats.

 - Loads anterograde volume and masks by subdivision
 - Calculates projection strength and projection density
 - Saves as a csv

"""
import os,re,csv
import urllib.request as request
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

def download_volumes(vol_list,volume_path):
    
    dir_path = Path(volume_path)
    
    for vol in vol_list:
        
        full_path = dir_path / str(vol)+'.nrrd'
        if full_path.exists():
            continue
        else:
            download_link = f'http://api.brain-map.org/grid_data/download_file/{vol}?image=projection_density&resolution=10'
            request.urlretrieve(download_link,filename=f'{volume_path}{vol}.nrrd')
            
def mask_volume_by_region(vol_directory,region_name,ontology_metadata):
    
    vols = os.listdir(vol_directory)
    annotation = sitk_load('../data/ccf_volumes/annotation_10.nrrd')
    mask_region_id = ontology_metadata.loc[ontology_metadata['Acronym']==region_name]['ID'].to_list()[0]
    mask = annotation==mask_region_id
    
    for vol in vols:
        img_vol = sitk_load(vol_directory+vol)
        mask = sitk.Cast(mask,img_vol.GetPixelID())
        mask.CopyInformation(img_vol)
        
        img_masked = sitk.Multiply(img_vol,mask)
        sitk.WriteImage(img_masked,f'{vol_directory}{region_name}/{vol}')

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
    
    half_vol_size = [volume.GetSize()[0],
                                     volume.GetSize()[1],
                                     volume.GetSize()[2]//2]
    half_vol_start = [0,0,volume.GetSize()[2]//2]
    volume = sitk.RegionOfInterest(volume,size=half_vol_size,index=half_vol_start)
    
    return volume
    
def threshold_image(volume_filename,volume_path,injection_hemisphere='Right'):
    
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
    
    vol1_filename,vol2_filename,volume_path=args
    
    vol1 = threshold_image(vol1_filename,volume_path)
    vol2 = threshold_image(vol2_filename,volume_path)
    
    n_overlap = get_overlap(vol1,vol2)[1]
    dice = get_dice_coef(vol1,vol2)
    
    row_info = [vol1_filename,vol2_filename,n_overlap,dice]
    
    with open(f'../data/anterograde_cp_overlap.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_info)


# Define the main function
def main():

    df = pd.read_table('../data/CP_anterograde_output.csv',delimiter=',')
    metadata = pd.read_csv('../data/all_experiments.csv')
    ontology_metadata = pd.read_csv('../data/MouseAtlas_ontologies_notree.csv')
    regions = ['GPe','GPi','SNr']
    
    vol_list = df['image series ID'].to_numpy()
    injection_div = df['manually annotated injection site'].to_numpy()
    
    injection_z = metadata.loc[metadata['id'].isin(vol_list)]['injection_z'].to_numpy()
    hemisphere = []
    for coord in injection_z:
        if coord<5700:
            hemisphere.append('Left')
        else:
            hemisphere.append('Right')
    df['injection_hemisphere'] = hemisphere

    volume_path = '../data/input/subcortical/CP_output/'
    download_volumes()
    
    for ccf_region in regions:
        mask_volume_by_region(volume_path,ccf_region,ontology_metadata)
    
    for ccf_region in regions:
        path = f'{volume_path}{ccf_region}/'
        vol_list = os.listdir(path)
    
        args_list = [(vol1,vol2,path) for i,vol1 in enumerate(vol_list) for vol2 in vol_list]
        
        with open(f'../data/anterograde_cp_overlap_{ccf_region}.csv', mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Volume1','Volume2','Overlapping_voxels','Dice_coefficient'])
        
        # Create a Pool of worker processes
        print(cpu_count())
        with Pool(processes=20) as pool:
            pool.map(projection_info,args_list)
        
if __name__ == '__main__':
    
    main()
