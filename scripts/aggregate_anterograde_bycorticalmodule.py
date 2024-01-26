"""
Aggregate anterograde injection data by cortical module and split into CP subdivisions.
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk

def hex_to_rgb(hex_color):
    """Convert hex values into a tuple of size 3 8-bit RGB values."""   
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def aggregate_volume(module_list,group_name,col,vol):
    """Aggregate volumetric imaging data into rgb_vol (numpy array) using SimpleITK.
    
    Inputs:
        module_list: list of cortical areas (str) to iterate through for merging into modules.
        group_name: str; name of CP subdivision for splitting.
        col: Color values assigned to each module (pandas dataframe)
        vol: Empty numpy array used for populating volume output
        
    Outputs:
        vol: Numpy array containing aggregated volume.
        """
    rgb_vol = np.zeros([1320,800,570,3],dtype=np.int16)
    
    for module in module_list:
        volume = sitk.ReadImage(f'../data/input/cortical/{module}_aggregrated.nrrd')
        vol_npy = sitk.GetArrayFromImage(sitk.PermuteAxes(volume,[2,1,0]))
        vol_npy = vol_npy[:,:,570:] #Extract one hemisphere
        
        assert(np.shape(vol_npy)==np.shape(rgb_vol[:,:,:,0]))
        
        color = col.loc[col['name']==module,'color_qual'].values[0]
        color = hex_to_rgb(color)
        
        signal = np.nonzero(vol_npy)
        normd_values = vol_npy[signal]/vol_npy[signal].max() #normalize by maximum value in volume
        
        rgb_vol[signal[0],signal[1],signal[2],0] = color[0]*normd_values
        rgb_vol[signal[0],signal[1],signal[2],1] = color[1]*normd_values
        rgb_vol[signal[0],signal[1],signal[2],2] = color[2]*normd_values
        del vol_npy, volume
        
    rgb_vol = sitk.PermuteAxes(sitk.GetImageFromArray(rgb_vol),[2,1,0])
    vol = sitk.Add(vol,rgb_vol)
    
    return vol

    
    
def main():

    col = pd.read_csv('../data/harris_order_smgrouped.csv')
    
    # CP regions and cortical regions picked by visual inspection. 
    regions_group = dict({'CPdm': ['ACAv'],
                          'CPvm': ['ORBm','ORBvl','ORBl'],
                          'CPl': ['SSp-m','SSp-n','SSp-bfd','SSp-ll','SSp-ul','MOp-m','MOp-ul','MOp-ll','MOs-bfd','MOs-n'],
                          'CPp': ['AUDp','AUDpo','AUDd'],
                          'CPiv': ['AId','AIp','AIv','GU']})
    
    ctx_regions = ['ACAv','ORBm','ORBvl','ORBl','SSp-m','SSp-n','SSp-bfd','SSp-ll','SSp-ul',
                  'MOp-m','MOp-ul','MOp-ll','MOs-bfd','MOs-n','AUDp','AUDpo','AUDd','AId','AIp','AIv','GU']
    
    vol = np.zeros([1320,800,570,3],dtype=np.int16)
    vol = sitk.PermuteAxes(sitk.GetImageFromArray(vol),[2,1,0])
    for key,value in regions_group.items():
        vol = aggregate_volume(value,key,col,vol)
    
    
        
if __name__ == '__main__':
    main()