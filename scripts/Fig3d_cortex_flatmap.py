# -*- coding: utf-8 -*-
"""
Generates cortical flatmap for anterograde injections, retrograde injections, and swc soma locations.
Used for Fig 3d and Fig 4a.
Uses ccf_streamlines package, documented at https://ccf-streamlines.readthedocs.io/en/latest/
"""

import json,os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import ccf_streamlines.projection as ccfproj

def get_layer_thickness(json_filename):

    with open(json_filename, "r") as f:
        layer_tops = json.load(f)
    
    layer_thicknesses = {
            'Isocortex layer 1': layer_tops['2/3'],
            'Isocortex layer 2/3': layer_tops['4'] - layer_tops['2/3'],
            'Isocortex layer 4': layer_tops['5'] - layer_tops['4'],
            'Isocortex layer 5': layer_tops['6a'] - layer_tops['5'],
            'Isocortex layer 6a': layer_tops['6b'] - layer_tops['6a'],
            'Isocortex layer 6b': layer_tops['wm'] - layer_tops['6b'],
    }
    
    return layer_thicknesses

def initialize_projector():
    
    proj_butterfly_slab = ccfproj.Isocortex3dProjector(
        # Similar inputs as the 2d version...
        "flatmap_butterfly.h5",
        "surface_paths_10_v3.h5",

        hemisphere="both",
        view_space_for_other_hemisphere='flatmap_butterfly',

        # Additional information for thickness calculations
        thickness_type="normalized_layers", # each layer will have the same thickness everwhere
        layer_thicknesses=layer_thicknesses,
        streamline_layer_thickness_file="cortical_layers_10_v2.h5",
    )

    bf_boundary_finder = ccfproj.BoundaryFinder(
        projected_atlas_file="flatmap_butterfly.nrrd",
        labels_file="labelDescription_ITKSNAPColor.txt",
    )

    # We get the left hemisphere region boundaries with the default arguments
    bf_left_boundaries = bf_boundary_finder.region_boundaries()

    # And we can get the right hemisphere boundaries that match up with
    # our projection if we specify the same configuration
    bf_right_boundaries = bf_boundary_finder.region_boundaries(
        # we want the right hemisphere boundaries, but located in the right place
        # to plot both hemispheres at the same time
        hemisphere='right_for_both',

        # we also want the hemispheres to be adjacent
        view_space_for_other_hemisphere='flatmap_butterfly',
    )
    
    return proj_butterfly_slab,bf_left_boundaries,bf_right_boundaries

def load_and_flip(image_filename):
    
    volume = sitk.ReadImage(image_filename)

    vol_shape = volume.GetSize()
    
    left_hemi = volume[:,:,:vol_shape[2]//2]
    right_hemi = volume[:,:,vol_shape[2]//2:vol_shape[2]]
    
    flipped_left = sitk.Flip(left_hemi,[False,False,True])
    flipped_left.CopyInformation(right_hemi)
    
    single_hemi = sitk.Or(right_hemi,flipped_left)
    
    new_vol = sitk.Image(vol_shape,sitk.sitkInt16)
    new_vol[:,:,vol_shape[2]//2:vol_shape[2]] = single_hemi
    
    new_vol = sitk.GetArrayFromImage(sitk.PermuteAxes(new_vol,[2,1,0]))
    
    return new_vol

def load_and_flatmap(image_filename):

    volume = load_and_flip(image_filename)
    flatmap = proj_butterfly_slab.project_volume(volume)

    main_max = flatmap.max(axis=2).T
    top_max = flatmap.max(axis=1).T
    left_max = flatmap.max(axis=0)

    return main_max,top_max,left_max    

os.chdir('../data/')

# Set up input filename. This should be an 8-bit volume of shape (1320,800,1140)
input_filename = 'swc_soma.nrrd'
output_filepath = '../figures/Fig4a_injection_locations.tif'

# Initialize flatmap parameters
layer_thicknesses = get_layer_thickness("avg_layer_depths.json")

proj_butterfly_slab,bf_left_boundaries,bf_right_boundaries = initialize_projector()

r_main,r_top,r_left = load_and_flatmap(input_filename)

main_shape = r_main.shape
top_shape = r_top.shape
left_shape = r_left.shape

#Add layer lines
total_thickness = sum(layer_thicknesses.values())
shape_scale = top_shape[0]/total_thickness
layer_depth = 0
for _, layers in layer_thicknesses.items():
    layer_depth += layers*shape_scale - 0.5
    r_top[int(layer_depth),int(top_shape[1]/2):] = 255
    r_left[:,int(layer_depth)] = 255

# Set up a figure to plot them together
fig, axes = plt.subplots(2, 2,
                          gridspec_kw=dict(
                              width_ratios=(left_shape[1], main_shape[1]),
                              height_ratios=(top_shape[0], main_shape[0]),
                              hspace=0.01,
                              wspace=0.01),
                          figsize=(19.4, 12))

# Plot the surface view
axes[1, 1].imshow(r_main, vmin=0, vmax=1, cmap='Greys_r', interpolation=None)

# plot region boundaries
for k, boundary_coords in bf_left_boundaries.items():
    axes[1, 1].plot(*boundary_coords.T, c="white", lw=0.5)
for k, boundary_coords in bf_right_boundaries.items():
    axes[1, 1].plot(*boundary_coords.T, c="white", lw=0.5)

axes[1, 1].set(xticks=[], yticks=[], anchor="NW")

# Plot the top view
axes[0, 1].imshow(r_top, vmin=0, vmax=1,cmap='Greys_r', interpolation=None)
axes[0, 1].set(xticks=[], yticks=[], anchor="SW")


# Plot the side view
axes[1, 0].imshow(r_left, vmin=0, vmax=1, cmap='Greys_r', interpolation=None)
axes[1, 0].set(xticks=[], yticks=[], anchor="NE")

# Remove axes from unused plot area
axes[0, 0].set(xticks=[], yticks=[])
sns.despine(ax=axes[0, 0], left=True, bottom=True)

plt.savefig(output_filepath)
