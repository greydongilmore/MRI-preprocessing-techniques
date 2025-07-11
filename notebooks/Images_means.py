#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
os.chdir('/mnt/c/Users/fisbain/Documents/GitHub/MRI-preprocessing-techniques/FUS/FUS_SegAll')
os.getcwd()


# In[7]:


import nibabel as nib
import numpy as np
# List of file paths to your segmentation masks
segmentation_files = ['sub-001-FUS_seg_registered.nii', 'sub-002-FUS_seg_registered.nii']

# Initialize an empty array to hold the summed values
summed_segmentation = None

# Loop through each segmentation file
for seg_file in segmentation_files:
    # Load the segmentation mask
    seg_img = nib.load(seg_file)
    seg_data = seg_img.get_fdata()

    # Add the segmentation mask to the summed segmentation
    if summed_segmentation is None:
        summed_segmentation = seg_data
    else:
        summed_segmentation += seg_data

# Calculate the mean segmentation
num_segmentations = len(segmentation_files)
mean_segmentation = summed_segmentation / num_segmentations

# Save the mean segmentation as a NIfTI file
mean_seg_img = nib.Nifti1Image(mean_segmentation, seg_img.affine)
nib.save(mean_seg_img, 'mean_segmentation.nii.gz')


# In[8]:


import nibabel as nib
import numpy as np

# List of file paths to your segmentation masks
segmentation_files = ['sub-001-FUS_seg_registered.nii', 'sub-002-FUS_seg_registered.nii']

# Initialize an empty array to hold the summed values
summed_segmentation = None

# Load and normalize the segmentation masks
for seg_file in segmentation_files:
    # Load the segmentation mask
    seg_img = nib.load(seg_file)
    seg_data = seg_img.get_fdata()

    # Normalize the segmentation mask
    max_intensity = np.max(seg_data)
    normalized_seg_data = seg_data / max_intensity

    # Add the normalized segmentation mask to the summed segmentation
    if summed_segmentation is None:
        summed_segmentation = normalized_seg_data
    else:
        summed_segmentation += normalized_seg_data

# Calculate the mean segmentation
num_segmentations = len(segmentation_files)
mean_segmentation = summed_segmentation / num_segmentations

# Save the mean segmentation as a NIfTI file
mean_seg_img = nib.Nifti1Image(mean_segmentation, seg_img.affine)
nib.save(mean_seg_img, 'normalized_mean_segmentation.nii.gz')


# In[11]:


import nibabel as nib
import numpy as np
import os
import csv

# Directory containing the segmentation masks
seg_dir = '/mnt/c/Users/fisbain/Documents/GitHub/MRI-preprocessing-techniques/FUS/FUS_SegAll'

# CSV file to export volumes
csv_file = 'segmentation_volumes.csv'

# Initialize a list to store volumes
volumes = []

# Iterate through segmentation masks in the directory
for filename in os.listdir(seg_dir):
    if filename.endswith('.nii'):
        # Load the segmentation mask
        seg_path = os.path.join(seg_dir, filename)
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()

        # Calculate the volume of the segmentation mask
        voxel_volume = np.prod(seg_img.header.get_zooms())
        volume = np.sum(seg_data > 0) * voxel_volume  # Count non-zero voxels

        # Extract segmentation label (assuming filename contains label)
        label = os.path.splitext(filename)[0]

        # Append volume and label to the volumes list
        volumes.append((label, volume))

# Write volumes to CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Segmentation Label', 'Volume (mm^3)'])
    writer.writerows(volumes)

print(f'Segmentation volumes exported to {csv_file}')

