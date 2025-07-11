#!/usr/bin/env python
# coding: utf-8

# In[9]:


# import stuff and check directory
import nibabel as nib
import numpy as np
import csv
import shutil
import ants
from helpers import *
import os
os.chdir('/mnt/c/Users/fisbain/Documents/GitHub/MRI-preprocessing-techniques/FUS')
BASE_DIR = os.getcwd()


# In[10]:


source_dir = "/mnt/c/Users/fisbain/Documents/GitHub/MRI-preprocessing-techniques/FUS"  # Path to the source directory
target_dir = "/mnt/c/Users/fisbain/Documents/GitHub/MRI-preprocessing-techniques/FUS_res/SegMNI"  # Path to the target directory
file_extension = "seg_corr_norm.nii"  # File extension to search for


# In[11]:


for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        # Check if the item is a directory
        if os.path.isdir(folder_path):
            # Iterate through files in the folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                # Check if the file has the specified extension
                if file_name.endswith(file_extension):
                    # Create target directory if it doesn't exist
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)

                    # Copy the file to the target directory
                    shutil.copy(file_path, target_dir)
                    print(f"Copied {file_name} to {target_dir}")


# In[12]:


# CSV file to export volumes
csv_file = 'FUS_MNI_segmentation_volumes.csv'

# Initialize a list to store volumes
volumes = []

# Iterate through segmentation masks in the directory
for filename in os.listdir(target_dir):
    if filename.endswith('.nii'):
        # Load the segmentation mask
        seg_path = os.path.join(target_dir, filename)
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


# In[ ]:




