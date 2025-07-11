#!/usr/bin/env python
# coding: utf-8

# ### Brain extraction(Skull stripping)

# **Learning outcomes:**
# - How to do quick brain extraction using ants(antspynet module)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
from helpers import *

import ants
import SimpleITK as sitk

print(f'AntsPy version = {ants.__version__}')
print(f'SimpleITK version = {sitk.__version__}')


# In[ ]:


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
print(f'project folder = {BASE_DIR}')


# In[ ]:


raw_examples = [
    'fsl-open-dev_sub-001_T1w.nii.gz',
    'wash-120_sub-001_T1w.nii.gz',
    'kf-panda_sub-01_ses-3T_T1w.nii.gz',
    'listen-task_sub-UTS01_ses-1_T1w.nii.gz',
    'brain-lesion_T1w.nii.gz'
]


# #### Raw Image

# In[ ]:


raw_example = raw_examples[4]
raw_img_path = os.path.join(BASE_DIR, 'assets', 'raw_examples', raw_example)
raw_img_ants = ants.image_read(raw_img_path, reorient='IAL')

print(f'shape = {raw_img_ants.numpy().shape} -> (Z, X, Y)')

explore_3D_array(arr=raw_img_ants.numpy(), cmap='nipy_spectral')


# #### Deep Learning based method

# #### Load Model via AntsPyNet API and predict

# In[ ]:


from antspynet.utilities import brain_extraction


# In[ ]:


prob_brain_mask = brain_extraction(raw_img_ants, verbose=True)


# #### Inspect probabilities array

# In[ ]:


print(prob_brain_mask)
explore_3D_array(prob_brain_mask.numpy())


# #### Generate final mask

# In[ ]:


brain_mask = ants.get_mask(prob_brain_mask, low_thresh=0.5)


# In[ ]:


explore_3D_array_with_mask_contour(raw_img_ants.numpy(), brain_mask.numpy())


# In[ ]:


out_folder =  os.path.join(BASE_DIR, 'assets', 'preprocessed')
out_folder = os.path.join(out_folder, raw_example.split('.')[0]) # create folder with name of the raw file
os.makedirs(out_folder, exist_ok=True) # create folder if not exists

out_filename = add_suffix_to_filename(raw_example, suffix='brainMaskByDL')
out_path = os.path.join(out_folder, out_filename)

print(raw_img_path[len(BASE_DIR):])
print(out_path[len(BASE_DIR):])


# In[ ]:


brain_mask.to_file(out_path)


# #### Generate brain masked

# In[ ]:


masked = ants.mask_image(raw_img_ants, brain_mask)

explore_3D_array(masked.numpy())


# In[ ]:


out_filename = add_suffix_to_filename(raw_example, suffix='brainMaskedByDL')
out_path = os.path.join(out_folder, out_filename)

print(raw_img_path[len(BASE_DIR):])
print(out_path[len(BASE_DIR):])


# In[ ]:


masked.to_file(out_path)

