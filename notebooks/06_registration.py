#!/usr/bin/env python
# coding: utf-8

# ### Registration

# **Learning outcomes:**
# - How to apply registration to an image using ants.

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
    'listen-task_sub-UTS01_ses-1_T1w.nii.gz'
]


# #### Raw image

# In[ ]:


raw_example = raw_examples[0]
raw_img_path = os.path.join(BASE_DIR, 'assets', 'raw_examples', raw_example)
raw_img_ants = ants.image_read(raw_img_path, reorient='IAL')

explore_3D_array(arr=raw_img_ants.numpy())


# #### Template image

# In[ ]:


template_img_path = os.path.join(BASE_DIR, 'assets', 'templates', 'mni_icbm152_t1_tal_nlin_sym_09a.nii')
template_img_ants = ants.image_read(template_img_path, reorient='IAL')

explore_3D_array(arr = template_img_ants.numpy())


# In[ ]:


print('\t\tRAW IMG')
print(raw_img_ants)

print('\t\tTEMPLATE IMG')
print(template_img_ants)


# #### Registration

# In[ ]:


transformation = ants.registration(
    fixed=template_img_ants,
    moving=raw_img_ants, 
    type_of_transform='SyN',
    verbose=True
)


# In[ ]:


print(transformation)


# In[ ]:


registered_img_ants = transformation['warpedmovout']

explore_3D_array(arr=registered_img_ants.numpy())


# In[ ]:


out_folder =  os.path.join(BASE_DIR, 'assets', 'preprocessed')
out_folder = os.path.join(out_folder, raw_example.split('.')[0]) # create folder with name of the raw file
os.makedirs(out_folder, exist_ok=True) # create folder if not exists

out_filename = add_suffix_to_filename(raw_example, suffix='registered')
out_path = os.path.join(out_folder, out_filename)

print(raw_img_path[len(BASE_DIR):])
print(out_path[len(BASE_DIR):])


# In[ ]:


registered_img_ants.to_file(out_path)

