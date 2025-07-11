#!/usr/bin/env python
# coding: utf-8

# ### Image Orientation

# **Learning outcomes:**
# - Load .nii.gz/.nii images using AntsPy and SITK using different orientations

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


# ### AntsPy

# In[ ]:


raw_img_path = os.path.join(BASE_DIR, 'assets', 'raw_examples', raw_examples[0])


# In[ ]:


raw_img_ants = ants.image_read(raw_img_path) 
print(raw_img_ants)


# In[ ]:


# LPI = Left-to-right, Posterior-to-anterior, Inferior-to-superior
print(raw_img_ants.get_orientation())


# In[ ]:


arr = raw_img_ants.numpy()
print(raw_img_ants.get_orientation())
print(arr.shape, '-> (Z,Y,X)')


# In[ ]:


# LPI = Left-to-right, Posterior-to-anterior, Inferior-to-superior
arr = raw_img_ants.numpy()
print(raw_img_ants.get_orientation())
print(arr.shape, '-> (Z,Y,X)')
explore_3D_array(arr=raw_img_ants.numpy()) 


# In[ ]:


# Pixel arrangement
# Z, Y, X = (↑,↓,→)


# In[ ]:


# LPI = Left-to-right, Posterior-to-anterior, Inferior-to-superior
# IAL = Inferior-to-superior, Anterior-to-posterior, Left-to-right
raw_img_ants = ants.image_read(raw_img_path, reorient='IAL') 

print(raw_img_ants.get_orientation())
print(arr.shape, '-> (Z,Y,X)')
explore_3D_array(arr=raw_img_ants.numpy()) 


# ### Simple ITK

# In[ ]:


raw_img_path = os.path.join(BASE_DIR, 'assets', 'raw_examples', raw_examples[0])
raw_img_sitk = sitk.ReadImage(raw_img_path, sitk.sitkFloat32)


# In[ ]:


raw_img_sitk_arr = sitk.GetArrayFromImage(raw_img_sitk)
print(raw_img_sitk_arr.shape)
explore_3D_array(raw_img_sitk_arr)


# For AntsPy:
# - Internal axis are (Z,Y,X). It means, when we get numpy array dimensions are (Z,Y,X)
# - When we define orientation, orientation string is according to internal axis.
# 
# For SimpleITK:
# - Internal axis are (X,Y,Z). It means, when we get numpy array dimensions are (Z,Y,X) i.e. shifted.
# - When we define orientation, orientation string is according to internal axis. 
# - The orientation string is set with the latest letter, e.g. : 
#     - "RPS" = (left-to-Right, anterior-to-Posterior, inferior-to-Superior)
#     - "PSR" = (anterior-to-Posterior, inferior-to-Superior, left-to-Right)

# In[ ]:


raw_img_sitk = sitk.ReadImage(raw_img_path, sitk.sitkFloat32)
raw_img_sitk = sitk.DICOMOrient(raw_img_sitk,'RPS')

raw_img_sitk_arr = sitk.GetArrayFromImage(raw_img_sitk)
print(raw_img_sitk_arr.shape)
explore_3D_array(raw_img_sitk_arr)


# In[ ]:


# Internal Pixel arrangement for SimpleItk
# (X, Y, Z) = (→, ↓, ↑)

