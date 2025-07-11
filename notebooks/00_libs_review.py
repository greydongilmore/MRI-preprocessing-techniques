#!/usr/bin/env python
# coding: utf-8

# ### **Review of the libraries**

# **Learning outcomes:**
# - Load .nii.gz/.nii images using AntsPy and SITK
# - Get basic information
# - Get numpy representation
# - Plot MRI images

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


raw_img_path = os.path.join(BASE_DIR, 'assets', 'raw_examples', 'fsl-open-dev_sub-001_T1w.nii.gz')
print(f'raw_img_path = {raw_img_path}')


# ### AntsPY

# In[ ]:


raw_img_ants = ants.image_read(raw_img_path)


# In[ ]:


print(raw_img_ants)


# In[ ]:


raw_img_ants_arr = raw_img_ants.numpy()

print(f'type = {type(raw_img_ants_arr)}')
print(f'shape = {raw_img_ants_arr.shape}')


# In[ ]:


ants.plot(raw_img_ants, figsize=3, axis=2)


# In[ ]:


explore_3D_array(raw_img_ants_arr)


# ### Simple ITK

# In[ ]:


raw_img_sitk = sitk.ReadImage(raw_img_path, sitk.sitkFloat32)


# In[ ]:


#print(raw_img_sitk)


# In[ ]:


show_sitk_img_info(raw_img_sitk)


# In[ ]:


raw_img_sitk_arr = sitk.GetArrayFromImage(raw_img_sitk)

print(f'type = {type(raw_img_sitk_arr)}')
print(f'shape = {raw_img_sitk_arr.shape}')


# In[ ]:


#sitk.Show(raw_img_sitk)


# In[ ]:


explore_3D_array(raw_img_sitk_arr)

