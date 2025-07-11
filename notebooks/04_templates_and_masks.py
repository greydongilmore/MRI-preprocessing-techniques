#!/usr/bin/env python
# coding: utf-8

# ### Templates and masks

# **Learning outcomes:**
# - How to load a template image and a mask with ants.
# - How to mask a brain using ants.
# - Inspect visually region delimited by a mask.

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


# #### Template example

# In[ ]:


template_img_path = os.path.join(BASE_DIR, 'assets', 'templates', 'mni_icbm152_t1_tal_nlin_sym_09a.nii')
template_img_ants = ants.image_read(template_img_path, reorient='IAL')

explore_3D_array(arr = template_img_ants.numpy())


# In[ ]:


print('\t\tTEMPLATE IMG')
print(template_img_ants)


# #### Brain mask

# In[ ]:


brain_mask_img_path = os.path.join(BASE_DIR, 'assets', 'templates', 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii')
brain_mask_img_ants = ants.image_read(brain_mask_img_path, reorient='IAL')

explore_3D_array(arr = brain_mask_img_ants.numpy())


# In[ ]:


print('\t\tTEMPLATE IMG')
print(template_img_ants)

print('\t\tBRAIN MASK IMG')
print(brain_mask_img_ants)


# #### Mask out the brain

# In[ ]:


brain_masked = ants.mask_image(template_img_ants, brain_mask_img_ants)

explore_3D_array_comparison(template_img_ants.numpy(), brain_masked.numpy())


# In[ ]:


explore_3D_array_with_mask_contour(template_img_ants.numpy(), brain_mask_img_ants.numpy())


# #### Brain Lesion example

# In[ ]:


raw_img_path = os.path.join(BASE_DIR, 'assets', 'raw_examples', 'brain-lesion_T1w.nii.gz')
raw_img_ants = ants.image_read(raw_img_path, reorient='IAL')

explore_3D_array(arr = raw_img_ants.numpy())


# #### Tissue mask

# In[ ]:


mask_img_path = os.path.join(BASE_DIR, 'assets', 'raw_examples', 'brain-lesion_T1w_mask.nii.gz')
mask_img_ants = ants.image_read(mask_img_path, reorient='IAL')

explore_3D_array(arr = mask_img_ants.numpy())


# In[ ]:


print('\t\tRAW IMG')
print(raw_img_ants)

print('\t\tMASK IMG')
print(mask_img_ants)


# In[ ]:


explore_3D_array_with_mask_contour(raw_img_ants.numpy(),mask_img_ants.numpy())

