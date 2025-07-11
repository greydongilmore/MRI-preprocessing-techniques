#!/usr/bin/env python
# coding: utf-8

# ### Brain extraction(Skull stripping)

# **Learning outcomes:**
# - How to do brain extraction using registration.

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
]


# #### Raw Image

# In[ ]:


raw_example = raw_examples[0]
raw_img_path = os.path.join(BASE_DIR, 'assets', 'raw_examples', raw_example)
raw_img_ants = ants.image_read(raw_img_path, reorient='IAL')

explore_3D_array(arr=raw_img_ants.numpy(), cmap='nipy_spectral')


# ### Template based method (Native space)

# #### Template Image

# In[ ]:


template_img_path = os.path.join(BASE_DIR, 'assets', 'templates', 'mni_icbm152_t1_tal_nlin_sym_09a.nii')
template_img_ants = ants.image_read(template_img_path, reorient='IAL')

explore_3D_array(arr = template_img_ants.numpy())


# #### Brain Mask of the template

# In[ ]:


mask_template_img_path = os.path.join(BASE_DIR, 'assets', 'templates', 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii')
mask_template_img_ants = ants.image_read(mask_template_img_path, reorient='IAL')

explore_3D_array(mask_template_img_ants.numpy())


# In[ ]:


np.unique(mask_template_img_ants.numpy())


# #### Register template to raw image

# In[ ]:


transformation = ants.registration(
    fixed=raw_img_ants,
    moving=template_img_ants, 
    type_of_transform='SyN',
    verbose=True
)


# In[ ]:


print(transformation)


# In[ ]:


registered_img_ants = transformation['warpedmovout']

explore_3D_array_comparison(
    arr_before=raw_img_ants.numpy(), 
    arr_after=registered_img_ants.numpy()
)


# #### Apply the generated transformations to the mask of template

# In[ ]:


brain_mask = ants.apply_transforms(
    fixed=transformation['warpedmovout'],
    moving=mask_template_img_ants,
    transformlist=transformation['fwdtransforms'],
    interpolator='nearestNeighbor',
    verbose=True
)


# In[ ]:


explore_3D_array(brain_mask.numpy())


# In[ ]:


explore_3D_array_with_mask_contour(raw_img_ants.numpy(), brain_mask.numpy())


# In[ ]:


brain_mask_dilated = ants.morphology(brain_mask, radius=4, operation='dilate', mtype='binary')

explore_3D_array_with_mask_contour(raw_img_ants.numpy(), brain_mask_dilated.numpy())


# #### Save brain mask

# In[ ]:


out_folder =  os.path.join(BASE_DIR, 'assets', 'preprocessed')
out_folder = os.path.join(out_folder, raw_example.split('.')[0]) # create folder with name of the raw file
os.makedirs(out_folder, exist_ok=True) # create folder if not exists

out_filename = add_suffix_to_filename(raw_example, suffix='brainMaskByTemplate')
out_path = os.path.join(out_folder, out_filename)

print(raw_img_path[len(BASE_DIR):])
print(out_path[len(BASE_DIR):])


# In[ ]:


brain_mask_dilated.to_file(out_path)


# #### Generate brain masked

# In[ ]:


masked = ants.mask_image(raw_img_ants, brain_mask_dilated)

explore_3D_array(masked.numpy())


# In[ ]:


out_filename = add_suffix_to_filename(raw_example, suffix='brainMaskedByTemplate')
out_path = os.path.join(out_folder, out_filename)

print(raw_img_path[len(BASE_DIR):])
print(out_path[len(BASE_DIR):])


# In[ ]:


masked.to_file(out_path)

