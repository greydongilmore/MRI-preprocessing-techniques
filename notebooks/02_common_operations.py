#!/usr/bin/env python
# coding: utf-8

# ### Common operations

# **Learning outcomes:**
# - Learn how to apply basic filters and transformations using AntsPy and SITK:
#     - Denoise
#     - Morphological operations
#     - Shrink
#     - Cropping
#     - Padding
#     - Blurring
#     - Thresholding
#     - Statistics

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

# #### Raw Image

# In[ ]:


raw_img_path = os.path.join(BASE_DIR, 'assets', 'raw_examples', raw_examples[0])
raw_img_ants = ants.image_read(raw_img_path, reorient='IAL') 

print(f'shape = {raw_img_ants.numpy().shape} -> (Z, X, Y)')

explore_3D_array(
    arr=raw_img_ants.numpy(),
    cmap='viridis'
)


# #### Denoise

# In[ ]:


transformed = ants.denoise_image(raw_img_ants, shrink_factor=8)

explore_3D_array_comparison(
    arr_before=raw_img_ants.numpy(),
    arr_after=transformed.numpy(),
    cmap='viridis'
)


# #### Morphological operations

# In[ ]:


"""
operation to apply
    "close" Morphological closing
    "dilate" Morphological dilation
    "erode" Morphological erosion
    "open" Morphological opening
"""

transformed = ants.morphology(raw_img_ants, radius=1, operation='erode', mtype='grayscale')

explore_3D_array_comparison(
    arr_before=raw_img_ants.numpy(),
    arr_after=transformed.numpy(),
    cmap='viridis'
)


# ### Simple ITK

# #### Raw Image

# In[ ]:


raw_img_path = os.path.join(BASE_DIR, 'assets', 'raw_examples', raw_examples[0])
raw_img_sitk = sitk.ReadImage(raw_img_path, sitk.sitkFloat32)
raw_img_sitk = sitk.DICOMOrient(raw_img_sitk,'RPS')

print(f'shape = {sitk.GetArrayFromImage(raw_img_sitk).shape} -> (Z, X, Y)')
explore_3D_array(
    arr=sitk.GetArrayFromImage(raw_img_sitk),
    cmap='viridis'
)


# #### Shrink

# In[ ]:


shrinkFactor = 3
transformed = sitk.Shrink( raw_img_sitk, [ shrinkFactor ] * raw_img_sitk.GetDimension() )

print(f'shape before = {sitk.GetArrayFromImage(raw_img_sitk).shape}')
print(f'shape after = {sitk.GetArrayFromImage(transformed).shape}')

explore_3D_array(sitk.GetArrayFromImage(transformed))


# #### Crop

# In[ ]:


# Cropping takes the orientation of the pixels for the reference of lower & upper boundaries vectors
# Pixel orientation = RPS = (left-to-Right, anterior-to-Posterior, inferior-to-Superior)

# crop nothing
#transformed = sitk.Crop(raw_img_sitk)
#transformed = sitk.Crop(raw_img_sitk, (0,0,0), (0,0,0))

# crop 20 from left to right             X,Y,Z
#transformed = sitk.Crop(raw_img_sitk, (20,0,0), (0,0,0))

# crop 20 from left to right, crop 30 from anterior to posterior
#transformed = sitk.Crop(raw_img_sitk, (20,30,0), (0,0,0))

# crop 20 from left to right, crop 30 from anterior to posterior, 
# crop 10 from right to left, crop 5 from posterior to anterior. 
#transformed = sitk.Crop(raw_img_sitk, (20,30,0), (10,5,0)) 

# crop 40 from inferior to superior, crop 50 from superior to inferior
transformed = sitk.Crop(raw_img_sitk, (0,0,40), (0,0,50)) 


print(f'shape before = {sitk.GetArrayFromImage(raw_img_sitk).shape}')
print(f'shape after = {sitk.GetArrayFromImage(transformed).shape}')

explore_3D_array(sitk.GetArrayFromImage(transformed))


# #### Padding

# In[ ]:


constant = int(sitk.GetArrayFromImage(raw_img_sitk).min())
constant


# In[ ]:


# Padding (as Cropping) takes the orientation of the pixels for the reference of lower & upper boundaries vectors
# Pixel orientation = RPS = (left-to-Right, anterior-to-Posterior, inferior-to-Superior)

# pad nothing
#transformed = sitk.ConstantPad(raw_img_sitk)
#transformed = sitk.ConstantPad(raw_img_sitk,(0,0,0),(0,0,0), constant)

# pad 10 from left to right
#transformed = sitk.ConstantPad(raw_img_sitk,(10,0,0),(0,0,0),constant)

# pad 10 from left to right, pad 15 from anterior to posterior
#transformed = sitk.ConstantPad(raw_img_sitk,(10,15,0),(0,0,0),constant)

# pad 10 from left to right, pad 15 from anterior to posterior, 
# pad 5 from right to left, pad 8 from posterior to anterior. 
transformed = sitk.ConstantPad(raw_img_sitk,(10,15,0),(5,8,0),constant)


print(f'shape before = {sitk.GetArrayFromImage(raw_img_sitk).shape}')
print(f'shape after = {sitk.GetArrayFromImage(transformed).shape}')

explore_3D_array(sitk.GetArrayFromImage(transformed), cmap='viridis')


# #### Denoise

# Curvature Flow filter

# In[ ]:


transformed = sitk.CurvatureFlow(raw_img_sitk)

explore_3D_array_comparison(
    arr_before=sitk.GetArrayFromImage(raw_img_sitk),
    arr_after=sitk.GetArrayFromImage(transformed),
    cmap='viridis'
)


# #### Morphological Operations

# In[ ]:


"""
sitk.GrayscaleMorphologicalClosing
sitk.GrayscaleDilate
sitk.GrayscaleErode
sitk.GrayscaleMorphologicalOpening

sitk.BinaryMorphologicalClosing
sitk.BinaryDilate
sitk.BinaryErode
sitk.BinaryMorphologicalOpening
"""

transformed = sitk.GrayscaleErode(raw_img_sitk)

explore_3D_array_comparison(
    arr_before=sitk.GetArrayFromImage(raw_img_sitk),
    arr_after=sitk.GetArrayFromImage(transformed),
    cmap='viridis'
)


# #### Blurring

# In[ ]:


transformed = sitk.DiscreteGaussian(raw_img_sitk)

explore_3D_array_comparison(
    arr_before=sitk.GetArrayFromImage(raw_img_sitk),
    arr_after=sitk.GetArrayFromImage(transformed),
    cmap='viridis'
)


# #### Thresholding

# In[ ]:


"""
sitk.OtsuThreshold
sitk.LiThreshold
sitk.TriangleThreshold
sitk.MomentsThreshold
"""

transformed = sitk.TriangleThreshold(raw_img_sitk, 0, 1)

explore_3D_array_comparison(
    arr_before=sitk.GetArrayFromImage(raw_img_sitk),
    arr_after=sitk.GetArrayFromImage(transformed)
)


# #### Statistics

# In[ ]:


stats = sitk.StatisticsImageFilter()
stats.Execute(raw_img_sitk)


print('\tRaw img')
print("min  =", stats.GetMinimum())
print("max  =", stats.GetMaximum())
print("mean =", stats.GetMean())

