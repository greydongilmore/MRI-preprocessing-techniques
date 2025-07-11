#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ls
from IPython.display import Audio
sound_file = '/mnt/c/Users/fisbain/Documents/GitHub//beep.wav'


# In[2]:


import ants
from helpers import *
import os
os.chdir('/mnt/c/Users/fisbain/Documents/GitHub/MRI-preprocessing-techniques/')
#os.getcwd()
BASE_DIR = os.getcwd()
print(f'Parent folder = {BASE_DIR}')


# In[3]:


pat= ["sub-014"]
pat_dir=os.path.join(BASE_DIR, 'FUS', pat[0])
os.chdir(pat_dir)
#os.getcwd()
print(f'Study folder = {pat_dir}')


# In[4]:


reference_image= pat[0]+'-t1.nii'
input_image= pat[0]+'-FUS.nii'
mask= pat[0]+'-FUS_seg.nii'
output_image = add_suffix_to_filename(input_image, suffix='corr')
output_image_mask = add_suffix_to_filename(mask, suffix='corr')
template_img_path = os.path.join(BASE_DIR, 'assets', 'templates', 'mni_icbm152_t1_tal_nlin_sym_09a.nii')
output_norm_image = add_suffix_to_filename(output_image, suffix='norm')
output_norm_image_mask = add_suffix_to_filename(output_image_mask, suffix='norm')


# In[5]:


print(output_image)
print(output_image_mask)


# In[6]:


# Load the reference MRI scans
fixed_image = ants.image_read(reference_image, reorient='IAL')
# plot
explore_3D_array(arr=fixed_image.numpy())


# In[7]:


# Load the input MRI scans and mask
moving_image = ants.image_read(input_image, reorient='IAL')
mask_image = ants.image_read(mask, reorient='IAL')
# plot
explore_3D_array_with_mask_contour(
arr=moving_image.numpy(),
mask=mask_image.numpy()
)


# In[16]:


# Perform linear registration using ANTs
registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid', verbose=False,
write_composite_transform=True)
redTx = ants.read_transform(registration['fwdtransforms'])
ants.write_transform( redTx, '/mnt/c/Users/fisbain/Documents/GitHub/MRI-preprocessing-techniques/FUS/'+pat[0]+'/redTx.txt')


# In[17]:


# Apply linear transform to input image
registered_image= ants.apply_transforms(moving=moving_image, fixed=registration['warpedmovout'],
                                              transformlist=registration['fwdtransforms'],
                                              verbose=False)


# In[23]:


# Apply linear transform to Mask
registered_image_mask = ants.apply_transforms(moving=mask_image, fixed=fixed_image,
                                              transformlist=registration['fwdtransforms'])


# In[24]:


# Save the registered MRI scan
ants.image_write(registered_image, output_image)
ants.image_write(registered_image_mask, output_image_mask)


# In[ ]:


template_img_ants = ants.image_read(template_img_path, reorient='IAL')
    # Register T1 pre to MNI
transformation = ants.registration(
    fixed=template_img_ants,
    moving=fixed_image, 
    type_of_transform='SyN',
    verbose=False,
    write_composite_transform=True
)


# In[ ]:


Tnorm = ants.read_transform(transformation['fwdtransforms'])
ants.write_transform( Tnorm, '/mnt/c/Users/fisbain/Documents/GitHub/MRI-preprocessing-techniques/FUS/'+pat[0]+'/Tnorm.txt')


# In[ ]:


image_corr= ants.image_read(pat[0]+'-FUS_corr.nii')
image_mask_corr= ants.image_read(pat[0]+'-FUS_seg_corr.nii')


# In[ ]:


# Apply Norm transform to  Corr moving image (FUS)
norm_image= ants.apply_transforms(moving=image_corr, fixed=transformation['warpedmovout'],
                                              transformlist=transformation['fwdtransforms'],
                                              verbose=False)

# Apply Norm transform to Corr mask
norm_image_mask = ants.apply_transforms(moving=image_mask_corr, fixed=transformation['warpedmovout'],
                                              transformlist=transformation['fwdtransforms'],
                                              verbose=False)


# In[ ]:


# Save the normalized MRI scan
ants.image_write(norm_image,output_norm_image )
ants.image_write(norm_image_mask, output_norm_image_mask)


# In[ ]:


Audio(sound_file, autoplay=True)

