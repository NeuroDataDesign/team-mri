#!/usr/bin/env python
# coding: utf-8

# In[17]:


import nibabel as nib
import skimage.io as io
import numpy as np
img = nib.load('d:/Downloads/neurodatadesign/ndmg/tests/ndmg_outputs/anat/preproc/t1w_brain.nii.gz')
print(img.shape)
print(img.affine.shape)
img_arr = img.get_fdata()
print(img_arr.shape)
# img_arr = np.squeeze(img_arr)
io.imshow(img_arr[46])


# In[19]:


img_csfprob = nib.load('d:/Downloads/neurodatadesign/ndmg/tests/ndmg_outputs/anat/preproc/t1w_seg_pve_0.nii.gz')
print(img_csfprob.shape)
print(img_csfprob.affine.shape)
img_csfprob_arr = img_csfprob.get_fdata()
print(img_csfprob_arr.shape)
# img_arr = np.squeeze(img_arr)
io.imshow(img_csfprob_arr[46])


# In[20]:


img_gmprob = nib.load('d:/Downloads/neurodatadesign/ndmg/tests/ndmg_outputs/anat/preproc/t1w_seg_pve_1.nii.gz')
print(img_gmprob.shape)
print(img_gmprob.affine.shape)
img_gmprob_arr = img_gmprob.get_fdata()
print(img_gmprob_arr.shape)
# img_arr = np.squeeze(img_arr)
io.imshow(img_gmprob_arr[46])


# In[21]:


img_vmprob = nib.load('d:/Downloads/neurodatadesign/ndmg/tests/ndmg_outputs/anat/preproc/t1w_seg_pve_2.nii.gz')
print(img_vmprob.shape)
print(img_vmprob.affine.shape)
img_vmprob_arr = img_vmprob.get_fdata()
print(img_vmprob_arr.shape)
# img_arr = np.squeeze(img_arr)
io.imshow(img_vmprob_arr[46])


# In[ ]:




