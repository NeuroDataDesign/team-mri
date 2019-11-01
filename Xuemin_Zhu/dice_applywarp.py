import warnings
warnings.simplefilter("ignore")
import ndmg
from ndmg.utils import gen_utils as mgu
from ndmg.utils import reg_utils as mgr
import nibabel as nib
import numpy as np
import nilearn.image as nl
import torch as torch
import os
import os.path as op
import pytest

#set input paths
warped_fsl = r"/Users/xueminzhu/Desktop/test/tests_output/desikan_space-MNI152NLin6_res-2x2x2_reor_RAS_nores_aligned_atlas_skull.nii.gz"
warped_dipy= r"/Users/xueminzhu/Desktop/test/dipywarp.nii.gz"
#warped_original = r"/Users/xueminzhu/Desktop/test/tests_input/desikan_space-MNI152NLin6_res-2x2x2_reor_RAS_nores_aligned_atlas_t1w_mni.nii.gz"

#warped_original_data=nib.load(str(warped_original)).get_fdata()
#X = torch.from_numpy(warped_original_data)

warped_dipy_data=nib.load(str(warped_dipy)).get_fdata()
Y1 = torch.from_numpy(warped_dipy_data)

warped_fsl_data=nib.load(str(warped_fsl)).get_fdata()
Y2 = torch.from_numpy(warped_fsl_data)

#iflat = X.contiguous().view(-1)
tflat1 = Y1.contiguous().view(-1)
tflat2 = Y2.contiguous().view(-1)

#intersection1 = (iflat * tflat1).sum()
#intersection2 = (iflat * tflat2).sum()
intersection2 = (tflat1 * tflat2).sum()
#X_sum = torch.sum(iflat * iflat)
Y1_sum = torch.sum(tflat1 * tflat1)
Y2_sum = torch.sum(tflat2 * tflat2)

#ans1 = 1 - ((2. * intersection1 + 1.) / (X_sum + Y1_sum + 1.))
ans2 = 1 - ((2. * intersection2 + 1.) / (Y1_sum + Y2_sum + 1.))

print (ans2)