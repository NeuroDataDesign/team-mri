
#Use dice to test apply_warp
import warnings
warnings.simplefilter("ignore")
from ndmg.utils import gen_utils as mgu
import nibabel as nib
import numpy as np
import nilearn.image as nl
import os
import os.path as op
import nilearn.image as nl
import pytest
import torch as torch


import ndmg
from ndmg.utils import gen_utils as mgu
from ndmg.utils import reg_utils as mgr
from ndmg.utils.reg_utils import apply_warp

def test_apply_warp_dice(tmp_path):
 #make a temporary path#
    d = tmp_path / "sub"
    d.mkdir()
    out_out_temp_path = d / "outnii.nii.gz"
    warp_out_temp_path = d/  "warpnii.nii.gz"
    
    #set input/output data path
    ref_in_path = r"/Users/xueminzhu/Desktop/test/tests_input/t1w_brain_nores.nii.gz"
    inp_in_path = r"/Users/xueminzhu/Desktop/test/tests_input/desikan_space-MNI152NLin6_res-2x2x2_reor_RAS_nores_aligned_atlas_t1w_mni.nii.gz"
    outnii_out_cntrl_path = r"/Users/xueminzhu/Desktop/test/tests_output/desikan_space-MNI152NLin6_res-2x2x2_reor_RAS_nores_aligned_atlas_skull.nii.gz"
    
    #call function
    ref=ref_in_path
    inp=inp_in_path
    out=out_out_temp_path
    warp=warp_out_temp_path
    apply_warp(str(ref),str(inp),str(out),str(warp),None,None,str('nn'),True)
    
    #load function output 
    outnii_out_temp = nib.load(str(out)).get_fdata()
    X = torch.from_numpy(outnii_out_temp)
    
    #load output data
    outnii_out_cntrl = nib.load(str(outnii_out_cntrl_path)).get_fdata()
    Y = torch.from_numpy(outnii_out_cntrl)
    
    #calculate dice loss function
    iflat = X.contiguous().view(-1)
    tflat = Y.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    X_sum = torch.sum(iflat * iflat)
    Y_sum = torch.sum(tflat * tflat)

    ans = 1 - ((2. * intersection + 1.) / (X_sum + Y_sum + 1.))

    # assert
    assert ans < 0.3