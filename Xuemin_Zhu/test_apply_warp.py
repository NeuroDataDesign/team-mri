#This is a function to test apply_warp
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


from ndmg.utils import reg_utils as rgu

def test_apply_warp():
 
    #trainout= r"/Users/xueminzhu/ndmg_outputs/tmp/reg_a/desikan_space-MNI152NLin6_res-2x2x2_reor_RAS_nores_aligned_atlas_skull.nii.gz"
    
    #ref = r"/Users/xueminzhu/ndmg_outputs/anat/preproc/t1w_brain_norse.nii.gz"
    #inp= r"/Users/xueminzhu/ndmg_outputs/tmp/reg_a/desikan_space-MNI152NLin6_res-2x2x2_reor_RAS_nores_aligned_atlas_t1w_mni.nii.gz"
    #out= r"/Users/xueminzhu/ndmg_outputs/tmp/reg_a"
    #warp=r"/Users/xueminzhu/ndmg_outputs/tmp/reg_a"

    #make a temporary path
    d = tmp_path / "sub"
    d.mkdir()
    temp_out1 = d / "outnii.nii.gz"
    temp_out2 = d/  "warpnii.nii.gz"

    #define correct input data path
    ref = '../test_data/inputs/t1w_brain_nores.nii.gz'
    inp = '../test_data/inputs/desikan_space-MNI152NLin6_res-2x2x2_reor_RAS_nores_aligned_atlas_t1w_mni.nii.gz'
    out = temp_out1
    warp = temp_out2

    rgu.apply_warp(ref,inp,out,warp)



    img = nib.load(str(warp))
    img_1 = img.get_fdata()
    result_warp = np.array(img_1)
    out_warp = result_warp.shape()
    ref_warp = (25, 30, 25, 3)

    assert out_warp == ref_warp
