#This is a function to test apply_warp
import warnings

warnings.simplefilter("ignore")

from ndmg.utils import gen_utils as mgu
import nibabel as nib
import numpy as np
import nilearn.image as nl
import os
import os.path as op
import scipy.io as scio

from ndmg.utils import reg_utils as rgu

def test_apply_warp()

ref= r"/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz"
inp= r"/Users/xueminzhu/Downloads/BNU1_2/input/inp.nii"
out= r"/Users/xueminzhu/Downloads/BNU1_2/output"
warp=r"/Users/xueminzhu/Downloads/BNU1_2/input/warp2.nii.gz"
test_ref=ref
test_inp=inp
test_out=scio.loadmat("Users/xueminzhu/Downloads/BNU1_2/output/func2struct.mat")
rgu.apply_warp(ref,inp,out,warp)
test_out_matrix=scio.loadmat(out)

assert test_out_matrix=test_out_matrix=test_out
