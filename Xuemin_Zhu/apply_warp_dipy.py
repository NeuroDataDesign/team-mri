#This is a pure python function for apply_warp
import dipy
import warnings

warnings.simplefilter("ignore")
from ndmg.utils import gen_utils as mgu
import nibabel as nib
import numpy as np
import nilearn.image as nl
import os
import os.path as op
import matplotlib.image as mpimg

from ndmg.utils import reg_utils as rgu

"""Applies a warp from the structural to reference space in a single step using information about
    the structural -> ref mapping as well as the functional to structural mapping.
    
    Parameters
    ----------
    in_nii : 
        the reference image to be aligned to
    nonlin_nii: 
        the input image to be aligned
    out_nii : 
        the resulting warped output image
    affine_mat : 
        the warp coefficent file to go from inp -> ref
"""
dipy.external.fsl.apply_warp(in_nii, affine_mat, nonlin_nii, out_nii)
in_nii = mping.imread(ref)
affine = mping.imread(warp)
nonlin_nii = mping.imread(inp)
out_nii=mping.imread(out)