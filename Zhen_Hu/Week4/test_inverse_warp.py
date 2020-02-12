"""This is the unit test of inverse_warp function"""
import warnings

warnings.simplefilter("ignore")
from ndmg.utils import reg_utils as rgu
import nibabel as nib
import numpy as np
import nilearn.image as nl
import os
import os.path as op
import scipy.io as scio
import sys.argv as argv


from ndmg.utils import reg_utils as rgu

def test_inverse_warp(argv):
"""Parameters
    ----------
    ref : str
        path to a file in target space, which is a different target space than warp (a image that has not been mapped to mni)
    out : str
        path to the output file, containing warps that are now inverted
    warp : str
        path to the warp/shiftmap transform volume wanting to be inverted
    """
    ref= r"/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz"
    out= r"/Users/zhenhu/Documents/Neuro_Data_Design/Downloads/BNU1/BNU1_2/output"
    warp=r"/Users/zhenhu/Documents/Neuro_Data_Design/Downloads/BNU1/BNU1_2/input/warp2.nii.gz"

    test_ref=argv.t1w_brain
    test_out=argv.mni2t1w_warp
    test_warp=argv.warp_t1w2mni

    Assume=rgu.inverse_warp(test_ref,test_out,test_warp)
    Real=rgu.inverse_warp(ref,out,warp)
    Real_matrix=scio.loadmat(Real)
    Test_matrix=scio.loadmat(Assume)

    assert Real_matrix=Test_matrix

test_inverse_warp(argv)
