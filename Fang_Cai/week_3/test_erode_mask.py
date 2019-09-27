import ndmg
import warnings

warnings.simplefilter("ignore")
from ndmg.utils import gen_utils as mgu
import nibabel as nib
import numpy as np
import nilearn.image as nl
import os
import os.path as op

import pytest

def test_erode_mask():
    mask = np.ones((3,3,3))
    mask[0][0][0] = -5
    mask[2][2][2] = 0.8
    v = 64
    output = ndmg.utils.reg_utils.erode_mask(mask, v)
    standard = np.ones((3,3,3))
    assert np.allclose(output, standard)