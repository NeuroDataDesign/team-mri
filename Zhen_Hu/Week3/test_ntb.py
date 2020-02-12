#This is the test code for nifti_to_binary.py
# Assume the original function is true. 
# What the unit test do is to verify whether your code is correct if you modify the original code.

from argparse import ArgumentParser
import warnings
from functools import reduce
warnings.simplefilter("ignore")

import pytest
import numpy as np 
import ndmg
import os
import nibabel as nibabel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imghdr
import os.path as op
import nibabel as nb

#We use the 1.nii as the example test_data. The output binary 
#Our goal is to determine if you modify the code correct or not.

#Step 1：try to read the real and correct document into python. 
# Attention: you have to first run nifti_to_binary.py on 1.nii image first on your own to get the binary document.
#Step2：get the test output document：
#Step 2.1 use the nifti_to binary.py run the 2.nii, which is our test image.
#Attention:2.nii is exactly the same as 1.nii, so the unit test result should be right.
#Step 2.2: after running, try to read the test output file into here.
#Step three：compare these three document.

def nib_to_bin(nii, dat):
    im = nb.load(nii)
    im_d = im.get_data()

    length = reduce(lambda x, y: x * y, im_d.shape)
    dat_d = np.reshape(im_d.astype(np.dtype("float32")), (1, length))
    with open(dat, "wb") as fl:
        fl.write(dat_d)

def test_nifti_to_binary():
    content_true = np.fromfile('1.dat', dtype=int)
    nii='2.nii'
    dat='2.dat'
    nib_to_bin(nii,dat)
    content_test = np.fromfile('2.dat', dtype=int)

    assert content_true.all() == content_test.all()