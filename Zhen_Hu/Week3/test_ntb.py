#This is the test code for nifti_to_binary.py
# Assume the original function is true. 
# What the unit test do is to verify whether your code is correct if you change the original code.

from argparse import ArgumentParser

import pytest
import numpy as np 
import ndmg
import os
import nibabel as nibabel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imghdr
import os.path as op

#First step：try to read the real and correct document. 
# Attention: you have to first run nifti_to_binary.py on 1.nii image first on your own to get the binary document.
#Second step：get the test output document：
#Second step-1.use the nifti_to binary.py run the 2.nii, which is our test image.
#Attention:2.nii is exactly the same as 1.nii, so the unit test result should be right.
#Second step-2: after running, try to read the test output file into here.
#Step three：compare these three document.

#create a temporary file 
#d = tmp_path"/Users/zhenhu/Documents/Neuro_Data_Design/Downloads/ndmg/ndmg/utils"
#d.mkdir()
#test_input = d/ "1.nii"

#test_output = d/ "1.dat"

def test_nifti_to_binary():
    #dir_true = '/Users/zhenhu/Documents/Neuro_Data_Design/Downloads/ndmg/ndmg/utils/1.dat'
    #content_true=open(dir_true,'r').read()
    content_true = np.fromfile('1.dat', dtype=int)
    #for f in glob.glob(dir + '*.dat'):
    #  content_true = open(f,'r').read()
    #dats = [os.path.splitext(os.path.splitext(fn)[0])[0] + ".dat" for fn in niis]
    ndmg.utils.nifti_to_binary.main()
    #dir_test = '/Users/zhenhu/Documents/Neuro_Data_Design/Downloads/ndmg/ndmg/utils/2.dat'
    #content_test=open(dir_test,'r').read()
    content_test = np.fromfile('2.dat', dtype=int)

    assert content_true == content_test