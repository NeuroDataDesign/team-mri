#This is a pytest file to nifti_to_png

from argparse import ArgumentParser
from spipy.misd import imsave
import ndmg
import pytest
import nibabel as nb 
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imghdr

#create a temporary file 
	d = tmp_path/"test"
	d.mkdir()
	test_input = d/ "test_nifiti.nii"

	test_output = d/ "test_png.png"
	

def test_nifti_to_png()
test_input_image  = mping.imread('test_nifti.nii')
test_output_image = mping.imread('test_png.png')

ndmg.utils.main(str(test_input),str(test_output),verbose=False)

test = np.loadtxt(str(test_output))
rtest= imghdr.what(str(test_output))
rctest = imghdr.what(test_output_image) 

assert rtest = rctest
