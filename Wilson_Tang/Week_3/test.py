import ndmg
from ndmg import preproc as mgp
from ndmg.utils import gen_utils as mgu
from ndmg.register import gen_reg as mgr
from ndmg.track import gen_track as mgt
from ndmg.graph import gen_graph as mgg
from ndmg.utils.bids_utils import name_resource
from unittest.mock import Mock
import numpy as np 
import pytest
import os

import nibabel as nib
#tmp_path = PosixPath('PYTEST_TMPDIR/test_createfile0')


# def test_reorient_dwi():
#     """Orients dwi data to the proper orientation (RAS+) using nibabel
    
#     Parameters
#     ----------
#     dwi_prep : str
#         Path to eddy corrected dwi file
#     bvecs : str
#         Path to the resaled b-vector file
#     namer : name_resource
#         name_resource variable containing relevant directory tree information
    
#     Returns
#     -------
#     str
#         Path to potentially reoriented dwi file
#     str
#         Path to b-vector file, potentially reoriented if dwi data was
#     """
#     #create temp file dir
# 	d = tmp_path/"sub"
# 	d.mkdir()
# 	temp_in1 = d/ "test1.bvec"

# 	#desired ouput
# 	output = np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
# 								 [-2.03325349e-01,  5.19142517e-01, -8.30150498e-01],
# 								 [ 1.97708117e-01,  5.19668166e-01, -8.31177778e-01],
# 								 [ 4.01805079e-01,  1.75685707e-01, -8.98714199e-01],
# 								 [-4.03123289e-01,  7.31595100e-01, -5.49781978e-01],
# 								 [-2.01737253e-01, 9.41754689e-01, -2.69072828e-01],
# 								 [-8.53421512e-01,  5.17921688e-01, -5.85563669e-02],
# 								 [-7.31238368e-01,  5.19749047e-01, -4.41759410e-01],
# 								 [-4.06789294e-01, 1.76111128e-01, -8.96385710e-01],
# 								 [-7.32634311e-01,  1.75959528e-01, -6.57484000e-01],
# 								 [-6.50504746e-01,  7.30204796e-01, 2.08912738e-01],
# 								 [-3.22355457e-01,  9.40912090e-01,  1.03785345e-01],
# 								 [-3.23905616e-01,  5.18818854e-01,  7.91146098e-01],
# 								 [-6.49987712e-01,  5.19871842e-01,  5.54300678e-01],
# 								 [-9.78756200e-01,  1.75626220e-01,  1.05790981e-01],
# 								 [-8.54325102e-01,  1.75781009e-01,  4.89111088e-01],
# 								 [ 7.35120011e-04,  7.31228091e-01,  6.82132640e-01],
# 								 [ 1.88436018e-03,  9.41592914e-01,  3.36748027e-01],
# 								 [ 6.54135065e-01,  5.18889560e-01,  5.50328031e-01],
# 								 [ 3.28634707e-01,  5.19986685e-01,  7.88424427e-01],
# 								 [ 1.97367835e-01, -1.75376433e-01, -9.64514927e-01],
# 								 [ 2.03872410e-01,  1.75468264e-01,  9.63144293e-01],
# 								 [ 6.51154629e-01,  7.30292430e-01,  2.06568670e-01],
# 								 [ 3.23619332e-01,  9.41025840e-01,  9.86959772e-02],
# 								 [ 1.99623536e-01,  9.40698784e-01, -2.74292255e-01],
# 								 [ 4.03123289e-01,  7.31595100e-01, -5.49781978e-01],
# 								 [ 7.31922351e-01,  1.75805873e-01, -6.58317527e-01],
# 								 [ 7.28641097e-01,  5.18680158e-01, -4.47272899e-01],
# 								 [ 8.52419436e-01,  5.18867075e-01, -6.44830465e-02],
# 								 [ 8.57252640e-01,  1.76136461e-01,  4.83832469e-01],
# 								 [ 9.79328763e-01,  1.75544517e-01,  1.00495257e-01]])
# 		#store ouput in a folder path
# 		np.savetxt(temp_in1,output)

# 		dwi_data = /eddy_corrected_data.nii.gz

# 		[dwi_prep,bvec] = mgu.reorient_dwi(  ,temp_in1, )


# def test_match_target_vox_res_1mm ():

# 	nifti_data = Mock()
# 	nifti_data.get_fdata = np.array([[1,2,3],[1,2,3]])
# 	nifti_data.header = Mock()
# 	nifti_data.header.get_zooms = np.array([1.0,2.0,3.0])
	
# 	assert np.allclose(nifti_data.get_fdata(),np.array([[1,2,3],[1,2,3]]))


def test_get_braindata_np():
	brain_data = Mock()
	brain_data.configure_mock(type = nib.nifti1.Nifti1Image)

	assert type(brain_data) is nib.nifti1.Nifti1Image

# def test_get_braindata_str():



	# assert np.allclose(nifti_data.header.getzooms(),np.array([1.0,2.0,3.0]))

# def test_match_target_vox_res_1mm(tmp_path):
# 	"""Reslices input MRI file if it does not match the targeted voxel resolution. Can take dwi or t1w scans.
    
#     Parameters
#     ----------
#     img_file : str
#         path to file to be resliced
#     vox_size : str
#         target voxel resolution ('2mm' or '1mm')
#     namer : name_resource
#         name_resource variable containing relevant directory tree information
#     sens : str
#         type of data being analyzed ('dwi' or 'func')
    
#     Returns
#     -------
#     str
#         location of potentially resliced image
#     """
# 	#create temp file dir

# 	assert 1 ==1



# def test_match_target_vox_res_2mm():
# 	"""Reslices input MRI file if it does not match the targeted voxel resolution. Can take dwi or t1w scans.
    
#     Parameters
#     ----------
#     img_file : str
#         path to file to be resliced
#     vox_size : str
#         target voxel resolution ('2mm' or '1mm')
#     namer : name_resource
#         name_resource variable containing relevant directory tree information
#     sens : str
#         type of data being analyzed ('dwi' or 'func')
    
#     Returns
#     -------
#     str
#         location of potentially resliced image
#     """




