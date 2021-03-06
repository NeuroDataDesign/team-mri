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
#tmp_path = PosixPath('PYTEST_TMPDIR/test_createfile0')


def test_reorient_dwi():
    """Orients dwi data to the proper orientation (RAS+) using nibabel
    
    Parameters
    ----------
    dwi_prep : str
        Path to eddy corrected dwi file
    bvecs : str
        Path to the resaled b-vector file
    namer : name_resource
        name_resource variable containing relevant directory tree information
    
    Returns
    -------
    str
        Path to potentially reoriented dwi file
    str
        Path to b-vector file, potentially reoriented if dwi data was
    """
    #create temp file dir
	d = tmp_path/"sub"
	d.mkdir()
	temp_in1 = d/ "test1.bvec"

	#desired ouput
	output = np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
								 [-2.03325349e-01,  5.19142517e-01, -8.30150498e-01],
								 [ 1.97708117e-01,  5.19668166e-01, -8.31177778e-01],
								 [ 4.01805079e-01,  1.75685707e-01, -8.98714199e-01],
								 [-4.03123289e-01,  7.31595100e-01, -5.49781978e-01],
								 [-2.01737253e-01, 9.41754689e-01, -2.69072828e-01],
								 [-8.53421512e-01,  5.17921688e-01, -5.85563669e-02],
								 [-7.31238368e-01,  5.19749047e-01, -4.41759410e-01],
								 [-4.06789294e-01, 1.76111128e-01, -8.96385710e-01],
								 [-7.32634311e-01,  1.75959528e-01, -6.57484000e-01],
								 [-6.50504746e-01,  7.30204796e-01, 2.08912738e-01],
								 [-3.22355457e-01,  9.40912090e-01,  1.03785345e-01],
								 [-3.23905616e-01,  5.18818854e-01,  7.91146098e-01],
								 [-6.49987712e-01,  5.19871842e-01,  5.54300678e-01],
								 [-9.78756200e-01,  1.75626220e-01,  1.05790981e-01],
								 [-8.54325102e-01,  1.75781009e-01,  4.89111088e-01],
								 [ 7.35120011e-04,  7.31228091e-01,  6.82132640e-01],
								 [ 1.88436018e-03,  9.41592914e-01,  3.36748027e-01],
								 [ 6.54135065e-01,  5.18889560e-01,  5.50328031e-01],
								 [ 3.28634707e-01,  5.19986685e-01,  7.88424427e-01],
								 [ 1.97367835e-01, -1.75376433e-01, -9.64514927e-01],
								 [ 2.03872410e-01,  1.75468264e-01,  9.63144293e-01],
								 [ 6.51154629e-01,  7.30292430e-01,  2.06568670e-01],
								 [ 3.23619332e-01,  9.41025840e-01,  9.86959772e-02],
								 [ 1.99623536e-01,  9.40698784e-01, -2.74292255e-01],
								 [ 4.03123289e-01,  7.31595100e-01, -5.49781978e-01],
								 [ 7.31922351e-01,  1.75805873e-01, -6.58317527e-01],
								 [ 7.28641097e-01,  5.18680158e-01, -4.47272899e-01],
								 [ 8.52419436e-01,  5.18867075e-01, -6.44830465e-02],
								 [ 8.57252640e-01,  1.76136461e-01,  4.83832469e-01],
								 [ 9.79328763e-01,  1.75544517e-01,  1.00495257e-01]])
		#store ouput in a folder path
		np.savetxt(temp_in1,output)

		dwi_data = /eddy_corrected_data.nii.gz

		[dwi_prep,bvec] = mgu.reorient_dwi(  ,temp_in1, )




def test_match_target_vox_res_1mm():
	"""Reslices input MRI file if it does not match the targeted voxel resolution. Can take dwi or t1w scans.
    
    Parameters
    ----------
    img_file : str
        path to file to be resliced
    vox_size : str
        target voxel resolution ('2mm' or '1mm')
    namer : name_resource
        name_resource variable containing relevant directory tree information
    sens : str
        type of data being analyzed ('dwi' or 'func')
    
    Returns
    -------
    str
        location of potentially resliced image
    """

    ###create mock of the image
    nifti_data_mock = Mock()
    


    test_match_target_vox_res_1mm(img)



def test_match_target_vox_res_2mm():
	"""Reslices input MRI file if it does not match the targeted voxel resolution. Can take dwi or t1w scans.
    
    Parameters
    ----------
    img_file : str
        path to file to be resliced
    vox_size : str
        target voxel resolution ('2mm' or '1mm')
    namer : name_resource
        name_resource variable containing relevant directory tree information
    sens : str
        type of data being analyzed ('dwi' or 'func')
    
    Returns
    -------
    str
        location of potentially resliced image
    """


 
def test_rescale_bvac(tmp_path):
	#create temp file dir
	d = tmp_path/"sub"
	d.mkdir()
	temp_in1 = d/ "test1.bvec"
	temp_in2 = d/ "test2.bvec"

	#create temp output dir
	temp_out1 = d/ "test1_new.bvec"
	temp_out2 = d/ "test2_new.bvec"

	#write cases for test1 Test w/ proper dimensions
	bvec_test_1 = np.array([[ 0.00000000e+00, -2.03325346e-01,  1.97708115e-01,
	         4.01805073e-01, -4.03123289e-01, -2.01737255e-01,
	        -8.53421509e-01, -7.31238365e-01, -4.06789303e-01,
	        -7.32634306e-01, -6.50504768e-01, -3.22355449e-01,
	        -3.23905617e-01, -6.49987698e-01, -9.78756189e-01,
	        -8.54325116e-01,  7.35120033e-04,  1.88436022e-03,
	         6.54135048e-01,  3.28634709e-01,  1.97367832e-01,
	         2.03872412e-01,  6.51154637e-01,  3.23619336e-01,
	         1.99623540e-01,  4.03123289e-01,  7.31922328e-01,
	         7.28641093e-01,  8.52419436e-01,  8.57252657e-01,
	         9.79328752e-01],
	       [ 0.00000000e+00,  5.19142509e-01,  5.19668162e-01,
	         1.75685704e-01,  7.31595099e-01,  9.41754699e-01,
	         5.17921686e-01,  5.19749045e-01,  1.76111132e-01,
	         1.75959527e-01,  7.30204821e-01,  9.40912068e-01,
	         5.18818855e-01,  5.19871831e-01,  1.75626218e-01,
	         1.75781012e-01,  7.31228113e-01,  9.41592932e-01,
	         5.18889546e-01,  5.19986689e-01, -1.75376430e-01,
	         1.75468266e-01,  7.30292439e-01,  9.41025853e-01,
	         9.40698802e-01,  7.31595099e-01,  1.75805867e-01,
	         5.18680155e-01,  5.18867075e-01,  1.76136464e-01,
	         1.75544515e-01],
	       [ 0.00000000e+00, -8.30150485e-01, -8.31177771e-01,
	        -8.98714185e-01, -5.49781978e-01, -2.69072831e-01,
	        -5.85563667e-02, -4.41759408e-01, -8.96385729e-01,
	        -6.57483995e-01,  2.08912745e-01,  1.03785343e-01,
	         7.91146100e-01,  5.54300666e-01,  1.05790980e-01,
	         4.89111096e-01,  6.82132661e-01,  3.36748034e-01,
	         5.50328016e-01,  7.88424432e-01, -9.64514911e-01,
	         9.63144302e-01,  2.06568673e-01,  9.86959785e-02,
	        -2.74292260e-01, -5.49781978e-01, -6.58317506e-01,
	        -4.47272897e-01, -6.44830465e-02,  4.83832479e-01,
	         1.00495256e-01]])
	#save file
	np.savetxt(temp_in1,bvec_test_1)

	#write cases for test2 Test non proper dimensions
	bvec_test_2 = np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
	       [-2.03325346e-01,  5.19142509e-01, -8.30150485e-01],
	       [ 1.97708115e-01,  5.19668162e-01, -8.31177771e-01],
	       [ 4.01805073e-01,  1.75685704e-01, -8.98714185e-01],
	       [-4.03123289e-01,  7.31595099e-01, -5.49781978e-01],
	       [-2.01737255e-01,  9.41754699e-01, -2.69072831e-01],
	       [-8.53421509e-01,  5.17921686e-01, -5.85563667e-02],
	       [-7.31238365e-01,  5.19749045e-01, -4.41759408e-01],
	       [-4.06789303e-01,  1.76111132e-01, -8.96385729e-01],
	       [-7.32634306e-01,  1.75959527e-01, -6.57483995e-01],
	       [-6.50504768e-01,  7.30204821e-01,  2.08912745e-01],
	       [-3.22355449e-01,  9.40912068e-01,  1.03785343e-01],
	       [-3.23905617e-01,  5.18818855e-01,  7.91146100e-01],
	       [-6.49987698e-01,  5.19871831e-01,  5.54300666e-01],
	       [-9.78756189e-01,  1.75626218e-01,  1.05790980e-01],
	       [-8.54325116e-01,  1.75781012e-01,  4.89111096e-01],
	       [ 7.35120033e-04,  7.31228113e-01,  6.82132661e-01],
	       [ 1.88436022e-03,  9.41592932e-01,  3.36748034e-01],
	       [ 6.54135048e-01,  5.18889546e-01,  5.50328016e-01],
	       [ 3.28634709e-01,  5.19986689e-01,  7.88424432e-01],
	       [ 1.97367832e-01, -1.75376430e-01, -9.64514911e-01],
	       [ 2.03872412e-01,  1.75468266e-01,  9.63144302e-01],
	       [ 6.51154637e-01,  7.30292439e-01,  2.06568673e-01],
	       [ 3.23619336e-01,  9.41025853e-01,  9.86959785e-02],
	       [ 1.99623540e-01,  9.40698802e-01, -2.74292260e-01],
	       [ 4.03123289e-01,  7.31595099e-01, -5.49781978e-01],
	       [ 7.31922328e-01,  1.75805867e-01, -6.58317506e-01],
	       [ 7.28641093e-01,  5.18680155e-01, -4.47272897e-01],
	       [ 8.52419436e-01,  5.18867075e-01, -6.44830465e-02],
	       [ 8.57252657e-01,  1.76136464e-01,  4.83832479e-01],
	       [ 9.79328752e-01,  1.75544515e-01,  1.00495256e-01]])
	#save file
	np.savetxt(temp_in2,bvec_test_2)

	# #desired ouput
	output = np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
								 [-2.03325349e-01,  5.19142517e-01, -8.30150498e-01],
								 [ 1.97708117e-01,  5.19668166e-01, -8.31177778e-01],
								 [ 4.01805079e-01,  1.75685707e-01, -8.98714199e-01],
								 [-4.03123289e-01,  7.31595100e-01, -5.49781978e-01],
								 [-2.01737253e-01, 9.41754689e-01, -2.69072828e-01],
								 [-8.53421512e-01,  5.17921688e-01, -5.85563669e-02],
								 [-7.31238368e-01,  5.19749047e-01, -4.41759410e-01],
								 [-4.06789294e-01, 1.76111128e-01, -8.96385710e-01],
								 [-7.32634311e-01,  1.75959528e-01, -6.57484000e-01],
								 [-6.50504746e-01,  7.30204796e-01, 2.08912738e-01],
								 [-3.22355457e-01,  9.40912090e-01,  1.03785345e-01],
								 [-3.23905616e-01,  5.18818854e-01,  7.91146098e-01],
								 [-6.49987712e-01,  5.19871842e-01,  5.54300678e-01],
								 [-9.78756200e-01,  1.75626220e-01,  1.05790981e-01],
								 [-8.54325102e-01,  1.75781009e-01,  4.89111088e-01],
								 [ 7.35120011e-04,  7.31228091e-01,  6.82132640e-01],
								 [ 1.88436018e-03,  9.41592914e-01,  3.36748027e-01],
								 [ 6.54135065e-01,  5.18889560e-01,  5.50328031e-01],
								 [ 3.28634707e-01,  5.19986685e-01,  7.88424427e-01],
								 [ 1.97367835e-01, -1.75376433e-01, -9.64514927e-01],
								 [ 2.03872410e-01,  1.75468264e-01,  9.63144293e-01],
								 [ 6.51154629e-01,  7.30292430e-01,  2.06568670e-01],
								 [ 3.23619332e-01,  9.41025840e-01,  9.86959772e-02],
								 [ 1.99623536e-01,  9.40698784e-01, -2.74292255e-01],
								 [ 4.03123289e-01,  7.31595100e-01, -5.49781978e-01],
								 [ 7.31922351e-01,  1.75805873e-01, -6.58317527e-01],
								 [ 7.28641097e-01,  5.18680158e-01, -4.47272899e-01],
								 [ 8.52419436e-01,  5.18867075e-01, -6.44830465e-02],
								 [ 8.57252640e-01,  1.76136461e-01,  4.83832469e-01],
								 [ 9.79328763e-01,  1.75544517e-01,  1.00495257e-01]])

	#run through data
	mgp.rescale_bvec(str(temp_in1),str(temp_out1))
	mgp.rescale_bvec(str(temp_in2),str(temp_out2))
	#open data 
	test1 = np.loadtxt(str(temp_out1))
	test2 = np.loadtxt(str(temp_out2))

	assert np.allclose(test1,output) 
	assert np.allclose(test2,output)


