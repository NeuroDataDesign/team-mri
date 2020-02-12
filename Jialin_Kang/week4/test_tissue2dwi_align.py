import warnings

warnings.simplefilter("ignore")
import os
import nibabel as nib
import numpy as np
from nilearn.image import load_img, math_img
from ndmg.utils import gen_utils as mgu
from ndmg.utils import reg_utils as mgru

import warnings

warnings.simplefilter("ignore")
from bids import BIDSLayout
import re
from itertools import product
import boto3
from ndmg.utils import gen_utils as mgu
import os

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Standard Library
import os
import shutil
import time
from datetime import datetime
from subprocess import Popen

# External Packages
import numpy as np
import nibabel as nib
from dipy.tracking.streamline import Streamlines

# Internal Imports
import ndmg
from ndmg import preproc as mgp
from ndmg.utils import gen_utils as mgu
from ndmg.register import gen_reg as mgr
from ndmg.track import gen_track as mgt
from ndmg.graph import gen_graph as mgg
from ndmg.utils.bids_utils import name_resource


dwi = '/mnt/d/Downloads/neurodatadesign/BNU1/sub-0025864/ses-1/dwi/sub-0025864_ses-1_dwi.nii.gz'
t1w = '/mnt/d/Downloads/neurodatadesign/BNU1//sub-0025864/ses-1/dwi/sub-0025864_ses-1_dwi.bval'
atlas = 'desikan'
outdir = '/mnt/d/Downloads/neurodatadesign/ndmg_outputs'

fbval = '/mnt/d/Downloads/neurodatadesign/ndmg_outputs/dwi/preproc/bval.bval'
fbvec = '/mnt/d/Downloads/neurodatadesign/ndmg_outputs/dwi/preproc/bvec.bvec'
dwi_prep = '/mnt/d/Downloads/neurodatadesign/ndmg_outputs/dwi/preproc/eddy_corrected_data_reor_RAS_res.nii.gz'
# (dwi_file, outdir) = namer.dirs["output"]["prep_dwi"]
nodif_B0 = '/mnt/d/Downloads/neurodatadesign/ndmg_outputs/dwi/preproc/nodif_B0.nii.gz'
nodif_B0_mask = '/mnt/d/Downloads/neurodatadesign/ndmg_outputs/dwi/preproc/nodif_B0_bet_mask.nii.gz'
t1w_in = '/mnt/d/Downloads/neurodatadesign/BNU1/sub-0025864/ses-1/dwi/sub-0025864_ses-1_dwi.bval'
vox_size = '2mm'

paths = {
    "prep_dwi": "dwi/preproc",
    "prep_anat": "anat/preproc",
    "reg_anat": "anat/registered",
    "fiber": "dwi/fiber",
    "tensor": "dwi/tensor",
    "conn": "dwi/roi-connectomes"
}
labels = ['/ndmg_atlases/atlases/label/Human/desikan_space-MNI152NLin6_res-2x2x2.nii.gz']
label_dirs = ["conn"]


def flatten(current, result=[]):
    """Flatten a folder heirarchy

    Parameters
    ----------
    current : dict
        path to directory you want to flatten
    result : list, optional
        Default is []

    Returns
    -------
    list
        All new directories created by flattening the current directory
    """
    if isinstance(current, dict):
        for key in current:
            flatten(current[key], result)
    else:
        result.append(current)
    return result

class dmri_reg(object):
    """Class containing relevant paths and class methods for analysing tractography

    Parameters
    ----------
    namer : name_resource
        name_resource variable containing relevant directory tree information
    nodif_B0 : str
        path to mean b0 image
    nodif_B0_mask : str
        path to mean b0 mask (nodif_B0....nii.gz)
    t1w_in : str
        path to t1w file
    vox_size : str
        voxel resolution ('2mm' or '1mm')
    simple : bool
        Whether you want to attempt non-linear registration when transforming between mni, t1w, and dwi space

    Raises
    ------
    ValueError
        FSL atlas for ventricle reference not found
    """

    def __init__(self, namer, nodif_B0, nodif_B0_mask, t1w_in, vox_size, simple):
        import os.path as op

        if os.path.isdir("/ndmg_atlases"):
            # in docker
            atlas_dir = "/ndmg_atlases"
        else:
            # local
            atlas_dir = op.expanduser("~") + "/.ndmg/ndmg_atlases"
        try:
            FSLDIR = os.environ["FSLDIR"]
        except KeyError:
            print("FSLDIR environment variable not set!")

        if vox_size == "2mm":
            vox_dims = "2x2x2"
        elif vox_size == "1mm":
            vox_dims = "1x1x1"

        self.simple = simple
        self.nodif_B0 = nodif_B0
        self.nodif_B0_mask = nodif_B0_mask
        self.t1w = t1w_in
        self.vox_size = vox_size
        self.t1w_name = "t1w"
        self.dwi_name = "dwi"
        self.namer = namer
        self.t12mni_xfm_init = "{}/xfm_t1w2mni_init.mat".format(
            self.namer.dirs["tmp"]["reg_m"]
        )
        self.mni2t1_xfm_init = "{}/xfm_mni2t1w_init.mat".format(
            self.namer.dirs["tmp"]["reg_m"]
        )
        self.t12mni_xfm = "{}/xfm_t1w2mni.mat".format(self.namer.dirs["tmp"]["reg_m"])
        self.mni2t1_xfm = "{}/xfm_mni2t1.mat".format(self.namer.dirs["tmp"]["reg_m"])
        self.mni2t1w_warp = "{}/mni2t1w_warp.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"]
        )
        self.warp_t1w2mni = "{}/warp_t12mni.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"]
        )
        self.t1w2dwi = "{}/{}_in_dwi.nii.gz".format(
            self.namer.dirs["output"]["reg_anat"], self.t1w_name
        )
        self.t1_aligned_mni = "{}/{}_aligned_mni.nii.gz".format(
            self.namer.dirs["output"]["prep_anat"], self.t1w_name
        )
        self.t1w_brain = "{}/{}_brain.nii.gz".format(
            self.namer.dirs["output"]["prep_anat"], self.t1w_name
        )
        self.dwi2t1w_xfm = "{}/dwi2t1w_xfm.mat".format(self.namer.dirs["tmp"]["reg_m"])
        self.t1w2dwi_xfm = "{}/t1w2dwi_xfm.mat".format(self.namer.dirs["tmp"]["reg_m"])
        self.t1w2dwi_bbr_xfm = "{}/t1w2dwi_bbr_xfm.mat".format(
            self.namer.dirs["tmp"]["reg_m"]
        )
        self.dwi2t1w_bbr_xfm = "{}/dwi2t1w_bbr_xfm.mat".format(
            self.namer.dirs["tmp"]["reg_m"]
        )
        self.t1wtissue2dwi_xfm = "{}/t1wtissue2dwi_xfm.mat".format(
            self.namer.dirs["tmp"]["reg_m"]
        )
        self.xfm_atlas2t1w_init = "{}/{}_xfm_atlas2t1w_init.mat".format(
            self.namer.dirs["tmp"]["reg_m"], self.t1w_name
        )
        self.xfm_atlas2t1w = "{}/{}_xfm_atlas2t1w.mat".format(
            self.namer.dirs["tmp"]["reg_m"], self.t1w_name
        )
        self.temp2dwi_xfm = "{}/{}_xfm_temp2dwi.mat".format(
            self.namer.dirs["tmp"]["reg_m"], self.dwi_name
        )
        self.input_mni = "%s%s%s%s" % (
            FSLDIR,
            "/data/standard/MNI152_T1_",
            vox_size,
            "_brain.nii.gz",
        )
        self.temp2dwi_xfm = "{}/{}_xfm_temp2dwi.mat".format(
            self.namer.dirs["tmp"]["reg_m"], self.dwi_name
        )
        self.map_path = "{}/{}_seg".format(
            self.namer.dirs["output"]["prep_anat"], self.t1w_name
        )
        self.wm_mask = "{}/{}_wm.nii.gz".format(
            self.namer.dirs["output"]["prep_anat"], self.t1w_name
        )
        self.wm_mask_thr = "{}/{}_wm_thr.nii.gz".format(
            self.namer.dirs["output"]["prep_anat"], self.t1w_name
        )
        self.wm_edge = "{}/{}_wm_edge.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"], self.t1w_name
        )
        self.csf_mask = "{}/{}_csf.nii.gz".format(
            self.namer.dirs["output"]["prep_anat"], self.t1w_name
        )
        self.gm_mask = "{}/{}_gm.nii.gz".format(
            self.namer.dirs["output"]["prep_anat"], self.t1w_name
        )
        self.xfm_roi2mni_init = "{}/roi_2_mni.mat".format(
            self.namer.dirs["tmp"]["reg_m"]
        )
        self.lvent_out_file = "{}/LVentricle.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"]
        )
        self.rvent_out_file = "{}/RVentricle.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"]
        )
        self.csf_mask_dwi = "{}/{}_csf_mask_dwi.nii.gz".format(
            self.namer.dirs["output"]["reg_anat"], self.t1w_name
        )
        self.gm_in_dwi = "{}/{}_gm_in_dwi.nii.gz".format(
            self.namer.dirs["output"]["reg_anat"], self.t1w_name
        )
        self.wm_in_dwi = "{}/{}_wm_in_dwi.nii.gz".format(
            self.namer.dirs["output"]["reg_anat"], self.t1w_name
        )
        self.csf_mask_dwi_bin = "{}/{}_csf_mask_dwi_bin.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"], self.t1w_name
        )
        self.gm_in_dwi_bin = "{}/{}_gm_in_dwi_bin.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"], self.t1w_name
        )
        self.wm_in_dwi_bin = "{}/{}_wm_in_dwi_bin.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"], self.t1w_name
        )
        self.vent_mask_dwi = "{}/{}_vent_mask_dwi.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"], self.t1w_name
        )
        self.vent_csf_in_dwi = "{}/{}_vent_csf_in_dwi.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"], self.t1w_name
        )
        self.vent_mask_mni = "{}/vent_mask_mni.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"]
        )
        self.vent_mask_t1w = "{}/vent_mask_t1w.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"]
        )

        self.input_mni = "%s%s%s%s" % (
            FSLDIR,
            "/data/standard/MNI152_T1_",
            vox_size,
            "_brain.nii.gz",
        )
        self.input_mni_mask = "%s%s%s%s" % (
            FSLDIR,
            "/data/standard/MNI152_T1_",
            vox_size,
            "_brain_mask.nii.gz",
        )
        self.wm_gm_int_in_dwi = "{}/{}_wm_gm_int_in_dwi.nii.gz".format(
            namer.dirs["output"]["reg_anat"], self.t1w_name
        )
        self.wm_gm_int_in_dwi_bin = "{}/{}_wm_gm_int_in_dwi_bin.nii.gz".format(
            namer.dirs["output"]["reg_anat"], self.t1w_name
        )
        self.input_mni_sched = "%s%s" % (FSLDIR, "/etc/flirtsch/T1_2_MNI152_2mm.cnf")
        self.mni_atlas = "%s%s%s%s" % (
            FSLDIR,
            "/data/atlases/HarvardOxford/HarvardOxford-sub-prob-",
            vox_size,
            ".nii.gz",
        )
        self.mni_vent_loc = (
                atlas_dir
                + "/atlases/mask/HarvardOxford-thr25_space-MNI152NLin6_variant-lateral-ventricles_res-"
                + vox_dims
                + "_descr-brainmask.nii.gz"
        )
        self.corpuscallosum = (
                atlas_dir + "/atlases/mask/CorpusCallosum_res_" + vox_size + ".nii.gz"
        )
        self.corpuscallosum_mask_t1w = "{}/{}_corpuscallosum.nii.gz".format(
            self.namer.dirs["output"]["reg_anat"], self.t1w_name
        )
        self.corpuscallosum_dwi = "{}/{}_corpuscallosum_dwi.nii.gz".format(
            self.namer.dirs["output"]["reg_anat"], self.t1w_name
        )

    def gen_tissue(self):
        """Extracts the brain from the raw t1w image (as indicated by self.t1w), uses it to create WM, GM, and CSF masks,
        reslices all 4 files to the target voxel resolution and extracts the white matter edge. Each mask is saved to
        location indicated by self.map_path
        """
        # BET needed for this, as afni 3dautomask only works on 4d volumes
        print("Extracting brain from raw T1w image...")
        mgru.t1w_skullstrip(self.t1w, self.t1w_brain)

        # Segment the t1w brain into probability maps
        self.maps = mgru.segment_t1w(self.t1w_brain, self.map_path)
        self.wm_mask = self.maps["wm_prob"]
        self.gm_mask = self.maps["gm_prob"]
        self.csf_mask = self.maps["csf_prob"]

        self.t1w_brain = mgu.match_target_vox_res(
            self.t1w_brain, self.vox_size, self.namer, sens="t1w"
        )
        self.wm_mask = mgu.match_target_vox_res(
            self.wm_mask, self.vox_size, self.namer, sens="t1w"
        )
        self.gm_mask = mgu.match_target_vox_res(
            self.gm_mask, self.vox_size, self.namer, sens="t1w"
        )
        self.csf_mask = mgu.match_target_vox_res(
            self.csf_mask, self.vox_size, self.namer, sens="t1w"
        )

        # Threshold WM to binary in dwi space
        self.t_img = load_img(self.wm_mask)
        self.mask = math_img("img > 0.2", img=self.t_img)
        self.mask.to_filename(self.wm_mask_thr)

        # Extract wm edge
        cmd = (
                "fslmaths "
                + self.wm_mask_thr
                + " -edge -bin -mas "
                + self.wm_mask_thr
                + " "
                + self.wm_edge
        )
        os.system(cmd)
        print(cmd)

        return

    def t1w2dwi_align(self):
        """Alignment from t1w to mni, making t1w_mni, and t1w_mni to dwi. A function to perform self alignment. Uses a local optimisation cost function to get the
        two images close, and then uses bbr to obtain a good alignment of brain boundaries. Assumes input dwi is already preprocessed and brain extracted.
        """

        # Create linear transform/ initializer T1w-->MNI
        mgru.align(
            self.t1w_brain,
            self.input_mni,
            xfm=self.t12mni_xfm_init,
            bins=None,
            interp="spline",
            out=None,
            dof=12,
            cost="mutualinfo",
            searchrad=True,
        )

        # Attempt non-linear registration of T1 to MNI template
        if self.simple is False:
            try:
                print("Running non-linear registration: T1w-->MNI ...")
                # Use FNIRT to nonlinearly align T1 to MNI template
                mgru.align_nonlinear(
                    self.t1w_brain,
                    self.input_mni,
                    xfm=self.t12mni_xfm_init,
                    out=self.t1_aligned_mni,
                    warp=self.warp_t1w2mni,
                    ref_mask=self.input_mni_mask,
                    config=self.input_mni_sched,
                )

                # Get warp from MNI -> T1
                mgru.inverse_warp(self.t1w_brain, self.mni2t1w_warp, self.warp_t1w2mni)

                # Get mat from MNI -> T1
                cmd = (
                        "convert_xfm -omat "
                        + self.mni2t1_xfm_init
                        + " -inverse "
                        + self.t12mni_xfm_init
                )
                print(cmd)
                os.system(cmd)

            except RuntimeError("Error: FNIRT failed!"):
                pass
        else:
            # Falling back to linear registration
            mgru.align(
                self.t1w_brain,
                self.input_mni,
                xfm=self.t12mni_xfm,
                init=self.t12mni_xfm_init,
                bins=None,
                dof=12,
                cost="mutualinfo",
                searchrad=True,
                interp="spline",
                out=self.t1_aligned_mni,
                sch=None,
            )

        # Align T1w-->DWI
        mgru.align(
            self.nodif_B0,
            self.t1w_brain,
            xfm=self.t1w2dwi_xfm,
            bins=None,
            interp="spline",
            dof=6,
            cost="mutualinfo",
            out=None,
            searchrad=True,
            sch=None,
        )
        cmd = "convert_xfm -omat " + self.dwi2t1w_xfm + " -inverse " + self.t1w2dwi_xfm
        print(cmd)
        os.system(cmd)

        if self.simple is False:
            # Flirt bbr
            try:
                print("Running FLIRT BBR registration: T1w-->DWI ...")
                mgru.align(
                    self.nodif_B0,
                    self.t1w_brain,
                    wmseg=self.wm_edge,
                    xfm=self.dwi2t1w_bbr_xfm,
                    init=self.dwi2t1w_xfm,
                    bins=256,
                    dof=7,
                    searchrad=True,
                    interp="spline",
                    out=None,
                    cost="bbr",
                    finesearch=5,
                    sch="${FSLDIR}/etc/flirtsch/bbr.sch",
                )
                cmd = (
                        "convert_xfm -omat "
                        + self.t1w2dwi_bbr_xfm
                        + " -inverse "
                        + self.dwi2t1w_bbr_xfm
                )
                os.system(cmd)

                # Apply the alignment
                mgru.align(
                    self.t1w_brain,
                    self.nodif_B0,
                    init=self.t1w2dwi_bbr_xfm,
                    xfm=self.t1wtissue2dwi_xfm,
                    bins=None,
                    interp="spline",
                    dof=7,
                    cost="mutualinfo",
                    out=self.t1w2dwi,
                    searchrad=True,
                    sch=None,
                )
            except RuntimeError("Error: FLIRT BBR failed!"):
                pass
        else:
            # Apply the alignment
            mgru.align(
                self.t1w_brain,
                self.nodif_B0,
                init=self.t1w2dwi_xfm,
                xfm=self.t1wtissue2dwi_xfm,
                bins=None,
                interp="spline",
                dof=6,
                cost="mutualinfo",
                out=self.t1w2dwi,
                searchrad=True,
                sch=None,
            )

        return

    def atlas2t1w2dwi_align(self, atlas, dsn=True):
        """alignment from atlas to t1w to dwi. A function to perform atlas alignmet. Tries nonlinear registration first, and if that fails, does a liner
        registration instead.
        Note: for this to work, must first have called t1w2dwi_align.

        Parameters
        ----------
        atlas : str
            path to atlas file you want to use
        dsn : bool, optional
            is your space for tractography native-dsn, by default True

        Returns
        -------
        str
            path to aligned atlas file
        """

        self.atlas = atlas
        self.atlas_name = self.atlas.split("/")[-1].split(".")[0]
        self.aligned_atlas_t1mni = "{}/{}_aligned_atlas_t1w_mni.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"], self.atlas_name
        )
        self.aligned_atlas_skull = "{}/{}_aligned_atlas_skull.nii.gz".format(
            self.namer.dirs["tmp"]["reg_a"], self.atlas_name
        )
        self.dwi_aligned_atlas = "{}/{}_aligned_atlas.nii.gz".format(
            self.namer.dirs["output"]["reg_anat"], self.atlas_name
        )
        # self.dwi_aligned_atlas_mask = "{}/{}_aligned_atlas_mask.nii.gz".format(self.namer.dirs['tmp']['reg_a'], self.atlas_name)

        mgru.align(
            self.atlas,
            self.t1_aligned_mni,
            init=None,
            xfm=None,
            out=self.aligned_atlas_t1mni,
            dof=12,
            searchrad=True,
            interp="nearestneighbour",
            cost="mutualinfo",
        )

        if (self.simple is False) and (dsn is False):
            try:
                # Apply warp resulting from the inverse of T1w-->MNI created earlier
                mgru.apply_warp(
                    self.t1w_brain,
                    self.aligned_atlas_t1mni,
                    self.aligned_atlas_skull,
                    warp=self.mni2t1w_warp,
                    interp="nn",
                    sup=True,
                )

                # Apply transform to dwi space
                mgru.align(
                    self.aligned_atlas_skull,
                    self.nodif_B0,
                    init=self.t1wtissue2dwi_xfm,
                    xfm=None,
                    out=self.dwi_aligned_atlas,
                    dof=6,
                    searchrad=True,
                    interp="nearestneighbour",
                    cost="mutualinfo",
                )
            except:
                print(
                    "Warning: Atlas is not in correct dimensions, or input is low quality,\nusing linear template registration."
                )
                # Create transform to align atlas to T1w using flirt
                mgru.align(
                    self.atlas,
                    self.t1w_brain,
                    xfm=self.xfm_atlas2t1w_init,
                    init=None,
                    bins=None,
                    dof=6,
                    cost="mutualinfo",
                    searchrad=True,
                    interp="spline",
                    out=None,
                    sch=None,
                )
                mgru.align(
                    self.atlas,
                    self.t1_aligned_mni,
                    xfm=self.xfm_atlas2t1w,
                    out=None,
                    dof=6,
                    searchrad=True,
                    bins=None,
                    interp="spline",
                    cost="mutualinfo",
                    init=self.xfm_atlas2t1w_init,
                )

                # Combine our linear transform from t1w to template with our transform from dwi to t1w space to get a transform from atlas ->(-> t1w ->)-> dwi
                mgru.combine_xfms(
                    self.xfm_atlas2t1w, self.t1wtissue2dwi_xfm, self.temp2dwi_xfm
                )

                # Apply linear transformation from template to dwi space
                mgru.applyxfm(
                    self.nodif_B0, self.atlas, self.temp2dwi_xfm, self.dwi_aligned_atlas
                )
        elif dsn is False:
            # Create transform to align atlas to T1w using flirt
            mgru.align(
                self.atlas,
                self.t1w_brain,
                xfm=self.xfm_atlas2t1w_init,
                init=None,
                bins=None,
                dof=6,
                cost="mutualinfo",
                searchrad=None,
                interp="spline",
                out=None,
                sch=None,
            )
            mgru.align(
                self.atlas,
                self.t1w_brain,
                xfm=self.xfm_atlas2t1w,
                out=None,
                dof=6,
                searchrad=True,
                bins=None,
                interp="spline",
                cost="mutualinfo",
                init=self.xfm_atlas2t1w_init,
            )

            # Combine our linear transform from t1w to template with our transform from dwi to t1w space to get a transform from atlas ->(-> t1w ->)-> dwi
            mgru.combine_xfms(
                self.xfm_atlas2t1w, self.t1wtissue2dwi_xfm, self.temp2dwi_xfm
            )

            # Apply linear transformation from template to dwi space
            mgru.applyxfm(
                self.nodif_B0, self.atlas, self.temp2dwi_xfm, self.dwi_aligned_atlas
            )
        else:
            pass

        # Set intensities to int
        if dsn is False:
            self.atlas_img = nib.load(self.dwi_aligned_atlas)
        else:
            self.atlas_img = nib.load(self.aligned_atlas_t1mni)
        self.atlas_data = np.around(self.atlas_img.get_data()).astype("int16")
        node_num = len(np.unique(self.atlas_data))
        self.atlas_data[self.atlas_data > node_num] = 0

        t_img = load_img(self.wm_gm_int_in_dwi)
        mask = math_img("img > 0", img=t_img)
        mask.to_filename(self.wm_gm_int_in_dwi_bin)

        if dsn is False:
            nib.save(
                nib.Nifti1Image(
                    self.atlas_data.astype(np.int32),
                    affine=self.atlas_img.affine,
                    header=self.atlas_img.header,
                ),
                self.dwi_aligned_atlas,
            )
            return self.dwi_aligned_atlas
        else:
            nib.save(
                nib.Nifti1Image(
                    self.atlas_data.astype(np.int32),
                    affine=self.atlas_img.affine,
                    header=self.atlas_img.header,
                ),
                self.aligned_atlas_t1mni,
            )
            return self.aligned_atlas_t1mni

    def tissue2dwi_align(self):
        """alignment of ventricle and CC ROI's from MNI space --> dwi and CC and CSF from T1w space --> dwi
        A function to generate and perform dwi space alignment of avoidance/waypoint masks for tractography.
        First creates ventricle and CC ROI. Then creates transforms from stock MNI template to dwi space.
        NOTE: for this to work, must first have called both t1w2dwi_align and atlas2t1w2dwi_align.

        Raises
        ------
        ValueError
            Raised if FSL atlas for ventricle reference not found
        """

        # Create MNI-space ventricle mask
        print("Creating MNI-space ventricle ROI...")
        if not os.path.isfile(self.mni_atlas):
            raise ValueError("FSL atlas for ventricle reference not found!")
        cmd = "fslmaths " + self.mni_vent_loc + " -thr 0.1 -bin " + self.mni_vent_loc
        os.system(cmd)

        cmd = "fslmaths " + self.corpuscallosum + " -bin " + self.corpuscallosum
        os.system(cmd)

        cmd = (
                "fslmaths "
                + self.corpuscallosum
                + " -sub "
                + self.mni_vent_loc
                + " -bin "
                + self.corpuscallosum
        )
        os.system(cmd)

        # Create a transform from the atlas onto T1w. This will be used to transform the ventricles to dwi space.
        mgru.align(
            self.mni_atlas,
            self.input_mni,
            xfm=self.xfm_roi2mni_init,
            init=None,
            bins=None,
            dof=6,
            cost="mutualinfo",
            searchrad=True,
            interp="spline",
            out=None,
        )

        # Create transform to align roi to mni and T1w using flirt
        mgru.applyxfm(
            self.input_mni, self.mni_vent_loc, self.xfm_roi2mni_init, self.vent_mask_mni
        )

        if self.simple is False:
            # Apply warp resulting from the inverse MNI->T1w created earlier
            mgru.apply_warp(
                self.t1w_brain,
                self.vent_mask_mni,
                self.vent_mask_t1w,
                warp=self.mni2t1w_warp,
                interp="nn",
                sup=True,
            )

            # Apply warp resulting from the inverse MNI->T1w created earlier
            mgru.apply_warp(
                self.t1w_brain,
                self.corpuscallosum,
                self.corpuscallosum_mask_t1w,
                warp=self.mni2t1w_warp,
                interp="nn",
                sup=True,
            )

        # Applyxfm tissue maps to dwi space
        mgru.applyxfm(
            self.nodif_B0,
            self.vent_mask_t1w,
            self.t1wtissue2dwi_xfm,
            self.vent_mask_dwi,
        )
        mgru.applyxfm(
            self.nodif_B0,
            self.corpuscallosum_mask_t1w,
            self.t1wtissue2dwi_xfm,
            self.corpuscallosum_dwi,
        )
        mgru.applyxfm(
            self.nodif_B0, self.csf_mask, self.t1wtissue2dwi_xfm, self.csf_mask_dwi
        )
        mgru.applyxfm(
            self.nodif_B0, self.gm_mask, self.t1wtissue2dwi_xfm, self.gm_in_dwi
        )
        mgru.applyxfm(
            self.nodif_B0, self.wm_mask, self.t1wtissue2dwi_xfm, self.wm_in_dwi
        )

        # Threshold WM to binary in dwi space
        thr_img = nib.load(self.wm_in_dwi)
        thr_img.get_data()[thr_img.get_data() < 0.15] = 0
        nib.save(thr_img, self.wm_in_dwi_bin)

        # Threshold GM to binary in dwi space
        thr_img = nib.load(self.gm_in_dwi)
        thr_img.get_data()[thr_img.get_data() < 0.15] = 0
        nib.save(thr_img, self.gm_in_dwi_bin)

        # Threshold CSF to binary in dwi space
        thr_img = nib.load(self.csf_mask_dwi)
        thr_img.get_data()[thr_img.get_data() < 0.99] = 0
        nib.save(thr_img, self.csf_mask_dwi)

        # Threshold WM to binary in dwi space
        self.t_img = load_img(self.wm_in_dwi_bin)
        self.mask = math_img("img > 0", img=self.t_img)
        self.mask.to_filename(self.wm_in_dwi_bin)

        # Threshold GM to binary in dwi space
        self.t_img = load_img(self.gm_in_dwi_bin)
        self.mask = math_img("img > 0", img=self.t_img)
        self.mask.to_filename(self.gm_in_dwi_bin)

        # Threshold CSF to binary in dwi space
        self.t_img = load_img(self.csf_mask_dwi)
        self.mask = math_img("img > 0", img=self.t_img)
        self.mask.to_filename(self.csf_mask_dwi_bin)

        # Create ventricular CSF mask
        print("Creating ventricular CSF mask...")
        cmd = (
                "fslmaths "
                + self.vent_mask_dwi
                + " -kernel sphere 10 -ero -bin "
                + self.vent_mask_dwi
        )
        os.system(cmd)
        print("Creating Corpus Callosum mask...")
        cmd = (
                "fslmaths "
                + self.corpuscallosum_dwi
                + " -mas "
                + self.wm_in_dwi_bin
                + " -bin "
                + self.corpuscallosum_dwi
        )
        os.system(cmd)
        cmd = (
                "fslmaths "
                + self.csf_mask_dwi
                + " -add "
                + self.vent_mask_dwi
                + " -bin "
                + self.vent_csf_in_dwi
        )
        os.system(cmd)

        # Create gm-wm interface image
        cmd = (
                "fslmaths "
                + self.gm_in_dwi_bin
                + " -mul "
                + self.wm_in_dwi_bin
                + " -add "
                + self.corpuscallosum_dwi
                + " -sub "
                + self.vent_csf_in_dwi
                + " -mas "
                + self.nodif_B0_mask
                + " -bin "
                + self.wm_gm_int_in_dwi
        )
        os.system(cmd)

        return

class name_resource(object):
    """
    A class for naming derivatives under the BIDs spec.
    Parameters
    ----------
    modf : str
        Path to subject MRI (dwi or func) data to be analyzed
    t1wf : str
        Path to subject t1w anatomical data
    tempf : str
        Path to atlas file(s) to be used during analysis
    opath : str
        Path to output directory
    """

    def __init__(self, modf, t1wf, tempf, opath):
        """__init__ containing relevant BIDS specified paths for relevant data
        """
        self.__subi__ = os.path.basename(modf).split(".")[0]
        self.__anati__ = os.path.basename(t1wf).split(".")[0]
        self.__suball__ = ""
        self.__sub__ = re.search(r"(sub-)(?!.*sub-).*?(?=[_])", modf)
        if self.__sub__:
            self.__sub__ = self.__sub__.group()
            self.__suball__ = "sub-{}".format(self.__sub__)
        self.__ses__ = re.search(r"(ses-)(?!.*ses-).*?(?=[_])", modf)
        if self.__ses__:
            self.__ses__ = self.__ses__.group()
            self.__suball__ = self.__suball__ + "_ses-{}".format(self.__ses__)
        self.__run__ = re.search(r"(run-)(?!.*run-).*?(?=[_])", modf)
        if self.__run__:
            self.__run__ = self.__run__.group()
            self.__suball__ = self.__suball__ + "_run-{}".format(self.__run__)
        self.__task__ = re.search(r"(task-)(?!.*task-).*?(?=[_])", modf)
        if self.__task__:
            self.__task__ = self.__task__.group()
            self.__suball__ = self.__suball__ + "_run-{}".format(self.__task__)
        self.__temp__ = os.path.basename(tempf).split(".")[0]
        self.__space__ = re.split(r"[._]", self.__temp__)[0]
        self.__res__ = re.search(r"(res-)(?!.*res-).*?(?=[_])", tempf)
        if self.__res__:
            self.__res__ = self.__res__.group()
        self.__basepath__ = opath
        self.__outdir__ = self._get_outdir()
        return

    def add_dirs(self, paths, labels, label_dirs):
        """
        creates tmp and permanent directories for the desired suffixes.
        **Positional Arguments:
            - paths:
                - a dictionary of keys to suffix directories desired.
        """
        self.dirs = {}
        if not isinstance(labels, list):
            labels = [labels]
        dirtypes = ["output", "tmp", "qa"]
        for dirt in dirtypes:
            olist = [self.get_outdir()]
            self.dirs[dirt] = {}
            if dirt in ["tmp", "qa"]:
                olist = olist + [dirt] + self.get_sub_info()
            self.dirs[dirt]["base"] = os.path.join(*olist)
            for kwd, path in paths.items():
                newdir = os.path.join(*[self.dirs[dirt]["base"], path])
                if kwd in label_dirs:  # levels with label granularity
                    self.dirs[dirt][kwd] = {}
                    for label in labels:
                        labname = self.get_label(label)
                        self.dirs[dirt][kwd][labname] = os.path.join(newdir, labname)
                else:
                    self.dirs[dirt][kwd] = newdir
        newdirs = flatten(self.dirs, [])
        cmd = "mkdir -p {}".format(" ".join(newdirs))
        mgu.execute_cmd(cmd)  # make the directories
        return

    def add_dirs_dwi(namer, paths, labels, label_dirs):
        """Creates tmp and permanent directories for the desired suffixes

        Parameters
        ----------
        namer : name_resource
            varibale of the name_resource class created by name_resource() containing path and settings information for the desired run. It includes: subject, anatomical scan, session, run number, task, resolution, output directory
        paths : dict
            a dictionary of keys to suffix directories
        labels : list
            path to desired atlas labeling file
        label_dirs : list
            label directories
        """

        namer.dirs = {}
        if not isinstance(labels, list):
            labels = [labels]
        dirtypes = ["output"]
        for dirt in dirtypes:
            olist = [namer.get_outdir()]
            namer.dirs[dirt] = {}
            if dirt in ["tmp"]:
                olist = olist + [dirt]
            namer.dirs[dirt]["base"] = os.path.join(*olist)
            for kwd, path in paths.items():
                newdir = os.path.join(*[namer.dirs[dirt]["base"], path])
                if kwd in label_dirs:  # levels with label granularity
                    namer.dirs[dirt][kwd] = {}
                    for label in labels:
                        labname = namer.get_label(label)
                        namer.dirs[dirt][kwd][labname] = os.path.join(newdir, labname)
                else:
                    namer.dirs[dirt][kwd] = newdir
        namer.dirs["tmp"] = {}
        namer.dirs["tmp"]["base"] = namer.get_outdir() + "/tmp"
        namer.dirs["tmp"]["reg_a"] = namer.dirs["tmp"]["base"] + "/reg_a"
        namer.dirs["tmp"]["reg_m"] = namer.dirs["tmp"]["base"] + "/reg_m"
        namer.dirs["qa"] = {}
        namer.dirs["qa"]["base"] = namer.get_outdir() + "/qa"
        namer.dirs["qa"]["adjacency"] = namer.dirs["qa"]["base"] + "/adjacency"
        namer.dirs["qa"]["fibers"] = namer.dirs["qa"]["base"] + "/fibers"
        namer.dirs["qa"]["graphs"] = namer.dirs["qa"]["base"] + "/graphs"
        namer.dirs["qa"]["graphs_plotting"] = (
                namer.dirs["qa"]["base"] + "/graphs_plotting"
        )
        namer.dirs["qa"]["mri"] = namer.dirs["qa"]["base"] + "/mri"
        namer.dirs["qa"]["reg"] = namer.dirs["qa"]["base"] + "/reg"
        namer.dirs["qa"]["tensor"] = namer.dirs["qa"]["base"] + "/tensor"
        newdirs = flatten(namer.dirs, [])
        cmd = "mkdir -p {}".format(" ".join(newdirs))
        mgu.execute_cmd(cmd)  # make the directories
        return

    def _get_outdir(self):
        """Called by constructor to initialize the output directory

        Returns
        -------
        list
            path to output directory
        """

        olist = [self.__basepath__]
        # olist.append(self.__sub__)
        # if self.__ses__:
        #    olist.append(self.__ses__)
        return os.path.join(*olist)

    def get_outdir(self):
        """Returns the base output directory for a particular subject and appropriate granularity.

        Returns
        -------
        str
            output directory
        """

        return self.__outdir__

    def get_template_info(self):
        """
        returns the formatted spatial information associated with a template.-
        """
        return "space-{}_{}".format(self.__space__, self.__res__)

    def get_template_space(self):
        return "space-{}_{}".format(self.__space__, self.__res__)

    def get_label(self, label):
        """
        return the formatted label information for the parcellation.
        """
        return mgu.get_filename(label)
        # return "label-{}".format(re.split(r'[._]',
        #                         os.path.basename(label))[0])

    def name_derivative(self, folder, derivative):
        """Creates derivative output file paths using os.path.join

        Parameters
        ----------
        folder : str
            Path of directory that you want the derivative file name appended too
        derivative : str
            The name of the file to be produced

        Returns
        -------
        str
            Derivative output file path
        """

        return os.path.join(*[folder, derivative])

    def get_mod_source(self):
        return self.__subi__

    def get_anat_source(self):
        return self.__anati__

    def get_sub_info(self):
        olist = []
        if self.__sub__:
            olist.append(self.__sub__)
        if self.__ses__:
            olist.append(self.__ses__)
        return olist

namer = name_resource(dwi, t1w, atlas, outdir)
# namer = name_resource(dwi, t1w, atlas, outdir)

def make_gtab_and_bmask(fbval, fbvec, dwi_file, outdir):
    """Takes bval and bvec files and produces a structure in dipy format while also using FSL commands

    Parameters
    ----------
    fbval : str
        b-value file
    fbvec : str
        b-vector file
    dwi_file : str
        dwi file being analyzed
    outdir : str
        output directory

    Returns
    -------
    GradientTable
        gradient table created from bval and bvec files
    str
        location of averaged b0 image file
    str
        location of b0 brain mask file
    """

    # Use B0's from the DWI to create a more stable DWI image for registration
    nodif_B0 = "{}/nodif_B0.nii.gz".format(outdir)
    nodif_B0_bet = "{}/nodif_B0_bet.nii.gz".format(outdir)
    nodif_B0_mask = "{}/nodif_B0_bet_mask.nii.gz".format(outdir)

    # loading bvecs/bvals
    print(fbval)
    print(fbvec)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    # Creating the gradient table
    gtab = gradient_table(bvals, bvecs, atol=1.0)

    # Correct b0 threshold
    gtab.b0_threshold = min(bvals)

    # Get B0 indices
    B0s = np.where(gtab.bvals == gtab.b0_threshold)[0]
    print("%s%s" % ("B0's found at: ", B0s))

    # Show info
    print(gtab.info)

    # Extract and Combine all B0s collected
    print("Extracting B0's...")
    cmds = []
    B0s_bbr = []
    for B0 in B0s:
        print(B0)
        B0_bbr = "{}/{}_B0.nii.gz".format(outdir, str(B0))
        cmd = "fslroi " + dwi_file + " " + B0_bbr + " " + str(B0) + " 1"
        cmds.append(cmd)
        B0s_bbr.append(B0_bbr)

    for cmd in cmds:
        print(cmd)
        os.system(cmd)

    # Get mean B0
    B0s_bbr_imgs = []
    for B0 in B0s_bbr:
        B0s_bbr_imgs.append(nib.load(B0))

    mean_B0 = mean_img(B0s_bbr_imgs)
    nib.save(mean_B0, nodif_B0)

    # Get mean B0 brain mask
    cmd = "bet " + nodif_B0 + " " + nodif_B0_bet + " -m -f 0.2"
    os.system(cmd)
    return gtab, nodif_B0, nodif_B0_mask

# [gtab, nodif_B0, nodif_B0_mask] = mgu.make_gtab_and_bmask(
#     fbval, fbvec, dwi_prep, namer.dirs["output"]["prep_dwi"]
# )

namer.add_dirs_dwi(paths, labels, label_dirs)

runniii = dmri_reg(namer, nodif_B0, nodif_B0_mask, t1w_in, vox_size,simple=False)
# dmri_reg(namer, nodif_B0, nodif_B0_mask, t1w, vox_size, simple=False)

# runniii.tissue2dwi_align()

def test_tissue2dwi_align():
    runniii.tissue2dwi_align()
    a = np.array(np.loadtxt(runniii.xfm_roi2mni_init))
    outarray = np.array([[ 1.30201748e-01, -1.82806115e-01, -9.74489331e-01,  1.77111309e+02],
                        [3.79709005e-02,  9.83054222e-01, -1.79339507e-01,  1.51448375e+01],
                        [9.90760194e-01, -1.36519317e-02,  1.34936684e-01, -3.70971904e+01],
                        [0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    assert np.allclose(a, outarray)

test_tissue2dwi_align()
