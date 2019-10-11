#This is a pure python function for apply_warp

import warnings

warnings.simplefilter("ignore")
from ndmg.utils import gen_utils as mgu
import nibabel as nib
import numpy as np
import nilearn.image as nl
import os
import os.path as op
import nilearn.image as nl


from __future__ import (print_function, division, unicode_literals,
                        absolute_import)
from builtins import open

import os.path as op
import re

from ... import logging
from .base import ElastixBaseInputSpec
from ..base import CommandLine, TraitedSpec, File, traits, InputMultiPath



class ApplyWarpOutputSpec(TraitedSpec):
    warped_file = File(desc='input moving image warped to fixed image')


class ApplyWarp(CommandLine):
    """
    Use ``transformix`` to apply a transform on an input image.
    The transform is specified in the transform-parameter file.
    """

    _cmd = 'transformix'
    input_spec = ApplyWarpInputSpec
    output_spec = ApplyWarpOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir = op.abspath(self.inputs.output_path)
        outputs['warped_file'] = op.join(out_dir, 'result.nii.gz')
        return outputs


class AnalyzeWarpInputSpec(ApplyWarpInputSpec):
    points = traits.Enum(
        'all',
        usedefault=True,
        position=0,
        argstr='-def %s',
        desc='transform all points from the input-image, which effectively'
             ' generates a deformation field.')
    jac = traits.Enum(
        'all',
        usedefault=True,
        argstr='-jac %s',
        desc='generate an image with the determinant of the spatial Jacobian')
    jacmat = traits.Enum(
        'all',
        usedefault=True,
        argstr='-jacmat %s',
        desc='generate an image with the spatial Jacobian matrix at each voxel')
    moving_image = File(
        exists=True,
        argstr='-in %s',
        desc='input image to deform (not used)')

class AnalyzeWarpOutputSpec(TraitedSpec):
    disp_field = File(desc='displacements field')
    jacdet_map = File(desc='det(Jacobian) map')
    jacmat_map = File(desc='Jacobian matrix map')


class AnalyzeWarp(ApplyWarp):
    """
    Use transformix to get details from the input transform (generate
    the corresponding deformation field, generate the determinant of the
    Jacobian map or the Jacobian map itself)
    """

    input_spec = AnalyzeWarpInputSpec
    output_spec = AnalyzeWarpOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir = op.abspath(self.inputs.output_path)
        outputs['disp_field'] = op.join(out_dir, 'deformationField.nii.gz')
        outputs['jacdet_map'] = op.join(out_dir, 'spatialJacobian.nii.gz')
        outputs['jacmat_map'] = op.join(out_dir, 'fullSpatialJacobian.nii.gz')
        return outputs
