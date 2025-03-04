#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Email: b.kettle@imperial.ac.uk
"""
import numpy as np

def calc_bragg_dispersion(central_eV, crystal_2d, dist_src_det, dists_pixels):
    """
        Work out the spectral disperion on a detector after a bragg crystal reflection
        (Initially assuming the detector is at a right angle to incoming rays, and first order reflection. Also using an approximation?)

        dist_src_det:   Source to detector distance (m) - along ray path, i.e. source-crystal + crystal-detector
        crystal_2d:     2d lattice spacing of the crystal (m - not angstroms!)
        dists_pixels:   array of distance points (i.e. pixel distances) along detector surface to return photon energy
        central_eV:     central photon energy at zero distance point (centre?) along the detector (eV) - this sets the crystal angle
    """

    # Using "mirror image" straight line geometry calc. See page 68 of B. Kettle Thesis
    central_lambda = 1.2398e-6 / central_eV;
    crystal_angle = np.arcsin(central_lambda / crystal_2d);
    px_angles = np.arctan(dists_pixels / dist_src_det)
    px_braggs = crystal_angle - px_angles
    px_lambdas = crystal_2d * np.sin(px_braggs)
    px_eV = 1.2398e-6 / px_lambdas

    return px_eV

    """
    OLD METHOD

    # work out bragg angle
    central_lambda = 1.2398e-6 / central_eV;
    bragg_angle = math.asin(central_lambda / crystal_2d);

    # Currently using simple equation from A. Pak et al (2004) "X-Ray Line Measurements with High Efficiency Bragg Crystals", 15th Topical Conference on High-Temperature Diagnostics
    # Is this too approximate?? Do we need better?
    px_size = dists_pixels[2] - dists_pixels[1] # again, currently assuming EQUAL SPACING! - bit of a bodge here but works for Andors for sure
    px_delta_theta = math.atan(px_size / dist_src_det); # anglular spread of one pixel
    px_delta_eV = (px_delta_theta / math.tan(bragg_angle)) * central_eV # eV/px
    px_num = dists_pixels / px_size # get pixel numbers - bodging this a little bit because the passed argument is actually pixel distance....
    px_eV_rel = px_num * px_delta_eV # relative energy dispersion across chip
    px_eV = px_eV_rel + central_eV # absolute energy across chip

    """
