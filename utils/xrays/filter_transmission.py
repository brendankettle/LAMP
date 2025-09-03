import numpy as np
import os
from pathlib import Path

# Use Chris Armstrong package? (saved locally)
import LAMP.utils.xrays.pyNIST as nist

# TODO: could try use pretabulated densities?

def filter_transmission(eV, material, thickness_um, density):
    """Function to return a materials transmission, as a function of energy, 
    using pretabulated values for the mass attenuation coefficient
    - density is g/cc
    """

    mat_obj = nist.Material(material, density, eV*1e-6, 'NIST') # energies in MeV
    trans = mat_obj.get_transmission(thickness_um*1e-3) # pass thickness in mm

    return trans

# def filter_transmission(eV, material, thickness_um, density):
#     """Function to return a materials transmission, as a function of energy, 
#     using pretabulated values for the mass attenuation coefficient
#     """

#     # assuming lookup data is in subfolder where this file is 
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     path_ma = Path(dir_path+"/mass_attenuation_data/mass_attenuation_%s.txt"%(material))
#     ma_data = np.genfromtxt(path_ma, delimiter='\t', skip_header=1)

#     # Combine different element transmissions if necessary
#     # To Do: Check this works!
#     num_elements = np.shape(ma_data)[1]-1
#     trans = 1
#     for ei in range(num_elements):
#         att_coeff = 1.0 / (density * ma_data[:, ei+1])
#         trans = trans * np.exp(-(thickness_um*1e-4) / att_coeff)
#     if num_elements > 1:
#         print('NEED TO DOUBLE CHECK COMBINING WORKS!')

#     # interpolate over given energy range
#     # Provided data is in keV...
#     # TODO: Could us scipy and extrpolate here?
#     interp_trans = np.interp(eV, ma_data[:, 0] * 1000, trans)

#     return interp_trans