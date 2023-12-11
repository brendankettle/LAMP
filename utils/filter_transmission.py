import numpy as np
import os

# TODO: Update mass atten data for sub keV (or extrapolate?? using scipy not np.interp)
# TODO: set the path/format for attenuation files
# TODO: could try use pretabulated densities?

def filter_transmission(eV, material, thickness_um, density):
    """Function to return a materials transmission, as a function of energy, 
    using pretabulated values for the mass attenuation coefficient
    """

    # assuming lookup data is in subfolder where this file is 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_ma = dir_path+"/mass_attenuation_data/mass_attenuation_%s.txt"%(material)
    ma_data = np.genfromtxt(path_ma, delimiter='\t', skip_header=1)

    # Combine different element transmissions if necessary
    # To Do: Check this works!
    num_elements = np.shape(ma_data)[1]-1
    trans = 1
    for ei in range(num_elements):
        att_coeff = 1.0 / (density * ma_data[:, ei+1])
        trans = trans * np.exp(-(thickness_um*1e-4) / att_coeff)
    if num_elements > 1:
        print('NEED TO DOUBLE CHECK COMBINING WORKS!')

    # interpolate over given energy range
    # Provided data is in keV...
    # TODO: Could us scipy and extrpolate here?
    interp_trans = np.interp(eV, ma_data[:, 0] * 1000, trans)

    return interp_trans