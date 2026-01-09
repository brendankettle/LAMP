import numpy as np
import os
from pathlib import Path

# Use Chris Armstrong package? (saved locally)
import LAMP.utils.xrays.pyNIST as nist

# TODO: could try use pretabulated densities?

#
# This needs work. Currently implementing a bit of two systems depending on energy or compound usage.
# NONE OF THE DATA GOES BELOW 1 KeV CURRENTLY!
#

def filter_transmission(keV, material, thickness_um, density_gcc, engine='pyNist'):
    """Function to return a materials transmission, as a function of energy, 
    using pretabulated values for the mass attenuation coefficient
    """

    if np.min(keV) < 1:
        print('Warning, filter_transmission currently has no data for <1 keV')

    # default to LAMP loading files
    # This doesn't have compounds yet, just relies on filenames
    # This also doesn't go below 1 keV???
    if engine.lower() == 'lamp': 
        # assuming lookup data is in subfolder where this file is 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path_ma = Path(dir_path+"/mass_attenuation_data/mass_attenuation_%s.txt"%(material))
        ma_data = np.genfromtxt(path_ma, delimiter='\t', skip_header=1)

        # Combine different element transmissions if necessary
        # To Do: Check this works!
        num_elements = np.shape(ma_data)[1]-1
        trans = 1
        for ei in range(num_elements):
            att_coeff = 1.0 / (density_gcc * ma_data[:, ei+1])
            trans = trans * np.exp(-(thickness_um*1e-4) / att_coeff)
        if num_elements > 1:
            print('NEED TO DOUBLE CHECK COMBINING WORKS!')

        # interpolate over given energy range
        # Provided data is in keV...
        # TODO: Could us scipy and extrpolate here?
        interp_trans = np.interp(keV, ma_data[:, 0], trans)

        return interp_trans
    # Can use pyNist, supports compounds, but this doesn't (currently) go below 1 keV!
    # Also pyNIST has interp errors below 1 keV? E.g. Ni? - I hacked a line to fix this in pyNIST.py, but will still miss edges etc. below 1 keV. I.e. Oxygen
    elif engine.lower() == 'pynist':

        # quick check, convert eV to array if given as range
        if isinstance(keV, range):
            keV = np.array(list(keV))

        mat_obj = nist.Material(material, density_gcc, keV*1e-3, 'NIST') # energies in MeV
        trans = mat_obj.get_transmission(thickness_um*1e-3) # pass thickness in mm

        return trans
    
    else:
        print('filter_transmission() error: Unrecognised engine value')
        return