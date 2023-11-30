import numpy as np

# TODO: set the path/format for attenuation files
# TODO: could try use pretabulated densities?

def filter_transmission(eV, material, thickness_um, density):

    path_ma = "./mass_attenuation_data/mass_attenuation_%s.txt"%(material)
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
    interp_trans = np.interp(eV, ma_data[:, 0] * 1000, trans)

    return interp_trans