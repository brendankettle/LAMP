import numpy as np
import collections.abc

def mindex(arr,val):
    """return index of closest matched value in array"""
    if isinstance(val, list):
        indices = []
        for this_val in val:
            indices.append(np.argmin(np.abs(np.array(arr)-this_val)))
        return indices
    elif isinstance(val,np.ndarray):
        indices = []
        for this_val in list(val):
            indices.append(np.argmin(np.abs(np.array(arr)-this_val)))
        return np.array(indices)
    else:
        return np.argmin(np.abs(np.array(arr)-val))

# make sure we join dictionaries on a recussive level
def dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def smooth_lin(lin, kernel):
    """ Sav Gol is somehwere??
        For smoothing a noisey lineout
        lin: Lineout to smoothing
        kernel: number of points to smooth over
    """
    # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean

    # no kernel? or zero? skip smoothing...
    if (kernel == 0) or (not kernel):
        return lin

    smoothed_lin = np.convolve(lin, np.ones(kernel), mode='same') / kernel

    # Bodge the start and end, padding to same length with some assumed values
    smoothed_lin[0:kernel] = smoothed_lin[(kernel+1)]
    smoothed_lin[-kernel:] = smoothed_lin[(-kernel-1)]

    return smoothed_lin


def first_index(haystack, needle, side='left', cut='low'):
    """ pass an array of values ('haystack'), and return the index of first value above/below 'needle'.
        The function returns the first index coming from either the start or end of the array.
        This is indicated by side='left' (first) or side='right' (last).
        Cut can also be defined to be low or high, i.e. whether you want the above/below."""

    haystack_thresholded = haystack.copy() # make sure we don't alter the passed array

    # looking for higher or lower?
    if cut == 'low':
        haystack_thresholded[haystack < needle] = 0
    elif cut == 'high':
        haystack_thresholded[haystack > needle] = 0
    else:
        print('Error, unknown "cut" argument for first_index. Allowed options are "low" or "high"')

    # do non-zero
    nonzero_indices = np.nonzero(haystack_thresholded)[0]

    if side == 'left':
        idx = nonzero_indices[0] # first above threshold
    elif side == 'right':
        idx = nonzero_indices[-1] # last above threshold
    else:
        print('Error, unknown "side" argument for first_index. Allowed options are "left" or "right"')

    return idx

def gaussian(x, A, x0, FWHM, H=0):
    sigma = FWHM / 2.35482 
    return H + A * np.exp(-(x - x0)**2 / (2 * sigma**2))

def gauss_conv(x,y,gFWHM,cmode='same'):
    """Convolve a gaussian of certain FWHM with x and y data.
    gFWHM units are in x units"""
    # make gaussian, with same x scale step
    dx = np.mean(np.diff(x))
    #print(dx)
    gx = np.linspace(-2*gFWHM,2*gFWHM, int((4*gFWHM)/dx))
    #print(np.mean(np.diff(gx)))
    gauss = gaussian(gx,A=1,x0=0,FWHM=gFWHM)
    # convolve with data
    conv_y = np.convolve(y,gauss,mode=cmode)
    # normalise??
    conv_y = conv_y / np.sum(gauss)
    return conv_y