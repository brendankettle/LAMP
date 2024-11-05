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