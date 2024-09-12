import numpy as np
import collections.abc

# return index of closest matched value in array
def mindex(arr,val):
    return np.argmin(np.abs(np.array(arr)-val))

# make sure we join dictionaries on a recussive level without overwriting
def dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d