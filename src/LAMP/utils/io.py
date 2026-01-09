import os
from pathlib import Path
from matplotlib import image
import numpy as np
import pickle
import json
import toml

def load_file(filepath, file_type=None, options=None):
    if not os.path.exists(Path(filepath)):
        print(f'IO Error; load_file(); {filepath} not found')
        return None
    filepath_no_ext, file_ext = os.path.splitext(filepath)
    # auto-detect type through file extension?
    if file_type is None:
        if file_ext.lower() == '.pickle' or file_ext.lower() == '.pkl':
            data = load_pickle(filepath)
        elif file_ext.lower() == '.json':
            data = load_json(filepath)
        elif file_ext.lower() == '.csv':
            data = load_csv(filepath)
        elif file_ext.lower() == '.npy':
            data = load_npy(filepath)
        elif file_ext.lower() == '.toml':
            data = load_toml(filepath)
        elif file_ext.lower() == '.tif':
            data = image.imread(filepath)
        else:
            print(f"IO error; load_file(); could not auto-read file type, please provide file_type arugment")
    elif file_type.lower() == 'pickle' or file_ext.lower() == 'pkl':
        data = load_pickle(filepath)
    elif file_type.lower() == 'json':
        data = load_json(filepath)
    elif file_type.lower() == 'csv':
        data = load_csv(filepath)
    elif file_type.lower() == 'numpy' or file_type.lower() == 'npy':
        data = load_npy(filepath)
    elif file_type.lower() == 'toml':
        data = load_toml(filepath)
    elif file_type.lower() == 'tif':
        data = image.imread(filepath)
    else:
        print(f"IO error; load_file(); no known type '{file_type}'")
    return data

def save_file(filepath, data, file_type=None, options=None):
    # auto-detect type through file extension?
    if file_type is None:
        filepath_no_ext, file_ext = os.path.splitext(filepath)
        if file_ext.lower() == '.pickle' or file_ext.lower() == '.pkl':
            save_pickle(filepath, data)
        elif file_ext.lower() == '.json':
            save_json(filepath, data)
        elif file_ext.lower() == '.csv':
            save_csv(filepath, data)
        else:
            print(f"IO error; save_file(); could not auto-read file type, please provide file_type arugment")
    elif file_type.lower() == 'pickle' or file_type.lower() == 'pkl':
        save_pickle(filepath, data)
    elif file_type.lower() == 'json':
        save_json(filepath, data)
    elif file_type.lower() == 'csv':
        save_csv(filepath, data)
    else:
        print(f"IO error; save_file(); no known type '{file_type}'")
    return

#
# TODO: Look for these functions automatically in some subfolder? (and import)
#

def load_npy(filepath):
    return np.load(Path(filepath))

def save_npy(filepath, data):
    return np.save(Path(filepath),data)

def load_csv(filepath, delimiter=',', col_dtypes=None, skip_header=0): # cold_dtypes=float
    # Pandas might be better here? problems with mixed data types...
    data = np.genfromtxt(Path(filepath), delimiter=delimiter, dtype=col_dtypes, skip_header=skip_header, encoding=None)
    if type(data[0]) == np.void: # if problems loading datatypes, try to return an array
        data = np.array(list(map(list, data))) 
    return data

def save_csv(filepath, data, delimiter=","):
    return np.savetxt(filepath, data, delimiter=delimiter)

def load_pickle(filepath):
    with open(Path(filepath), 'rb') as handle:
        return pickle.load(handle)

def save_pickle(filepath, data):
    with open(Path(filepath), 'wb') as handle:
        pickle.dump(data, handle, protocol=4)#pickle.HIGHEST_PROTOCOL) # use version 4 so some people don't break the loading for earlier python versions
    return

def load_json(filepath):
    with open(Path(filepath)) as handle:
        return json.load(handle)

def save_json(filepath, data):
    with open(Path(filepath), "w") as handle:
        json.dump(data, handle)
    return

def load_toml(filepath):
    # with open(filepath, "rb") as handle:
    #     return toml.load(handle)
    #return toml.load(Path(filepath))
    # the following JSON to and fro is for fixing a pickling issue?
    return json.loads(json.dumps(toml.load(Path(filepath))))

def save_toml(filepath, data):
    with open(Path(filepath), 'w') as handle:
        toml.dump(data, handle)
    return