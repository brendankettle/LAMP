import os
from matplotlib import image
import numpy as np
import pickle
import json
import toml

class DAQ():
    """Base class for DAQs
    """

    def __init__(self, exp_obj):
        self.ex = exp_obj # pass in experiment object
        self.data_folder = self.ex.config['paths']['data_folder']
        return
    
    # def get_calib(self,calib_id,diag=False): 
    #     # assume calib_id is filepath in root calib folder? 
    #     calib_filepath = os.path.join(self.ex.config['paths']['calibs_folder'], calib_id)
    #     if os.path.exists(calib_filepath):
    #         return self.load_file(calib_filepath)
    #     else:
    #         return None

    def load_imdata(self, shot_filepath, data_type=float):
        imdata = image.imread(shot_filepath).astype(data_type)
        return imdata

    def load_file(self, filepath, file_type=None, options=None):
        if not os.path.exists(filepath):
            print(f'DAQ Error; load_file(); {filepath} not found')
            return None
        # auto-detect type through file extension?
        if file_type is None:
            filepath_no_ext, file_ext = os.path.splitext(filepath)
            if file_ext.lower() == '.pickle' or file_ext.lower() == '.pkl':
                data = self.load_pickle(filepath)
            elif file_ext.lower() == '.json':
                data = self.load_json(filepath)
            elif file_ext.lower() == '.csv':
                data = self.load_csv(filepath)
            elif file_ext.lower() == '.npy':
                data = self.load_npy(filepath)
            elif file_ext.lower() == '.toml':
                data = self.load_toml(filepath)
            else:
                print(f"DAQ error; load_file(); could not auto-read file type, please provide type= arugment")
        elif file_type.lower() == 'pickle' or file_ext.lower() == 'pkl':
            data = self.load_pickle(filepath)
        elif file_type.lower() == 'json':
            data = self.load_json(filepath)
        elif file_type.lower() == 'csv':
            data = self.load_csv(filepath)
        elif file_type.lower() == 'numpy' or file_type.lower() == 'npy':
            data = self.load_npy(filepath)
        elif file_type.lower() == 'toml':
            data = self.load_toml(filepath)
        else:
            print(f"DAQ error; load_file(); no known type '{type}'")
        return data
    
    def save_file(self, filepath, data, type=None):
        # auto-detect type through file extension?
        if type is None:
            filepath_no_ext, file_ext = os.path.splitext(filepath)
            if file_ext.lower() == '.pickle' or file_ext.lower() == '.pkl':
                self.save_pickle(filepath, data)
            elif file_ext.lower() == '.json':
                self.save_json(filepath, data)
            else:
                print(f"DAQ error; save_file(); could not auto-read file type, please provide type= arugment")
        elif type.lower() == '.pickle' or file_ext.lower() == '.pkl':
            self.save_pickle(filepath, data)
        elif type.lower() == '.json':
            self.save_json(filepath, data)
        else:
            print(f"DAQ error; save_file(); no known type '{type}'")
        return
    
    def load_npy(self, filepath):
        return np.load(filepath)

    def load_csv(self, filepath, delimiter=',', col_dtypes=None):
        # Pandas might be better here? problems with mixed data types...
        return np.genfromtxt(filepath, delimiter=delimiter, dtype=col_dtypes, encoding=None)

    def save_csv(self, filepath, data):
        print('TODO: CSV writer!')
        return
    
    def load_pickle(self, filepath):
        with open(filepath, 'rb') as handle:
            return pickle.load(handle)

    def save_pickle(self, filepath, data):
        with open(filepath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def load_json(self, filepath):
        with open(filepath) as json_file:
            return json.load(json_file)
    
    def save_json(self, filepath, data):
        with open(filepath, "w") as outfile:
            json.dump(data, outfile)
        return
    
    def load_toml(self, filepath):
        # https://docs.python.org/3/library/tomllib.html
        with open(filepath, "rb") as toml_file:
            return toml.load(toml_file)
    
    def save_toml(self, filepath, data):
        with open(filepath, 'w') as outfile:
            toml.dump(data, outfile)
        return