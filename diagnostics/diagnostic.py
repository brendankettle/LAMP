"""Base class for Diagnostics
"""
import os
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
import pickle
import json

class Diagnostic():

    def __init__(self, exp_obj, config_filepath):
        self.ex = exp_obj # pass in experiment object
        self.DAQ = self.ex.DAQ # shortcut to DAQ
        self.load_config(config_filepath)
        return

    def __repr__(self):
        return f"{self.config['setup']['type']}(name='{self.config['setup']['name']}')"
    
    def load_config(self, config_filepath):
        if not os.path.exists(config_filepath):
            raise Exception(f'Problem finding experiment config file: {config_filepath}')
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read(config_filepath)
        return
    
    # TODO: Shouldn't auto save calib to the object? we load different calibrations...
    # TODO: Passing arguments such as delimiters
    def load_calib_file(self, filepath, type=None, options=None):
        # auto-detect type through file extension?
        if type is None:
            filepath_no_ext, file_ext = os.path.splitext(filepath)
            if file_ext.lower() == '.pickle':
                self.calib = self.load_pickle(filepath)
            elif file_ext.lower() == '.json':
                self.calib = self.load_json(filepath)
            elif file_ext.lower() == '.csv':
                self.calib = self.load_csv(filepath)
            else:
                print(f"Diagnostic error; load_calib_file(); could not auto-read file type, please provide type= arugment")
        elif type.lower() == '.pickle':
            self.calib = self.load_pickle(filepath)
        elif type.lower() == '.json':
            self.calib = self.load_json(filepath)
        elif type.lower() == '.csv':
            self.calib = self.load_csv(filepath)
        else:
            print(f"Diagnostic error; load_calib_file(); no known type '{type}'")
        return self.calib
    
    def save_calib_file(self, filepath, calib_data, type=None):
        # auto-detect type through file extension?
        if type is None:
            filepath_no_ext, file_ext = os.path.splitext(filepath)
            if file_ext.lower() == '.pickle':
                self.save_pickle(filepath, calib_data)
            elif file_ext.lower() == '.json':
                self.save_json(filepath, calib_data)
            else:
                print(f"Diagnostic error; save_calib_file(); could not auto-read file type, please provide type= arugment")
        elif type.lower() == '.pickle':
            self.save_pickle(filepath, calib_data)
        elif type.lower() == '.json':
            self.save_json(filepath, calib_data)
        else:
            print(f"Diagnostic error; save_calib_file(); no known type '{type}'")
        return
    
    def load_csv(self, filepath, delimiter=','):
        return np.loadtxt(filepath, delimiter=delimiter)

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
    