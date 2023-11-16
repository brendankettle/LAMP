"""Base class for Diagnostics
"""
import os
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
import pickle
import json

# TODO: Some way of auto picking calibration by date? a calib history file dictates files by dates?

class Diagnostic():

    calib_dict = {}
    calib_input = {}
    calib_hist = {}

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

    def get_calib(self, calib_id=None, input=False):
        """Take a calibration id of some form, and return calibration dictionary
            - = None: Try and use pre-saved calibration dict within object
            - = Dictionary: Assuming calibration is already given by input and simply return
            - = Filepath: Look for a dictionary within a calibration file
            - = ID String:  Look for a dictionary within the master calibration (input) file, with this ID
        """
        # If none passed, try use pre-saved calib input
        if calib_id is None:
            if input and self.calib_input:
                return self.calib_input
            elif self.calib_dict:
                return self.calib_dict
            else:
                print("get_calib() error; None passed and no calibration pre-set")
        # passing dictionary directly? This needs to have the required keys!
        elif isinstance(calib_id, dict):
            return calib_id
        # passing string?
        else:
            # let's look for a file first
            calib_filepath = os.path.join(self.ex.config['paths']['calibs_folder'], self.config['setup']['calib_folder'], calib_id)
            if os.path.exists(calib_filepath):
                return self.load_calib_file(calib_id)
            # no file, so let's look for ID key within the master calibration input file (if set in config)
            elif input and 'calib_inputs' in self.config['setup']:
                all_calib_input_dicts = self.load_calib_file(self.config['setup']['calib_inputs'])
                if calib_id in all_calib_input_dicts:
                    return all_calib_input_dicts[calib_id]
                else:
                    print(f"get_calib() error; No calibration input ID found for {calib_id} in master calib input file")
            else:
                print(f"get_calib() error; Unknown calibration input found for {calib_id}")

    def set_calib(self, calib_id):
        self.calib_dict = self.get_calib(calib_id)
        return
    
    def save_calib(self, calib_filename, calib_dict = None):
        """Save all the current calibration information to file, or save specific dictionary if passed"""
        if calib_dict is None:
            calib_dict = self.calib_dict
        self.save_calib_file(calib_filename, calib_dict)
        return
    
    def load_calib(self, calib_filename, set=True, file_type=None, options=None):
        calib = self.load_calib_file(calib_filename, file_type=file_type, options=options)
        if set:
            self.set_calib(calib)
        return
    
    def get_calib_input(self, calib_input=None):
        """Wrapper for get_calib, but specifying an input calibration
        """
        return self.get_calib(calib_input, input=True)

    def set_calib_input(self, calib_input):
        self.calib_input = self.get_calib_input(calib_input)
        return self.calib_input
    
    # TODO: Passing arguments such as delimiters
    def load_calib_file(self, filename, file_type=None, options=None):
        filepath = os.path.join(self.ex.config['paths']['calibs_folder'], self.config['setup']['calib_folder'], filename)
        # auto-detect type through file extension?
        if file_type is None:
            filepath_no_ext, file_ext = os.path.splitext(filepath)
            if file_ext.lower() == '.pickle' or file_ext.lower() == '.pkl':
                calib = self.load_pickle(filepath)
            elif file_ext.lower() == '.json':
                calib = self.load_json(filepath)
            elif file_ext.lower() == '.csv':
                calib = self.load_csv(filepath)
            else:
                print(f"Diagnostic error; load_calib_file(); could not auto-read file type, please provide type= arugment")
        elif file_type.lower() == '.pickle' or file_ext.lower() == '.pkl':
            calib = self.load_pickle(filepath)
        elif file_type.lower() == '.json':
            calib = self.load_json(filepath)
        elif file_type.lower() == '.csv':
            calib = self.load_csv(filepath)
        else:
            print(f"Diagnostic error; load_calib_file(); no known type '{type}'")
        return calib

    def save_calib_file(self, filename, calib_data, type=None):
        filepath = os.path.join(self.ex.config['paths']['calibs_folder'], self.config['setup']['calib_folder'], filename)
        # auto-detect type through file extension?
        if type is None:
            filepath_no_ext, file_ext = os.path.splitext(filepath)
            if file_ext.lower() == '.pickle' or file_ext.lower() == '.pkl':
                self.save_pickle(filepath, calib_data)
            elif file_ext.lower() == '.json':
                self.save_json(filepath, calib_data)
            else:
                print(f"Diagnostic error; save_calib_file(); could not auto-read file type, please provide type= arugment")
        elif type.lower() == '.pickle' or file_ext.lower() == '.pkl':
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
    