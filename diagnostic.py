import os
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
from pathlib import Path
from .utils.io import *
from .utils.dict_update import *

class Diagnostic():
    """Base class for Diagnostics. 
    Currently this mostly handles loading/saving calibrations.
    """

    calib_dict = {}

    def __init__(self, exp_obj, config):
        self.ex = exp_obj # pass in experiment object
        self.DAQ = self.ex.DAQ # shortcut to DAQ
        # ToDo: if diagnostic config filepath passed as opposed to array, go get config
        # self.load_config(config_filepath)
        self.config = config
        return

    def __repr__(self):
        return f"{self.config['type']}(name='{self.config['name']}')"

    def get_shot_data(self, shot_dict):
        """Wrapper for getting shot data through DAQ"""
        return self.DAQ.get_shot_data(self.config['name'], shot_dict)
    
    def get_calib(self, calib_id=None):
        """Take a calibration id of some form, and return calibration dictionary.
            - = None: Try and use pre-saved calibration dict within object
            - = Dictionary: Assume a shot dictionary, and look for configuration using dates
            - = (string) ID String:  Look for a dictionary within the master calibration file, with this ID
            - = (string) Filepath: Look for a dictionary within a calibration file?
        """
        # If none passed, try use pre-saved calib input
        if calib_id is None:
            if self.calib_dict:
                calib_dict = self.calib_dict
                return calib_dict # exiting now instead of loading savefile down below? (again?)
            else:
                print('get_calib() error; None passed and no calibration loaded yet')
                return None

        # passing dictionary? assume shot dictionary and try load using dates
        if isinstance(calib_id, dict):
            shot_dict = calib_id
            shot_time = self.DAQ.build_time_point(shot_dict)
            # load all calibrations in master file
            all_calibs = self.load_calib_file(self.config['calib_file'])
            # loop through calibrations and 
            calib_dict = None
            for this_calib_id in all_calibs:
                start_shot_dict  = all_calibs[this_calib_id]['start']
                calib_start  = self.DAQ.build_time_point(start_shot_dict)
                end_shot_dict = all_calibs[this_calib_id]['end']
                calib_end  = self.DAQ.build_time_point(end_shot_dict)
                if shot_time > calib_start and shot_time < calib_end:
                    calib_dict = all_calibs[this_calib_id]
                    break
            if not calib_dict:
                print("get_calib() error; Could not place shot in calibration timeline")        
                return None

        # passing string?
        if isinstance(calib_id, str):
            # let's look for a file first
            if os.path.exists(self.build_calib_filepath(calib_id)):
                calib_dict = self.load_calib_file(calib_id)
            # no file, so let's look for ID key within the master calibration input file (if set in config)
            elif 'calib_file' in self.config:
                all_calibs = self.load_calib_file(self.config['calib_file'])
                if calib_id in all_calibs:
                    calib_dict = all_calibs[calib_id]
                else:
                    print(f"get_calib() error; No calibration input ID found for {calib_id} in master calib input file")
                    return None
            else:
                print(f"get_calib() error; Unknown calibration found for {calib_id}")
                return None

        # before returning, if processed file is set, try load it and return contents with dictionary
        if 'proc_file' in calib_dict:
            if os.path.exists(self.build_calib_filepath(calib_dict['proc_file'])):
                proc_calib_dict = self.load_calib_file(calib_dict['proc_file'])
                dict_update(calib_dict, proc_calib_dict)
            # might not be processed yet, just print a warnging and move on
            else:
                print(f"get_calib() warning; no processed file found for '{calib_dict['proc_file']}'")

        return calib_dict
    
    def build_calib_filepath(self, filename):
        if 'calib_subfolder' in self.config:
            calib_subfolder = self.config['calib_subfolder']
        else:
            calib_subfolder = '/'
        return Path(os.path.join(self.ex.config['paths']['calibs_folder'], calib_subfolder, filename))

    def load_calib_file(self, filename, file_type=None, options=None):
        calibs  = load_file(self.build_calib_filepath(filename), file_type=file_type, options=options)
        return calibs

    def save_calib_file(self, filename, calib_data, file_type=None, options=None):
        save_file(self.build_calib_filepath(filename), calib_data, file_type=file_type, options=options)
        return
