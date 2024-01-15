import os
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
from pathlib import Path

class Diagnostic():
    """Base class for Diagnostics
    """

    calib_dict = {}
    calib_input = {}
    calib_hist = {}

    def __init__(self, exp_obj, config_filepath):
        self.ex = exp_obj # pass in experiment object
        self.DAQ = self.ex.DAQ # shortcut to DAQ
        self.load_config(config_filepath)
        # load calibration history?
        if 'calib_history' in self.config['setup']:
            self.calib_hist = self.load_calib_file(self.config['setup']['calib_history'])
        # TODO: load calibration input or calibration default file?
        self.diag_name = self.config['setup']['name'] # shortcut
        return

    def __repr__(self):
        return f"{self.config['setup']['type']}(name='{self.config['setup']['name']}')"
    
    def load_config(self, config_filepath):
        if not os.path.exists(Path(config_filepath)):
            raise Exception(f'Problem finding experiment config file: {Path(config_filepath)}')
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read(Path(config_filepath))
        return

    def get_shot_data(self, shot_dict):
        """Wrapper for getting shot data through DAQ"""
        return self.DAQ.get_shot_data(self.config['setup']['name'], shot_dict)
    
    def get_calib(self, calib_id=None, shot_dict=None, input=False):
        """Take a calibration id of some form, and return calibration dictionary
            - = None: Try and use pre-saved calibration dict within object
            - = Dictionary: Assuming calibration is already given by input and simply return
            - = Filepath: Look for a dictionary within a calibration file
            - = ID String:  Look for a dictionary within the master calibration (input) file, with this ID
        """
        # If none passed, try use pre-saved calib input
        if calib_id is None:
            # TODO: Should this check if shot dictionary is set first?
            if input and self.calib_input:
                return self.calib_input
            elif self.calib_dict:
                return self.calib_dict
            else:
                # try using calibration history with a shot dictionary?
                if self.calib_hist and shot_dict:
                    shot_time = self.DAQ.build_time_point(shot_dict)
                    for calib_name in self.calib_hist:
                        calib = self.calib_hist[calib_name]
                        start_shot_dict  = calib['start']
                        calib_start  = self.DAQ.build_time_point(start_shot_dict)
                        end_shot_dict = calib['end']
                        calib_end  = self.DAQ.build_time_point(end_shot_dict)
                        if shot_time > calib_start and shot_time < calib_end:
                            if input:
                                calib_id = calib['input_id']
                            else:
                                calib_id = calib['calib_file']
                                calib_input_id = calib['input_id']
                            break
                    if not calib_id:
                        print("get_calib() error; Could not place shot in calibration history timeline")
                else:
                    print("get_calib() error; None passed and no calibration pre-set")
        # passing dictionary directly? This needs to have the required keys!
        if isinstance(calib_id, dict):
            return calib_id
        # passing string? (or got a calibration filepath  using history above)
        if isinstance(calib_id, str):
            # let's look for a file first
            calib_filepath = os.path.join(self.ex.config['paths']['calibs_folder'], self.config['setup']['calib_folder'], calib_id)
            if os.path.exists(Path(calib_filepath)):
                return self.load_calib_file(calib_id)
            # no file, so let's look for ID key within the master calibration input file (if set in config)
            elif 'calib_inputs' in self.config['setup']:
                all_calib_input_dicts = self.load_calib_file(self.config['setup']['calib_inputs'])
                if input and calib_id in all_calib_input_dicts:
                    return all_calib_input_dicts[calib_id]
                elif 'calib_input_id' in locals() and calib_input_id in all_calib_input_dicts:
                    print(f"Warning; get_calib(); Could not find calibration: {calib_id}, so using input: {calib_input_id}")
                    return all_calib_input_dicts[calib_input_id]
                else:
                    print(f"get_calib() error; No calibration input ID found for {calib_id} in master calib input file")
            else:
                print(f"get_calib() error; Unknown calibration found for {calib_id}")

    def set_calib(self, calib_id=None, shot_dict=None):
        self.calib_dict = self.get_calib(calib_id, shot_dict=shot_dict)
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
        return calib
    
    def get_calib_input(self, calib_input=None, shot_dict=None):
        """Wrapper for get_calib, but specifying an input calibration
        """
        return self.get_calib(calib_input, shot_dict=shot_dict, input=True)

    def set_calib_input(self, calib_input=None, shot_dict=None):
        self.calib_input = self.get_calib_input(calib_input, shot_dict=shot_dict)
        return self.calib_input
    
    # TODO: Passing arguments such as delimiters
    def load_calib_file(self, filename, file_type=None, options=None):
        filepath = os.path.join(self.ex.config['paths']['calibs_folder'], self.config['setup']['calib_folder'], filename)
        calib  = self.DAQ.load_file(Path(filepath), file_type=file_type, options=options)
        return calib

    def save_calib_file(self, filename, calib_data, type=None):
        filepath = os.path.join(self.ex.config['paths']['calibs_folder'], self.config['setup']['calib_folder'], filename)
        self.DAQ.save_file(Path(filepath), calib_data, type=type)
        return
