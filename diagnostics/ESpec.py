import os
import configparser
from .diagnostic import Diagnostic
import cv2

class ESpec(Diagnostic):

    version = 0.1

    # TODO: Move stuff into the diagnostic base class

    def __init__(self, exp_parent, config_filepath):
        print('Starting up ESpec diagnostic')
        # pass in experiment object
        self.ex = exp_parent

        # ?? Get all shared attributes and funcs from base Diagnostic
        Diagnostic.__init__(self)

        # read config info  - use root folder??
        if not os.path.exists(config_filepath):
            raise Exception(f'Problem finding experiment config file: {config_filepath}')
        self.config = configparser.ConfigParser()
        self.config.read(config_filepath)

        return

    def __repr__(self):
        return "ESpec(name=" + self.config['name'] + ")"
    
    # move to base class?
    def get_shotdata(self, shot_dict):

        self.ex.DAQ.get_shotdata(shot_dict)

        return

    def load_calib(self, timestamp):
        print(f'Loading some calibration from file: {timestamp}')
        print(self.ex.config['setup']['calib_folder'])

        return
