"""Base class for Diagnostics
"""
import os
from configparser import ConfigParser, ExtendedInterpolation
import pandas as pd

class Diagnostic():

    def __init__(self, config_filepath):
        # load config file
        self.load_config(config_filepath)
        return

    def __repr__(self):
        return f"{self.config['setup']['type']}(name='{self.config['setup']['name']}')"
    
    def load_config(self, config_filepath):
         # read config info
        if not os.path.exists(config_filepath):
            raise Exception(f'Problem finding experiment config file: {config_filepath}')
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read(config_filepath)
        return
    
    def load_calib(self, filename):

        self.calib = pd.read_pickle(filename)
        return