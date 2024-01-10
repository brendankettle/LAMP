"""Main entry point for analysis scripts; Experiment class
"""
import os
from configparser import ConfigParser, ExtendedInterpolation
import importlib

class Experiment:

    def __init__(self, root_folder, config_filepath):
        self.root_folder = root_folder
        self.config_filepath = root_folder + config_filepath

        if not os.path.exists(config_filepath):
            raise Exception(f'Problem finding experiment config file: {config_filepath}')
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read(config_filepath)

        # setup DAQ
        if self.config['setup']['DAQ'].lower() == 'none':
            print('To Do: Support single file analysis...')
        else:
            DAQ_module = 'LAMP.DAQs.' + self.config['setup']['DAQ']
            try:
                DAQ_lib = importlib.import_module(DAQ_module)
            except ImportError:
                raise Exception(f'Could not find DAQ module: {DAQ_module}')
            #self.DAQ = DAQ_lib.DAQ(self)
            if callable(DAQ_class := getattr(DAQ_lib, self.config['setup']['DAQ'])):
                print(f"Using DAQ: {self.config['setup']['DAQ']}")
                self.DAQ = DAQ_class(self)

        # loop through diagnostics and add
        self.diags = {}
        if 'diagnostics' in self.config.keys():
            for diag_name in self.config['diagnostics']:  
                self.add_diagnostic(diag_name, self.config['diagnostics'][diag_name])

    def add_diagnostic(self, diag_name, diag_config_filepath):

        # read config file (need type at least)
        if not os.path.exists(diag_config_filepath):
            raise Exception(f'Problem finding config file for: {diag_config_filepath}')
        diag_config = ConfigParser(interpolation=ExtendedInterpolation())
        diag_config.read(diag_config_filepath)

        if 'name' in diag_config['setup']:
            diag_name = diag_config['setup']['name']
        if 'type' in diag_config['setup']:
            diag_type = diag_config['setup']['type']
        else:
            raise Exception(f'No diagnostic type defined for: {diag_name}')

        diag_module = 'LAMP.diagnostics.' + diag_type
        try:
            diag_lib = importlib.import_module(diag_module)
        except ImportError:
            raise Exception(f'Could not find Diagnostics module: {diag_module}')

        if callable(diag_class := getattr(diag_lib, diag_type)):
            print(f'Adding Diagnostic: {diag_name} ({diag_config_filepath})')
            self.diags[diag_name] = diag_class(self, diag_config_filepath)
        else:
            raise Exception(f'Could not find Diagnostic object: {diag_type}')

        return self.get_diagnostic(diag_name)
    
    def get_diagnostic(self, diag_name):
        return self.diags[diag_name]

    def list_diagnostics(self):
        for diag_name in self.diags.keys():
            print(diag_name)
        return

