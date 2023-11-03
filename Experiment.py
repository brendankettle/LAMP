import os
import configparser
import importlib

class Experiment:

    def __init__(self, root_folder, config_filepath):
        self.root_folder = root_folder
        self.config_filepath = root_folder + config_filepath
        self.diags = {}

        # read config info
        if not os.path.exists(config_filepath):
            raise Exception(f'Problem finding experiment config file: {config_filepath}')
        self.config = configparser.ConfigParser()
        self.config.read(config_filepath)

        # setup DAQ
        # TODO: Can be "none"? For accessing random single fiels etc.
        DAQ_module = 'LAMP.DAQs.' + self.config['setup']['DAQ']
        try:
            DAQ_lib = importlib.import_module(DAQ_module)
        except ImportError:
            raise Exception(f'Could not find DAQ module: {DAQ_module}')
        self.DAQ = DAQ_lib.DAQ()

        # loop through diagnostics and add
        if 'diagnostics' in self.config.keys():
            for diag_name in self.config['diagnostics']:  
                self.add_diagnostic(diag_name, self.config['diagnostics'][diag_name])

    def add_diagnostic(self, diag_name, diag_config_filepath):

        # read config file
        if not os.path.exists(diag_config_filepath):
            raise Exception(f'Problem finding config file for: {diag_config_filepath}')
        diag_config = configparser.ConfigParser()
        diag_config.read(diag_config_filepath)

        if 'name' in diag_config['info']:
            diag_name = diag_config['info']['name']
        if 'type' in diag_config['info']:
            diag_type = diag_config['info']['type']
        else:
            raise Exception(f'No diagnostic type defined for: {diag_name}')

        diag_module = 'LAMP.diagnostics.' + diag_type
        try:
            diag_lib = importlib.import_module(diag_module)
        except ImportError:
            raise Exception(f'Could not find Diagnostics module: {diag_module}')

        if callable(diag_func := getattr(diag_lib, diag_type)):
            print(f'Adding Diagnostic: {diag_name} ({diag_config_filepath})')
            self.diags[diag_name] = diag_func(self, diag_config_filepath)
        else:
            raise Exception(f'Could not find Diagnostic object: {diag_type}')

        return
    
    # or in DAQ?
    def shot_series(self):

        return
