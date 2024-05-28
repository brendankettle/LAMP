"""Main entry point for analysis scripts; Experiment class
"""
import os
from pathlib import Path
import importlib
from .utils.io import load_file

class Experiment:

    def __init__(self, root_folder):
        """Load config, load DAQ, add diagnostics"""

        # Load local config
        local_config_filepath = Path(root_folder + 'local.toml')
        if not os.path.exists(local_config_filepath):
            raise Exception(f'Problem finding local config file: {local_config_filepath}')
        local_config = load_file(local_config_filepath)

        # load global config and save to object
        global_config_filepath = Path(root_folder + 'global.toml')
        if not os.path.exists(global_config_filepath):
            raise Exception(f'Problem finding global config file: {global_config_filepath}')
        self.config = load_file(global_config_filepath)

        # Add contents of local config to (global) config
        for section_key in local_config: 
            for config_key in local_config[section_key]:
                self.config[section_key][config_key] =  local_config[section_key][config_key]

        # save paths to config
        self.config['paths']['root'] = Path(root_folder)
        self.config['paths']['local_config'] = local_config_filepath
        self.config['paths']['global_config'] = global_config_filepath

        # setup DAQ
        if self.config['setup']['DAQ'].lower() == 'none':
            print('To Do: Support single file analysis... This could just be another DAQ module called None...')
        else:
            DAQ_module = 'LAMP.DAQs.' + self.config['setup']['DAQ']
            try:
                DAQ_lib = importlib.import_module(DAQ_module)
            except ImportError:
                raise Exception(f'Error importing DAQ module: {DAQ_module}')
            #self.DAQ = DAQ_lib.DAQ(self)
            if callable(DAQ_class := getattr(DAQ_lib, self.config['setup']['DAQ'])):
                print(f"Using DAQ: {self.config['setup']['DAQ']}")
                self.DAQ = DAQ_class(self)

        # loop through diagnostics and add
        self.diags = {}
        diag_config_filepath = Path(root_folder + 'diagnostics.toml')
        if os.path.exists(diag_config_filepath):
            self.diag_config = load_file(diag_config_filepath)
            for diag_name in self.diag_config: 
                self.add_diagnostic(diag_name)

    def add_diagnostic(self, diag_name):

        # TODO: add from seperate file?
        if not self.diag_config:
            raise Exception('No diagnostics config file loaded')
        if diag_name not in self.diag_config:
            raise Exception(f'Could not find diagnostic: {diag_name}')

        self.diag_config[diag_name]['name'] = diag_name

        if 'type' in self.diag_config[diag_name]:
            diag_type = self.diag_config[diag_name]['type']
        else:
            raise Exception(f'No diagnostic type defined for: {diag_name}')

        diag_module = 'LAMP.diagnostics.' + diag_type
        try:
            diag_lib = importlib.import_module(diag_module)
        except ImportError:
            raise Exception(f'Could not find Diagnostics module: {diag_module}')

        if callable(diag_class := getattr(diag_lib, diag_type)):
            print(f'Adding Diagnostic: {diag_name}')
            self.diags[diag_name] = diag_class(self, self.diag_config[diag_name])
        else:
            raise Exception(f'Could not find Diagnostic object: {diag_type}')

        return self.get_diagnostic(diag_name)
    
    def get_diagnostic(self, diag_name):
        if diag_name not in self.diags:
            raise Exception(f'Could not find Diagnostic: {diag_name}')
        return self.diags[diag_name]

    def list_diagnostics(self):
        for diag_name in self.diags.keys():
            print(f"{diag_name} [{self.diag_config[diag_name]['type']}]")
        return

