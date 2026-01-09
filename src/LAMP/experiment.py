"""Main entry point for analysis scripts; Experiment class
"""
import os
from pathlib import Path
import importlib
from LAMP.utils.io import load_file
from LAMP.results import Results

class Experiment:

    def __init__(self, root_folder, local_config='local.toml', global_config='global.toml', diagnostics='diagnostics.toml'):
        """LAMP entry point. Load config, load DAQ, add diagnostics"""

        # Load local config
        local_config_filepath = Path(root_folder) / Path(local_config)
        if not os.path.exists(local_config_filepath):
            raise Exception(f'Problem finding local config file: {local_config_filepath}')
        local_config = load_file(local_config_filepath)

        # load global config and save to object
        global_config_filepath = Path(root_folder) / Path(global_config)
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
            except ImportError as exc:
                # User supplied DAQ?
                if 'user_DAQs' in self.config['paths']:
                    user_DAQ_module = self.config['paths']['user_DAQs'] + self.config['setup']['DAQ']
                    try:
                        DAQ_lib = importlib.import_module(user_DAQ_module)
                    except ImportError as user_exc:
                        print(f'Warning! Could not import DAQ module: {DAQ_module} (or {user_DAQ_module}). Exceptions given below.')
                        print('Exception when tried user module:')
                        print(user_exc)
                        print('Exception when tried LAMP module:')
                        raise Exception(exc)
                    else:
                        user_text = '(User) ' 
                else:
                    print(f'Warning! Could not import Diagnostics module: {DAQ_module}. Exception given below.')
                    raise Exception(exc)
            else:
                user_text = ''
          
            #self.DAQ = DAQ_lib.DAQ(self)
            #if callable(DAQ_class := getattr(DAQ_lib, self.config['setup']['DAQ'])):
            DAQ_class = getattr(DAQ_lib, self.config['setup']['DAQ'])
            if callable(DAQ_class):
                print(f"Using {user_text}DAQ: {self.config['setup']['DAQ']}")
                if 'DAQ_config' in self.config:
                    self.DAQ = DAQ_class(self, self.config['DAQ_config'])
                else:
                    self.DAQ = DAQ_class(self)
            else:
                print(f"Error! Could not call {user_text}DAQ: {self.config['setup']['DAQ']}")

        # loop through diagnostics and add
        self.diags = {}
        diag_config_filepath = Path(root_folder) / Path(diagnostics)
        if os.path.exists(diag_config_filepath):
            self.diag_config = load_file(diag_config_filepath)
            for diag_name in self.diag_config: 
                if ('on_startup' in self.diag_config[diag_name] and self.diag_config[diag_name]['on_startup']) or ('on_startup' not in self.diag_config[diag_name]):
                    self.add_diagnostic(diag_name)

        # loop through metas and add
        # To Do: Not really made use of this yet... Meant for shot sheets etc.
        self.metas = {}
        meta_config_filepath = Path(root_folder) / Path('metas.toml')
        if os.path.exists(meta_config_filepath):
            self.meta_config = load_file(meta_config_filepath)
            for meta_name in self.meta_config: 
                self.add_meta(meta_name)

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

        found = False
        diag_module = 'LAMP.diagnostics.' + diag_type
        try: 
            diag_lib = importlib.import_module(diag_module)
        except ImportError as exc:
            # User module?
            if 'user_diagnostics' in self.config['paths']:
                user_diag_module = self.config['paths']['user_diagnostics'] + diag_type
                try:
                    diag_lib = importlib.import_module(user_diag_module)
                except ImportError as user_exc:
                    print(f'Warning! Could not import Diagnostics module: {diag_module} (or {user_diag_module}). Exceptions given below.')
                    print('Exception when tried user module:')
                    print(user_exc)
                    print('Exception when tried LAMP module:')
                    print(exc)
                    self.diags[diag_name] = False
                else:
                    found = True
                    user_text = '(User) ' 
            else:
                print(f'Warning! Could not import Diagnostics module: {diag_module}. Exception given below.')
                print(exc)
                self.diags[diag_name] = False
                #raise Exception(f'Could not import Diagnostics module: {diag_module}')
        else:
            found = True
            user_text = ''
        
        if found:
            #if callable(diag_class := getattr(diag_lib, diag_type)):
            diag_class = getattr(diag_lib, diag_type)
            if callable(diag_class):
                print(f'Adding {user_text}Diagnostic: {diag_name} [{diag_type}]')
                self.diags[diag_name] = diag_class(self, self.diag_config[diag_name])
            else:
                # what's different about this to above?
                print(f'Warning! Could not find Diagnostic object: {diag_type}')
                self.diags[diag_name] = False
                #raise Exception(f'Could not find Diagnostic object: {diag_type}')

        return self.get_diagnostic(diag_name)
    
    def get_diagnostic(self, diag_name):
        if diag_name not in self.diags:
            # if it doesn't exist, try to add diagnostic. Will error out inside if still not found
            self.add_diagnostic(diag_name)
        return self.diags[diag_name]

    def list_diagnostics(self):
        for diag_name in self.diags.keys():
            print(f"{diag_name} [{self.diag_config[diag_name]['type']}]")
        return
    
    def add_meta(self, meta_name):

        if not self.meta_config:
            raise Exception('No meta config file loaded')
        if meta_name not in self.meta_config:
            raise Exception(f'Could not find diagnostic: {meta_name}')

        self.meta_config[meta_name]['name'] = meta_name

        if 'type' in self.meta_config[meta_name]:
            meta_type = self.meta_config[meta_name]['type']
        else:
            raise Exception(f'No meta type defined for: {meta_name}')

        meta_module = 'LAMP.metas.' + meta_type
        try:
            meta_lib = importlib.import_module(meta_module)
        except ImportError:
            raise Exception(f'Could not find Meta module: {meta_module}')

        meta_class = getattr(meta_lib, meta_type)
        if callable(meta_class):
            print(f'Adding Meta: {meta_name} [{meta_type}]')
            self.metas[meta_name] = meta_class(self, self.meta_config[meta_name])
        else:
            raise Exception(f'Could not find Meta object: {meta_type}')

        return self.get_meta(meta_name)
    
    def get_meta(self, meta_name):
        if meta_name not in self.metas:
            raise Exception(f'Could not find Meta: {meta_name}')
        return self.metas[meta_name]

    def list_metas(self):
        for meta_name in self.metas.keys():
            print(f"{meta_name} [{self.meta_config[meta_name]['type']}]")
        return
    
    def open_results(self, db_name):
        db = Results(self, self.config, db_name)
        return db

    def make_all_calibs(self):
        """Loop through all diagnostics, for each, loop through calibrations, if proc file set, make"""
        for diag_name in self.diags.keys():
            print(f"Making calibrations for {diag_name} [{self.diag_config[diag_name]['type']}]")
            diag = self.get_diagnostic(diag_name)
            for calib_id in diag.list_calibs():
                calib = diag.get_calib(calib_id, no_proc=True)
                if 'proc_file' in calib:
                    print(f'Processing [{calib_id}]')
                    diag.make_calib(calib_id, save=True, view=False)
                else:
                    print(f'Skipping [{calib_id}] (No processed savepath provided)')
        return

