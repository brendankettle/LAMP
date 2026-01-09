import os
import numpy as np
import re
from pathlib import Path
from LAMP.DAQ import DAQ

class NoDAQ(DAQ):
    """Interface layer for no DAQ; filename based. Probably missing functionality.
    THIS ISN'T FINISHED AND IS UNTESTED. NOT SURE HOW IT WOULD WORK...
    Still have to set a subfolder for diagnostics? (can be blank?)
    What about file extensions?
    """

    __version = 0.1
    __name = 'NoDAQ'
    __authors = ['Brendan Kettle']


    def __init__(self, exp_obj, config=None):
        """Initiate parent base DAQ class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config=config)
        return

    def _build_shot_filepath(self, diagnostic, shot_dict, ext):
        """This is used internally, and so can be DAQ specific"""
        shot_path = Path(f'{self.data_folder}/{diagnostic}/{shot_dict}.{ext}')
        return shot_path

    def get_shot_info(self, shot_dict):
        print('No DAQ set; no info associated with shot.')
        return None

    def build_time_point(self, shot_dict):
        """Universal function to return a point in time for DAQ, for comparison, say in calibrations
        """
        # COULD HAVE A GO AT IDENTIFYING NUMBERS IN FILENAME?
        time_point = None
        return  time_point
    
    def file_exists(self, diag_name, shot_dict):
        diag_config = self.ex.diags[diag_name].config
        ext = diag_config['data_ext']
        folder = diag_config['data_folder']
        if os.path.isfile(self._build_shot_filepath(folder, shot_dict, ext)):
            return True
        else:
            return False

    def get_filepath(self, diag_name, shot_dict):
        """Required function to return shot filepath, given the diagnostic name and a shot dict (or filename str)"""
        diag_config = self.ex.diags[diag_name].config
        shot_filepath = self._build_shot_filepath(diag_config['data_folder'], shot_dict, diag_config['data_ext'])
        return shot_filepath
