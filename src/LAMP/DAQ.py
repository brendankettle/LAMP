from matplotlib import image
import numpy as np
from pathlib import Path
from LAMP.utils.io import load_file

class DAQ():
    """Base class for DAQs.
    DAQs that call this function should have (at least), the following functions:
        - build_time_point(shot_dict)
        - get_filepath(diag_name, shot_dict) [or get_shot_data()?]
        - get_shot_dicts(diag_name, timeframe) ?
    """

    def __init__(self, exp_obj, config=None):
        self.ex = exp_obj # pass in experiment object
        self.data_folder = self.ex.config['paths']['data_folder'] # Path()?
        self.config = config
        return

    def load_imdata(self, shot_filepath, data_type=float):
        try:
            imdata = image.imread(Path(shot_filepath)).astype(data_type)
        except (FileNotFoundError) as Error:
            print(f'load_imdata() error: Could not find {shot_filepath}')
            imdata = None
        except (TypeError) as Error:
            print(f'load_imdata() error: File type error, could not find {shot_filepath}')
            imdata = None
        return imdata

    def load_data(self, shot_filepath, file_type=None, options=None):
        # To Do: need to implement proper IO class 
        data = load_file(Path(shot_filepath), file_type=file_type, options=options)
        return data
    
    def get_shot_data(self, diag_name, shot_dict):
        """Requires DAQ to provide get_filepath()
        """
        shot_filepath = self.get_filepath(diag_name, shot_dict)

        diag_config = self.ex.diags[diag_name].config
    
        if diag_config['data_type'] == 'image':
            shot_data = self.load_imdata(shot_filepath)
        elif diag_config['data_type'] == 'text':
            if 'data_options' in diag_config:
                options = diag_config['data_options'] 
            else:
                options = None
            shot_data = self.load_data(shot_filepath, file_type=diag_config['data_ext'], options=options)
        else:
            print(f"warning, data_type not recognised: {diag_config['data_type']}, for {diag_name}")
            shot_data = None

        return shot_data
    
    def shot_string(self, shot_dict):

        if isinstance(shot_dict, dict):
            if 'date' in shot_dict:
                datestr = 'Date: ' + str(shot_dict['date'])
            else:
                datestr = ''
            if 'run' in shot_dict:
                runstr = ', Run: ' + str(shot_dict['run'])
            else:
                runstr = ''
            if 'burst' in shot_dict:
                burststr = ', Burst: ' + str(shot_dict['burst'])
            else:
                burststr = ''
            if 'shotnum' in shot_dict:
                shotstr = ', Shot: ' + str(shot_dict['shotnum'])
            else:
                shotstr = ''
            return f"{datestr} {runstr} {burststr} {shotstr}"
        elif isinstance(shot_dict, str):
            # if it's a string, just return that again
            return shot_dict
    