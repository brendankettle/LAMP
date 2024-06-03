from matplotlib import image
import numpy as np
from pathlib import Path

class DAQ():
    """Base class for DAQs
    """

    def __init__(self, exp_obj):
        self.ex = exp_obj # pass in experiment object
        self.data_folder = Path(self.ex.config['paths']['data_folder'])
        return

    def load_imdata(self, shot_filepath, data_type=float):
        imdata = image.imread(Path(shot_filepath)).astype(data_type)
        return imdata
    
    def shot_string(self, shot_dict):
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

    