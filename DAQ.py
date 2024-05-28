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

    