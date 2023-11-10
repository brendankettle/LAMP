"""Base class for DAQs
"""
from matplotlib import image

class DAQ():

    def __init__(self):
        return
    
    def load_imdata(self, shot_filepath, data_type=float):
        imdata = image.imread(shot_filepath).astype(data_type)
        return imdata