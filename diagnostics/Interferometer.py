import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..diagnostic import Diagnostic
from ..utils.image_proc import ImageProc

class Interferometer(Diagnostic):
    """Interferometer
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''
    data_type = 'image'

    curr_img = None
    img_units = 'Counts'
    x_mm, y_mm = None, None
    calib_dict = None

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

