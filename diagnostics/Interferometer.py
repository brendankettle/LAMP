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

    curr_img = None
    img_units = 'Counts'
    x_mm, y_mm = None, None
    calib_dict = None

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

    def get_proc_shot(self, shot_dict, calib_id=None, roi=None, debug=False):
        """Return a processed shot using saved or passed calibrations.
        """
        # set calibration dictionary
        if calib_id:
            self.calib_dict = self.get_calib(calib_id)
        else:
            self.calib_dict = self.get_calib(shot_dict)
        
        # do standard image calibration. Transforms, background, ROIs etc.
        img, x, y = self.run_img_calib(shot_dict, debug=debug)

        self.curr_img = img
        self.x = x
        self.y = y

        return img, x, y
    
