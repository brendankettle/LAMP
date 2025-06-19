import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..diagnostic import Diagnostic


class Camera(Diagnostic):
    """
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''
    data_type = 'image'

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

    def get_proc_shot(self, shot_dict, calib_id=None, debug=False):
        """Return a processed shot using saved or passed calibrations.
        """
        # set calibration dictionary
        if calib_id:
            self.calib_dict = self.get_calib(calib_id)
        else:
            self.calib_dict = self.get_calib(shot_dict)

        # do standard image calibration. Transforms, background, ROIs etc.
        # minimum calibration is spatial transform
        img, x, y = self.run_img_calib(shot_dict, debug=debug)

        return img, x, y
