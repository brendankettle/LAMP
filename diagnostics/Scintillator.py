import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re

from ..diagnostic import Diagnostic
from ..utils.plotting import *

class Scintillator(Diagnostic):
    """
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return
    
    def get_proc_shot(self, shot_dict, calib_id=None):
        """Return a processed shot using saved or passed calibrations.
        """

        # set calibration dictionary
        if calib_id:
            self.calib_dict = self.get_calib(calib_id)
        else:
            self.calib_dict = self.get_calib(shot_dict)

        img_data = self.get_shot_data(shot_dict)

        if 'bkg_type' in self.calib_dict:
            if self.calib_dict['bkg_type'] == 'flat':
                if 'bkg_roi' in self.calib_dict:
                    bkg_roi = self.calib_dict['bkg_roi']
                    bkg_value = np.mean(img_data[bkg_roi[0][1]:bkg_roi[1][1],bkg_roi[0][0]:bkg_roi[1][0]])
                    img_data = img_data - bkg_value
                else:
                    print(f"{self.config['name']}: No bkg_roi provided")
            else:
                print(f"{self.config['name']}: Unknown background correction type '{self.calib_dict['bkg_type']}'")

        if 'roi' in self.calib_dict:
            roi = self.calib_dict['roi']
            img_data = img_data[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]

        rows, cols = np.shape(img_data)
        self.x = np.arange(cols)
        self.y = np.arange(rows)

        return img_data, self.x, self.y
    
    # ------------------------------------------------------ #
    # PLOTTING FUNCTIONS
    # TODO: Move some of this to shared plotting class
    # ------------------------------------------------------ #

    def montage(self, timeframe, x_roi=None, y_roi=None, x_downsample=1, y_downsample=1, exceptions=None, vmin=None, vmax=None, transpose=True, num_rows=1):

        # calling 'universal' DAQ function here, that is probably DAQ specific
        shot_dicts = self.DAQ.get_shot_dicts(self.config['name'],timeframe,exceptions=exceptions)

        shot_labels = []
        for shot_dict in shot_dicts:
            img, x, y = self.get_proc_shot(shot_dict)

            if 'images' in locals():
                images = np.concatenate((images, np.atleast_3d(img)), axis=2)
            else:
                images = np.atleast_3d(img)

            if 'burst' in shot_dict:
                m = re.search(r'\d+$', str(shot_dict['burst'])) # gets last numbers
                burst = int(m.group())
                burst_str = str(burst) + '|'
            else:
                burst_str = ''
            if 'shotnum' in shot_dict:
                shot_str = str(shot_dict['shotnum'])
            else:
                shot_str = ''
            shot_labels.append(burst_str + shot_str)

        # or y_MeV?
        fig, ax = plot_montage(images, x_roi=x_roi, y_roi=y_roi, x_downsample=x_downsample, y_downsample=y_downsample, 
                               title=self.shot_string(timeframe), vmin=vmin, vmax=vmax, 
                               transpose=transpose, num_rows=num_rows, shot_labels=shot_labels)
        #ax.set_ylabel(r'$E$ [MeV]')

        return fig, ax
    
        # def plot_histogram(self, timeframe, num_bins=100):

    #     shot_dicts = self.DAQ.get_shot_dicts(self.diag_name, timeframe)

    #     raw_data = []
    #     for shot_dict in shot_dicts:
    #         print(shot_dict)
    #         raw_data.append(self.get_shot_data(shot_dict))

    #     # plt.figure()
    #     # plt.imshow(raw_img)
    #     # plt.show(block=False)

    #     #print(len(np.array(raw_data).flatten()))

    #     plt.figure()
    #     plt.hist(np.array(raw_data).flatten(), bins=num_bins, log=True)
    #     plt.show(block=False)

    #    return