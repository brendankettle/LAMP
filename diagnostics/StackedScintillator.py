import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .diagnostic import Diagnostic


class StackedScintillator(Diagnostic):
    """
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

    def get_integrated_signal(self, shot_dict, roi=None):
        imdata = self.get_shot_data(shot_dict)
        if not roi:
            height, width = np.shape(imdata)
            roi = [0,height,0,width]
        return np.sum(imdata[roi[0]:roi[1],roi[2]:roi[3]])
    
    def get_integrated_signals(self, timeframe, roi=None):

        shot_dicts = self.DAQ.get_shot_dicts(self.diag_name, timeframe)

        int_data = []
        for shot_dict in shot_dicts:
            print(shot_dict)
            int_data.append(self.get_integrated_signal(shot_dict, roi=roi))
        
        return int_data

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

        return
