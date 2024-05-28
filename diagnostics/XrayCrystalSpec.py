import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..diagnostic import Diagnostic


class XrayCrystalSpec(Diagnostic):
    """
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

    def plot_histogram(self, timeframe, bins=100):

        shot_dicts = self.DAQ.get_shot_dicts(self.diag_name, timeframe)

        raw_data = []
        for shot_dict in shot_dicts:
            print(shot_dict)
            raw_data.append(self.get_shot_data(shot_dict))

        # plt.figure()
        # plt.imshow(raw_img)
        # plt.show(block=False)

        #print(len(np.array(raw_data).flatten()))

        fig = plt.figure()
        n, bins, patches = plt.hist(np.array(raw_data).flatten(), bins=bins, log=True)
        plt.show(block=False)

        return n, bins

    def get_hist_sig(self, shot_dict, lefti, righti, bin_edges = range(0,5000), view=False):
        """Written on the fly for experiment, definitely needs cleaned up!"""

        raw_data = self.get_shot_data(shot_dict)

        n, bins = np.histogram(np.array(raw_data).flatten(), bins=bin_edges)

        x = (bins[1:] + bins[:-1])/2
        fitn = np.concatenate((n[lefti[0]:lefti[1]], n[righti[0]:righti[1]]))
        fitx = np.concatenate((x[lefti[0]:lefti[1]], x[righti[0]:righti[1]]))
        z = np.polyfit(fitx, fitn, 3)
        p = np.poly1d(z)
        sig = n[lefti[1]+1:righti[0]] - p(x[lefti[1]+1:righti[0]])

        #print(np.sum(sig))

        if view:
            plt.figure()
            plt.semilogy(x,n)
            plt.semilogy(fitx,fitn)
            plt.semilogy(x[lefti[1]+1:righti[0]],p(x[lefti[1]+1:righti[0]]))
            plt.show(block=False)
            
            plt.figure()
            plt.plot(x[lefti[1]+1:righti[0]],sig)
            plt.ylabel('No. photons per ADU')
            plt.xlabel('ADUs')
            plt.show(block=False)

        return np.sum(sig)
    
    def get_hist_sigs(self, timeframe, lefti, righti, bin_edges = range(0,5000)):

        shot_dicts = self.DAQ.get_shot_dicts(self.diag_name, timeframe)

        sigs = []
        for shot_dict in shot_dicts:
            print(shot_dict)
            sigs.append(self.get_hist_sig(shot_dict, lefti, righti, bin_edges = bin_edges))
        
        return sigs
