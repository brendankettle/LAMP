import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re

from .diagnostic import Diagnostic


class ScintillatorArray(Diagnostic):
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
    
    def get_proc_shot(self, shot_dict):
        """Return a processed shot using saved or passed calibrations.
        """

        # get calibration dictionary
        self.calib_dict = self.get_calib(shot_dict=shot_dict, input=True)

        img_data = self.get_shot_data(shot_dict)

        if 'roi' in self.calib_dict:
            roi = self.calib_dict['roi']
            img_data = img_data[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]

        rows, cols = np.shape(img_data)
        self.x = np.arange(cols)
        self.y = np.arange(rows)

        return img_data, self.x, self.y


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

    def plot_montage(self, timeframe, x_roi=None, y_roi=None, x_downsample=1, y_downsample=1, exceptions=None):

        # calling 'universal' DAQ function here, that is probably DAQ specific
        shot_dicts = self.DAQ.get_shot_dicts(self.diag_name,timeframe,exceptions=exceptions)

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

        # print(images.shape)

        # TODO: create montage should be outside of this class
        montage, x_ax = self.create_montage(images, x_roi, y_roi, x_downsample, y_downsample)

        # TODO: optional argument for function
        brightness_scale = np.percentile(montage, 99)

        if not x_roi:
            x_roi = [0,img.shape[1]]
        if not y_roi:
             y_roi = [0,img.shape[0]]

        # TODO: Assuming X axis here???
        xaxis = self.x[x_roi[0]:x_roi[1]:x_downsample]

        num_shots = len(shot_dicts)
        shotnum_tick_locs = range(int((montage.shape[1]/num_shots)/2),montage.shape[1]+int((montage.shape[1]/num_shots)/2),int(montage.shape[1]/num_shots))

        fig = plt.figure()
        ax = plt.gca()
        im = ax.pcolormesh(np.arange(montage.shape[1]), xaxis, montage, vmin=0.0, vmax=brightness_scale, shading='auto')
        ax.set_ylabel(r'X')
        ax.set_title(self.plot_make_title(timeframe), y=-0.2)
        divider = make_axes_locatable(ax)
        ax.set_xticks(shotnum_tick_locs)
        ax.set_xticklabels(shot_labels)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cb=plt.colorbar(im, cax=cax)
        #cb.set_label(r'Ed$^2$counts/d$\theta$d$E$ [counts mrad$^{-1}$]')
        plt.tight_layout()

        return fig
    
    # ------------------------------------------------------ #
    # PLOTTING FUNCTIONS
    # TODO: Move some of this to shared plotting class
    # ------------------------------------------------------ #
    
    def plot_make_title(self, shot_dict):
        if 'date' in shot_dict:
            datestr = ', Date: ' + str(shot_dict['date'])
        else:
            datestr = ''
        if 'run' in shot_dict:
            runstr = ', Run: ' + str(shot_dict['run'])
        else:
            runstr = ''
        if 'shotnum' in shot_dict:
            shotstr = ', Shot: ' + str(shot_dict['shotnum'])
        else:
            shotstr = ''
        return f"{self.diag_name} {datestr} {runstr} {shotstr}"

    # TODO: This needs cleaned up and moved to utils?
    def create_montage(self, image, x_roi=None, y_roi=None, x_downsample=1, y_downsample=1, transpose=True):
        #count, m, n = image.shape
        #mm = int(ceil(sqrt(count)))
        #nn = mm
        m, n, count = image.shape
        
        #print(m) # num y pixels
        #print(n) # num x pixels
        #print(count) # num images

        if x_roi:
            n = x_roi[1] - x_roi[0]
        else:
            x_roi = [0,image.shape[1]]
        if y_roi:
            m = y_roi[1] - y_roi[0]
        else:
            y_roi = [0,image.shape[0]]
        
        # m is energy axis, n is y axis
        m = int(m /  y_downsample)
        n = int(n / x_downsample)
        
        mm=count
        nn=1
        M = np.zeros((nn * n, mm * m))
        x_ax=np.linspace(0, m*(mm-1), count)+m/2.0

        image_id = 0
        for j in range(mm):
            for k in range(nn):
                if image_id >= count:
                    break
                sliceM, sliceN = j * m, k * n
                # TODO: Not sure this downsampling is bug free - can have rounding error?
                if transpose:
                    M[sliceN:sliceN + n, sliceM:sliceM + m] = image[y_roi[0]:y_roi[1]:y_downsample, x_roi[0]:x_roi[1]:x_downsample, image_id].T
                else:
                    M[sliceN:sliceN + n, sliceM:sliceM + m] = image[y_roi[0]:y_roi[1]:y_downsample, x_roi[0]:x_roi[1]:x_downsample, image_id]
                image_id += 1
        return M, x_ax