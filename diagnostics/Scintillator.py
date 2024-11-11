import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from scipy.optimize import curve_fit
from scipy.ndimage import binary_dilation, binary_fill_holes

from ..diagnostic import Diagnostic
from ..utils.plotting import *
from ..utils.general import mindex

class Scintillator(Diagnostic):
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

        #img_data = self.get_shot_data(shot_dict)
        # do standard image loading and calibration. Transforms, background, ROIs etc.
        img_data, x, y = self.run_img_calib(shot_dict, debug=debug)
        self.x = x
        self.y = y

        return img_data, self.x, self.y
    
    def get_scint_sigs(self, shot_dict, calib_id=None, debug=False):

        # if not self.calib_dict and not calib_id:
        #     print('Missing Calibration before using get_scint_sigs... Please set using set_calib(calib_id) or pass a calib_id')
        #     return False
        if 'extraction' not in self.calib_dict:
            print('No extraction calibration set')
            return False        
        ext_dict = self.calib_dict['extraction']

        img, x, y = self.get_proc_shot(shot_dict, calib_id=calib_id, debug=debug)

        scint_sigs = np.zeros([ext_dict['num_cols'],ext_dict['num_rows']])

        # make scintiallator crystal coordinates
        scint_centres_x = ext_dict['start_x'] + ext_dict['spacing_x'] * np.array(range(ext_dict['num_cols'])) 
        scint_centres_y = ext_dict['start_y'] + ext_dict['spacing_y'] * np.array(range(ext_dict['num_rows'])) 
        # convert to pixel numbers from spatial definitions
        scx_px = mindex(x, scint_centres_x)
        scy_px = mindex(y, scint_centres_y)

        wr = int(ext_dict['sample_width']/2)
        hr = int(ext_dict['sample_height']/2)

        # TODO: Would be better to integrate signal in crystal masked area around each of these points
        # i.e. get all the signal from a crystal, then give average counts per pixel?

        if(debug):
            plt.figure()
            im = plt.imshow(img, vmax=np.percentile(img,90))
            cb = plt.colorbar(im)
            ax = plt.gca()
            img_masked = np.copy(img)

        for ci in range(len(scx_px)):
            for ri in range(len(scy_px)):
                x = scx_px[ci]
                y = scy_px[ri]
                # do some masking?
                if 'threshold' in ext_dict:
                    threshold_val = ext_dict['threshold']
                    bottom = y-hr
                    top = y+hr
                    left = x-wr
                    right = x+wr
                    if (bottom < 0): bottom = 0
                    if (left < 0): left = 0
                    if (top > np.shape(img)[0]): top = np.shape(img)[0]
                    if (right > np.shape(img)[1]): np.shape(img)[1]
                    masked_element = np.copy(img[bottom:top,left:right])
                    masked_element_max = np.percentile(masked_element,95)
                    masked_element_min = np.percentile(masked_element,5)
                    masked_element[masked_element<(((masked_element_max-masked_element_min)*threshold_val)+masked_element_min)] = 0
                    # try filling out mask for holes using morphology
                    binary_mask = np.copy(masked_element)
                    binary_mask[binary_mask>0] = 1
                    binary_mask = binary_dilation(binary_mask, iterations=1).astype(int)
                    #binary_mask = binary_fill_holes(binary_mask).astype(int)
                    masked_element = img[bottom:top,left:right]*binary_mask
                    num_pixels = np.count_nonzero(masked_element)
                    scint_sigs[ci,ri] = np.sum(masked_element) / num_pixels
                    if(debug):
                        masked_element_hl = masked_element
                        masked_element_hl[masked_element_hl==0] = -100
                        img_masked[bottom:top,left:right] = masked_element_hl
                else:
                    scint_sigs[ci,ri] = np.mean(img[y-hr:y+hr,x-wr:x+wr])
                if(debug):
                    rect = patches.Rectangle((x-wr, y-hr), 2*wr, 2*hr, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
        #if(debug): cb = plt.colorbar(im)

        if(debug):
            plt.figure()
            im = plt.imshow(img_masked, vmax=np.percentile(img_masked,90))
            cb = plt.colorbar(im)

        # THIS IS A BIT CRUDE? Would be better to calculate at each point...
        # Also probably doesn't work if there is an offset...
        mm2_per_px = np.mean(scint_centres_x / scx_px) * np.mean(scint_centres_y / scy_px)
        #print(mm2_per_px)
        #mm2_per_px = self.calib_dict['scale_x'] * self.calib_dict['scale_y']

        # returning average counts per mm2 of scintalator?
        return scint_sigs.T / mm2_per_px

    def circ_to_depth(self,img):
        # assuming equal vert/hor viewing
        radius = int((np.shape(img)[0]+np.shape(img)[1])/2/2)

        centre_x = radius

        depth_img = np.zeros((np.shape(img)[0],np.shape(img)[1]))
        # loop through rows
        for ri in range(0,np.shape(img)[0]):
            height = radius - ri
            first_px = radius - int(np.sqrt(radius**2-height**2))
            depth_img[ri,0:(2*radius)-(2*first_px)] = img[ri,first_px:(2*radius)-first_px]

        return depth_img

    # ------------------------------------------------------ #
    # PLOTTING FUNCTIONS
    # ------------------------------------------------------ #
        
    def Plot_scan_average_over_shots(self, timeframe, skipshots, dist_to_screen, n_per_pos, exceptions=None, x_axis=True, y_axis=False):
    # Function written by CMcA to plot horizontal lineout and calculate divergence of the beam from FWHM calc
    # TODO = Get skipshots to work to avoid misfires
    
        def gaus(x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        
        shot_dicts = self.DAQ.get_shot_dicts(self.config['name'],timeframe,exceptions=exceptions)
        shot_labels = []
        scan_average = []
        scan_sum = []
        sum_potter = []
        fwhm_vals = []
        sum_pixels = []
        k = 0
        fig1 = plt.figure(figsize=(3,2), dpi=150)
        for n, shot_dict in enumerate(shot_dicts):
            img, x, y = self.get_proc_shot(shot_dict)
            print(shot_dict)
            # sum_pixels.append(np.sum(img[200:800, 200:800]))
            if x_axis == True:
                lineout = img[490:510,:]
            elif y_axis == True:
                lineout = img[:,490:510]

            average = np.mean(lineout, axis=0)
            scan_average.append(average)
            # scan_sum.append(sum_pixels)
            if n == 0:
                pass
            elif (n+1)%n_per_pos == 0:
                k+=1
                average = np.mean(scan_average, axis=0)
                #             	summer = np.mean(scan_sum)
                err = np.std(scan_average, axis=0)
                plt.plot(np.arange(len(average))*0.305, average, label='Set '+str(k))
                plt.fill_between(np.arange(len(average))*0.305, average+err, average-err, alpha=0.2)

                x = np.arange(len(average))*0.305
                x1 = x
                isel = (x<=125)|(x>=175)
                x = x[isel]
                y = average
                y = y[isel]

                popt,pcov = curve_fit(gaus,x,y,p0=[3000,150,30])
                #             	plt.plot(x1,gaus(x1,*popt),linestyle=':')

                fwhm = np.where(y>=np.max(y)/2)
                fwhm = fwhm[0]

                fwhm_vals.append((fwhm[-1]-fwhm[0])*0.305)
                scan_average = []
            	
            	
        plt.legend(fontsize=6, handlelength=0.2)
        plt.xlabel('X [mm]')
        plt.title('Averaged lineouts')
        plt.tight_layout()
        
        fig2 = plt.figure(figsize=(3,2), dpi=150)
        plt.plot(np.arctan((np.array(fwhm_vals)*0.305)/dist_to_screen)*1e3)
        plt.ylabel('Divergence [mrad]')
        plt.xlabel('Set no.')
        plt.title('Divergence with scan')
        
#         plt.twinx()
#         plt.plot(sum_potter)
#         plt.ylabel('Sum pixel count')
        
        plt.tight_layout()
    

        return fig1, fig2
    
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

    def montage(self, timeframe, calib_id=None, x_roi=None, y_roi=None, x_downsample=1, y_downsample=1, exceptions=None, vmin=None, vmax=None, transpose=True, num_rows=1, debug=False):
        """This actually builds the montage shots, and feeds into the plotting functions"""

        axis_label = ''
        axis = None

        if calib_id:
            self.calib_dict = self.get_calib(calib_id)
        # ???
        # if not self.calib_dict:
        #     print('Missing Calibration before using Montage...')
        #     return False

        # calling 'universal' DAQ function here, that is probably DAQ specific
        shot_dicts = self.DAQ.get_shot_dicts(self.config['name'],timeframe,exceptions=exceptions)

        shot_labels = []
        for shot_dict in shot_dicts:
            # To Do, if no get_proc_shot, just use raw data?
            # To Do: Pass ROIs here not montage function below?
            img, x, y = self.get_proc_shot(shot_dict, calib_id=calib_id, debug=debug)

            if 'images' in locals():
                images = np.concatenate((images, np.atleast_3d(img)), axis=2)
            else:
                images = np.atleast_3d(img)

            # try build a shot label
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

        if 'transform' in self.calib_dict:
            axis_label = 'mm'
            axis = x
            # or y??? need some logic above... different for transpose for example?

        cb_label = 'Counts?'

        fig, ax = plot_montage(images, x_roi=x_roi, y_roi=y_roi, axis=axis, x_downsample=x_downsample, 
                               y_downsample=y_downsample, title=self.shot_string(timeframe), vmin=vmin, vmax=vmax, 
                               transpose=transpose, num_rows=num_rows, cb_label=cb_label, y_label=axis_label, shot_labels=shot_labels)

        return fig, ax