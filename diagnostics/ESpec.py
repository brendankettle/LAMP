import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import re

from ..diagnostic import Diagnostic
from ..utils.image_proc import ImageProc
from ..utils.general import dict_update
from ..utils.plotting import *

class ESpec(Diagnostic):
    """Electron (charged particle?) Spectrometer.
        TODO: Background correction
        TODO: Charge calibration
        TODO: Tracking sims
        TODO: Two screens?
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = 'cv2'

    # my (BK) thinking is that it is better to keep track of all the different units for the x/y axis
    # also, sticking to the same units (mm/MeV/mrad) helps make it easier to convert from different calibrations and simplify plotting
    curr_img = None
    img_units = 'Counts'
    x_mm, y_mm = None, None
    x_mrad, y_mrad = None, None
    x_MeV, y_MeV = None, None

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

    def get_proc_shot(self, shot_dict, calib_id=None, roi=None, apply_disp=True, apply_div=True, debug=False):
        """Return a processed shot using saved or passed calibrations.
        """
        # TO DO: SHOULD ALOT OF THIS BE IN DIAGNOSTIC CLASS? ALOT COULD APPLY TO ANY IMAGE PROCESSING

        # set calibration dictionary
        if calib_id:
            self.calib_dict = self.get_calib(calib_id)
        else:
            self.calib_dict = self.get_calib(shot_dict)

        # minimum calibration is spatial transform
        img, x, y = self.transform(shot_dict, self.calib_dict['transform'])
        self.curr_img = img
        self.x_mm = x
        self.y_mm = y

        if debug:
            plt.figure()
            im = plt.imshow(img, vmax=(0.2*np.max(img)))
            cb = plt.colorbar(im)
            plt.xlabel('new pixels')
            plt.ylabel('new pixels')
            plt.tight_layout()
            plt.show(block=False)

        # To Do: background correction should be handled elsewhere and shared?
        if 'bkg_type' in self.calib_dict:
            if self.calib_dict['bkg_type'] == 'flat':
                if 'bkg_roi' in self.calib_dict:
                    bkg_roi = self.calib_dict['bkg_roi']
                    bkg_value = np.mean(img[bkg_roi[0][1]:bkg_roi[1][1],bkg_roi[0][0]:bkg_roi[1][0]])
                    img = img - bkg_value
                else:
                    print(f"{self.config['name']}: No bkg_roi provided")
            if self.calib_dict['bkg_type'] == 'horizontal_poly':
                if 'bkg_roi' in self.calib_dict:
                    bkg_roi = self.calib_dict['bkg_roi']

                    bkg_px = np.arange(bkg_roi[0][0],bkg_roi[1][0])
                    bkg_lin = np.mean(img[bkg_roi[0][1]:bkg_roi[1][1],bkg_roi[0][0]:bkg_roi[1][0]], 0)
                    bkg_fit = np.polyfit(bkg_px, bkg_lin, 4)
                    bkg_func = np.poly1d(bkg_fit)
                    all_px = np.arange(0,np.shape(img)[1])
                    bkg_img = np.tile(bkg_func(all_px), (np.shape(img)[0],1))

                    if debug:
                        plt.figure()
                        plt.plot(bkg_px, bkg_lin)
                        plt.plot(all_px,bkg_func(all_px))
                        plt.xlabel('new pixels')
                        plt.ylabel('mean counts')
                        plt.tight_layout()
                        plt.title('Background correction')
                        plt.show(block=False)

                        plt.figure()
                        im = plt.imshow(bkg_img)
                        cb = plt.colorbar(im)
                        #cb.set_label(self.img_units, rotation=270, labelpad=20)
                        plt.tight_layout()
                        plt.title('Background correction')
                        plt.show(block=False)

                    img = img - bkg_img
                else:
                    print(f"{self.config['name']}: No bkg_roi provided")
            else:
                print(f"{self.config['name']}: Unknown background correction type '{self.calib_dict['bkg_type']}'")

        # dispersion?
        if apply_disp and 'dispersion' in self.calib_dict:
            img, MeV = self.apply_dispersion(img, self.calib_dict['dispersion'])

            # default to applying to X axis unless set
            if 'axis' in self.calib_dict['dispersion']:
                if self.calib_dict['dispersion']['axis'].lower() == 'y':
                    axis = 'y'
                else:
                    axis = 'x'
            else:
                axis = 'x'

            # ROI?
            if 'roi_MeV' in self.calib_dict:
                MeV_min = np.min(self.calib_dict['roi_MeV'])
                MeV_max = np.max(self.calib_dict['roi_MeV'])
                if axis == 'y':
                    self.y_mm = self.y_mm[(MeV > MeV_min)]# update spatial axis with ROI selection
                    img = img[(MeV > MeV_min), :]
                    MeV = MeV[(MeV > MeV_min)]
                    self.y_mm = self.y_mm[(MeV < MeV_max)] # update spatial axis with ROI selection
                    img = img[(MeV < MeV_max), :]
                    MeV = MeV[(MeV < MeV_max)]
                else:
                    # I'm sure this cam be done in one line, but I'm being lazy...
                    self.x_mm = self.x_mm[(MeV > MeV_min)] # update spatial axis with ROI selection
                    img = img[:, (MeV > MeV_min)]
                    MeV = MeV[(MeV > MeV_min)]
                    self.x_mm = self.x_mm[(MeV < MeV_max)] # update spatial axis with ROI selection
                    img = img[:, (MeV < MeV_max)]
                    MeV = MeV[(MeV < MeV_max)]

            # save axis to object
            if axis == 'y':
                self.y_MeV = MeV 
                y = MeV
            else:
                self.x_MeV = MeV 
                x = MeV

        # divergence?
        if apply_div and 'divergence' in self.calib_dict:
            img, mrad = self.apply_divergence(img, self.calib_dict['divergence'])

            # default to Y axis
            if 'axis' in self.calib_dict['divergence']:
                if self.calib_dict['divergence']['axis'].lower() == 'x':
                    axis = 'x'
                else:
                    axis = 'y'
            else:
                axis = 'y'

            # ROI?
            if 'roi_mrad' in self.calib_dict:
                mrad_min = np.min(self.calib_dict['roi_mrad'])
                mrad_max = np.max(self.calib_dict['roi_mrad'])
                if axis == 'y':
                    self.y_mm = self.y_mm[(mrad > mrad_min)] # update spatial axis with ROI selection
                    img = img[(mrad > mrad_min), :]
                    mrad = mrad[(mrad > mrad_min)]
                    self.y_mm = self.y_mm[(mrad < mrad_max)] # update spatial axis with ROI selection
                    img = img[(mrad < mrad_max), :]
                    mrad = mrad[(mrad < mrad_max)]
                else:
                    self.x_mm = self.x_mm[(mrad > mrad_min)] # update spatial axis with ROI selection
                    img = img[:, (mrad > mrad_min)]
                    mrad = mrad[(mrad > mrad_min)]
                    self.x_mm = self.x_mm[(mrad < mrad_max)] # update spatial axis with ROI selection
                    img = img[:, (mrad < mrad_max)]
                    mrad = mrad[(mrad < mrad_max)]

            # save axis to object
            if axis == 'y':
                self.y_mrad = mrad
                y = mrad
            else:
                self.x_mrad = mrad
                x = mrad

        return img, x, y
    
    def get_spectrum(self, shot_dict, calib_id=None, roi=None):
        """Integrate across the non-dispersive axis and return a spectral lineout"""
        img, x, y = self.get_proc_shot(shot_dict, calib_id=calib_id)

        if roi is None:
            roi = [[0,0],[np.shape(img)[1],np.shape(img)[0]]]

        spec = np.sum(img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]], 0)
        if 'axis' in  self.calib_dict['dispersion'] and self.calib_dict['dispersion']['axis'].lower() == 'y':
            MeV = y
        else:
            MeV = x
        MeV = MeV[roi[0][0]:roi[1][0]]
        return spec, MeV
    
    def get_div(self, shot_dict, calib_id=None):
        """Currently integrating across the spatial axis. Could be something more involved?"""
        img, x, y = self.get_proc_shot(shot_dict, calib_id=calib_id)
        sum_lineout = np.sum(img, 1)
        if 'axis' in  self.calib_dict['divergence'] and self.calib_dict['divergence']['axis'].lower() == 'x':
            mrad = x
        else:
            mrad = y
        return sum_lineout, mrad
    
    def get_div_FWHM(self, shot_dict, calib_id=None):
        lineout, mrad = self.get_div(shot_dict, calib_id=calib_id)
        # TODO: write library function to find FWHM from lineout
        # TODO: Return Error estimate as well
        FWHM = None
        return FWHM

    def make_dispersion(self, disp_dict, view=False):
        """"""

        # get dispersion curve from file
        disp_curve = self.load_calib_file(disp_dict['filename'])

        # TODO: Still asuming its spatial then spectral in data
        m,n = np.shape(disp_curve)
        if m > n:
            disp_spat = disp_curve[:,0]
            disp_spec = disp_curve[:,1]
        else:
            disp_spat = disp_curve[0,:]
            disp_spec = disp_curve[1,:]

        # remove any missing points
        disp_spat = disp_spat[~np.isnan(disp_spec)]
        disp_spec = disp_spec[~np.isnan(disp_spec)]
        disp_spat = disp_spat[disp_spec>0]
        disp_spec = disp_spec[disp_spec>0]

        if 'spatial_units' in disp_dict:
            spat_units = disp_dict['spatial_units']
        else:
            spat_units = 'mm'
        if 'spectral_units' in disp_dict:
            spec_units = disp_dict['spectral_units']
        else:
            spec_units = 'MeV'
        if 'axis' in disp_dict:
            axis = disp_dict['axis'].lower()
        else:
            axis = 'x'

        disp_fit = interp1d(self.to_mm(disp_spat,spat_units), self.to_MeV(disp_spec, spec_units),bounds_error=False, fill_value="extrapolate")

        if axis.lower() == 'x':
            MeV = disp_fit(self.x_mm)
            mm = self.x_mm
            self.x_MeV = MeV
        elif axis.lower() == 'y':
            MeV = disp_fit(self.y_mm)
            mm = self.y_mm
            self.y_MeV = MeV

        if view:
            plt.figure()
            plt.title('Displacement curve')
            plt.xlabel('mm')
            plt.ylabel('MeV')
            plt.plot(mm,MeV)
            plt.show(block=False)

        if 'dispersion' not in self.calib_dict:
            self.calib_dict['dispersion'] = {}

        # save details to calib dictionary
        dict_update(self.calib_dict['dispersion'],{
            "calib_curve": disp_curve,
            "calib_filename": disp_dict['filename'],
            "calib_spatial_units": spat_units,
            "calib_spectral_units": spec_units,
            "mm": mm,
            "MeV": MeV,
            "axis": axis
        })
            
        return MeV

    def apply_dispersion(self, img_data, disp_dict):
        """"""

        MeV = disp_dict['MeV']
        dMeV = abs(np.gradient(MeV)) # gradient is like diff, but calculates as average of differences either side

        if disp_dict['axis'] == 'x':
            self.x_MeV = MeV
            dMeV_matrix = np.tile(dMeV, (len(self.y_mm),1))
        elif disp_dict['axis'] == 'y':
            self.y_MeV = MeV
            dMeV_matrix = np.transpose(np.tile(dMeV, (len(self.y_mm),1)))

        # convert from counts to counts per MeV
        img_data = img_data / dMeV_matrix

        self.img_units = self.img_units + '_per_MeV'

        return img_data, MeV

    def make_divergence(self, div_dict, view=False):
        """"""

        mm_to_screen = div_dict['mm_to_screen']

        if 'axis' in div_dict:
            axis = div_dict['axis'].lower()
        else:
            axis = 'y'

        # could this be more complicated? like a function for distance to angle...
        if axis.lower() == 'x':
            mrad = np.arctan(self.x_mm / mm_to_screen) * 1000
            mm = self.x_mm
            self.x_mrad = mrad
        elif axis.lower() == 'y':
            mrad = np.arctan(self.y_mm / mm_to_screen) * 1000
            mm = self.y_mm
            self.y_mrad = mrad

        if 'divergence' not in self.calib_dict:
            self.calib_dict['divergence'] = {}

        # save details to calib dictionary
        dict_update(self.calib_dict['divergence'], {
            "mm_to_screen": mm_to_screen,
            "mm": mm,
            "mrad": mrad,
            "axis": axis
        })
            
        return mrad
    
    def apply_divergence(self, img_data, div_dict):
        """"""

        mrad = div_dict['mrad']
        dmrad = np.mean(np.diff(mrad)) # assuming linear for now...

        if div_dict['axis'] == 'x':
            self.x_mrad = mrad
        elif div_dict['axis'] == 'y':
            self.y_mrad = mrad

        # convert counts to per mrad
        img_data = img_data / dmrad
        self.img_units = self.img_units + '_per_mrad'

        return img_data, mrad

    # ------------------------------------------------------ #
    # PLOTTING FUNCTIONS
    # TODO: Move some of this to shared plotting class?
    # ------------------------------------------------------ #

    def montage(self, timeframe, x_roi=None, y_roi=None, x_downsample=1, y_downsample=1, exceptions=None, vmin=None, vmax=None, transpose=True, num_rows=1):

        # calling 'universal' DAQ function here, that is probably DAQ specific
        shot_dicts = self.DAQ.get_shot_dicts(self.config['name'],timeframe,exceptions=exceptions)

        shot_labels = []
        for shot_dict in shot_dicts:
            espec_img, x_MeV, y_mrad = self.get_proc_shot(shot_dict)

            if 'images' in locals():
                images = np.concatenate((images, np.atleast_3d(espec_img)), axis=2)
            else:
                images = np.atleast_3d(espec_img)

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
        fig, ax = plot_montage(images, x_roi=x_roi, y_roi=y_roi, axis=self.x_MeV, x_downsample=x_downsample, 
                               y_downsample=y_downsample, title=self.shot_string(timeframe), vmin=vmin, vmax=vmax, 
                               transpose=transpose, num_rows=num_rows, shot_labels=shot_labels)
        ax.set_ylabel(r'$E$ [MeV]')

        return fig, ax

    def plot_proc_shot(self, shot_dict, vmin=None, vmax=None, debug=False):

        # below still assumes X = spectral, Y =  divergence
        espec_img, x, y = self.get_proc_shot(shot_dict, debug=debug)

        if vmin is None:
            vmin = np.nanmin(espec_img)
        if vmax is None:
            vmax = np.nanmax(espec_img)

        fig = plt.figure()
        im = plt.pcolormesh(x, y, espec_img, vmin=vmin, vmax=vmax, shading='auto')
        cb = plt.colorbar(im)
        cb.set_label(self.img_units, rotation=270, labelpad=20)
        plt.title(self.shot_string(shot_dict))
        plt.xlabel('Electron energy [MeV]') # These could be wrong...
        plt.ylabel('Beam divergence [mrad]') # These could be wrong...
        plt.tight_layout()
        plt.show(block=False)

        return fig, plt.gca()
    
    def plot_spectrum(self, shot_dict, roi=None):

        spec, MeV = self.get_spectrum(shot_dict, roi=roi)

        fig = plt.figure()
        im = plt.plot(MeV, spec)
        plt.title(self.shot_string(shot_dict))
        plt.xlabel('MeV') 
        plt.ylabel('Counts per MeV')
        plt.tight_layout()
        plt.show(block=False)

        return fig