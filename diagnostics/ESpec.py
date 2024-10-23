import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import re

from ..diagnostic import Diagnostic
from ..utils.image_proc import ImageProc
from ..utils.general import dict_update, mindex
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
    data_type = 'image'

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

        # set calibration dictionary
        if calib_id:
            self.calib_dict = self.get_calib(calib_id)
        else:
            self.calib_dict = self.get_calib(shot_dict)

        # do standard image calibration. Transforms, background, ROIs etc.
        img, x, y = self.run_img_calib(shot_dict, debug=debug)
        # minimum calibration is spatial transform
        #img, x, y = self.transform(shot_dict, self.calib_dict['transform'])

        # assuming mm here? 
        # either don't or use conversion functions...
        self.curr_img = img
        self.x_mm = x
        self.y_mm = y

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
            #if 'roi_MeV' in self.calib_dict:
            if ('roi' in self.calib_dict) and ('MeV' in self.calib_dict['roi']):
                MeV_min = np.min(self.calib_dict['roi']['MeV'])
                MeV_max = np.max(self.calib_dict['roi']['MeV'])
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
            #if 'roi_mrad' in self.calib_dict:
            if ('roi' in self.calib_dict) and ('mrad' in self.calib_dict['roi']):
                mrad_min = np.min(self.calib_dict['roi']['mrad'])
                mrad_max = np.max(self.calib_dict['roi']['mrad'])
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

    # Charge calibration functions
    def QLtoPSL(self, X, R=25, S=4000, L=5, G=16, scanner='GE'):
        if scanner.lower() == 'ge':
            # Maddox. For use on .gel files!
            # For S, you will need to know the PMT value at the time of scanning and use a calibration for S=4000/h(V)
            # Example values are for example; https://doi.org/10.1063/1.4886390
            # However, the livermore report gives more details on actual fit? LLNL-JRNL-606753
            # from ImageJ script 'PSL Convert from .gel';
            # (0.000023284*X*X/100000)*(Res/100)*(Res/100)*(4000/S)*316.2
            # below is the same, without assuming L=5, or G=16
            # the difference between gel and tif is a sqrt, then linear scale factor. For dynamic range reasons.
            return ((X/((pow(2,G))-1))**2)*((R/100)**2)*(4000/S)*pow(10,(L/2))
        elif scanner.lower() == 'fuji':
            # Vlad. For use on .tif files from FUJI machines
            g = pow(2,G) - 1
            return (R/100)**2 * (4000/S) * pow(10, L*(X/g - 0.5))

    def PSLtofC(self,PSL_val,IP_type='MS'):
        # https://dx.doi.org/10.1063/1.4936141
        if IP_type == 'TR':
            # above claims 0.005 PSL per electron for TR type. Error bar is 20%
            # 1 Coulumb is 6.241509×10^18 electrons
            # Therefore 1 C = 3.1207545e+16 PSL
            # or 0.032043 fC per PSL
            # From Jon Woods thesis, it takes 350 electrons to produce 1 PSL for TR.
            # 350 electrons is 0.056076183 fC (per PSL)
            # but the paper above is experimental measurements... gonna use it. Sorry Jon!
            return (PSL_val/(0.005*6.241509e18))*1e15 
        elif IP_type == 'SR':
            # as per above, but 0.0065 PSL per electron
            return (PSL_val/(0.0065*6.241509e18))*1e15
        elif IP_type == 'MS':
            # as per above, but 0.023 PSL per electron
            # ~0.007 fc per PSL
            return (PSL_val/(0.023*6.241509e18))*1e15 
        else:
            print('Error in PSLtofC(): Unkown Image Plate type')
            return None

    def IP_fade(self, t, IP_type='MS'):
        """ This is a normalisation factor 0->1 for signal fading on Image plate. Used on PSL values. https://dx.doi.org/10.1063/1.4936141"""
        if IP_type == 'TR':
            A1=0.535
            B1=23.812
            A2=0.465
            B2=3837.2
        elif IP_type == 'MS':
            A1=0.334
            B1=107.32
            A2=0.666
            B2=33974
        elif IP_type == 'SR':
            A1=0.579
            B1=15.052
            A2=0.421
            B2=3829.5
        else:
            print('Error in fade_time(): Unkown Image Plate type')
            return None

        if t > 4000:
            print('Warning, fade time factor fit not confirmed for t > 4000. ')
        
        f=A1*np.exp(-t/B1)+A2*np.exp(-t/B2)
        return f

    def IP_rescan_factor(self, filepath1, filepath2, roi=None, R=25, S=4000, bins=200, debug=True):
        imgA = ImageProc(filepath1)
        imgA_orig = imgA.get_img()
        imgA_res= imgA_orig # resampling??? be careful with R below, etc.
        imgA_PSL = self.QLtoPSL(imgA_res, R=R, S=S)
        #imgA_PSL = imgA_PSL / self.IP_fade(fade_t) # fade times cancel anyway in ratio (if they are close)? this rescan factor takes any difference into account anyway... Would also need IP type
        imgA_PSL[imgA_PSL < 1e-6] = 1e-6
        imgB = ImageProc(filepath2)
        imgB_orig = imgB.get_img()
        imgB_res= imgB_orig # resampling??? be careful with R below, etc.
        imgB_PSL = self.QLtoPSL(imgB_res, R=R, S=S)
        #imgB_PSL = imgB_PSL / self.IP_fade(fade_t)
        imgB_PSL[imgB_PSL < 1e-6] = 1e-6

        if roi is None:
            roi = [[0,0],[np.shape(imgA_orig)[1],np.shape(imgA_orig)[0]]]

        img_ratio = imgB_PSL[int(roi[0][1]):int(roi[1][1]),int(roi[0][0]):int(roi[1][0])] / imgA_PSL[int(roi[0][1]):int(roi[1][1]),int(roi[0][0]):int(roi[1][0])]
        img_ratio[img_ratio>2] = 0
        hist_data, bin_edges = np.histogram(img_ratio.flatten(), bins=bins) # this might need a bit of playing!
        bin_edges = (bin_edges[1:] + bin_edges[:-1])/2
        bin_edges_roi = bin_edges[(bin_edges>0.1) & (bin_edges<0.9)]
        maxi = np.argmax(hist_data[(bin_edges>0.1) & (bin_edges<0.9)]) 

        if debug:
            plt.figure()
            plt.plot(bin_edges, hist_data) 
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of Flattened 2D Array')
            plt.show(block=False)

        return bin_edges_roi[maxi]

    def IP_rescan_product(self, filenames, roi=None, R=25, S=4000, bins=200, debug=True):
        """Assuming they are in order from first scan to last"""
        rescan_product = 1
        for fi in range(1,len(filenames)):
            rescan_product = rescan_product * self.IP_rescan_factor(filenames[fi-1],filenames[fi], roi=roi, R=R, S=S, bins=bins, debug=debug)

        return rescan_product

    # ------------------------------------------------------ #
    # PLOTTING FUNCTIONS
    # ------------------------------------------------------ #

    def montage(self, timeframe, x_roi=None, MeV_roi=None, y_roi=None, mrad_roi=None, x_downsample=1, y_downsample=1, exceptions=None, vmin=None, vmax=None, transpose=True, num_rows=1):
        """Wrapper for diagnostic make_montage() function, mainly to set axis"""

        if not self.calib_dict:
            print('Missing Calibration before using Montage... Please set using set_calib(calib_id)')
            return False

        # convert rois to MeV mrad indices
        if 'dispersion' in self.calib_dict:
            MeV = self.calib_dict['dispersion']['MeV']
            
            # Apply the preset ROIs first. Otherwise trouble below!
            if ('roi' in self.calib_dict) and ('MeV' in self.calib_dict['roi']):
                MeV_min = np.min(self.calib_dict['roi']['MeV'])
                MeV_max = np.max(self.calib_dict['roi']['MeV'])
                MeV = MeV[(MeV > MeV_min)]
                MeV = MeV[(MeV < MeV_max)]

            axis = MeV
            axis_label = r'$E$ [MeV]'

            if MeV_roi is not None:
                MeV_min_i = np.min([mindex(MeV,MeV_roi[0]),mindex(MeV,MeV_roi[1])])
                MeV_max_i = np.max([mindex(MeV,MeV_roi[0]),mindex(MeV,MeV_roi[1])])
                if self.calib_dict['dispersion']['axis'].lower() == 'x':
                    x_roi=[MeV_min_i,MeV_max_i]
                else:
                    y_roi=[MeV_min_i,MeV_max_i]
        else:
            axis = None # self.x #??? or y?
            axis_label = '?'

        if 'divergence' in self.calib_dict and mrad_roi is not None:
            mrad = self.calib_dict['divergence']['mrad']

            # Apply the preset ROIs first. Otherwise trouble below?
            if ('roi' in self.calib_dict) and ('mrad' in self.calib_dict['roi']):
                mrad_min = np.min(self.calib_dict['roi']['mrad'])
                mrad_max = np.max(self.calib_dict['roi']['mrad'])
                mrad = mrad[(mrad > mrad_min)]
                mrad = mrad[(mrad < mrad_max)]

            mrad_min_i = np.min([mindex(mrad,mrad_roi[0]),mindex(mrad,mrad_roi[1])])
            mrad_max_i = np.max([mindex(mrad,mrad_roi[0]),mindex(mrad,mrad_roi[1])])

            if self.calib_dict['divergence']['axis'].lower() == 'y':
                y_roi=[mrad_min_i,mrad_max_i]
            else:
                x_roi=[mrad_min_i,mrad_max_i]

        fig, ax = self.make_montage(timeframe, x_roi=x_roi, y_roi=y_roi, axis=axis, axis_label=axis_label, x_downsample=x_downsample, 
                               y_downsample=y_downsample, vmin=vmin, vmax=vmax, transpose=transpose, num_rows=num_rows)

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