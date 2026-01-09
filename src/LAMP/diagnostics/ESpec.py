import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import re

from LAMP.diagnostic import Diagnostic
from LAMP.utils.image_proc import ImageProc
from LAMP.utils.general import dict_update, mindex
from LAMP.utils.plotting import *

class ESpec(Diagnostic):
    """Electron (charged particle?) Spectrometer.
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
    img_units = ['Counts']
    x_mm, y_mm = None, None
    x_mrad, y_mrad = None, None
    x_MeV, y_MeV = None, None

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

    def get_proc_shot(self, shot_dict, calib_id=None, apply_disp=True, apply_div=True, apply_charge=True, roi_mm=None, roi_MeV=None, roi_mrad=None, debug=False):
        """Return a processed shot using saved or passed calibrations.
        Wraps base diagnostic class function, adding dispersion, divergence, charge.
        """

        # use diagnostic base function
        # loads calib id and run_img_calib for standard calibration routines
        img, x, y = super().get_proc_shot(shot_dict, calib_id=calib_id, debug=debug)

        if img is None:
            return None, None, None

        # assuming mm here for units
        # either don't or use conversion functions...
        self.x_mm = x
        self.y_mm = y

        # TO DO: roi_mm? only use if not setting dispersion or divergence below...

        # dispersion?
        if apply_disp and 'dispersion' in self.calib_dict:
            img, MeV = self.apply_dispersion(img, self.calib_dict['dispersion'])

            # default to applying to X axis unless set
            if 'axis' in self.calib_dict['dispersion'] and self.calib_dict['dispersion']['axis'].lower() == 'y':
                axis = 'y'
            else:
                axis = 'x'

            # no ROI passed? use defaults from calibration
            if not roi_MeV:
                # defaults in calibration dictionary?
                if 'roi' in self.calib_dict and 'MeV' in self.calib_dict['roi']:
                    MeV_min = np.min(self.calib_dict['roi']['MeV'])
                    MeV_max = np.max(self.calib_dict['roi']['MeV'])   
                # nope, no ROIs...
                else:
                    MeV_min = np.min(MeV)
                    MeV_max = np.max(MeV)     
            # ROI passed...
            else:
                MeV_min = np.min(roi_MeV)
                MeV_max = np.max(roi_MeV)

            # Apply ROIs...
            # NB: No other ROIs should be applied until this 'final' step, so there are no conflicts. If wrapping this function, just pass in ROI values
            if axis == 'y':
                self.y_mm = self.y_mm[(MeV >= MeV_min)]# update spatial axis with ROI selection
                img = img[(MeV >= MeV_min), :]
                MeV = MeV[(MeV >= MeV_min)]
                self.y_mm = self.y_mm[(MeV <= MeV_max)] # update spatial axis with ROI selection
                img = img[(MeV <= MeV_max), :]
                MeV = MeV[(MeV <= MeV_max)]
                self.y_MeV = MeV 
                y = MeV
            else:
                # I'm sure this cam be done in one line, but I'm being lazy...
                self.x_mm = self.x_mm[(MeV >= MeV_min)] # update spatial axis with ROI selection
                img = img[:, (MeV >= MeV_min)]
                MeV = MeV[(MeV >= MeV_min)]
                self.x_mm = self.x_mm[(MeV <= MeV_max)] # update spatial axis with ROI selection
                img = img[:, (MeV <= MeV_max)]
                MeV = MeV[(MeV <= MeV_max)]
                self.x_MeV = MeV 
                x = MeV

        # divergence?
        if apply_div and 'divergence' in self.calib_dict:
            img, mrad = self.apply_divergence(img, self.calib_dict['divergence'])

            # default to Y axis
            if 'axis' in self.calib_dict['divergence'] and self.calib_dict['divergence']['axis'].lower() == 'x':
                    axis = 'x'
            else:
                axis = 'y'

            # no ROI passed? use defaults from calibration
            if not roi_mrad:
                # defaults in calibration dictionary?
                if 'roi' in self.calib_dict and 'mrad' in self.calib_dict['roi']:
                    mrad_min = np.min(self.calib_dict['roi']['mrad'])
                    mrad_max = np.max(self.calib_dict['roi']['mrad'])   
                # nope, no ROIs...
                else:
                    mrad_min = np.min(mrad)
                    mrad_max = np.max(mrad)     
            # ROI passed...
            else:
                mrad_min = np.min(roi_mrad)
                mrad_max = np.max(roi_mrad)

            # Apply ROIs...
            # NB: No other ROIs should be applied until this 'final' step, so there are no conflicts. If wrapping this function, just pass in ROI values
            if axis == 'y':
                self.y_mm = self.y_mm[(mrad > mrad_min)] # update spatial axis with ROI selection
                img = img[(mrad > mrad_min), :]
                mrad = mrad[(mrad > mrad_min)]
                self.y_mm = self.y_mm[(mrad < mrad_max)] # update spatial axis with ROI selection
                img = img[(mrad < mrad_max), :]
                mrad = mrad[(mrad < mrad_max)]
                self.y_mrad = mrad
                y = mrad
            else:
                self.x_mm = self.x_mm[(mrad > mrad_min)] # update spatial axis with ROI selection
                img = img[:, (mrad > mrad_min)]
                mrad = mrad[(mrad > mrad_min)]
                self.x_mm = self.x_mm[(mrad < mrad_max)] # update spatial axis with ROI selection
                img = img[:, (mrad < mrad_max)]
                mrad = mrad[(mrad < mrad_max)]
                self.x_mrad = mrad
                x = mrad

        # charge calibration?
        if apply_charge and 'charge' in self.calib_dict:
            if 'fC_per_count' in self.calib_dict['charge']:
                img = img * self.calib_dict['charge']['fC_per_count']
                if 'Counts' in self.img_units:
                    self.img_units.remove('Counts')
                if 'fC' not in self.img_units:
                    self.img_units.insert(0,'fC')

        return img, x, y
    
    def get_spectrum(self, shot_dict, calib_id=None, roi_MeV=None, roi_mrad=None,  debug=False):
        """Integrate across the non-dispersive axis and return a spectral lineout"""
        img, x, y = self.get_proc_shot(shot_dict, calib_id=calib_id, roi_MeV=roi_MeV, roi_mrad=roi_mrad, debug=debug)

        if img is None:
            return None, None

        if 'axis' in self.calib_dict['dispersion'] and self.calib_dict['dispersion']['axis'].lower() == 'y':
            spec = np.sum(img, 1)
            MeV = y
        else:
            spec = np.sum(img, 0)
            MeV = x
        
        # normalise out the /mrad units
        if 'divergence' in self.calib_dict:
            mrad = self.calib_dict['divergence']['mrad']
            dmrad = np.mean(np.diff(mrad)) # assuming linear for now...
            # convert counts per mrad back to counts
            spec = spec * dmrad

        # Units?; if charge is set, it will be fC/MeV

        # let's sort the arrays to make sure MeV is increasing
        spec = [x for _, x in sorted(zip(MeV, spec))]
        MeV = sorted(MeV)

        return spec, MeV
    
    def get_spectra(self, timeframe, calib_id=None, roi_MeV=None, roi_mrad=None,  debug=False):

        shot_dicts = self.DAQ.get_shot_dicts(self.config['name'],timeframe)
        specs = []
        MeVs = []
        for shot_dict in shot_dicts:
            spec, MeV = self.get_spectrum(shot_dict, calib_id=None, roi_MeV=None, roi_mrad=None,  debug=False)
            specs.append(spec)
            MeVs.append(MeV)
        return specs, MeVs
    
    def get_spectrum_metrics(self, shot_dict, calib_id=None, roi_MeV=None, roi_mrad=None, percentile=95, debug=False):
        """"""
        spec, MeV = self.get_spectrum(shot_dict, calib_id=calib_id, roi_MeV=roi_MeV, roi_mrad=roi_mrad,  debug=debug)

        if spec is None:
            return None, None, None

        # first apply some smoothing, to reduce noise effects. These details could be passed as options?
        spec = savgol_filter(spec, int(len(MeV)/50), 2)

        # should we do this?
        spec[spec<0] = 0

        # ?? normalise spectrum, multiply by energy, then find mean
        #E_mean = np.mean(spec * MeV) / np.mean(spec)
        # following code is taken from GeminiRR21 code
        # normalise distribution by area under, then find mean using under under weighted spectrum?
        spec_dist = np.abs(spec/np.trapz(spec, MeV))
        if debug:
            plt.figure()
            plt.plot(MeV,spec_dist)
            plt.xlabel('Electron Energy [MeV]')
            plt.ylabel('Normalised spectral distribution')
            plt.show(block=False)
        E_mean = np.abs(np.trapz(spec_dist*MeV, MeV))
        E_std = np.sqrt(np.abs(np.trapz(spec_dist*(MeV-E_mean)**2, MeV)))

        # find last array position over threshold
        # spec_thres = np.max(spec) * ((100-percentile)/100)
        # E_max = np.max(MeV[np.where(spec > spec_thres)])

        # following code is taken from GeminiRR21 code
        div = 10.0 # resolution of step in scans?
        target_percentile = percentile / 100
        N = int(len(spec_dist)/div)
        percentile, energy = np.zeros(N), np.zeros(N)
        for i in range(0, N):
            max_index = len(MeV)-1-int(div)*i # work backwards
            percentile[i] = np.abs(np.trapz(spec_dist[0:max_index], MeV[0:max_index]))
            energy[i] = MeV[max_index]
            if percentile[i] < target_percentile-0.05: # are we going past to interpolate back?
                break
        percentile_cut = percentile[percentile!=0.0]
        energy_cut = energy[percentile!=0.0]
        energy_cut = energy_cut[percentile_cut<0.999]
        percentile_cut = percentile_cut[percentile_cut<0.999]
        if debug:
            plt.figure()
            plt.plot(percentile_cut,energy_cut)
            plt.xlabel('Percentile of total counts')
            plt.xlabel('Electron Energy [MeV]')
            plt.show(block=False)
        # interpolate back to answer at exact percentile. interp needs sorted arrays!
        energy_cut = [x for _, x in sorted(zip(percentile_cut, energy_cut))]
        percentile_cut = sorted(percentile_cut)
        E_percentile = np.interp(target_percentile, percentile_cut, energy_cut)

        return E_mean, E_std, E_percentile 
    
    # def mean_and_std_beam_energy(self,img_raw):
    #     """ Gets mean and std of electron energy. Returns electron energy at 90th percentile of charge distribution.
    #     """
    #     img_pC_permm2 = self.espec_data2screen(img_raw)
    #     img_pC_per_MeV = np.trapz(self.espec_screen2spec(img_pC_permm2), self.screen_y_mm, axis=0)
    #     spec_charge_dist= img_pC_per_MeV/np.trapz(img_pC_per_MeV, self.eAxis_MeV)
    #     spec_charge_dist[spec_charge_dist<=0]=0.0
    #     mean_energy = np.trapz(spec_charge_dist*self.eAxis_MeV, self.eAxis_MeV)
    #     #Exp_energy_sqrd = np.trapz(spec_charge_dist*self.eAxis_MeV**2, self.eAxis_MeV)
    #     variance = np.trapz(spec_charge_dist*(self.eAxis_MeV-mean_energy)**2, self.eAxis_MeV)

    #     div=10.0
    #     N=int(len(spec_charge_dist)/div)
    #     percentile, energy=np.zeros(N), np.zeros(N)
    #     target_percentile=0.9

    #     for i in range(0, N):
    #         percentile[i]=np.trapz(spec_charge_dist[0:len(self.eAxis_MeV)-1-int(div)*i], self.eAxis_MeV[0:len(self.eAxis_MeV)-1-int(div)*i])
    #         energy[i]=self.eAxis_MeV[len(self.eAxis_MeV)-1-int(div)*i]
    #         if percentile[i]<target_percentile-0.05:
    #             break
    #     percentile_cut=percentile[percentile!=0.0]
    #     energy_cut=energy[percentile!=0.0]
    #     energy_at_90th_percentile=np.interp(target_percentile, percentile_cut, energy_cut)
    #     return np.array([mean_energy, variance**0.5, energy_at_90th_percentile])
    
    def get_spectra_metrics(self, timeframe, calib_id=None, roi_MeV=None, roi_mrad=None, percentile=95, debug=False):
        """"""
        shot_dicts = self.DAQ.get_shot_dicts(self.config['name'],timeframe)
        E_means = []
        E_stds = []
        E_percentiles = []
        E_charges = []
        for shot_dict in shot_dicts:
            E_mean, E_std, E_percentile = self.get_spectrum_metrics(shot_dict, calib_id=calib_id, roi_MeV=roi_MeV, roi_mrad=roi_mrad, percentile=percentile, debug=debug)
            E_means.append(E_mean)
            E_stds.append(E_std)
            E_percentiles.append(E_percentile)
            if 'charge' in self.calib_dict:
                E_charge = self.get_charge(shot_dict, calib_id=calib_id, roi_MeV=roi_MeV, roi_mrad=roi_mrad, debug=debug)
            else:
                E_charge = 0
            E_charges.append(E_charge)

        return E_means, E_stds, E_percentiles, E_charges
    
    def get_div(self, shot_dict, calib_id=None, roi_MeV=None,  roi_mrad=None, debug=False):
        """Currently integrating across the spatial axis. Could be something more involved?"""
        img, x, y = self.get_proc_shot(shot_dict, calib_id=calib_id, roi_MeV=roi_MeV, roi_mrad=roi_mrad, debug=debug)
        if img is None:
            return None, None
        if 'axis' in  self.calib_dict['divergence'] and self.calib_dict['divergence']['axis'].lower() == 'x':
            mrad = x
            sum_lineout = np.sum(img, 0)
        else:
            mrad = y
            sum_lineout = np.sum(img, 1)
        return sum_lineout, mrad
    
    def get_div_FWHM(self, shot_dict, calib_id=None, roi_MeV=None, roi_mrad=None, debug=False):
        # TODO: write library function to find FWHM from lineout
        def FWHM(X,Y):
            half_max = max(Y) / 2.
            #find when function crosses line half_max (when sign of diff flips)
            #take the 'derivative' of signum(half_max - Y[])
            d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
            #plot(X[0:len(d)],d) #if you are interested
            #find the left and right most indexes
            left_idx = np.where(d > 0)[0]
            right_idx = np.where(d < 0)[-1]
            fwhm = X[right_idx] - X[left_idx] #return the difference (full width)
            return fwhm[0] #return the difference (full width)
        # TODO: Return Error estimate as well
        lineout, mrad = self.get_div(shot_dict, calib_id=calib_id, roi_MeV=roi_MeV, roi_mrad=roi_mrad, debug=debug)
        if lineout is None:
            return None, None
        lineout_smoothed = savgol_filter(lineout, int(len(mrad)/10), 2)
        peak_location_i = np.argmax(lineout_smoothed)
        peak_location = mrad[peak_location_i]
        if debug:
            plt.figure()
            plt.plot(mrad, lineout, label='Raw')
            plt.plot(mrad, lineout_smoothed, label='Smoothed')
            plt.title(shot_dict)
            plt.xlabel('mrad') 
            plt.ylabel('fc/mrad')
            plt.tight_layout()
            plt.legend()
            plt.show(block=False)

        return FWHM(mrad, lineout_smoothed), peak_location
    
    def get_charge(self, shot_dict, calib_id=None, roi_MeV=None, roi_mrad=None, debug=False):
        """Integrate and return total charge (pC)"""
        if not self.calib_dict and not calib_id:
            print('Missing Calibration before using get_charge... Please set using set_calib(calib_id) or pass a calib_id')
            return False

        if 'charge' not in self.calib_dict:
            print('No charge calibration set')
            return False

        img, x, y = self.get_proc_shot(shot_dict, calib_id=calib_id, roi_MeV=roi_MeV, roi_mrad=roi_mrad, debug=debug)
        if img is None:
            return None

        # we have to unfold count changes again for dMeV and dmrad
        if 'axis' in self.calib_dict['dispersion'] and self.calib_dict['dispersion']['axis'].lower() == 'y':
            # not tested??
            MeV = y
            dMeV = abs(np.gradient(MeV)) # gradient is like diff, but calculates as average of differences either side
            dMeV_matrix = np.transpose(np.tile(dMeV, (len(x),1))) # len(y)??
        else:
            MeV = x
            dMeV = abs(np.gradient(MeV)) # gradient is like diff, but calculates as average of differences either side
            dMeV_matrix = np.tile(dMeV, (len(y),1))
        # convert from counts per MeV to counts
        img_res = img * dMeV_matrix
        # we have to unfold count changes again for dMeV and dmrad
        if 'axis' in self.calib_dict['divergence'] and self.calib_dict['divergence']['axis'].lower() == 'x':
            mrad = x
            dmrad = np.mean(np.diff(mrad)) # assuming linear for now...
        else:
            mrad = y
            dmrad = np.mean(np.diff(mrad)) # assuming linear for now...
        # convert counts per mrad back to counts
        img_res = img_res * dmrad

         # return pC, not fC
        charge = np.sum(img_res) / 1000
        return charge

    def make_dispersion(self, disp_dict, debug=False):
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

        if debug:
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

        # Should this be calculated on the fly rather than storing array in processed file? Could store dispersion curve
        MeV = disp_dict['MeV']
        dMeV = abs(np.gradient(MeV)) # gradient is like diff, but calculates as average of differences either side

        if disp_dict['axis'] == 'x':
            self.x_MeV = MeV
            dMeV_matrix = np.tile(dMeV, (len(self.y_mm),1))
        elif disp_dict['axis'] == 'y':
            self.y_MeV = MeV
            dMeV_matrix = np.transpose(np.tile(dMeV, (len(self.x_mm),1)))

        # convert from counts to counts per MeV
        img_data = img_data / dMeV_matrix

        if '/MeV' not in self.img_units:
            self.img_units.append('/MeV')

        return img_data, MeV

    def make_divergence(self, div_dict, debug=False):
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
        if '/mrad' not in self.img_units:
            self.img_units.append('/mrad')

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
        """ This is a normalisation factor 0->1 for signal fading on Image plate. 
        Used on PSL values. https://dx.doi.org/10.1063/1.4936141
        ~5% error on these values juding from paper?"""
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
            rescan_factor = self.IP_rescan_factor(filenames[fi-1],filenames[fi], roi=roi, R=R, S=S, bins=bins, debug=debug)
            rescan_product = rescan_product * rescan_factor
            print(f'Rescan {fi}: {rescan_factor}')

        return rescan_product

    # ------------------------------------------------------ #
    # PLOTTING FUNCTIONS
    # ------------------------------------------------------ #

    def montage(self, timeframe, calib_id=None, roi_MeV=None, roi_mrad=None, x_downsample=1, y_downsample=1, exceptions=None, vmin=None, vmax=None, transpose=True, num_rows=1, colormap='electron_beam', colormap_option=5, debug=False):
        """Wrapper for diagnostic make_montage() function, mainly to set axis"""

        if calib_id:
            self.calib_dict = self.get_calib(calib_id)
        # ???
        # if not self.calib_dict:
        #     print('Missing Calibration before using Montage... Please set using set_calib(calib_id), or pass calib_id')
        #     return False
        
        # calling 'universal' DAQ function here, that is probably DAQ specific
        shot_dicts = self.DAQ.get_shot_dicts(self.config['name'],timeframe,exceptions=exceptions)

        shot_labels = []
        for shot_dict in shot_dicts:

            img, x, y = self.get_proc_shot(shot_dict, calib_id=calib_id, roi_MeV=roi_MeV, roi_mrad=roi_mrad, debug=debug)

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

        if 'dispersion' in self.calib_dict:
            axis_label = r'$E$ [MeV]'
            if 'axis' in self.calib_dict['dispersion'] and self.calib_dict['dispersion']['axis'].lower() == 'y':
                axis = y
            else:
                axis = x
        else:
            axis = x # default??
            axis_label = 'mm?'

        cb_label = self.make_units(self.img_units)

        fig, ax = plot_montage(images, axis=axis, x_downsample=x_downsample, y_downsample=y_downsample, title=self.shot_string(timeframe), 
                               vmin=vmin, vmax=vmax, transpose=transpose, cb_label=cb_label, y_label=axis_label, num_rows=num_rows, shot_labels=shot_labels, colormap=colormap, colormap_option=colormap_option)

        return fig, ax

    def make_units(self, units):
        return ''.join(units)


    def plot_proc_shot(self, shot_dict, calib_id=None, roi_MeV=None, roi_mrad=None, vmin=None, vmax=None, colormap='electron_beam', colormap_option=5, debug=False):

        # below still assumes X = spectral, Y =  divergence
        espec_img, x, y = self.get_proc_shot(shot_dict,calib_id=calib_id,roi_MeV=roi_MeV, roi_mrad=roi_mrad, debug=debug)

        if vmin is None:
            vmin = np.nanmin(espec_img)
        if vmax is None:
            #vmax = np.nanmax(espec_img)
            vmax = np.percentile(espec_img,99)

        fig = plt.figure()
        im = plt.pcolormesh(x, y, espec_img, vmin=vmin, vmax=vmax, cmap=get_colormap(colormap, option=colormap_option), shading='auto')
        cb = plt.colorbar(im)
        cb.set_label(self.make_units(self.img_units), rotation=270, labelpad=20)
        plt.title(self.shot_string(shot_dict))
        if 'dispersion' in self.calib_dict and 'axis' in self.calib_dict['dispersion'] and self.calib_dict['dispersion']['axis'].lower() == 'y':
            plt.ylabel('Electron energy [MeV]')
        else:
            plt.xlabel('Electron energy [MeV]')
        if 'divergence' in self.calib_dict and 'axis' in self.calib_dict['divergence'] and self.calib_dict['divergence']['axis'].lower() == 'y':
            plt.ylabel('Beam divergence [mrad]')
        else:
            plt.xlabel('Beam divergence [mrad]')
        plt.tight_layout()
        plt.show(block=False)

        return fig, plt.gca()
    
    def plot_spectrum(self, shot_dict, roi=None):

        spec, MeV = self.get_spectrum(shot_dict, roi=roi)

        if 'charge' in self.calib_dict:
            units = 'fC/MeV'
        else:
            units = 'Counts/MeV'

        fig = plt.figure()
        im = plt.plot(MeV, spec)
        plt.title(self.shot_string(shot_dict))
        plt.xlabel('MeV') 
        plt.ylabel(units) 
        plt.tight_layout()
        plt.show(block=False)

        return fig
    
    # To Do: plot_spectra(self, timeframe):