import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as patches

from ..diagnostic import Diagnostic
from ..utils.xrays.calc_bragg_dispersion import calc_bragg_dispersion
from ..utils.general import smooth_lin
from ..utils.general import first_index

class XAS(Diagnostic):
    """X-ray Absorption Spectrometer
    
    Based on XASA (Gemini 2020): X-ray Absorption Spectroscopy Analysis of laser-plasma experiment data.

    For processing and analysing raw XAS diagnostic image data. Each object
    instance is intended to generate a single absorption spectra. File lists
    for a shot series or run are feed into the object, along with processing
    and analysis parameters before preforming the routines to generate the
    various profiles and statistics for a given absorption spectrum. This
    includes reference profiles (unattenuated signal shapes), signal,
    transmission, and absorption profiles, and normalised profiles and errors.

    This class extends the base AnalysisObject class. It uses set_param() and
    get_param() for defining analysis options. Also shared diagnostic access
    to common results (TODO).

    Written by Brendan Kettle, b.kettle@imperial.ac.uk, Jan 2021.

    Attributes:
        ref_files: Array of reference shot filepaths (unattenuated)
        sig_files: Array of absorbed signal shot filepaths
        eV: array of photon energy per pixel of image spectral lineout data
        roi_eV: as above, but for region of interest
        nabs_eV: as above, but limited to normalised signal region
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''
    data_type = 'image'

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

    def set_dispersion(self, central_eV=None, dist_src_det=None, crystal_2d=None, dists_pixels=None):
        """Convert pixels to eV using basic geometry calc.

        Currently using the diagnostic libray function calc_bragg_dispersion(...)

        Note:
            The function uses a `central` detector photon energy at zero pixel
            distance (usually with pixels +/- either side of this).
            This will most likely be different to the absorption edge energy
            which will most likely be offset from the detector centre.
            If arugments not set, will try load config file parameters.

        Args:
            dist_src_det: Distance from source to dtector (m)
            crystal_2d: 2D lattice spacing of the crystal used (m)
            dists_pixels: Array of pixel distances (m)
            central_eV: Photon energy (eV) of zero pixel distance. Default to
                edge energy if not set.

        Returns:
            Array of dispersion (eV) across the detector pixels.
        """
        # If vars not passed, use calibration dict
        if not central_eV:
            central_eV = self.calib_dict['edge_eV']
        if not dist_src_det:
            dist_src_det = self.calib_dict['detector']['dist_source']
        if not crystal_2d:
            crystal_2d = self.calib_dict['crystal_2d']
        if not dists_pixels:
            dists_pixels = (np.arange(self.calib_dict['detector']['num_pixels']) - (self.calib_dict['detector']['num_pixels']/2)) * self.calib_dict['detector']['pixel_size']
        # Use library function
        self.eV = calc_bragg_dispersion(central_eV, crystal_2d, dist_src_det, dists_pixels)
        return self.eV
    
    def feed_dispersion(self, eV):
        """Feed a pre-calculated detector spectral dispersion into XASA object.

        Note:
            Useful for comparing across objects with a known fixed dispersion.
            MUST be done before taking ROI or normalisations (for nabs_eV)
        Args:
            eV: Array of dispersion (eV) across the detector pixels.
        """
        self.eV = eV

    def add_ref_shots(self, shot_dicts, debug=False):
        """Add to list of reference images for processing """
        # and a nice label
        if isinstance(shot_dicts, dict):
            shot_dicts = [shot_dicts]
        if not hasattr(self, 'ref_shot_dicts'):
            self.ref_shot_dicts = [] 
        for shot_dict in shot_dicts:
            if not isinstance(shot_dict, dict) :
                print('Error, add_ref_shots() argument should be a shot_dict or list of shot_dicts')
                return False
            # Check file exists
            if self.DAQ.file_exists(self.config['name'], shot_dict):
                self.ref_shot_dicts.append(shot_dict)
            elif debug:
                print(f'Skipping ref shot, no file found; {shot_dict}')

    def reset_ref_shots(self):
        """This is a temporary fix! I think?"""
        # del_list = ['ref_lin','ref_lin_err','sig_lins','sig_lin_err','trans_lins','trans_lin_err',
        #             'abs_lins','abs_lin_err','nabs_lins','nabs_lin_err']
        self.ref_shot_dicts = []
        del_list = ['ref_lin','ref_lins','ref_lin_err', 'ref_lin_x', 'ref_shot_dicts', 'ref_shot_dicts_used', 'trans_lins', 'abs_lins', 'nabs_lins']
        for del_name in del_list:
            if hasattr(self, del_name):
                delattr(self, del_name)
        return
    
    def add_sig_shots(self, shot_dicts):
        """Add to list of signal images for processing """
        if not hasattr(self, 'sig_shot_dicts'):
            self.sig_shot_dicts = [] 
        # and a nice label
        if isinstance(shot_dicts, list):
            for shot_dict in shot_dicts:
                self.sig_shot_dicts.append(shot_dicts)
        elif isinstance(shot_dicts, dict):
            self.sig_shot_dicts.append(shot_dicts)
        else:
            print('Error, add_sig_shots() argument should be a shot_dict or list of shot_dicts')
            return False

    def reset_sig_shots(self):
        """This is a temporary fix! I think?"""
        # del_list = ['ref_lin','ref_lin_err','sig_lins','sig_lin_err','trans_lins','trans_lin_err',
        #             'abs_lins','abs_lin_err','nabs_lins','nabs_lin_err']
        self.sig_shot_dicts = []
        del_list = ['sig_lins', 'trans_lins', 'abs_lins', 'nabs_lins', 'sig_shot_dicts', 'sig_shot_dicts_used']
        for del_name in del_list:
            if hasattr(self, del_name):
                delattr(self, del_name)
        return

    def get_avg_img(self, shot_set='sig', debug=False):
        """Return an image array of the average shots """
        # What type of shot set?
        if shot_set == 'sig':
            if not self.sig_shot_dicts:
                raise Exception('You must set absorbed signal shot files first')
            shot_dicts = self.sig_shot_dicts
        elif shot_set == 'ref':
            if not self.ref_shot_dicts:
                raise Exception('You must set reference shot files first')
            shot_dicts = self.ref_shot_dicts
        else:
            raise Exception('get_avg_img shot set not recognised. Options are "sig" or "ref".')
        # loop through files and build image
        for shot_dict in shot_dicts:
            shot_img, x, y = self.run_img_calib(shot_dict) # No debug, don't want to flood plots. Can use check_calib() seperately if you'd like
            if shot_img is None:
                print(f'Could not get shot data; {shot_dict}')
                return False
            if 'sum_img' in locals():
                sum_img += shot_img
            else:
                sum_img = shot_img
        avg_img = sum_img / len(shot_dicts)
        return avg_img
    
    def make_ref(self, roi=None, ref_threshold=None, fitting='Smooth', debug=True):
        """Generate averaged reference signal shape """

        if not roi and 'roi' in self.calib_dict:
            roi = self.calib_dict['roi']['pixels']

        if not ref_threshold:
            if 'ref_threshold' in self.calib_dict:
                ref_threshold = self.calib_dict['ref_threshold']
            else:
                ref_threshold = 0 # still no luck? just go to zero threshold

        # Loop through ref shots
        self.ref_lins = []
        skipped = 0
        si = 0
        shot_dicts_used = []
        for shot_dict in self.ref_shot_dicts:

            shot_img, x, y = self.run_img_calib(shot_dict) # No debug, don't want to flood plots. Can use check_calib() seperately if you'd like

            if shot_img is not None:
                # print(f'Could not get shot data; {shot_dict}')
                # return False
                if roi:
                    ref_lin = np.sum(shot_img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]],0)
                else:
                    ref_lin = np.sum(shot_img,0)
                # skip if too low?
                #ref_sum = np.sum(shot_img[roi[0]:roi[1],roi[2]:roi[3]])
                if np.max(ref_lin) < ref_threshold: # this use to be sum, but max easier for debugging lineout
                    if debug: print(f'Skipping Ref (Max={int(np.max(ref_lin))}, below Threshold): {shot_dict}')
                    skipped += 1
                else:
                    self.ref_lins.append(ref_lin)
                    shot_dicts_used.append(shot_dict)
                si += 1
        self.ref_shot_dicts_used = shot_dicts_used

        # no lineouts?
        if not self.ref_lins:
            print('Warning, no reference lineouts above threshold.')
            ref_lin_x = []
            self.ref_lin = np.array([])
        else:
            ref_lin_raw = np.mean(self.ref_lins,0)

            # progress update
            if debug: print(f'Making reference signal shape. Averaging over {str(len(self.ref_lins))} files ({str(skipped)} skipped). Threshold: {ref_threshold}.')
            # Normalise
            if 'norm_kernel' in self.calib_dict:
                norm_kernel = self.calib_dict['norm_kernel']
            else:
                norm_kernel = int(len(ref_lin_raw)/10)
            self.ref_lin_raw = ref_lin_raw / np.max(smooth_lin(ref_lin_raw, norm_kernel))

            # Fitting smooth curve to data?
            if fitting == 'SplitPoly9':
                smoothing = 30
                poly_order = 9
                # look at two halfs separately. Find "peak" using middle of high order polyfit
                px = np.arange(0,len(self.ref_lin_raw))
                pf = np.poly1d(np.polyfit(px, self.ref_lin_raw, 20))
                break_i = np.argmax(pf(px))
                break_px = px[break_i]
                #print('Break Pixel: ' + str(break_px))
                px1 = px[0:break_i]
                ref_lin1 = self.ref_lin_raw[0:break_i]
                px2 = px[break_i:]
                ref_lin2 = self.ref_lin_raw[break_i:]
                pf1 = np.poly1d(np.polyfit(px1, ref_lin1, poly_order))
                pf2 = np.poly1d(np.polyfit(px2, ref_lin2, poly_order))
                joined_pf = np.concatenate([pf1(px1),pf2(px2)]) # join back up
                self.ref_lin = smooth_lin(joined_pf,smoothing) # smooth any kinks
                #plt.figure()
                #plt.plot(px, self.ref_lin_raw-self.ref_lin, 'green', linewidth=1)
                #plt.title('Reference Fit Differences (Data-Fit)')
            # TODO: Other options for fitting?
            # No; smoothing?
            elif fitting == 'Poly9':
                smoothing = 30
                poly_order = 15
                px = np.arange(0,len(self.ref_lin_raw))
                pf = np.poly1d(np.polyfit(px, self.ref_lin_raw, poly_order))
                self.ref_lin = smooth_lin(pf(px),smoothing) 
            elif fitting == 'Smooth' or fitting == None:
                fitting = 'Smooth'
                ref_kernel = self.calib_dict['ref_kernel']
                self.ref_lin = smooth_lin(self.ref_lin_raw, ref_kernel)
            # If dispersion is set, set and return spectral range across this selection
            if not hasattr(self, 'eV'):
                ref_lin_x = np.arange(0,len(self.ref_lin))
                xlabel = 'Pixels'
            else:
                if roi and (len(self.eV) > (roi[1][0]-roi[0][0])):
                    self.roi_eV = self.eV[roi[0][0]:roi[1][0]]
                else:
                    self.roi_eV = self.eV
                ref_lin_x = self.roi_eV
                xlabel = 'Photon Energy (eV)'

            if debug:
                plt.figure()
                for ri in range(len(self.ref_lins)):
                    plt.plot(ref_lin_x, self.ref_lins[ri], label=shot_dicts_used[ri]['shotnum'])
                plt.title('Reference Lineouts')
                plt.xlabel(xlabel)
                plt.legend()

                plt.figure()
                plt.plot(ref_lin_x, self.ref_lin_raw, label='Raw Data')
                plt.plot(ref_lin_x, self.ref_lin, label=f'Fit (Using:{fitting})')
                plt.title('Average Reference Lineout')
                plt.legend()
                plt.xlabel(xlabel)
                self.plot_avg_img(shot_set='ref', roi=roi)
        return ref_lin_x, self.ref_lin
    
    def make_sig(self, roi=None, sig_threshold=None, debug=True):
        """Generate list of signal profiles from absorption shots """

        if not roi and 'roi' in self.calib_dict:
            roi = self.calib_dict['roi']['pixels']

        if not sig_threshold:
            if 'sig_threshold' in self.calib_dict:
                sig_threshold = self.calib_dict['sig_threshold']
            else:
                sig_threshold = 0 # still no luck? just go to zero threshold

        # loop through
        self.sig_lins = []
        skipped = 0
        sig_posi = 0
        shot_dicts_used = []
        for shot_dict in self.sig_shot_dicts:

            shot_img, x, y = self.run_img_calib(shot_dict) # No debug, don't want to flood plots. Can use check_calib() seperately if you'd like

            if shot_img is None:
                print(f'Could not get shot data; {shot_dict}')
                return False

            if roi:
                sig_lin = np.sum(shot_img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]],0)
            else:
                sig_lin = np.sum(shot_img,0)

            # skip if too low?
            if np.max(sig_lin) < sig_threshold:
                if debug: print(f'Skipping Sig (Max={int(np.max(sig_lin))}; below threshold): {shot_dict}')
                skipped += 1
            else:
                # before adding signal lineout, do we need to stabilise?
                if hasattr(self,'sig_shiftpxs'):
                    this_shiftpx = self.sig_shiftpxs[sig_posi]
                    #print(this_shiftpx)
                    if this_shiftpx > 0:
                        sig_lin[this_shiftpx:] = sig_lin[0:-this_shiftpx]
                        sig_lin[0:this_shiftpx] = sig_lin[this_shiftpx]
                    elif this_shiftpx < 0:
                        this_shiftpx = -this_shiftpx # make postive for readable logic below
                        sig_lin[0:-this_shiftpx] = sig_lin[this_shiftpx:]
                        sig_lin[-this_shiftpx:] = sig_lin[-this_shiftpx]
                self.sig_lins.append(sig_lin)
                shot_dicts_used.append(shot_dict)
            sig_posi = sig_posi + 1
        self.sig_shot_dicts_used = shot_dicts_used
        # If dispersion is set, set and return spectral range across this selection
        if not hasattr(self, 'eV'):
            sig_lin_x = np.arange(0,len(sig_lin))
            xlabel = 'Pixels'
        else:
            if roi:
                self.roi_eV = self.eV[roi[0][0]:roi[1][0]]
            else:
                self.roi_eV = self.eV
            sig_lin_x = self.roi_eV
            xlabel = 'Photon Energy (eV)'

        # progress update
        if debug: 
            print(f'Made signal profiles: {str(len(self.sig_lins))} files ({str(skipped)} skipped). Threshold: {sig_threshold}')
            plt.figure()
            for si in range(len(self.sig_lins)):
                plt.plot(sig_lin_x, self.sig_lins[si], label=shot_dicts_used[si]['shotnum'])
            plt.title('Signal Lineouts')
            plt.xlabel(xlabel)
            plt.legend()
        
            plt.figure()
            plt.plot(sig_lin_x, np.mean(self.sig_lins,0))
            plt.title('Average Signal Lineout')
            plt.xlabel(xlabel)
            self.plot_avg_img(shot_set='sig', roi=roi)

        return sig_lin_x, self.sig_lins
    
    def make_trans(self, roi=None, anchor_width_eV=10, debug=True):
        """Generate list of transmission profiles using signal profiles and reference shape """
        # If reference lineout not set, generate it.
        if not hasattr(self, 'ref_lin'):
            self.roi_eV, self.ref_lin = self.make_ref(roi=roi,debug=debug)
        # Make signal Lineouts?
        if not hasattr(self, 'sig_lins'):
            self.roi_eV, self.sig_lins = self.make_sig(roi=roi,debug=debug)
        # Make sure rough dispersion has been calculated
        if not hasattr(self, 'eV'):
            #raise Exception('Error! set_dispersion() or feed_dispersion() needs to be called before match_dispersion()')
            print('No prior dispersion set, attempting to use default config values...')
            self.set_dispersion()
        lin_x = self.eV
        if not hasattr(self, 'roi_eV'):
            # roi = self.load_param('sig_roi', False)
            # if not roi:
            #     img_size = np.shape(self.__load_data(self.ref_files[0]))
            #     roi = [0,img_size[0],0,img_size[1]]
            #     self.set_param('sig_roi', roi)
            if roi:
                self.roi_eV = self.eV[roi[0][0]:roi[0][1]]
            else:
                self.roi_eV = self.eV
        lin_x = self.roi_eV
        # Now rescale reference lineout at anchor point with assumed transmission
        anchor_eV = self.calib_dict['anchor_eV']
        anchor_trans = self.calib_dict['anchor_trans']
        if not anchor_eV:
            raise Exception('Error! anchor_eV needs to be set before making a transmission profile')
        if not anchor_trans:
            raise Exception('Error! anchor_trans needs to be set before making a transmission profile')
        # find index of anchor and get ref value
        anchor_i = (np.abs(self.roi_eV - anchor_eV)).argmin()
        #anchor_ref_val = self.ref_lin[anchor_i]
        anchor_ref_val = np.mean(self.ref_lin[anchor_i-2:anchor_i+2]) # small average over some pixels
        # calculate over how many pixels we should average for anchor
        deV = np.mean(np.diff(lin_x))
        anchor_width_px = anchor_width_eV / deV
        if debug: print(f'Making Transmission: Scaling sigs/refs at Anchor = {str(anchor_eV)} eV (idx={str(anchor_i)}), over {anchor_width_eV} eV ({int(anchor_width_px)} pixels), assumed transmission = {str(anchor_trans*100)}%')
        # # removing a fraction of reference and signal? (perhaps pinhole leakage or harmonic contamination)
        # # PRETTY SURE THIS STILL DOESN'T DO ANYTHING. JUST CANCELS BELOW?
        # if not self.load_param('leakage_frac',False):
        #     leakage_frac = 0
        # else:
        #     leakage_frac = self.load_param('leakage_frac')
        #     print('Removing leakage fraction.... ' +str(leakage_frac))
        # loop through signal lineouts and divide out reference shapes
        leakage_frac = 0 # not sure if this works, setting to zero for now... then doesn't effect below

        self.trans_lins = []
        # go through sign lineouts and make transmissions
        for sig_lin in self.sig_lins:
            anchor_sig_val = np.mean(sig_lin[anchor_i-int(anchor_width_px/2):anchor_i+int(anchor_width_px/2)]) 
            anchor_sig_val_unfilt = anchor_sig_val / (anchor_trans + leakage_frac)
            ref_scale_factor = anchor_sig_val_unfilt / anchor_ref_val
            scaled_ref = self.ref_lin * ref_scale_factor
            scaled_ref_leakage = scaled_ref * leakage_frac
            trans_lin = (sig_lin - scaled_ref_leakage) / scaled_ref
            self.trans_lins.append(trans_lin)
        
        # instaead, make sig average first? then one transmission lineout
        # I think this is basically the same? only then we lose the individual transmission lineouts
        # sig_lin = np.mean(self.sig_lins,0)
        # anchor_sig_val = np.mean(sig_lin[anchor_i-int(anchor_width_px/2):anchor_i+int(anchor_width_px/2)])
        # ref_scale_factor = (anchor_sig_val / anchor_trans) / anchor_ref_val
        # scaled_ref = self.ref_lin * ref_scale_factor
        # self.trans_lins.append(sig_lin / scaled_ref)

        if debug:
            plt.figure()
            for ti in range(len(self.trans_lins)):
                plt.plot(lin_x, self.trans_lins[ti], label=self.sig_shot_dicts_used[ti]['shotnum'])
            plt.title('Transmission profiles')
            plt.xlabel('Photon Energy [eV]')
            plt.ylabel('Transmission')
            plt.legend()

            plt.figure()
            plt.plot(lin_x, np.mean(self.trans_lins,0))
            plt.title('Average Transmission profile')
            plt.xlabel('Photon Energy [eV]')
            plt.ylabel('Transmission')

        return lin_x, self.trans_lins
    
    def make_abs(self, roi=None, debug=True):
        """Generate list of absorption profiles using transmission profiles """
        # Make transmission Lineouts?
        if not hasattr(self, 'trans_lins'):
            self.roi_eV, self.trans_lins = self.make_trans(roi=roi, debug=debug)
        # NOTE! we are not caluclating the absorption (1 - transmission), but rather mu * t
        self.abs_lins = []
        for trans_lin in self.trans_lins:
            if min(trans_lin) < 1e-2:
                print('Warning, transmission under 1%; capped at 1% for absorption profile')
                trans_lin[trans_lin<1e-2] = 1e-2 # avoid erroring below... bodge?
            abs_lin = -np.log(trans_lin)
            self.abs_lins.append(abs_lin)
        # progress update
        if debug: 
            print('Making absorption profiles (' + str(len(self.trans_lins)) + ' files)')

            plt.figure()
            for ai in range(len(self.abs_lins)):
                plt.plot(self.roi_eV, self.abs_lins[ai], label=self.sig_shot_dicts_used[ai]['shotnum'])
            plt.title('Absorption profiles')
            plt.xlabel('Photon Energy [eV]')
            plt.ylabel('-Ln(Transmission)')
            plt.legend()

            plt.figure()
            plt.plot(self.roi_eV, np.mean(self.abs_lins,0))
            plt.title('Average Absorption Profile')
            plt.xlabel('Photon Energy [eV]')
            plt.ylabel('-Ln(Transmission)')
        return self.roi_eV, self.abs_lins
    
    def make_nabs(self, roi=None, method='average', preedge_buffer_eV = None, postedge_buffer_eV = None, frac_threshold = None, debug=True, flat_preedge=False):
        """Generate (list of) normalised absorption profile(s).
        You can do this either individually or with the signal averaged first  """
        # Make absorption lineouts? (includes reference making etc.)
        if not hasattr(self, 'abs_lins'):
            self.roi_eV, self.abs_lins = self.make_abs(roi=roi, debug=debug)
        if not preedge_buffer_eV:
            preedge_buffer_eV = self.calib_dict['preedge_buffer_eV']
        if not postedge_buffer_eV:
            postedge_buffer_eV = self.calib_dict['postedge_buffer_eV']
        if not frac_threshold:
            frac_threshold = self.calib_dict['frac_threshold']
        edge_eV = self.calib_dict['edge_eV']
        # Truncate the absorption profile to reigions of good signal (using reference as guide)
        # find first and last ref signal indices above threshold
        ref_threshold_val = max(self.ref_lin) * frac_threshold
        ref_start_ti = first_index(self.ref_lin,ref_threshold_val,'left')
        ref_end_ti = first_index(self.ref_lin,ref_threshold_val,'right')
        # apply same threshold for signal?
        avg_sig_lin = np.mean(np.array(self.sig_lins),0)
        sig_threshold_val = max(avg_sig_lin) * frac_threshold
        sig_start_ti = first_index(avg_sig_lin,sig_threshold_val,'left')
        sig_end_ti = first_index(avg_sig_lin,sig_threshold_val,'right')
        start_eV = max(self.roi_eV[ref_start_ti],self.roi_eV[sig_start_ti])
        end_eV = min(self.roi_eV[ref_end_ti],self.roi_eV[sig_end_ti])
        # are there any other limits set? (for energies missing sample for example)
        # eV_lower_limit = self.load_param('eV_lower_limit', False)
        # eV_upper_limit = self.load_param('eV_upper_limit', False)
        # if eV_lower_limit:
        #     if ref_start_eV < eV_lower_limit:
        #         ref_start_eV = eV_lower_limit
        # if eV_upper_limit:
        #     if ref_end_eV > eV_upper_limit:
        #         ref_end_eV = eV_upper_limit
        start_i = (np.abs(self.roi_eV - start_eV)).argmin()
        end_i = (np.abs(self.roi_eV - end_eV)).argmin()
        nabs_eV = self.roi_eV[start_i:end_i]
        if debug: print('Truncating normalised absorption signal at ' + str(frac_threshold*100) + '% max ref/sig level: Between ' + str(start_eV) + ' eV and ' + str(end_eV) + ' eV')
        self.nabs_i = np.arange(start_i,end_i)
        # Which method?
        if method == 'average':
            avg_abs_lin = np.mean(np.array(self.abs_lins),0)
            self.nabs_eV, self.nabs_lin, self.nabs_pre_fit, self.nabs_post_fit = self.normalise_lin(nabs_eV,avg_abs_lin[start_i:end_i],edge_eV,preedge_buffer_eV,postedge_buffer_eV,debug=debug,return_fits=True,flat_preedge=flat_preedge)
            self.nabs_lins = self.nabs_lin
            self.nabs_pre_fits = self.nabs_pre_fit
            self.nabs_post_fits = self.nabs_post_fit
        elif method == 'individual':
            self.nabs_lins = []
            self.nabs_pre_fits = []
            self.nabs_post_fits = []
            for abs_lin in self.abs_lins:
                nabs_eV, nabs_lin, nabs_pre_fit, nabs_post_fit = self.normalise_lin(nabs_eV,abs_lin[start_i:end_i],edge_eV,preedge_buffer_eV,postedge_buffer_eV,debug=debug,return_fits=True,flat_preedge=flat_preedge)
                self.nabs_lins.append(nabs_lin)
                self.nabs_pre_fits.append(nabs_pre_fit)
                self.nabs_post_fits.append(nabs_post_fit)
            self.nabs_eV = nabs_eV
        else:
            raise Exception('Unknown method for make_nabs()')
        return self.nabs_eV, self.nabs_lins
    
    def normalise_lin(self, abs_eV, abs_lin, edge_eV, preedge_buffer_eV=None, postedge_buffer_eV=None, return_fits=False, flat_preedge=False, debug=True):
        """Normalise an absorption profile by fitting above and below the edge """
        if not preedge_buffer_eV:
            preedge_buffer_eV = self.calib_dict['preedge_buffer_eV']
        if not postedge_buffer_eV:
            postedge_buffer_eV = self.calib_dict['postedge_buffer_eV']
        # workout edge limits
        edge_start_eV = edge_eV - preedge_buffer_eV
        edge_end_eV = edge_eV + postedge_buffer_eV
        # find indices positions in abs signal for our four fitting limits
        # NB: energy going up or down might cause issues???
        preedge_end_i = (np.abs(abs_eV - edge_start_eV)).argmin()
        postedge_start_i = (np.abs(abs_eV - edge_end_eV)).argmin()
        # ... and express again as eV
        preedge_start_eV = abs_eV[0]
        preedge_end_eV = abs_eV[preedge_end_i]
        postedge_start_eV = abs_eV[postedge_start_i]
        postedge_end_eV = abs_eV[-1]
        if debug: print('Fitting pre-edge from ' + str(preedge_start_eV) + ' eV (idx=0) to ' + str(preedge_end_eV) + ' eV (idx=' + str(preedge_end_i) + ')')
        if debug: print('Fitting post-edge from ' + str(postedge_start_eV) + ' eV (idx=' + str(postedge_start_i) + ') to ' + str(postedge_end_eV) + ' eV (idx=' + str(len(abs_lin)) + ')')
        # Fit to our pre-edge
        if flat_preedge or ('flat_preedge' in self.calib_dict and self.calib_dict['flat_preedge']):
            # assume flat constant (as not enough signal?)
            preedge_avg = np.mean(abs_lin[:preedge_end_i+1])
            pre_fit = np.ones(len(abs_eV)) * preedge_avg
        else:
            x = abs_eV[:preedge_end_i+1]
            y = abs_lin[:preedge_end_i+1]
            pre_pfit = np.polyfit(x, y, 1)
            pre_fit = np.polyval(pre_pfit, abs_eV)
        # And fit to post edge
        x = abs_eV[postedge_start_i:]
        y = abs_lin[postedge_start_i:]
        post_pfit = np.polyfit(x, y, 1)
        post_fit = np.polyval(post_pfit, abs_eV)
        # If things go awry, set debugging=True in arugments for some helpful plotting
        if debug:
            plt.figure()
            plt.plot(abs_eV,abs_lin, color='g', linewidth=1)
            plt.plot(abs_eV[:preedge_end_i],abs_lin[:preedge_end_i], color='b', linewidth=1)
            plt.plot(abs_eV[postedge_start_i:],abs_lin[postedge_start_i:], color='r', linewidth=1)
            plt.plot(abs_eV,pre_fit, color='c', linewidth=1)
            plt.plot(abs_eV,post_fit, color='m', linewidth=1)
            plt.title('normalise_lin() Debugging')
            plt.show(block=False)
        # normalise using fits
        nabs_eV = abs_eV
        nabs_lin = (abs_lin - pre_fit) / (post_fit - pre_fit)
        if return_fits:
            return nabs_eV, nabs_lin, pre_fit, post_fit
        else:
            return nabs_eV, nabs_lin

    def match_dispersion(self, edge_nabs=None, preedge_buffer_eV = None, postedge_buffer_eV = None, debug=True):
        """Use the absorption edge of the data to fix the spectral dispersion.

        Starts from the prior estimated edge position (using the set fixed
        dispersion) and splits the normalised absorption lineout in two.
        Then locates the real edge position searching from this central point.

        Notes:
            if set_disperion(...) not already called, try use default configs
            Will calculate normalised absorption profile if not already done so.
        Args:
            edge_nabs: Normalised absorption value at edge energy.
            debugging: Flag for plotting debug information. Defaults to False.
        Returns:
            Array of matched dispersion (eV) across the detector pixels.
        """
        print('Matching dispersion...')
        if not edge_nabs:
            edge_nabs = self.calib_dict['edge_nabs']
        if not preedge_buffer_eV:
            preedge_buffer_eV = self.calib_dict['preedge_buffer_eV']
        if not postedge_buffer_eV:
            postedge_buffer_eV = self.calib_dict['postedge_buffer_eV']
        # Make sure rough dispersion has been calculated
        if not hasattr(self, 'eV'):
            #raise Exception('Error! set_dispersion() or feed_dispersion() needs to be called before match_dispersion()')
            print('No prior dispersion set, attempting to use default config values...')
            self.set_dispersion()
        # Get normalised absorption lineouts and take average (if needed)
        # Need large edge buffers as edge probably in wrong place??
        if not hasattr(self, 'nabs_lins'):
            self.make_nabs(preedge_buffer_eV=preedge_buffer_eV, postedge_buffer_eV=postedge_buffer_eV, debug=debug)
        if len(np.shape(self.nabs_lins)) > 1:
            nabs_lin = np.mean(np.array(self.nabs_lins),0)
        else:
            nabs_lin = np.array(self.nabs_lins)
        # Begin our search for the real edge, using the initial set dispersion.
        edge_eV = self.calib_dict['edge_eV']
        est_edge_i = (np.abs(self.nabs_eV - edge_eV)).argmin()
        if 'sig_kernel' in self.calib_dict:
            sig_kernel = self.calib_dict['sig_kernel']
        else:
            sig_kernel = int(len(self.nabs_eV)/50)
        # If things go awry, set debugging=True in arugments for some helpful plotting
        if debug:
            plt.figure()
            plt.plot(self.nabs_eV[:est_edge_i],smooth_lin(nabs_lin[:est_edge_i], sig_kernel), color='b', linewidth=1)
            plt.plot(self.nabs_eV[est_edge_i:],smooth_lin(nabs_lin[est_edge_i:], sig_kernel), color='r', linewidth=1)
            #plt.plot(self.nabs_eV,nabs_lin, color='g', linewidth=1)
            plt.title('match_dispersion() Debugging')
            plt.show(block=False)
        # Split the Normalised ROI in two at the estimated prior edge position
        # Then using edge value of absorption edge, locate it's position, looking either left or right from edge
        right_avg = np.mean(nabs_lin[est_edge_i:(est_edge_i+5)])
        if right_avg > edge_nabs:
            # signal immediately to right is above edge, so our edge is lower energy
            print('Real edge actually at lower position than dispersion prior')
            # for looking left, we find first index from right, BELOW the edge value.
            # Smoothing first to avoid noise?
            left_eval_i = first_index(smooth_lin(nabs_lin[:est_edge_i], sig_kernel), edge_nabs, side='right',cut='high')
            edge_i = left_eval_i
        else:
            # signal immediately to right is below edge, so our edge is higher in energy
            print('Real edge actually at higher position than dispersion prior')
            print('UNTESTED?')
            # for looking right, we are are looking from first index from left, above edge value (default of function)
            # Smoothing first to avoid noise?
            right_eval_i = first_index(smooth_lin(nabs_lin[est_edge_i:], sig_kernel), edge_nabs)
            edge_i = est_edge_i + right_eval_i
        # Calculate shift
        edge_old_eV = self.nabs_eV[edge_i]
        shift_eV = edge_eV - edge_old_eV
        print('Dispersion matched. Edge taken at absorption = ' + str(edge_nabs) + '. Resulting spectral shift of ' + str(shift_eV) + ' eV')
        # Change spectra calibrations
        self.eV = self.eV + shift_eV
        self.roi_eV = self.roi_eV + shift_eV
        self.nabs_eV = self.nabs_eV + shift_eV
        # Too big a shift?
        if shift_eV > 50:
            print('WARNING! large shift with dispersion match; anchor point eV and level might have changed too much?')
        # Delete saved transmission and absorption lineouts.
        # These will need to be generated again as anchor point has changed
        del_list = ['ref_lin','ref_lin_err','sig_lins','sig_lin_err','trans_lins','trans_lin_err',
                    'abs_lins','abs_lin_err','nabs_lins','nabs_lin_err']
        for del_name in del_list:
            if hasattr(self, del_name):
                delattr(self, del_name)
        # Return dispersion for feeding into other objects for same dispersion
        return self.eV

    def get_spat_lins(self, shot_set='sig'):
        """Return spatial sum lineouts (across whole width) """
        # What type of shot set?
        if shot_set == 'sig':
            if not self.sig_files:
                raise Exception('You must set absorbed signal shot files first')
            filepaths = self.sig_files
        elif shot_set == 'ref':
            if not self.ref_files:
                raise Exception('You must set reference shot files first')
            filepaths = self.ref_files
        else:
            raise Exception('get_spat_lins shot set not recognised. Options are "sig" or "ref".')
        # get roi
        roi = self.load_param('sig_roi', False)
        # loop through files
        spat_lins = []
        for shot_filepath in filepaths:
            shot_img = self.__load_data(shot_filepath);
            if roi:
                spat_lins.append(np.sum(shot_img[roi[0]:roi[1],:],0))
            else:
                spat_lins.append(np.sum(shot_img,0))
        return spat_lins

    def get_spat_lin(self, shot_set='sig'):
        """Return the average spatial sum lineout (across whole width) """
        spat_lins = self.get_spat_lins(shot_set)
        spat_lin = np.mean(spat_lins,0)
        return spat_lin
    
    #===========================================
    # Plotting
    #===========================================
    def plot_avg_img(self, shot_set='sig', roi=None, vmin=0, vmax=None, debug=False):
        img = self.get_avg_img(shot_set=shot_set, debug=debug)
        x = np.arange(0,np.shape(img)[1]+1)
        y = np.arange(0,np.shape(img)[0]+1)
        fig = plt.figure()
        if not vmax:
            vmax = np.percentile(img, 99) # use 99% precentile max as default
        im = plt.pcolormesh(x, y, img, vmin=vmin, vmax=vmax, shading='auto')
        cb = plt.colorbar(im)
        if roi:
            ax = plt.gca()
            rect = patches.Rectangle((roi[0][0], roi[0][1]), (roi[1][0]-roi[0][0]), (roi[1][1]-roi[0][1]), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.tight_layout()
        plt.title(f'Average Image, shot set: {shot_set}')
        plt.show(block=False)
        return fig, plt.gca()
    
    def plot_dispersion(self):
        fig = plt.figure()
        plt.plot(self.eV)
        plt.xlabel('Pixel')
        plt.ylabel('Photon Energy [eV]')
        return fig, plt.gca()