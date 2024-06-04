import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import re

from ..diagnostic import Diagnostic
from ..utils.image_proc import ImageProc
from ..utils.dict_update import *
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

    def get_proc_shot(self, shot_dict, calib_id=None, roi=None, debug=False):
        """Return a processed shot using saved or passed calibrations.
        """

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
                        plt.show(block=False)

                        plt.figure()
                        plt.imshow(bkg_img)
                        plt.tight_layout()
                        plt.show(block=False)

                    img = img - bkg_img
                else:
                    print(f"{self.config['name']}: No bkg_roi provided")
            else:
                print(f"{self.config['name']}: Unknown background correction type '{self.calib_dict['bkg_type']}'")


        # dispersion?
        if 'dispersion' in self.calib_dict:
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
                    img = img[(MeV > MeV_min), :] # I'm sure this cam be done in one line, but I'm being lazy...
                    MeV = MeV[(MeV > MeV_min)]
                    img = img[(MeV < MeV_max), :]
                    MeV = MeV[(MeV < MeV_max)]
                else:
                    img = img[:, (MeV > MeV_min)]
                    MeV = MeV[(MeV > MeV_min)]
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
        if 'divergence' in self.calib_dict:
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
                    img = img[(mrad > mrad_min), :]
                    mrad = mrad[(mrad > mrad_min)]
                    img = img[(mrad < mrad_max), :]
                    mrad = mrad[(mrad < mrad_max)]
                else:
                    img = img[:, (mrad > mrad_min)]
                    mrad = mrad[(mrad > mrad_min)]
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


    def make_calib(self, calib_id=None, save=False, view=True):
        """Master function for generating procssed portion of calibration file
            E.g transform, dispersion, etc. 
        """

        # Get calibration input
        self.calib_dict = self.get_calib(calib_id)

        # Make a spatial transform (required)
        self.make_transform(self.calib_dict['transform'], view=view)

        # Apply dispersion?
        if 'dispersion' in self.calib_dict:
            self.make_dispersion(self.calib_dict['dispersion'], view=view)

        # Apply divergence?
        if 'divergence' in self.calib_dict:
            self.make_divergence(self.calib_dict['divergence'], view=view)

        # transfer other values
        # NOTE: we save the whole existing dictionary anyway...
        save_vars = ['roi_mm','roi_MeV','roi_mrad']
        for var in save_vars:
            if var in self.calib_dict:
                self.calib_dict[var] = self.calib_dict[var]

        # save the full calibration?
        if save:
            self.save_calib_file(self.calib_dict['proc_file'], self.calib_dict)

        return self.get_calib()

    def make_transform(self, tcalib_input, view=False):
        """Generate a transform dictionary for use with spatially transforming raw shot images.
            This is a wrapper for ImageProc make_transform()

            tcalib_input: A dictionary containing the required information for the transform, or calibration file/id for loading...
                        Required dictionary keys; 
                            - tpoints; list of [X,Y], where the first pair is raw pixel, the next is the corresponding transform point, and repeat...
                            - raw_img; shot dictionary or filepath to raw untransformed calibration image 
                            - img_size_t; [X,Y] size of plane being transformed, in it's coords (mm?)
                            - img_size_px; [X,Y] new size of transformed image in pixels (can upsample)
                            - offsets; [X,Y] offset of plane being transformed, in it's coords (mm?)
                            - e_offsets; [X,Y] shift of transformed plane from electron beam axis
                        Optional dictionary keys; description, notes
            save_path:
            view:
        """

        # points are (by convention) passed in a list of [X,Y], where the first is in the pixel point, 
        # the next is the corresponding transform point, and repeat
        # so here we pick out every other value for the appropriate seperate arrays
        points = np.array(tcalib_input['tpoints'])
        p_px, p_t =  points[::2], points[1::2]

        # get raw image using shot dictionary or filepath
        raw_img = self.get_shot_data(tcalib_input['raw_img'])

        # optionals?
        if 'description' in tcalib_input:
            description = tcalib_input['description']
        else:
            description = ''
        if 'notes' in tcalib_input:
            notes = tcalib_input['notes']
        else:
            notes = ''

        # Use image processing library to generate a transform dictionary 
        img = ImageProc(raw_img)

        if self.calib_dict is None:
            self.calib_dict = {}
        if 'transform' not in self.calib_dict:
            self.calib_dict['transform'] = {}

        # update dictionary with new dictionary values
        dict_update(self.calib_dict['transform'], img.make_transform(p_px, p_t, tcalib_input['img_size_t'], tcalib_input['img_size_px'], 
                                        tcalib_input['offsets'], notes=notes, description=description))
        # Add electron beam axis offset
        self.calib_dict['transform']['e_offsets'] = tcalib_input['e_offsets']

        # perform transform to check
        timg, tx, ty = self.transform(raw_img)

        # save current processed image to object along with x and y values
        self.curr_img = timg
        self.x_mm = tx
        self.y_mm = ty

        if view:
            # if viewing, plot raw image
            plt.figure()
            im = plt.imshow(raw_img)
            plt.plot(p_px[:,0],p_px[:,1],'r+')
            cb = plt.colorbar(im)
            cb.set_label('Counts on CCD', rotation=270, labelpad=20)
            plt.title(description)
            plt.xlabel('pixels')
            plt.ylabel('pixels')
            plt.tight_layout()
            plt.show(block=False)
            # then plot transformed
            plt.figure()
            im = plt.imshow(timg, extent= (np.min(self.x_mm), np.max(self.x_mm), np.max(self.y_mm), np.min(self.y_mm)))
            plt.plot(p_t[:,0]-tcalib_input['e_offsets'][0],p_t[:,1]-tcalib_input['e_offsets'][1],'r+')
            cb = plt.colorbar(im)
            cb.set_label('Counts on CCD', rotation=270, labelpad=20)
            plt.title(description)
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.tight_layout()
            plt.show(block=False)

        return self.calib_dict['transform']

    def transform(self, img_data, tform_dict=None):
        """"""

        # if not passed, use stored tform_dict, or complain
        if tform_dict is None:
            if self.calib_dict['transform'] is None:
                print('ESpec Error, transform dictionary needs to be passed or loaded')
                return
        else:
            self.calib_dict['transform'] = tform_dict
        # if img_data is passed as a shot dictionary, grab the actual image
        # TODO: could pass a filepath?
        if isinstance(img_data, dict):
            img_data = self.get_shot_data(img_data)

        img = ImageProc(img_data)
        timg, tx, ty = img.transform(self.calib_dict['transform'])
        # E beam offset? shifts the xy cords on transformed screen
        ex = tx - self.calib_dict['transform']['e_offsets'][0]
        ey = ty - self.calib_dict['transform']['e_offsets'][1]

        return timg, ex, ey

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

    def to_mm(self, value, units):
        if units.lower() == 'mm':
            return value
        elif units.lower() == 'cm':
            return (value * 10)
        elif units.lower() == 'm':
            return (value * 1e3)
        else:
            print(f"to_mm error; unknown spatial units {units}")

    def to_MeV(self, value, units):
        if units == 'MeV':
            return value
        elif units == 'GeV':
            return (value * 1e3)
        elif units == 'eV':
            return (value * 1e-3)
        else:
            print(f"to_MeV error; unknown spectral units {units}")

    def to_mrad(self, value, units):
        if units.lower() == 'mrad':
            return value
        elif units.lower() == 'rad':
            return (value * 1e3)
        elif units.lower() == 'deg':
            return (value * (np.pi() / 180) * 1e3)
        else:
            print(f"to_mrad error; unknown angular units {units}")

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

    def plot_proc_shot(self, shot_dict, vmin=None,vmax=None, debug=False):

        espec_img, x_MeV, y_mrad = self.get_proc_shot(shot_dict, debug=debug)

        if not vmin:
            vmin = np.min(espec_img)
        if not vmax:
            vmax = np.max(espec_img)

        fig = plt.figure()
        im = plt.pcolormesh(x_MeV, y_mrad, espec_img, vmin=vmin, vmax=vmax, shading='auto')
        cb = plt.colorbar(im)
        cb.set_label(self.img_units, rotation=270, labelpad=20)
        plt.title(self.shot_string(shot_dict))
        plt.xlabel('Electron energy [MeV]') 
        plt.ylabel('Beam divergence [mrad]')
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