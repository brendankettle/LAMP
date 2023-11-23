import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib
import os
from scipy.interpolate import interp1d

from .diagnostic import Diagnostic
from ..lib.image_proc import ImageProc

# ESpec results class / objects?

class ESpec(Diagnostic):
    """Electron (charged particle?) Spectrometer. This can potentially expand to cover a lot more actions;
        - Montage creation (lib function?)
        - Two screen processing
        - Charge calibration?
        - Calculating trajectories and dispersions?
    """

    # TODO: Background correction?

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''

    # my (BK) thinking is that it is better to keep track of all the different units for the x/y axis
    # also, sticking to the same units (mm/MeV/mrad) helps make it easier to convert from different calibrations and simplify plotting
    # but I'm sure I can can be convinced otehrwise!
    curr_img = None
    img_units = 'Counts'
    x_mm, y_mm = None, None
    x_mrad, y_mrad = None, None
    x_MeV, y_MeV = None, None

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return
    
    def get_shot_data(self, shot_dict):
        """Wrapper for getting shot data through DAQ"""
        return self.DAQ.get_shot_data(self.config['setup']['name'], shot_dict)

    def get_proc_shot(self, shot_dict, calib_id=None):
        """Return a processed shot using saved or passed calibrations.
        """

        # get calibration dictionary
        calib_dict = self.get_calib(calib_id)

        # minimum calibration is spatial transform
        img, x, y = self.transform(shot_dict, calib_dict['transform'])

        # dispersion?
        if 'dispersion' in calib_dict:
            img, MeV = self.apply_dispersion(img, calib_dict)

            # default to applying to X axis unless set
            if 'axis' in calib_dict['dispersion']:
                if calib_dict['dispersion']['axis'].lower() == 'y':
                    axis = 'y'
                else:
                    axis = 'x'
            else:
                axis = 'x'

            # ROI?
            if 'roi_MeV' in calib_dict:
                MeV_min = np.min(calib_dict['roi_MeV'])
                MeV_max = np.max(calib_dict['roi_MeV'])
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
        if 'divergence' in calib_dict:
            img, mrad = self.apply_divergence(img, calib_dict)

            # default to Y axis
            if 'axis' in calib_dict['divergence']:
                if calib_dict['divergence']['axis'].lower() == 'x':
                    axis = 'x'
                else:
                    axis = 'y'
            else:
                axis = 'y'

            # ROI?
            if 'roi_mrad' in calib_dict:
                mrad_min = np.min(calib_dict['roi_mrad'])
                mrad_max = np.max(calib_dict['roi_mrad'])
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

    def get_spectra(self, shot_dict, calib_id=None):

        return
    
    def get_div(self, shot_dict, calib_id=None):

        return

    def make_calib(self, calib_input, calib_filename=None, view=True):
        """Master function for generating a calibration file using a calibration input
            E.g transform, dispersion, etc. 
        """

        # Get and set calibration input 
        calib_input = self.set_calib_input(calib_input)

        # Make a spatial transform
        self.make_transform(view=view)

        # Apply dispersion?
        if 'dispersion' in calib_input:
            self.make_dispersion()

        # Apply divergence?
        if 'divergence' in calib_input:
            self.make_divergence()

        # transfer other values
        save_vars = ['roi_mm','roi_MeV','roi_mrad']
        for var in save_vars:
            if var in calib_input:
                self.calib_dict[var] = calib_input[var]

        # save the full calibration?
        if calib_filename:
            self.save_calib(calib_filename)

        return self.get_calib()

    def make_transform(self, calib_input=None, view=False):
        """Generate a transform dictionary for use with spatially transforming raw shot images.
            This is a wrapper for ImageProc make_transform()

            calib_input: A dictionary containing the required information for the transform, or calibration file/id for loading...
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

        # get dictionary from whatever form of calib_input was passed 
        tcalib_input = self.get_calib_input(calib_input)['transform']

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
        self.calib_dict['transform'] = img.make_transform(p_px, p_t, tcalib_input['img_size_t'], tcalib_input['img_size_px'], 
                                        tcalib_input['offsets'], notes=notes, description=description)
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

    def make_dispersion(self, calib_input=None):
        """"""

        # get dispersion curve from file
        disp_dict = self.get_calib_input(calib_input)['dispersion']
        disp_curve = self.load_calib_file(disp_dict['filename'])

        # TODO: Still asuming its spatial then spectral in data
        m,n = np.shape(disp_curve)
        if m > n:
            disp_spat = disp_curve[:,0]
            disp_spec = disp_curve[:,1]
        else:
            print("BODGING DISPERSION! NEED TO REWRITE THE NUMPY FILES FROM JASON!")
            disp_spat = 0.03 + (0.19 - disp_curve[0,:])
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

        # save details to calib dictionary
        self.calib_dict['dispersion'] = {
            "calib_curve": disp_curve,
            "calib_filename": disp_dict['filename'],
            "calib_spatial_units": spat_units,
            "calib_spectral_units": spec_units,
            "mm": mm,
            "MeV": MeV,
            "axis": axis
        }
            
        return MeV

    def apply_dispersion(self, img_data, calib_id=None, disp_dict=None):
        """"""

        if disp_dict is None:
            disp_dict = self.get_calib(calib_id)['dispersion']

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

    def make_divergence(self, calib_input=None):
        """"""

        div_dict = self.get_calib_input(calib_input)['divergence']
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

        # save details to calib dictionary
        self.calib_dict['divergence'] = {
            "mm_to_screen": mm_to_screen,
            "mm": mm,
            "mrad": mrad,
            "axis": axis
        }
            
        return mrad
    
    def apply_divergence(self, img_data, calib_id=None, div_dict=None):
        """"""

        # either used passed dictionary, or load from ID
        if div_dict is None:
            div_dict = self.get_calib(calib_id)['divergence']
        
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