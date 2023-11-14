import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib
import os

from .diagnostic import Diagnostic
from ..lib.image_proc import ImageProc

# what about calling diagnostic analysis outside of an experiment environment? 

class ESpec(Diagnostic):
    """Electron (charged particle?) Spectrometer
    This can potentially expand to cover a lot more actions;
        - Montage creation (lib function?)
        - Two screen processing
        - Click tools for making spatial calibrations
        - Charge calibration?
        - Even maybe calculating trajectories and dispersions?
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''

    calib_input = {}
    calib_dict = {}
    curr_img = None
    x_mm, y_mm = None, None
    x_mrad, y_mrad = None, None
    x_MeV, y_MeV = None, None

    def __init__(self, exp_obj, config_filepath):
        # Initiate parent base Diagnostic class to get all shared attributes and funcs
        super().__init__(exp_obj, config_filepath)
        return
    
    def get_shot_data(self, shot_dict, process=False):
        """Wrapper for getting shot data through DAQ"""
        if process:
            print('TODO: Need to implement applying calibrations...')
            # Need to load calibration... use shot_dict to determine? OR the contents of "process" variable
        else:
            shot_data = self.DAQ.get_shot_data(self.config['setup']['name'], shot_dict)
        return shot_data

    def make_transform(self, calib_input, view=False):
        """Generate a transform dictionary for use with spatially transforming raw shot images.

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

        # sort calibration input format
        if isinstance(calib_input, dict):
            # passing dictionary directly; This needs to have the required keys!
            self.calib_input['transform'] = calib_input
        else:
            # passing a string, let's look for a file first
            calib_input_filepath = os.path.join(self.ex.config['paths']['calibs_folder'], self.config['setup']['calib_folder'], calib_input)
            if os.path.exists(calib_input_filepath):
                self.calib_input = self.load_calib_file(calib_input_filepath)
            elif 'calib_inputs' in self.config['setup']:
                # or, if it can't be found, look for ID key within the master calibration input file (if set in config)
                calib_input_filepath = os.path.join(self.ex.config['paths']['calibs_folder'], self.config['setup']['calib_folder'], self.config['setup']['calib_inputs'])
                all_cal_dicts = self.load_calib_file(calib_input_filepath)
                if calib_input in all_cal_dicts:
                    self.calib_input = all_cal_dicts[calib_input]
                else:
                    print(f"ESpec Error; make_transform(); No calibration input found for {calib_input}")
            else:
                print(f"ESpec Error; make_transform(); Unknown calibration input found for {calib_input}")
        tcalib_input = self.calib_input['transform']

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
        # if not passed, use stored tform_dict, or complain
        if tform_dict is None:
            if self.calib_dict['transform'] is None:
                print('ESpec Error, transform dictionary needs to be passed or loaded')
                return
        else:
            self.calib_dict['transform'] = tform_dict
        # if img_data is passed as a shot dictionary, grab the actual image
        if isinstance(img_data, dict):
            img_data = self.get_shot_data(img_data)
        img = ImageProc(img_data)
        timg, tx, ty = img.transform(self.calib_dict['transform'])
        # E beam offset? shifts the xy cords on transformed screen
        ex = tx - self.calib_dict['transform']['e_offsets'][0]
        ey = ty - self.calib_dict['transform']['e_offsets'][1]
        return timg, ex, ey

    def apply_dispersion(self, calib_input, axis='x', units='mm'):

        # TODO: ... work out calib_input format... this should be in seperate function?
        calib_filepath = os.path.join(self.ex.config['paths']['calibs_folder'], self.config['setup']['calib_folder'], calib_input)
        disp_curve = self.load_calib_file(calib_filepath)

        # TODO: Units??

        # TODO: Need to convert counts to per MeV?

        if axis.lower() == 'x':
            self.x_MeV = np.interp(self.x_mm/1000, disp_curve[:,0], disp_curve[:,1])
            return self.x_MeV
        elif axis.lower() == 'y':
            self.y_MeV = np.interp(self.y_mm/1000, disp_curve[:,0], disp_curve[:,1])
            return self.y_MeV

    def apply_divergence(self, calib_input, axis='y', units='mm'):

        # TODO: Units??

        # TODO: Need to convert counts to per mrad?

        mrad_per_mm = calib_input['mrad_per_mm']

        if axis.lower() == 'x':
            self.x_mrad = self.x_mm * mrad_per_mm
            return self.x_mrad
        elif axis.lower() == 'y':
            self.y_mrad = self.y_mm * mrad_per_mm
            return self.y_mrad

    def save_calib(self, calib_filename):
        """Save all the current calibration information to file"""
        calib_filepath = os.path.join(self.ex.config['paths']['calibs_folder'], self.config['setup']['calib_folder'], calib_filename)
        self.save_calib_file(calib_filepath, self.calib_dict)
        return

    def load_calib(self, calib_id):
        # TO DO: Pass some time frame and work out calibration file? 
        return
